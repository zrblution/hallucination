"""
Qwen3-VL 自定义模型，用于视觉证据记忆（VEM）训练
适配 transformers >= 4.57.0 的 Qwen3VLForConditionalGeneration
"""
from typing import List, Optional, Tuple, Union, Type
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Qwen3-VL 模块导入
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLPreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
import os
from safetensors.torch import load_file, save_file
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from torch.nn import LayerNorm

logger = logging.get_logger(__name__)


class FixedMerger(nn.Module):
    """
    修复merger的维度不匹配问题
    在forward中添加投影层，将visual blocks的输出投影到merger期望的输入
    """
    def __init__(self, original_merger, input_dim, target_dim):
        super().__init__()
        self.original_merger = original_merger
        self.proj = nn.Linear(input_dim, target_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.original_merger.ln_q.weight.shape[0]:
            x = self.proj(x)
        return self.original_merger(x)


class EvidenceRetriever(nn.Module):
    """Cross-attention retrieval from visual evidence memory."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_mem: torch.Tensor,
        v_mem_mask: Optional[torch.Tensor] = None,
    ):
        q = self.w_q(hidden_states)
        k = self.w_k(v_mem)
        v = self.w_v(v_mem)

        attn_logits = torch.matmul(q, k.transpose(-1, -2))

        if v_mem_mask is not None:
            mask = v_mem_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(
                mask == 0, torch.finfo(attn_logits.dtype).min
            )

        alpha = torch.softmax(attn_logits, dim=-1)
        e = torch.matmul(alpha, v)
        return e, alpha


class EvidenceAnalyzer(nn.Module):
    """Query-conditioned analysis: split candidate evidence into aligned (a) and residual (r)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, e: torch.Tensor, h: torch.Tensor):
        x = torch.cat([e, h], dim=-1)
        a = self.mlp(x)
        r = e - a
        return a, r


class EvidenceUtilization(nn.Module):
    """Estimate utilization strength u in (0,1) for each token."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, a: torch.Tensor, h: torch.Tensor):
        x = torch.cat([a, h], dim=-1)
        u = torch.sigmoid(self.mlp(x))
        return u


class EvidenceCorrector(nn.Module):
    """Residual decoding correction: h_tilde = h + u * W_c(a)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, a: torch.Tensor, u: torch.Tensor):
        return h + u * self.w_c(a)


class CustomQwen3VLTextDecoderLayer(nn.Module):
    """
    自定义的 Qwen3-VL 文本解码器层，添加视觉证据推理模块
    """
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        # 注意：enable_vision_gate 和 gate_layers 不再缓存为实例变量
        # 而是在 forward 中动态从 self.config 读取，以支持配置的后期更新

        # 标准 Qwen3-VL 组件
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 视觉证据推理模块
        self.retriever = EvidenceRetriever(config.hidden_size)
        self.analyzer = EvidenceAnalyzer(config.hidden_size)
        self.util = EvidenceUtilization(config.hidden_size)
        self.corrector = EvidenceCorrector(config.hidden_size)
        self.concat_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mem: Optional[torch.Tensor] = None,
        v_mem_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 视觉证据推理
        # 动态从 config 读取配置，支持配置的后期更新（如 from_pretrained 后同步配置）
        enable_vision_gate = getattr(self.config, "enable_vision_gate", True)
        gate_layers = getattr(self.config, "gate_layers", None)
        
        aux = None
        if (
            enable_vision_gate
            and v_mem is not None
            and (gate_layers is None or self.layer_idx in gate_layers)
        ):
            inject_op = getattr(self.config, "inject_op", "ours").lower()
            use_u = getattr(self.config, "use_utilization", True)
            evidence_source = getattr(self.config, "evidence_source", "aligned").lower()

            e_t, alpha = self.retriever(hidden_states, v_mem, v_mem_mask)
            a_t, r_t = self.analyzer(e_t, hidden_states)

            src = e_t if evidence_source == "candidate" else a_t

            if use_u:
                u_t = self.util(a_t, hidden_states)
            else:
                u_t = None

            if inject_op == "ours":
                if u_t is None:
                    hidden_states = hidden_states + self.corrector.w_c(src)
                else:
                    hidden_states = self.corrector(hidden_states, src, u_t)
            elif inject_op == "add":
                delta = self.corrector.w_c(src)
                if u_t is None:
                    hidden_states = hidden_states + delta
                else:
                    hidden_states = hidden_states + u_t * delta
            elif inject_op == "concat":
                cat = torch.cat([hidden_states, src], dim=-1)
                delta = self.concat_proj(cat)
                if u_t is None:
                    hidden_states = hidden_states + delta
                else:
                    hidden_states = hidden_states + u_t * delta
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            aux = {
                "e": e_t,
                "a": a_t,
                "r": r_t,
                "u": u_t if u_t is not None else hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1),
                "alpha": alpha,
            }

        outputs = (hidden_states, aux)
        return outputs


class CustomQwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    自定义的 Qwen3-VL 文本模型，使用自定义解码器层
    """
    config_class = Qwen3VLTextConfig

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CustomQwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_mem: Optional[torch.Tensor] = None,
        v_mem_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 处理 4D position_ids（Qwen3-VL 使用 [text, temporal, height, width] 4个维度）
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]  # 变成 [3, batch, seq_len] 用于 rotary_emb
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        if not getattr(self.config, "enable_vision_gate", True) or getattr(self.config, "inject_position", "per_layer") == "none":
            v_mem = None
            v_mem_mask = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_aux = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                v_mem=v_mem,
                v_mem_mask=v_mem_mask,
            )

            hidden_states = layer_outputs[0]

            if layer_outputs[1] is not None:
                all_aux.append({"layer_idx": getattr(decoder_layer, "layer_idx", None), **layer_outputs[1]})

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        setattr(out, "aux", all_aux)
        return out

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3VL."
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=getattr(self.config, "sliding_window", None),
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2_5_CustomVLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    自定义的 Qwen3-VL 条件生成模型，添加视觉证据记忆（VEM）支持
    类名保持为 Qwen2_5_CustomVLForConditionalGeneration 以兼容现有训练代码
    """
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    # 需要从主 config 同步到 text_config 的实验开关
    _EXPERIMENT_CONFIG_KEYS = [
        "enable_vision_gate",
        "gate_layers", 
        "inject_position",
        "inject_op",
        "use_utilization",
        "evidence_source",
        "export_u_stats",
        "export_u_stats_path",
    ]

    def __init__(self, config: Qwen3VLConfig):
        # 在调用 super().__init__ 之前，先同步配置到 text_config
        self._sync_config_to_text_config(config)
        super().__init__(config)
        # 替换文本模型为自定义版本（只替换语言模型部分，保留视觉模型）
        self.model.language_model = CustomQwen3VLTextModel(config.text_config)
        
        # 用于在 generate 过程中缓存 v_mem，确保自回归生成时新模块能持续生效
        self._cached_v_mem = None
        self._cached_v_mem_mask = None

    @staticmethod
    def _sync_config_to_text_config(config: Qwen3VLConfig):
        """将主 config 中的实验开关同步到 text_config，确保 decoder layer 能读取到"""
        if not hasattr(config, 'text_config') or config.text_config is None:
            return
        
        for key in Qwen2_5_CustomVLForConditionalGeneration._EXPERIMENT_CONFIG_KEYS:
            if hasattr(config, key):
                setattr(config.text_config, key, getattr(config, key))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs['ignore_mismatched_sizes'] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 加载后再次同步配置（确保从 checkpoint 加载的配置也能正确传递）
        cls._sync_config_to_text_config(model.config)
        
        # 同步到已创建的 language_model 的 config
        if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'config'):
            for key in cls._EXPERIMENT_CONFIG_KEYS:
                if hasattr(model.config, key):
                    setattr(model.model.language_model.config, key, getattr(model.config, key))
        
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        v_mem = None
        v_mem_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.dtype)
                image_embeds, _ = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                # 构造 Visual Evidence Memory (VEM)
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required to build visual evidence memory")

                thw = image_grid_thw.to(inputs_embeds.device)
                n_tokens_per_sample = (thw[:, 0] * thw[:, 1] * thw[:, 2]).tolist()

                splits = []
                start = 0
                for n_b in n_tokens_per_sample:
                    n_b = int(n_b)
                    end = start + n_b
                    splits.append(image_embeds[start:end, :])
                    start = end

                max_n = max([s.shape[0] for s in splits]) if len(splits) > 0 else 0
                bsz = len(splits)
                d = image_embeds.shape[-1]

                v_mem = image_embeds.new_zeros((bsz, max_n, d))
                v_mem_mask = image_embeds.new_zeros((bsz, max_n), dtype=torch.long)

                for i, s in enumerate(splits):
                    n = s.shape[0]
                    if n > 0:
                        v_mem[i, :n, :] = s
                        v_mem_mask[i, :n] = 1

                v_mem = v_mem.to(dtype=inputs_embeds.dtype)
            else:
                v_mem = None
                v_mem_mask = None

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
                video_embeds, _ = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # first_layer_input injection
        inject_position = getattr(self.config, "inject_position", "per_layer").lower()
        if inject_position == "first_layer_input" and v_mem is not None and getattr(self.config, "enable_vision_gate", True):
            layer0 = self.model.language_model.layers[0]
            inject_op = getattr(self.config, "inject_op", "add").lower()
            use_u = getattr(self.config, "use_utilization", False)
            evidence_source = getattr(self.config, "evidence_source", "candidate").lower()

            e_t, alpha = layer0.retriever(inputs_embeds, v_mem, v_mem_mask)
            a_t, r_t = layer0.analyzer(e_t, inputs_embeds)

            src = e_t if evidence_source == "candidate" else a_t

            if use_u:
                u_t = layer0.util(a_t, inputs_embeds)
            else:
                u_t = None

            if inject_op == "ours":
                if u_t is None:
                    inputs_embeds = inputs_embeds + layer0.corrector.w_c(src)
                else:
                    inputs_embeds = layer0.corrector(inputs_embeds, src, u_t)
            elif inject_op == "add":
                delta = layer0.corrector.w_c(src)
                if u_t is None:
                    inputs_embeds = inputs_embeds + delta
                else:
                    inputs_embeds = inputs_embeds + u_t * delta
            elif inject_op == "concat":
                if not hasattr(layer0, "concat_proj"):
                    raise ValueError("concat_proj not found in layer0")
                cat = torch.cat([inputs_embeds, src], dim=-1)
                delta = layer0.concat_proj(cat)
                if u_t is None:
                    inputs_embeds = inputs_embeds + delta
                else:
                    inputs_embeds = inputs_embeds + u_t * delta
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            v_mem = None
            v_mem_mask = None

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            v_mem=v_mem,
            v_mem_mask=v_mem_mask,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            vocab_size = getattr(self.config, "vocab_size", None)
            if vocab_size is None and hasattr(self.config, "text_config"):
                vocab_size = getattr(self.config.text_config, "vocab_size", None)
            if vocab_size is None:
                raise AttributeError("Config missing vocab_size for loss computation.")
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # NOTE: aux stats are attached to the inner text-model outputs (BaseModelOutputWithPast)
        # in CustomQwen3VLTextModel.forward(). We MUST propagate it to the outer
        # Qwen3VLCausalLMOutputWithPast; otherwise Trainer-side getattr(outputs, "aux", None)
        # will be None and regularizers (L_orth/L_ctr) will be silently skipped.
        aux = getattr(outputs, "aux", None)

        if not return_dict:
            output = (logits,) + outputs[1:]
            # (Optional) if you rely on tuple outputs, you can append aux here.
            # However, the EvidenceTrainer expects return_dict=True to access outputs.aux.
            return (loss,) + output if loss is not None else output

        final_out = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )
        if aux is not None:
            setattr(final_out, "aux", aux)
        return final_out
