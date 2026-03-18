"""
Qwen2.5-VL -> Qwen3-VL-30B-A3B-Instruct (MoE) 适配版模型入口
======================================================

本文件遵循“仅做适配 + 保证可运行”的约束：不改动原训练流程/损失/数据处理，只对与 Qwen2.5-VL
实现/接口耦合的部分做系统性对齐，并将你的 evidence 注入模块安全地接入 Qwen3-VL-MoE。

----------------
耦合点清单（审计）
----------------
以下耦合点来自对 Qwen2.5-VL 版本 `train.py` + `qwen_vl_model.py` 的抽取（非穷尽，覆盖关键契约）：

1) 训练脚本 <-> 模型 forward 契约
   - Trainer/Collator 传入字段：`input_ids`, `attention_mask`, `labels`, `pixel_values`, `image_grid_thw`
   - loss 在模型 forward 内部计算（causal LM shift loss），Trainer 直接取 `outputs.loss`
   - 证据正则（lambda_orth/lambda_ctr）在 Trainer.compute_loss 内计算，模型需通过 `outputs.aux`
     暴露每层的中间张量（至少包含 `a` 与 `r`）：
       `outputs.aux: List[Dict]`, 每个元素形如：
       `{layer_idx, e, a, r, u, alpha}`

2) 多模态输入/特征契约（processor/tokenizer/vision）
   - 文本侧由 `processor.apply_chat_template(...)` 产生，其中包含图像占位 token
   - 图像侧由 processor 输出 `pixel_values` 与 `image_grid_thw`：
       pixel_values: `[B, C, H, W]`
       image_grid_thw: `[B, 3]`（每样本 1 图时）或 `[num_images, 3]`（多图时）
   - 占位 token 数量必须与视觉特征 token 数量一致（否则抛错）

3) CLI -> config 的单一事实来源（训练侧写入，模型侧读取）
   - `enable_evidence` -> `config.enable_vision_gate`
   - `gate_layers` -> `config.gate_layers`（None=all, []=none, [idx...]=指定层）
   - `inject_position`: none|first_layer_input|per_layer
   - `inject_op`: ours|add|concat
   - `use_utilization`: True/False
   - `evidence_source`: candidate|aligned
   - Qwen3-VL 系列有 `config.text_config`：上述开关必须同步到 `text_config`（decoder layer 读取的 config）

4) checkpoint 加载/保存契约
   - 使用 HF `from_pretrained(...)` 加载 Qwen3VLMoe 权重
   - 新增模块参数需要稳定命名与清晰注册（便于冻结/统计），允许 missing keys（新模块随机初始化）

5) 分布式/Deepspeed/torchrun 耦合
   - forward 需兼容 `past_key_values: Cache`, `cache_position`, `use_cache` 等生成/加速参数
   - 支持 bf16/fp16 dtype；不在 forward 内强制 dtype 转换（保持 Trainer/Deepspeed 统一控制）

6) MoE 安全插入点
   - 不改动 MoE router/experts 的 forward；证据注入在 decoder layer 的标准 SelfAttn+MLP 残差之后
     以“额外残差”形式生效，避免破坏路由与专家计算逻辑。

7) 行为兼容降级要求
   - 当 `enable_vision_gate=False` 或 `inject_position=='none'`：不注入、不产生 aux（尽量贴近原生 Qwen3VLMoe）
   - 当 `inject_position=='first_layer_input'`：仅在进入 layer0 前做一次注入，并显式关闭后续 per-layer 注入

----------------
实现说明（Qwen3-VL-MoE）
----------------
* 以 `transformers.Qwen3VLMoeForConditionalGeneration` 为基座。
* 替换 `self.model.language_model` 为自定义 TextModel：
  - DecoderLayer 复用 transformers 官方 MoE 层（注意 mlp 是否为 SparseMoEBlock），确保权重键名一致。
  - 在每层增加 `retriever/analyzer/util/corrector/concat_proj` 五个子模块（新增参数清晰可冻结）。
* 为生成阶段提供 v_mem 缓存：HF generate 会在后续 step 不再传 pixel_values，
  若不缓存则新 token 无法继续使用 evidence（本实现仅在 eval/generate 时缓存，避免训练阶段图计算泄露）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import Qwen3VLMoeForConditionalGeneration
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers.integrations import use_experts_implementation
from transformers.integrations.moe import batched_mm_experts_forward
from transformers.utils import logging

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeModel,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextTopKRouter,
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoePreTrainedModel,
)

logger = logging.get_logger(__name__)


# =========================
# MoE experts (checkpoint-compat)
# =========================


@use_experts_implementation(is_transposed=True, has_bias=False)
class Qwen3VLMoeTextExpertsTransposed(nn.Module):
    """Experts weights stored in transposed layout (matches ModelScope ckpt for Qwen3-VL-30B-A3B).

    Note:
      - The upstream Transformers implementation expects weights shaped as:
          gate_up_proj: [E, 2I, H], down_proj: [E, H, I]
      - This checkpoint stores them transposed:
          gate_up_proj: [E, H, 2I], down_proj: [E, I, H]
      - By storing transposed weights and setting `is_transposed=True`, GroupedMM can run without runtime transpose.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # Transposed storage: (in_dim, out_dim)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, 2 * self.intermediate_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        # Eager fallback implementation (kernel backends are dispatched by the decorator)
        return batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)


class Qwen3VLMoeTextSparseMoeBlockTransposed(nn.Module):
    """Sparse MoE block using transposed experts weights (checkpoint-compatible)."""

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.experts = Qwen3VLMoeTextExpertsTransposed(config)
        self.gate = Qwen3VLMoeTextTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


# =========================
# Evidence modules
# =========================


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.w_q(hidden_states)
        k = self.w_k(v_mem)
        v = self.w_v(v_mem)

        attn_logits = torch.matmul(q, k.transpose(-1, -2))
        if v_mem_mask is not None:
            attn_logits = attn_logits.masked_fill(v_mem_mask.unsqueeze(1) == 0, torch.finfo(attn_logits.dtype).min)

        alpha = torch.softmax(attn_logits, dim=-1)
        e = torch.matmul(alpha, v)
        return e, alpha


class EvidenceAnalyzer(nn.Module):
    """Split candidate evidence into aligned (a) and residual (r)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, e: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([e, h], dim=-1)
        a = self.mlp(x)
        r = e - a
        return a, r


class EvidenceUtilization(nn.Module):
    """Estimate utilization strength u in (0, 1) for each token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, h], dim=-1)
        return torch.sigmoid(self.mlp(x))


class EvidenceCorrector(nn.Module):
    """Residual correction: h_tilde = h + u * W_c(a)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h + u * self.w_c(a)


# =========================
# Custom MoE decoder / text model
# =========================


class CustomQwen3VLMoeTextDecoderLayer(nn.Module):
    """Qwen3-VL-MoE decoder layer + evidence modules (safe post-MLP residual injection)."""

    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)

        self.self_attn = Qwen3VLMoeTextAttention(config, layer_idx)

        # Keep the exact MoE/dense selection logic to preserve checkpoint key compatibility.
        if (self.layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # Use checkpoint-compatible experts layout for Qwen3-VL-30B-A3B.
            self.mlp = Qwen3VLMoeTextSparseMoeBlockTransposed(config)
        else:
            self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Evidence modules (new params)
        self.retriever = EvidenceRetriever(config.hidden_size)
        self.analyzer = EvidenceAnalyzer(config.hidden_size)
        self.util = EvidenceUtilization(config.hidden_size)
        self.corrector = EvidenceCorrector(config.hidden_size)
        self.concat_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_mem: Optional[torch.Tensor] = None,
        v_mem_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        aux = None

        enable_vision_gate = bool(getattr(self.config, "enable_vision_gate", True))
        inject_position = str(getattr(self.config, "inject_position", "per_layer")).strip().lower()
        gate_layers = getattr(self.config, "gate_layers", None)  # None -> all, [] -> none

        if (
            enable_vision_gate
            and inject_position == "per_layer"
            and v_mem is not None
            and (gate_layers is None or self.layer_idx in gate_layers)
        ):
            inject_op = str(getattr(self.config, "inject_op", "ours")).strip().lower()
            use_u = bool(getattr(self.config, "use_utilization", True))
            evidence_source = str(getattr(self.config, "evidence_source", "aligned")).strip().lower()

            e_t, alpha = self.retriever(hidden_states, v_mem, v_mem_mask)
            a_t, r_t = self.analyzer(e_t, hidden_states)

            src = e_t if evidence_source == "candidate" else a_t

            u_t = self.util(a_t, hidden_states) if use_u else None

            if inject_op == "ours":
                if u_t is None:
                    hidden_states = hidden_states + self.corrector.w_c(src)
                else:
                    hidden_states = self.corrector(hidden_states, src, u_t)
            elif inject_op == "add":
                delta = self.corrector.w_c(src)
                hidden_states = hidden_states + (delta if u_t is None else u_t * delta)
            elif inject_op == "concat":
                cat = torch.cat([hidden_states, src], dim=-1)
                delta = self.concat_proj(cat)
                hidden_states = hidden_states + (delta if u_t is None else u_t * delta)
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            aux = {
                "e": e_t,
                "a": a_t,
                "r": r_t,
                "u": (
                    u_t
                    if u_t is not None
                    else hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1)
                ),
                "alpha": alpha,
            }

        return hidden_states, aux


class CustomQwen3VLMoeTextModel(Qwen3VLMoePreTrainedModel):
    """Custom text model that collects `aux` for training regularizers."""

    config_class = Qwen3VLMoeTextConfig
    _no_split_modules = ["CustomQwen3VLMoeTextDecoderLayer"]

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Transformers>=5 uses a heuristic to decide if MoE experts implementation can be set.

        Our custom TextModel reuses the official Qwen3-VL-MoE SparseMoeBlock/Experts modules (which already implement
        the experts backends). Override the heuristic so configs that request `experts_implementation=grouped_mm`
        won't fail during init.
        """

        return True

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CustomQwen3VLMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(config=config)
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
        cache_position: Optional[torch.LongTensor] = None,
        # deepstack args (kept for behavior parity with upstream)
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        # evidence memory
        v_mem: Optional[torch.Tensor] = None,
        v_mem_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = bool(use_cache) if use_cache is not None else False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        # hard-coded 3: temporal/height/width
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # Disable per-layer evidence injection if globally disabled or not per-layer.
        if (not bool(getattr(self.config, "enable_vision_gate", True))) or (
            str(getattr(self.config, "inject_position", "per_layer")).strip().lower() != "per_layer"
        ):
            v_mem = None
            v_mem_mask = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_aux: List[Dict[str, torch.Tensor]] = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_hidden, aux = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                v_mem=v_mem,
                v_mem_mask=v_mem_mask,
                **kwargs,
            )
            hidden_states = layer_hidden

            if aux is not None:
                all_aux.append({"layer_idx": getattr(decoder_layer, "layer_idx", layer_idx), **aux})

            # deepstack (upstream behavior)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])

        hidden_states = self.norm(hidden_states)

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        setattr(out, "aux", all_aux)
        return out

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        if visual_pos_masks is None or visual_embeds is None:
            return hidden_states
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states


# =========================
# Custom multimodal base model (avoid double-instantiating the base text model)
# =========================


class Qwen3VLMoeCustomModel(Qwen3VLMoeModel):
    """Same as `transformers.Qwen3VLMoeModel`, but uses `CustomQwen3VLMoeTextModel` from the start.

    Why:
      - If we call `Qwen3VLMoeForConditionalGeneration.__init__` and then replace `self.model.language_model`,
        the original (very large) text model is constructed first.
      - Under Deepspeed ZeRO-3 init, that temporary construction can lead to extreme GPU memory pressure and OOM during
        `_initialize_missing_keys` (especially when many evidence params are missing and need init).
    """

    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock", "CustomQwen3VLMoeTextDecoderLayer"]

    def __init__(self, config: Qwen3VLMoeConfig):
        # Bypass `Qwen3VLMoeModel.__init__` to avoid instantiating the original text model.
        Qwen3VLMoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLMoeVisionModel._from_config(config.vision_config)
        self.language_model = CustomQwen3VLMoeTextModel(config.text_config)
        self.rope_deltas = None

        # Initialize weights and apply final processing (keeps HF conventions)
        self.post_init()


# =========================
# Top-level model
# =========================


class Qwen3VLMoeCustomVLForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    """Qwen3-VL-MoE + evidence modules (API compatible with HF Trainer)."""

    # Keep the upstream tied-weights mapping
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Prevent `device_map=auto` from splitting inside a decoder layer (would break residual adds).
    _no_split_modules = ["Qwen3VLMoeVisionBlock", "Qwen3VLMoeTextDecoderLayer", "CustomQwen3VLMoeTextDecoderLayer"]

    # experiment flags that must be visible in text_config
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

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # See the note in `CustomQwen3VLMoeTextModel._can_set_experts_implementation`.
        return True

    def __init__(self, config: Qwen3VLMoeConfig):
        # Ensure experiment flags propagate to text_config before modules are created.
        self._sync_config_to_text_config(config)

        # IMPORTANT: don't call `Qwen3VLMoeForConditionalGeneration.__init__` (would instantiate the base text model).
        Qwen3VLMoePreTrainedModel.__init__(self, config)
        self.model = Qwen3VLMoeCustomModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

        # Generation-time cache for evidence memory (avoid training graph retention).
        self._cached_v_mem: Optional[torch.Tensor] = None
        self._cached_v_mem_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _sync_config_to_text_config(config: Qwen3VLMoeConfig) -> None:
        if not hasattr(config, "text_config") or config.text_config is None:
            return
        for key in Qwen3VLMoeCustomVLForConditionalGeneration._EXPERIMENT_CONFIG_KEYS:
            if hasattr(config, key):
                setattr(config.text_config, key, getattr(config, key))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load checkpoint with a ZeRO-3 safe workaround.

        Why needed:
          - Transformers 5.0.0 applies *weight conversions* during ZeRO-3 loading via
            `transformers.integrations.deepspeed._apply_weight_conversions_to_state_dict`.
          - The current implementation has a bug in the converter path: when any `WeightConverter` is registered for
            this `model_type`, it can drop non-converted keys (most keys), leading to a LOAD REPORT where *all* base
            weights are marked MISSING (i.e., the model is effectively randomly initialized).

        Workaround (minimal & reversible):
          - During ZeRO-3 loading only, temporarily set `config.model_type` to a dummy value so that
            `get_model_conversion_mapping(...)` does not register the model-specific `WeightConverter` entries.
          - This keeps renamings-only mapping (legacy), which uses the fast path and does not drop keys.
          - After loading, restore `model.config.model_type` to the original value.
        """

        # Default: do NOT ignore mismatched sizes, to avoid silently re-initializing base weights.
        # (New evidence modules are fine: they appear as MISSING keys, not mismatches.)
        kwargs.setdefault("ignore_mismatched_sizes", False)

        # Workaround for ZeRO-3 weight-conversion bug (see docstring above).
        from transformers.integrations import is_deepspeed_zero3_enabled

        orig_model_type = None
        if is_deepspeed_zero3_enabled():
            import copy

            cfg = kwargs.get("config", None)
            if cfg is None:
                cfg = cls.config_class.from_pretrained(pretrained_model_name_or_path)
            cfg = copy.deepcopy(cfg)
            orig_model_type = getattr(cfg, "model_type", None)
            cfg.model_type = "qwen3_vl_moe_no_weight_converters"
            kwargs["config"] = cfg

        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        load_info = None
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, load_info = loaded
        else:
            model = loaded

        if orig_model_type is not None:
            model.config.model_type = orig_model_type

        # Sync loaded config into text_config and the instantiated language_model.config
        cls._sync_config_to_text_config(model.config)
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
            for key in cls._EXPERIMENT_CONFIG_KEYS:
                if hasattr(model.config, key):
                    setattr(model.model.language_model.config, key, getattr(model.config, key))
        return (model, load_info) if load_info is not None else model

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """ZeRO-3 friendly missing-keys init.

        HF default behavior gathers *all* not-initialized params at once under ZeRO-3, which can OOM when the set of
        missing params is large (our evidence modules add several Linear weights per layer).

        We initialize evidence-module params layer-by-layer to keep peak gathered memory low.
        """

        from transformers.integrations import is_deepspeed_zero3_enabled

        if (not is_deepspeed_zero3_enabled()) or bool(is_quantized):
            return super()._initialize_missing_keys(is_quantized)

        import deepspeed
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0

        evidence_attrs = ("retriever", "analyzer", "util", "corrector", "concat_proj")

        # Init only evidence params (missing from base ckpt) in small chunks.
        for layer in getattr(self.model.language_model, "layers", []):
            for attr in evidence_attrs:
                mod = getattr(layer, attr, None)
                if mod is None:
                    continue
                params = [p for p in mod.parameters() if not getattr(p, "_is_hf_initialized", False)]
                if len(params) == 0:
                    continue

                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if rank == 0:
                        mod.apply(self._init_weights)
                for p in params:
                    p._is_hf_initialized = True

        # NOTE: Under ZeRO-3, some backends may drop arbitrary tensor attributes (like `_is_hf_initialized`) during
        # partition/gather cycles. We therefore avoid asserting on that flag here; true missing-keys are still reported
        # by HF's load report right after this hook returns.

    def _build_vmem_from_features(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        image_embeds_list: Optional[Tuple[torch.Tensor, ...]],
        video_embeds_list: Optional[Tuple[torch.Tensor, ...]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build padded v_mem/v_mem_mask per sample from per-image/video token embeddings."""

        if image_embeds_list is None and video_embeds_list is None:
            return None, None
        if input_ids is None:
            # Fallback: assume 1 image per sample if shapes allow.
            if image_embeds_list is not None:
                bsz = len(image_embeds_list)
            elif video_embeds_list is not None:
                bsz = len(video_embeds_list)
            else:
                return None, None
            per_sample = []
            for i in range(bsz):
                chunks = []
                if image_embeds_list is not None and i < len(image_embeds_list):
                    chunks.append(image_embeds_list[i])
                if video_embeds_list is not None and i < len(video_embeds_list):
                    chunks.append(video_embeds_list[i])
                if len(chunks) == 0:
                    per_sample.append(None)
                else:
                    per_sample.append(torch.cat(chunks, dim=0))
            return self._pad_vmem(per_sample)

        bsz = input_ids.shape[0]
        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)
        vision_start_token_id = getattr(self.config, "vision_start_token_id", None)

        # Infer how many images/videos per sample by scanning <vision_start> blocks (same strategy as upstream).
        image_counts = [0] * bsz
        video_counts = [0] * bsz
        if vision_start_token_id is not None and image_token_id is not None and video_token_id is not None:
            for i in range(bsz):
                ids = input_ids[i]
                if attention_mask is not None:
                    ids = ids[attention_mask[i] == 1]
                start_idx = torch.argwhere(ids == vision_start_token_id).squeeze(1)
                if start_idx.numel() == 0:
                    continue
                # token right after <vision_start> indicates modality type
                nxt = start_idx + 1
                nxt = nxt[nxt < ids.numel()]
                vision_tokens = ids[nxt]
                image_counts[i] = int((vision_tokens == image_token_id).sum().item())
                video_counts[i] = int((vision_tokens == video_token_id).sum().item())

        # Fallbacks for common single-image-per-sample training data
        total_images = len(image_embeds_list) if image_embeds_list is not None else 0
        total_videos = len(video_embeds_list) if video_embeds_list is not None else 0
        if total_images > 0 and sum(image_counts) != total_images:
            if total_images == bsz:
                image_counts = [1] * bsz
            else:
                # best-effort: assign all remaining to first sample
                image_counts = [total_images] + [0] * (bsz - 1)
        if total_videos > 0 and sum(video_counts) != total_videos:
            if total_videos == bsz:
                video_counts = [1] * bsz
            else:
                video_counts = [total_videos] + [0] * (bsz - 1)

        per_sample: List[Optional[torch.Tensor]] = []
        img_ptr = 0
        vid_ptr = 0
        for i in range(bsz):
            chunks = []
            n_img = image_counts[i]
            n_vid = video_counts[i]
            if image_embeds_list is not None and n_img > 0:
                imgs = image_embeds_list[img_ptr : img_ptr + n_img]
                img_ptr += n_img
                chunks.append(torch.cat(list(imgs), dim=0))
            if video_embeds_list is not None and n_vid > 0:
                vids = video_embeds_list[vid_ptr : vid_ptr + n_vid]
                vid_ptr += n_vid
                chunks.append(torch.cat(list(vids), dim=0))
            if len(chunks) == 0:
                per_sample.append(None)
            else:
                per_sample.append(torch.cat(chunks, dim=0))

        return self._pad_vmem(per_sample)

    @staticmethod
    def _pad_vmem(per_sample: List[Optional[torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(per_sample) == 0:
            return None, None
        # Determine device/dtype from first non-empty chunk
        ref = next((x for x in per_sample if x is not None and x.numel() > 0), None)
        if ref is None:
            return None, None
        bsz = len(per_sample)
        hidden = ref.shape[-1]
        lengths = [int(x.shape[0]) if x is not None else 0 for x in per_sample]
        max_n = max(lengths) if lengths else 0
        v_mem = ref.new_zeros((bsz, max_n, hidden))
        v_mask = torch.zeros((bsz, max_n), device=ref.device, dtype=torch.long)
        for i, x in enumerate(per_sample):
            if x is None:
                continue
            n = int(x.shape[0])
            if n <= 0:
                continue
            v_mem[i, :n, :] = x.to(device=ref.device, dtype=ref.dtype)
            v_mask[i, :n] = 1
        return v_mem, v_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[Tuple, Qwen3VLMoeCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        image_embeds_list = None
        video_embeds_list = None

        # ===== vision features (upstream behavior) =====
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_outputs = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds_list = tuple(image_outputs.pooler_output)
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds_list = tuple(video_outputs.pooler_output)
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ===== deepstack aggregation (upstream behavior) =====
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # ===== build v_mem =====
        enable_vision_gate = bool(getattr(self.config, "enable_vision_gate", True))
        inject_position = str(getattr(self.config, "inject_position", "per_layer")).strip().lower()

        v_mem = None
        v_mem_mask = None

        if enable_vision_gate and inject_position != "none":
            v_mem, v_mem_mask = self._build_vmem_from_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds_list=image_embeds_list,
                video_embeds_list=video_embeds_list,
            )

        # Generation-time cache: reuse v_mem when pixel_values is dropped by HF generate.
        if not self.training:
            if v_mem is not None:
                self._cached_v_mem = v_mem.detach()
                self._cached_v_mem_mask = v_mem_mask.detach() if v_mem_mask is not None else None
            elif self._cached_v_mem is not None and enable_vision_gate and inject_position != "none":
                v_mem = self._cached_v_mem
                v_mem_mask = self._cached_v_mem_mask

        # ===== position ids (upstream behavior) =====
        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.model.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.model.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ===== first_layer_input injection (+M) =====
        if inject_position == "first_layer_input" and enable_vision_gate and v_mem is not None:
            layer0 = self.model.language_model.layers[0]
            inject_op = str(getattr(self.config, "inject_op", "add")).strip().lower()
            use_u = bool(getattr(self.config, "use_utilization", False))
            evidence_source = str(getattr(self.config, "evidence_source", "candidate")).strip().lower()

            e_t, alpha = layer0.retriever(inputs_embeds, v_mem, v_mem_mask)
            a_t, r_t = layer0.analyzer(e_t, inputs_embeds)
            src = e_t if evidence_source == "candidate" else a_t
            u_t = layer0.util(a_t, inputs_embeds) if use_u else None

            if inject_op == "ours":
                inputs_embeds = inputs_embeds + layer0.corrector.w_c(src) if u_t is None else layer0.corrector(inputs_embeds, src, u_t)
            elif inject_op == "add":
                delta = layer0.corrector.w_c(src)
                inputs_embeds = inputs_embeds + (delta if u_t is None else u_t * delta)
            elif inject_op == "concat":
                cat = torch.cat([inputs_embeds, src], dim=-1)
                delta = layer0.concat_proj(cat)
                inputs_embeds = inputs_embeds + (delta if u_t is None else u_t * delta)
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            # Disable per-layer injection afterwards (match Qwen2.5-VL behavior)
            v_mem = None
            v_mem_mask = None

        # ===== language model forward =====
        text_outputs: BaseModelOutputWithPast = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            v_mem=v_mem,
            v_mem_mask=v_mem_mask,
            **kwargs,
        )

        hidden_states = text_outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            # router_logits is recorded by the transformers OutputRecorder mechanism
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import load_balancing_loss_func

            router_logits = getattr(text_outputs, "router_logits", None)
            aux_loss = load_balancing_loss_func(
                router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        out = Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=text_outputs.past_key_values,
            hidden_states=getattr(text_outputs, "hidden_states", None),
            attentions=getattr(text_outputs, "attentions", None),
            rope_deltas=self.model.rope_deltas,
        )

        # Expose aux for Trainer-side regularizers (lambda_orth/lambda_ctr)
        setattr(out, "aux", getattr(text_outputs, "aux", None))
        return out


# Backward-compat export name (optional)
Qwen3VLMoe_CustomVLForConditionalGeneration = Qwen3VLMoeCustomVLForConditionalGeneration
