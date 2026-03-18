import os
import sys
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers import PreTrainedTokenizerBase

from qwen_vl_utils import process_vision_info

# 添加父目录到 Python 路径，以便正确导入 model 模块（与 train_qwen_modified.py 保持一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.qwen_vl_moe_model import Qwen3VLMoeCustomVLForConditionalGeneration


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./basemodel")
    lambda_orth: float = field(default=0.0, metadata={"help": "Weight for L_orth regularizer."})
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for L_ctr regularizer."})
    tau: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss."})
    aux_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated layer indices to regularize; None means all layers returned."},
    )

    # ===== Experiment control (must be CLI-driven) =====
    enable_evidence: bool = field(
        default=True,
        metadata={"help": "Enable evidence modules (retrieval/analysis/util/correction). False -> Base."},
    )

    # Layer selection for per-layer injection: "all" / "none" / "0,1,2"
    gate_layers: str = field(
        default="all",
        metadata={"help": "Which decoder layers apply evidence injection: all|none|comma-separated indices."},
    )

    # Injection position: per_layer -> +M+A/Ours; first_layer_input -> +M; none -> Base
    inject_position: str = field(
        default="per_layer",
        metadata={"help": "Evidence injection position: none|first_layer_input|per_layer."},
    )

    # Injection operator for per-layer (and first-layer input): ours|add|concat
    inject_op: str = field(
        default="ours",
        metadata={"help": "Evidence injection operator: ours|add|concat."},
    )

    # Whether to use utilization strength u (sigmoid MLP). +M requires False.
    use_utilization: bool = field(
        default=True,
        metadata={"help": "Use utilization strength u. False -> unweighted injection."},
    )

    # Evidence source for injection: candidate(e) or aligned(a)
    evidence_source: str = field(
        default="aligned",
        metadata={"help": "Which evidence vector to inject: candidate|aligned."},
    )

    # Export case-study stats (layer-wise mean u)
    export_u_stats: bool = field(
        default=False,
        metadata={"help": "Export layer-wise mean utilization strength u during training."},
    )
    export_u_stats_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output path (json) for exported u stats. Default: <output_dir>/u_stats.json"},
    )

    # ===== Fine-tuning control =====
    finetune_type: str = field(
        default="full",
        metadata={"help": "Finetune strategy: full|lora."},
    )

    freeze_base_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze base model params (only effective when finetune_type=full; "
                "LoRA mode always freezes base weights)."
            )
        },
    )

    train_evidence_modules: bool = field(
        default=True,
        metadata={"help": "If freeze_base_model=True, train evidence modules (retriever/analyzer/util/corrector/concat_proj)."},
    )

    # ===== LoRA config (effective when finetune_type=lora) =====
    lora_r: int = field(default=8, metadata={"help": "LoRA rank r."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules (comma-separated) or 'auto'."},
    )

    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias: none|all|lora_only."},
    )

    lora_modules_to_save: str = field(
        default="retriever,analyzer,util,corrector,concat_proj",
        metadata={"help": "PEFT modules_to_save (comma-separated) or 'none'."},
    )


@dataclass
class DataArguments:
    training_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    training_image_dir: str = field(default=None, metadata={"help": "Path to the image directory."})


class MultiModalCollator:
    """实时处理图像的 Collator，避免预保存大量 pixel_values（与 train_qwen_modified.py 保持一致）"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, processor, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text_features = []
        for i, f in enumerate(features):
            if not isinstance(f.get("input_ids"), list):
                print(f"[DEBUG] feature index={i} has non-list input_ids:", f["input_ids"])
            if not isinstance(f.get("attention_mask"), list):
                print(f"[DEBUG] feature index={i} has non-list attention_mask:", f["attention_mask"])
            if not isinstance(f.get("labels"), list):
                print(f"[DEBUG] feature index={i} has non-list labels:", f["labels"])

            text_features.append(
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"], "labels": f["labels"]}
            )

        try:
            batch_text = self.tokenizer.pad(text_features, padding=True, return_tensors="pt")
        except Exception as e:
            print("\n[ERROR] tokenizer.pad(...) failed. Below is the text_features content:\n")
            for i, tf in enumerate(text_features):
                print(f"  === Sample {i} ===")
                print("  input_ids:", tf["input_ids"])
                print("  attention_mask:", tf["attention_mask"])
                print("  labels:", tf["labels"])
                print("  ----------------")
            raise e

        pixel_values_list = []
        image_grid_thw_list = []

        for f in features:
            img_path = os.path.join(self.data_args.training_image_dir, f["img"])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            image_pil = Image.open(img_path).convert("RGB")
            fixed_size = (224, 224)
            image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

            user_text = f["user_text"]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                            "resize_height": 224,
                            "resize_width": 224,
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                do_resize=True,
                padding=True,
            )

            pv = inputs["pixel_values"]
            if pv.dim() == 4 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            pixel_values_list.append(pv)

            gthw = inputs["image_grid_thw"]
            if gthw.dim() == 2 and gthw.shape[0] == 1:
                gthw = gthw.squeeze(0)
            elif gthw.dim() == 1:
                pass
            else:
                gthw = gthw.squeeze()
            image_grid_thw_list.append(gthw)

        pixel_values = torch.stack(pixel_values_list, dim=0)
        image_grid_thw = torch.stack(image_grid_thw_list, dim=0)

        return {
            "input_ids": batch_text["input_ids"],
            "attention_mask": batch_text["attention_mask"],
            "labels": batch_text["labels"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def process_func(example, data_args, tokenizer, processor, max_length=32000):
    """轻量级预处理：只保存图片路径和文本信息，不处理图像（与 train_qwen_modified.py 保持一致）"""

    user_text = example["text"] + "\n"
    label_text = example["labels"]

    img_path = os.path.join(data_args.training_image_dir, example["img"])
    image_pil = Image.open(img_path).convert("RGB")
    fixed_size = (224, 224)
    image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "resize_height": 224,
                    "resize_width": 224,
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    text_input_ids = inputs["input_ids"][0].tolist()

    response = tokenizer([label_text], add_special_tokens=False)

    input_ids = text_input_ids + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(text_input_ids) + response["input_ids"][0] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "img": example["img"],
        "user_text": user_text,
    }


def _parse_gate_layers(spec: str):
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    if spec == "none":
        return []
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out


def _parse_csv_str(spec: Optional[str]) -> List[str]:
    if spec is None:
        return []
    spec = str(spec).strip()
    if spec == "":
        return []
    return [x.strip() for x in spec.split(",") if x.strip() != ""]


def _unwrap_peft_model(m):
    if hasattr(m, "base_model") and hasattr(getattr(m, "base_model"), "model"):
        return m.base_model.model
    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass
    return m


_EVIDENCE_MODULE_ATTRS = ["retriever", "analyzer", "util", "corrector", "concat_proj"]


def _get_text_layers(m):
    base = _unwrap_peft_model(m)
    if hasattr(base, "model") and hasattr(base.model, "language_model") and hasattr(base.model.language_model, "layers"):
        return base.model.language_model.layers
    raise AttributeError("Cannot locate text decoder layers at: model.model.language_model.layers")


def _infer_active_evidence_layers(cfg, n_layers: int) -> List[int]:
    if not bool(getattr(cfg, "enable_vision_gate", True)):
        return []
    inject_position = str(getattr(cfg, "inject_position", "per_layer")).strip().lower()
    if inject_position == "first_layer_input":
        return [0] if n_layers > 0 else []
    if inject_position == "per_layer":
        gate_layers = getattr(cfg, "gate_layers", None)
        if gate_layers is None:
            return list(range(n_layers))
        if isinstance(gate_layers, (list, tuple, set)):
            out = []
            for x in gate_layers:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if 0 <= xi < n_layers:
                    out.append(xi)
            return sorted(set(out))
        return list(range(n_layers))
    return []


def _set_evidence_trainable(m, active_layers: List[int], enabled: bool = True):
    layers = _get_text_layers(m)
    active = set(int(i) for i in active_layers)
    for i, layer in enumerate(layers):
        flag = bool(enabled) and (i in active)
        for attr in _EVIDENCE_MODULE_ATTRS:
            mod = getattr(layer, attr, None)
            if mod is None:
                continue

            # PEFT ModulesToSaveWrapper compatibility (no hard dependency on PEFT)
            modules_to_save = getattr(mod, "modules_to_save", None)
            if modules_to_save is not None and hasattr(modules_to_save, "items"):
                for _, sub in modules_to_save.items():
                    for p in sub.parameters():
                        p.requires_grad = flag
                if hasattr(mod, "original_module") and getattr(mod, "original_module") is not None:
                    for p in mod.original_module.parameters():
                        p.requires_grad = False
            else:
                for p in mod.parameters():
                    p.requires_grad = flag


def _freeze_all_params(m):
    for p in m.parameters():
        p.requires_grad = False


def _count_trainable_params(m):
    base = _unwrap_peft_model(m)
    trainable = 0
    total = 0
    for p in base.parameters():
        # DeepSpeed ZeRO-3 can leave parameters partitioned/unavailable on a given rank
        # such that p.numel() returns 0. In that case, prefer the original element
        # count exposed by DeepSpeed (ds_numel) so we can correctly detect that
        # trainable params exist and avoid false negatives.
        n = p.numel()
        if n == 0 and hasattr(p, "ds_numel"):
            try:
                n = int(getattr(p, "ds_numel"))
            except Exception:
                pass
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _infer_lora_targets_from_model(m) -> List[str]:
    common = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj"}

    found_common = set()
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in common:
                found_common.add(leaf)

    if len(found_common) > 0:
        return sorted(found_common)

    leaves = set()
    for name, module in m.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(x in name for x in _EVIDENCE_MODULE_ATTRS):
            continue
        leaf = name.split(".")[-1]
        leaves.add(leaf)
    return sorted(leaves)


class EvidenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        aux = getattr(outputs, "aux", None)
        if aux is None:
            return (loss, outputs) if return_outputs else loss

        aux_layers = None
        if getattr(self.args, "aux_layers", None) is not None:
            try:
                aux_layers = set(int(x.strip()) for x in self.args.aux_layers.split(",") if x.strip() != "")
            except Exception:
                aux_layers = None

        lambda_orth = getattr(self.args, "lambda_orth", 0.0)
        if lambda_orth and lambda_orth > 0:
            l_orth = 0.0
            count = 0
            for item in aux:
                if aux_layers is not None and item.get("layer_idx", None) not in aux_layers:
                    continue
                a = item.get("a", None)
                r = item.get("r", None)
                if a is None or r is None:
                    continue

                eps = 1e-6
                num = (a * r).sum(dim=-1)
                den = (a.norm(p=2, dim=-1) * r.norm(p=2, dim=-1) + eps)
                cos = (num / den).abs().mean()

                l_orth = l_orth + cos
                count += 1

            if count > 0:
                loss = loss + lambda_orth * (l_orth / count)

        lambda_ctr = getattr(self.args, "lambda_ctr", 0.0)
        tau = getattr(self.args, "tau", 0.07)
        if lambda_ctr and lambda_ctr > 0:
            pooled = []
            for item in aux:
                if aux_layers is not None and item.get("layer_idx", None) not in aux_layers:
                    continue
                a = item.get("a", None)
                if a is None:
                    continue
                pooled.append(a.mean(dim=1))  # [B,T,D] -> [B,D]

            if len(pooled) > 0:
                z = torch.stack(pooled, dim=0).mean(dim=0)
                z = torch.nn.functional.normalize(z, p=2, dim=-1)
                sim = torch.matmul(z, z.transpose(0, 1)) / tau
                labels = torch.arange(sim.size(0), device=sim.device)
                l_ctr = torch.nn.functional.cross_entropy(sim, labels)
                loss = loss + lambda_ctr * l_ctr

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # ========= One-time freeze/update sanity (ZeRO-3 friendly) =========
        if self.state.global_step == 0 and not hasattr(self, "_update_sanity_cached"):
            self._update_sanity_cached = True

            freeze_base_model = bool(getattr(self.args, "freeze_base_model", False))
            finetune_type = str(getattr(self.args, "finetune_type", "full")).strip().lower()
            train_evidence_modules = bool(getattr(self.args, "train_evidence_modules", True))

            # Find a representative trainable param to verify it changes after the first optimizer step.
            probe_name = None
            probe_param = None

            def _pick_probe(predicate):
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if predicate(n.lower()):
                        return n, p
                return None, None

            if train_evidence_modules:
                probe_name, probe_param = _pick_probe(lambda n_low: any(x in n_low for x in _EVIDENCE_MODULE_ATTRS))
            if probe_param is None and finetune_type == "lora":
                probe_name, probe_param = _pick_probe(lambda n_low: "lora" in n_low)
            if probe_param is None:
                probe_name, probe_param = _pick_probe(lambda _: True)

            base_param = None
            if freeze_base_model:
                try:
                    underlying = model.module if hasattr(model, "module") else model
                    base = _unwrap_peft_model(underlying)
                    base_param = base.model.language_model.layers[0].input_layernorm.weight
                except Exception:
                    base_param = None

            def _gather_to_cpu(param):
                try:
                    import torch.distributed as dist
                    from transformers.integrations import is_deepspeed_zero3_enabled

                    if param is None:
                        return None
                    if bool(is_deepspeed_zero3_enabled()) and dist.is_initialized():
                        import deepspeed

                        dist.barrier()
                        with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                            if getattr(self, "is_world_process_zero", lambda: True)():
                                out = param.detach().cpu().to(torch.float32).clone()
                            else:
                                out = None
                        dist.barrier()
                        return out
                    if getattr(self, "is_world_process_zero", lambda: True)():
                        return param.detach().cpu().to(torch.float32).clone()
                    return None
                except Exception:
                    return None

            self._update_sanity_base_param = base_param
            self._update_sanity_probe_name = probe_name
            self._update_sanity_probe_param = probe_param
            self._update_sanity_base_before = _gather_to_cpu(base_param)
            self._update_sanity_probe_before = _gather_to_cpu(probe_param)

            if getattr(self, "is_world_process_zero", lambda: True)():
                print(
                    f"[FreezeSanity][BeforeStep] finetune_type={finetune_type} freeze_base_model={freeze_base_model} "
                    f"train_evidence_modules={train_evidence_modules} probe={probe_name}",
                    flush=True,
                )

        return loss

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
        **kwargs,
    ):
        out = super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_closure=optimizer_closure,
            **kwargs,
        )

        if hasattr(self, "_update_sanity_cached") and not hasattr(self, "_update_sanity_checked"):
            self._update_sanity_checked = True

            freeze_base_model = bool(getattr(self.args, "freeze_base_model", False))

            base_param = getattr(self, "_update_sanity_base_param", None)
            probe_name = getattr(self, "_update_sanity_probe_name", None)
            probe_param = getattr(self, "_update_sanity_probe_param", None)

            base_before = getattr(self, "_update_sanity_base_before", None)
            probe_before = getattr(self, "_update_sanity_probe_before", None)

            def _gather_to_cpu(param):
                try:
                    import torch.distributed as dist
                    from transformers.integrations import is_deepspeed_zero3_enabled

                    if param is None:
                        return None
                    if bool(is_deepspeed_zero3_enabled()) and dist.is_initialized():
                        import deepspeed

                        dist.barrier()
                        with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                            if getattr(self, "is_world_process_zero", lambda: True)():
                                out = param.detach().cpu().to(torch.float32).clone()
                            else:
                                out = None
                        dist.barrier()
                        return out
                    if getattr(self, "is_world_process_zero", lambda: True)():
                        return param.detach().cpu().to(torch.float32).clone()
                    return None
                except Exception:
                    return None

            base_after = _gather_to_cpu(base_param)
            probe_after = _gather_to_cpu(probe_param)

            if getattr(self, "is_world_process_zero", lambda: True)():
                msg = "[FreezeSanity][AfterStep]"
                if freeze_base_model and base_before is not None and base_after is not None:
                    base_diff = (base_after - base_before).abs().max().item()
                    msg += f" base_max_abs_update={base_diff}"
                if probe_before is not None and probe_after is not None:
                    probe_diff = (probe_after - probe_before).abs().max().item()
                    msg += f" probe_max_abs_update={probe_diff} probe={probe_name}"
                print(msg, flush=True)

        return out


def _print_checkpoint_weight_sanity(model, model_name_or_path: str, rank: int = 0):
    """One-time sanity check: verify at least one small base tensor matches the checkpoint.

    This does NOT affect training logic; it only prints diagnostics for auditing.
    """

    try:
        import json
        import os

        import torch.distributed as dist
        from safetensors import safe_open
        from transformers.integrations import is_deepspeed_zero3_enabled

        key = "model.language_model.layers.0.input_layernorm.weight"
        ckpt = None
        if rank == 0:
            index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
            with open(index_path, "r") as f:
                idx = json.load(f)
            shard = idx["weight_map"][key]
            shard_path = os.path.join(model_name_or_path, shard)

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                ckpt = f.get_tensor(key).to(torch.float32)

        param = model.model.language_model.layers[0].input_layernorm.weight

        if bool(is_deepspeed_zero3_enabled()) and dist.is_initialized():
            import deepspeed

            # All ranks must participate in the gather collective.
            dist.barrier()
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if rank == 0:
                    cur = param.detach().cpu().to(torch.float32)
                else:
                    cur = None
            dist.barrier()
        else:
            cur = param.detach().cpu().to(torch.float32) if rank == 0 else None

        if rank == 0:
            if ckpt is None or cur is None:
                print("[CkptSanity][WARN] sanity check skipped (missing ckpt/cur).", flush=True)
                return

            max_diff = (cur - ckpt).abs().max().item()
            print(f"[CkptSanity] max_abs_diff({key})={max_diff}", flush=True)
            if max_diff > 1e-3:
                print(
                    "[CkptSanity][WARN] Large diff detected; base weights may be randomly initialized (or key mapping differs).",
                    flush=True,
                )
    except Exception as e:
        if rank == 0:
            print(f"[CkptSanity][WARN] failed to run sanity check: {e}", flush=True)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Attach evidence-regularization hyperparameters to training_args so the custom Trainer can access them
    training_args.lambda_orth = getattr(model_args, "lambda_orth", 0.0)
    training_args.lambda_ctr = getattr(model_args, "lambda_ctr", 0.0)
    training_args.tau = getattr(model_args, "tau", 0.07)
    training_args.aux_layers = getattr(model_args, "aux_layers", None)

    # -------- distributed helpers (保持与原脚本一致) --------
    import torch.distributed as dist

    is_distributed = dist.is_initialized() if hasattr(dist, "is_initialized") else False
    rank = int(os.environ.get("RANK", -1)) if is_distributed else 0

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    cache_dir = os.path.join(training_args.output_dir, "preprocessed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    done_file = os.path.join(cache_dir, ".preprocessing_done")
    cache_exists = os.path.exists(done_file) and os.path.exists(os.path.join(cache_dir, "dataset_info.json"))

    if is_distributed:
        if rank == 0:
            if cache_exists:
                print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
                train_dataset = Dataset.load_from_disk(cache_dir)
                print(f"[Rank {rank}] Dataset loaded from cache.")
                dist.barrier()
            else:
                print(f"[Rank {rank}] No cache found, starting data preprocessing...")
                with open(data_args.training_data_path, "r", encoding="utf-8") as f:
                    training_data = json.load(f)
                train_ds = Dataset.from_list(training_data)

                import threading
                import time

                heartbeat_interval = 30
                heartbeat_stop = threading.Event()

                def heartbeat_worker():
                    while not heartbeat_stop.is_set():
                        try:
                            dummy_tensor = torch.tensor([1.0], device="cpu")
                            dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                            time.sleep(heartbeat_interval)
                        except Exception:
                            time.sleep(heartbeat_interval)

                heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
                heartbeat_thread.start()

                try:
                    train_dataset = train_ds.map(
                        lambda ex: process_func(ex, data_args, tokenizer, processor),
                        num_proc=1,
                        desc="Preprocessing dataset",
                    )
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=5)

                print(f"[Rank {rank}] Saving preprocessed dataset to {cache_dir}...")
                train_dataset.save_to_disk(cache_dir)
                del train_ds
                import gc

                gc.collect()

                with open(done_file, "w") as f:
                    f.write(str(time.time()))
                print(f"[Rank {rank}] Preprocessing completed and saved.")
                dist.barrier()
        else:
            if cache_exists:
                print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
                dist.barrier()
                train_dataset = Dataset.load_from_disk(cache_dir)
                print(f"[Rank {rank}] Dataset loaded from cache.")
            else:
                print(f"[Rank {rank}] Waiting for rank 0 to finish preprocessing...")
                import time

                max_wait_time = 3600 * 2
                start_time = time.time()
                check_interval = 30

                while True:
                    if os.path.exists(done_file):
                        break
                    if time.time() - start_time > max_wait_time:
                        raise RuntimeError(
                            f"Timeout waiting for rank 0 to finish preprocessing (waited {max_wait_time}s)"
                        )
                    try:
                        dummy_tensor = torch.tensor([1.0], device="cpu")
                        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                    except Exception:
                        pass
                    time.sleep(check_interval)

                dist.barrier()
                print(f"[Rank {rank}] Loading preprocessed dataset from {cache_dir}...")
                train_dataset = Dataset.load_from_disk(cache_dir)
                print(f"[Rank {rank}] Dataset loaded.")
    else:
        if cache_exists:
            print(f"Preprocessed cache found, loading from {cache_dir}...")
            train_dataset = Dataset.load_from_disk(cache_dir)
            print("Dataset loaded from cache.")
        else:
            print("No cache found, starting data preprocessing...")
            with open(data_args.training_data_path, "r", encoding="utf-8") as f:
                training_data = json.load(f)
            train_ds = Dataset.from_list(training_data)
            train_dataset = train_ds.map(
                lambda ex: process_func(ex, data_args, tokenizer, processor),
                num_proc=1,
                desc="Preprocessing dataset",
            )
            print(f"Saving preprocessed dataset to {cache_dir}...")
            train_dataset.save_to_disk(cache_dir)
            import time

            with open(done_file, "w") as f:
                f.write(str(time.time()))
            print("Preprocessing completed and saved.")

    # -------- model --------
    # IMPORTANT: to make Qwen3-VL-30B-A3B (MoE) loadable under ZeRO-3, we must instantiate weights directly in the
    # training dtype (bf16/fp16) to avoid transient fp32 memory spikes during `from_pretrained`.
    if bool(getattr(training_args, "bf16", False)):
        torch_dtype = torch.bfloat16
    elif bool(getattr(training_args, "fp16", False)):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    use_low_cpu_mem_usage = True

    model = Qwen3VLMoeCustomVLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=use_low_cpu_mem_usage,
        use_safetensors=True,
    )
    _print_checkpoint_weight_sanity(model, model_args.model_name_or_path, rank=rank)

    # ---- Write CLI controls into model.config (single source of truth) ----
    model.config.enable_vision_gate = bool(getattr(model_args, "enable_evidence", True))
    model.config.gate_layers = _parse_gate_layers(getattr(model_args, "gate_layers", "all"))

    model.config.inject_position = getattr(model_args, "inject_position", "per_layer").strip().lower()
    model.config.inject_op = getattr(model_args, "inject_op", "ours").strip().lower()
    model.config.use_utilization = bool(getattr(model_args, "use_utilization", True))
    model.config.evidence_source = getattr(model_args, "evidence_source", "aligned").strip().lower()

    model.config.export_u_stats = bool(getattr(model_args, "export_u_stats", False))
    model.config.export_u_stats_path = getattr(model_args, "export_u_stats_path", None)

    valid_pos = {"none", "first_layer_input", "per_layer"}
    valid_op = {"ours", "add", "concat"}
    valid_src = {"candidate", "aligned"}
    if model.config.inject_position not in valid_pos:
        raise ValueError(f"inject_position must be one of {valid_pos}, got {model.config.inject_position}")
    if model.config.inject_op not in valid_op:
        raise ValueError(f"inject_op must be one of {valid_op}, got {model.config.inject_op}")
    if model.config.evidence_source not in valid_src:
        raise ValueError(f"evidence_source must be one of {valid_src}, got {model.config.evidence_source}")

    # ---- 同步配置到 text_config 和 language_model.config ----
    Qwen3VLMoeCustomVLForConditionalGeneration._sync_config_to_text_config(model.config)
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
        for key in Qwen3VLMoeCustomVLForConditionalGeneration._EXPERIMENT_CONFIG_KEYS:
            if hasattr(model.config, key):
                setattr(model.model.language_model.config, key, getattr(model.config, key))
    print("[Config Sync] Synced experiment configs to text_config and language_model.config")

    # =========================
    # Fine-tuning strategy: full / freeze-base / LoRA
    # =========================

    finetune_type = str(getattr(model_args, "finetune_type", "full")).strip().lower()
    if finetune_type not in {"full", "lora"}:
        raise ValueError(f"finetune_type must be one of {{'full','lora'}}, got: {finetune_type}")

    freeze_base_model = bool(getattr(model_args, "freeze_base_model", False))
    if finetune_type == "lora":
        freeze_base_model = True

    train_evidence_modules = bool(getattr(model_args, "train_evidence_modules", True))

    # Expose these switches on TrainingArguments for one-time diagnostics inside Trainer (no effect on training).
    training_args.finetune_type = finetune_type
    training_args.freeze_base_model = freeze_base_model
    training_args.train_evidence_modules = train_evidence_modules
    training_args.enable_evidence = bool(getattr(model_args, "enable_evidence", True))

    text_layers = _get_text_layers(model)
    active_evidence_layers = _infer_active_evidence_layers(model.config, len(text_layers))

    if freeze_base_model:
        _freeze_all_params(model)

    if finetune_type == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:
            raise RuntimeError("finetune_type=lora requires the 'peft' package. Install it with: pip install peft") from e

        target_spec = str(getattr(model_args, "lora_target_modules", "auto")).strip()
        if target_spec.lower() == "auto":
            target_modules = _infer_lora_targets_from_model(model)
            if len(target_modules) == 0:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                print("[LoRA] Could not infer target_modules automatically; fallback =", target_modules)
            else:
                print("[LoRA] Auto-inferred target_modules =", target_modules)
        else:
            target_modules = _parse_csv_str(target_spec)

        bias = str(getattr(model_args, "lora_bias", "none")).strip().lower()
        if bias not in {"none", "all", "lora_only"}:
            raise ValueError(f"lora_bias must be one of {{'none','all','lora_only'}}, got: {bias}")

        modules_to_save_spec = str(getattr(model_args, "lora_modules_to_save", "")).strip()
        if modules_to_save_spec.lower() in {"", "none"}:
            modules_to_save = None
            if train_evidence_modules:
                print(
                    "[LoRA][WARN] lora_modules_to_save is 'none' but train_evidence_modules=True. "
                    "Evidence module weights may NOT be saved in adapter checkpoints."
                )
        else:
            modules_to_save = _parse_csv_str(modules_to_save_spec)

        lora_cfg = LoraConfig(
            r=int(getattr(model_args, "lora_r", 8)),
            lora_alpha=int(getattr(model_args, "lora_alpha", 16)),
            lora_dropout=float(getattr(model_args, "lora_dropout", 0.05)),
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_cfg)

    if freeze_base_model:
        _set_evidence_trainable(model, active_evidence_layers, enabled=train_evidence_modules)

    trainable, total = _count_trainable_params(model)
    print(
        f"[Finetune] finetune_type={finetune_type} "
        f"freeze_base_model={freeze_base_model} train_evidence_modules={train_evidence_modules} "
        f"enable_evidence={bool(model.config.enable_vision_gate)} "
        f"active_evidence_layers={active_evidence_layers} "
        f"trainable_params={trainable:,} / total_params={total:,} "
        f"({100.0 * trainable / max(total, 1):.6f}%)"
    )

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[Trainable Names] showing first 30:")
    for n in trainable_names[:30]:
        print("  ", n)

    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # Under ZeRO-3, local parameter shards can temporarily report numel()==0 on each rank
    # (especially during partitioned init), which can make a numeric count look like 0.
    # Use the actual requires_grad list as the final source of truth.
    if len(trainable_names) == 0:
        raise RuntimeError(
            "No trainable parameters found. Please check finetune_type/freeze_base_model/enable_evidence settings."
        )

    data_collator = MultiModalCollator(tokenizer=tokenizer, processor=processor, data_args=data_args)

    if not hasattr(training_args, "dataloader_num_workers") or training_args.dataloader_num_workers is None:
        training_args.dataloader_num_workers = 0

    trainer = EvidenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    _base_model = _unwrap_peft_model(trainer.model)
    _base_model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
