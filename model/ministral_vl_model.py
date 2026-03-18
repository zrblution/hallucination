"""Ministral-3 VL custom model with Visual Evidence Memory (VEM).

This file is a *fix* of the previously adapted `ministral_vl_model.py`.

Key fixes vs the broken adaptation:
  - VEM modules are injected *inside* the decoder forward pass (so they affect logits/loss).
  - VEM is built from the *actual image-token embeddings* (hidden_states entering layer-0),
    instead of incorrectly taking the first N tokens from hidden_states.
  - No second/duplicate decoder pass.

The class name is kept as `Qwen2_5_CustomVLForConditionalGeneration` for compatibility
with your training scripts.
"""

from __future__ import annotations

import os
import inspect
import types
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn


# -----------------------------
# Utilities
# -----------------------------


def _try_import_safetensors():
    try:
        from safetensors.torch import load_file as safetensors_load_file  # type: ignore

        return safetensors_load_file
    except Exception:
        return None


def _split_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# -----------------------------
# Evidence / VEM modules
# -----------------------------


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
            # v_mem_mask: [B, N] -> [B, 1, N]
            mask = v_mem_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask == 0, torch.finfo(attn_logits.dtype).min)

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

    def forward(self, e: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, h], dim=-1)
        u = torch.sigmoid(self.mlp(x))
        return u


class EvidenceCorrector(nn.Module):
    """Residual decoding correction: h_tilde = h + u * W_c(a)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h + u * self.w_c(a)


# -----------------------------
# Decoder-layer forward patch
# -----------------------------


def _patched_decoder_layer_forward(self, *args, **kwargs):
    """Monkey-patched forward() for each decoder layer.

    We keep the original forward() implementation intact (called via self._vem_orig_forward),
    and apply VEM injection either:
      - on the *input* of layer-0 (inject_position == 'first_layer_input')
      - on the *output* of each selected layer (inject_position == 'per_layer')

    Aux stats are collected into parent._aux.
    """
    # _vem_parent is a weakref, so we need to dereference it
    parent_ref = getattr(self, "_vem_parent", None)
    if parent_ref is None:
        return self._vem_orig_forward(*args, **kwargs)
    parent = parent_ref() if callable(parent_ref) else parent_ref
    if parent is None:
        return self._vem_orig_forward(*args, **kwargs)

    cfg = getattr(parent, "config", None)
    enable = bool(getattr(cfg, "enable_vision_gate", True))
    inject_position = str(getattr(cfg, "inject_position", "per_layer")).strip().lower()
    inject_op = str(getattr(cfg, "inject_op", "ours")).strip().lower()
    use_u = bool(getattr(cfg, "use_utilization", True))
    evidence_source = str(getattr(cfg, "evidence_source", "aligned")).strip().lower()
    gate_layers = getattr(cfg, "gate_layers", None)  # None => all, [] => none

    layer_idx = int(getattr(self, "layer_idx", -1))

    # Resolve hidden_states input (positional or kw)
    def _get_hidden_states_in() -> Optional[torch.Tensor]:
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        if len(args) >= 1:
            return args[0]
        return None

    def _set_hidden_states_in(new_h: torch.Tensor):
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = new_h
        else:
            # args is tuple -> rebuild
            if len(args) == 0:
                raise RuntimeError("Cannot set hidden_states: forward() received no positional args")
            new_args = list(args)
            new_args[0] = new_h
            return tuple(new_args)
        return args

    # v_mem is stored on parent. It can be built lazily from layer-0 input hidden_states.
    v_mem = getattr(parent, "_current_v_mem", None)
    v_mem_mask = getattr(parent, "_current_v_mem_mask", None)

    # Lazy-build VEM from layer-0 input embeddings and image-token mask.
    if enable and v_mem is None and layer_idx == 0:
        h0 = _get_hidden_states_in()
        if h0 is not None:
            parent._maybe_build_v_mem_from_layer0_input(h0)
            v_mem = getattr(parent, "_current_v_mem", None)
            v_mem_mask = getattr(parent, "_current_v_mem_mask", None)

    # ---- first_layer_input injection (layer-0 only, before original layer forward) ----
    if (
        enable
        and inject_position == "first_layer_input"
        and layer_idx == 0
        and v_mem is not None
        and (gate_layers is None or layer_idx in gate_layers)
    ):
        h_in = _get_hidden_states_in()
        if h_in is not None:
            e_t, alpha = self.retriever(h_in, v_mem, v_mem_mask)
            a_t, _r_t = self.analyzer(e_t, h_in)
            src = e_t if evidence_source == "candidate" else a_t

            if use_u:
                u_t = self.util(a_t, h_in)
            else:
                u_t = None

            if inject_op == "ours":
                if u_t is None:
                    h_in = h_in + self.corrector.w_c(src)
                else:
                    h_in = self.corrector(h_in, src, u_t)
            elif inject_op == "add":
                delta = self.corrector.w_c(src)
                h_in = h_in + (delta if u_t is None else u_t * delta)
            elif inject_op == "concat":
                cat = torch.cat([h_in, src], dim=-1)
                delta = self.concat_proj(cat)
                h_in = h_in + (delta if u_t is None else u_t * delta)
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            # Update the hidden_states arg
            new_args = _set_hidden_states_in(h_in)
            if new_args is not args:
                args = new_args

            # Disable per-layer injection afterwards (matches Qwen2.5/Qwen3 behavior)
            parent._current_v_mem = None
            parent._current_v_mem_mask = None
            v_mem = None
            v_mem_mask = None

    # ---- run original layer forward ----
    outputs = self._vem_orig_forward(*args, **kwargs)
    if not (enable and inject_position == "per_layer"):
        return outputs

    # If gating says "none", skip
    if gate_layers is not None and layer_idx not in gate_layers:
        return outputs

    v_mem = getattr(parent, "_current_v_mem", None)
    v_mem_mask = getattr(parent, "_current_v_mem_mask", None)
    if v_mem is None:
        return outputs

    # outputs can be a tuple or tensor
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        rest = outputs[1:]
    else:
        hidden_states = outputs
        rest = None

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
        hidden_states = hidden_states + (delta if u_t is None else u_t * delta)
    elif inject_op == "concat":
        cat = torch.cat([hidden_states, src], dim=-1)
        delta = self.concat_proj(cat)
        hidden_states = hidden_states + (delta if u_t is None else u_t * delta)
    else:
        raise ValueError(f"Unknown inject_op: {inject_op}")

    # Collect aux
    aux_item = {
        "layer_idx": layer_idx,
        "e": e_t,
        "a": a_t,
        "r": r_t,
        "u": u_t if u_t is not None else hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1),
        "alpha": alpha,
    }
    try:
        parent._aux.append(aux_item)
    except Exception:
        pass

    if isinstance(outputs, tuple):
        return (hidden_states,) + rest
    return hidden_states


# -----------------------------
# Main wrapper model
# -----------------------------


def _get_ministral3_model_class():
    """Try to import a Mistral-3 multimodal class from transformers.

    We keep this dynamic because different environments may expose different class names.
    """
    try:
        from transformers import Mistral3ForConditionalGeneration  # type: ignore

        return Mistral3ForConditionalGeneration
    except Exception:
        pass

    # Fall back to AutoModel if needed
    try:
        from transformers import AutoModelForCausalLM  # type: ignore

        return AutoModelForCausalLM
    except Exception as e:
        raise ImportError(
            "Cannot import Mistral3ForConditionalGeneration (or AutoModelForCausalLM). "
            "Please ensure transformers supports Ministral/Mistral-3 in your environment."
        ) from e


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layers ModuleList for the language model."""

    candidates = [
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
    ]

    for chain in candidates:
        cur: Any = model
        ok = True
        for name in chain:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, nn.ModuleList):
            return cur

    raise RuntimeError(
        "Cannot find decoder layers on the base model. "
        "Please update _find_decoder_layers() to match your Ministral-3 model structure."
    )


class Qwen2_5_CustomVLForConditionalGeneration(nn.Module):
    """Ministral-3 wrapper with in-place VEM injection."""

    # Keys we mirror from Qwen code (for CLI control)
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

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self._model = base_model
        self.config = getattr(base_model, "config", None)

        # runtime containers
        self._aux: List[Dict[str, Any]] = []
        self._current_v_mem: Optional[torch.Tensor] = None
        self._current_v_mem_mask: Optional[torch.Tensor] = None
        self._current_image_token_mask: Optional[torch.Tensor] = None

        # Patch decoder layers (add VEM modules + forward patch)
        self._patch_decoder_layers_inplace()

    # ------------------------
    # Construction / patching
    # ------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        ModelClass = _get_ministral3_model_class()
        base_model = ModelClass.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        wrapper = cls(base_model)

        # If the checkpoint already contains VEM weights, load them after patching.
        wrapper._load_vem_weights_from_checkpoint(pretrained_model_name_or_path)
        return wrapper

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # 默认使用 safetensors 格式
        kwargs.setdefault("safe_serialization", True)
        if hasattr(self._model, "save_pretrained"):
            return self._model.save_pretrained(save_directory, **kwargs)
        # Fallback: save raw state_dict as safetensors
        try:
            from safetensors.torch import save_file
            state_dict = self.state_dict()
            # 处理共享张量：克隆共享的张量
            seen_ptrs = {}
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                ptr = value.data_ptr()
                if ptr in seen_ptrs:
                    # 这是一个共享张量，需要克隆
                    cleaned_state_dict[key] = value.clone()
                else:
                    seen_ptrs[ptr] = key
                    cleaned_state_dict[key] = value
            save_file(cleaned_state_dict, os.path.join(save_directory, "model.safetensors"))
        except Exception as e:
            print(f"Warning: Failed to save as safetensors ({e}), falling back to pytorch_model.bin")
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    def generate(self, *args, **kwargs):
        if hasattr(self._model, "generate"):
            return self._model.generate(*args, **kwargs)
        raise AttributeError("Base model does not implement generate().")

    def _infer_hidden_size(self) -> int:
        cfg = self.config
        for key in ["hidden_size", "d_model", "n_embd"]:
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return v
        # nested
        for key in ["text_config", "language_config"]:
            sub = getattr(cfg, key, None)
            if sub is None:
                continue
            for k2 in ["hidden_size", "d_model", "n_embd"]:
                v = getattr(sub, k2, None)
                if isinstance(v, int) and v > 0:
                    return v
        raise RuntimeError("Cannot infer hidden_size from config.")

    def _patch_decoder_layers_inplace(self) -> None:
        layers = _find_decoder_layers(self._model)
        hidden_size = self._infer_hidden_size()

        for idx, layer in enumerate(layers):
            # Add VEM modules only once
            if not hasattr(layer, "retriever"):
                layer.retriever = EvidenceRetriever(hidden_size)
            if not hasattr(layer, "analyzer"):
                layer.analyzer = EvidenceAnalyzer(hidden_size)
            if not hasattr(layer, "util"):
                layer.util = EvidenceUtilization(hidden_size)
            if not hasattr(layer, "corrector"):
                layer.corrector = EvidenceCorrector(hidden_size)
            if not hasattr(layer, "concat_proj"):
                layer.concat_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

            layer.layer_idx = idx
            # Use weakref to avoid circular reference (which causes RecursionError in model.to())
            layer._vem_parent = weakref.ref(self)

            if not hasattr(layer, "_vem_orig_forward"):
                layer._vem_orig_forward = layer.forward
                layer.forward = types.MethodType(_patched_decoder_layer_forward, layer)


    def get_decoder_layers(self) -> nn.ModuleList:
        """Expose decoder layers for training scripts (freezing / enabling VEM grads)."""
        return _find_decoder_layers(self._model)


    # ------------------------
    # VEM building + loading
    # ------------------------

    def _get_image_token_id(self) -> Optional[int]:
        cfg = self.config
        # common keys
        for k in ["image_token_id", "image_token_index", "image_token"]:
            v = getattr(cfg, k, None)
            if isinstance(v, int):
                return v

        # nested configs
        for parent_key in ["vision_config", "text_config", "language_config"]:
            sub = getattr(cfg, parent_key, None)
            if sub is None:
                continue
            for k in ["image_token_id", "image_token_index", "image_token"]:
                v = getattr(sub, k, None)
                if isinstance(v, int):
                    return v
        return None

    def _maybe_build_v_mem_from_layer0_input(self, hidden_states_layer0_in: torch.Tensor) -> None:
        if self._current_v_mem is not None:
            return
        if self._current_image_token_mask is None:
            return
        if hidden_states_layer0_in is None:
            return

        mask = self._current_image_token_mask
        if mask.dtype != torch.bool:
            mask = mask.bool()

        if mask.ndim != 2 or hidden_states_layer0_in.ndim != 3:
            return

        bsz, seq_len, dim = hidden_states_layer0_in.shape
        if mask.shape[0] != bsz or mask.shape[1] != seq_len:
            return

        counts = mask.sum(dim=1).tolist()
        max_n = int(max(counts)) if counts else 0
        if max_n <= 0:
            return

        v_mem = hidden_states_layer0_in.new_zeros((bsz, max_n, dim))
        v_mem_mask = hidden_states_layer0_in.new_zeros((bsz, max_n), dtype=torch.long)

        for i in range(bsz):
            idxs = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            n = int(idxs.numel())
            if n > 0:
                v_mem[i, :n, :] = hidden_states_layer0_in[i, idxs, :]
                v_mem_mask[i, :n] = 1

        self._current_v_mem = v_mem
        self._current_v_mem_mask = v_mem_mask

    def _load_vem_weights_from_checkpoint(self, ckpt_path: str) -> None:
        # Only meaningful for local directories.
        if not (ckpt_path and os.path.isdir(ckpt_path)):
            return

        weight_files: List[str] = []
        for fn in os.listdir(ckpt_path):
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                weight_files.append(os.path.join(ckpt_path, fn))

        if not weight_files:
            return

        safetensors_load = _try_import_safetensors()
        vem_substrings = [".retriever.", ".analyzer.", ".util.", ".corrector.", ".concat_proj."]
        to_load: Dict[str, torch.Tensor] = {}

        for wf in sorted(weight_files):
            state: Dict[str, torch.Tensor]
            if wf.endswith(".safetensors") and safetensors_load is not None:
                state = safetensors_load(wf)
            else:
                obj = torch.load(wf, map_location="cpu")
                # Handle checkpoints that wrap state dict
                if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    state = obj["state_dict"]
                elif isinstance(obj, dict):
                    state = obj
                else:
                    continue

            for k, v in state.items():
                if any(ss in k for ss in vem_substrings):
                    # 处理键名前缀：保存时可能带有 "_model." 前缀，需要去掉
                    # 因为 self._model.load_state_dict 期望的键名不带 "_model." 前缀
                    normalized_key = k
                    if k.startswith("_model."):
                        normalized_key = k[len("_model."):]
                    to_load[normalized_key] = v

        if not to_load:
            return

        missing, unexpected = self._model.load_state_dict(to_load, strict=False)
        print(
            f"[Ministral VEM] Loaded {len(to_load)} VEM tensors from checkpoint. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    # ------------------------
    # Forward
    # ------------------------

    def forward(self, *args, **kwargs):
        # Reset per-forward containers
        self._aux = []
        self._current_v_mem = None
        self._current_v_mem_mask = None
        self._current_image_token_mask = None

        # Compute image token mask (used to lazily build VEM from layer-0 input embeddings)
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]

        image_token_id = self._get_image_token_id()
        if image_token_id is not None and isinstance(input_ids, torch.Tensor):
            try:
                self._current_image_token_mask = (input_ids == int(image_token_id))
            except Exception:
                self._current_image_token_mask = None

        # Filter unexpected kwargs for maximum compatibility across transformers versions.
        # Some models do not accept keys like image_grid_thw, video_grid_thw, etc.
        filtered_kwargs = kwargs
        try:
            sig = inspect.signature(self._model.forward)
            params = sig.parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_var_kw:
                accepted = set(params.keys())
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            filtered_kwargs = kwargs

        outputs = self._model(*args, **filtered_kwargs)

        # Store aux in model instance for DDP compatibility
        # (DDP cannot handle extra attributes on ModelOutput objects)
        # EvidenceTrainer should access model._last_aux instead of outputs.aux
        self._last_aux = self._aux

        return outputs
