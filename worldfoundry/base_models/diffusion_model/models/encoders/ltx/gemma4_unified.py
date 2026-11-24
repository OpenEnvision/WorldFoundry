# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Native PyTorch Gemma-4 Unified text encoder for LTX-2.5 split checkpoints.

Allows loading ``gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`` directly in
``transformers < 5`` environments (such as ``transformers==4.57.x``) where
``transformers.models.gemma4_unified`` is not yet bundled, while preserving
exact Gemma-4 Unified text-backbone numerics:

- Unshifted ``Gemma4RMSNorm`` (``x_norm * weight`` when ``with_scale=True``;
  unscaled ``v_norm`` with ``with_scale=False``).
- Hybrid sliding-window (``head_dim=256, num_kv_heads=8, theta=10000``) and
  global full-attention (``global_head_dim=512, num_global_kv_heads=1,
  attention_k_eq_v=True, proportional RoPE partial_rotary_factor=0.25,
  theta=1000000``) layers with explicit attention ``scale=1.0``.
- Per-layer output scaling by ``layer_scalar`` after the post-MLP residual add.
- Left-padded ``position_ids`` and causal+window masks matching HuggingFace
  ``LTXGemmaTextEncoder.encode`` (returning all 49 layer hidden states).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

from .gemma import (
    TOKENIZER_MAX_LENGTH,
    GemmaAssets,
    _chat_template_from_assets,
)
from .tokenizer import (
    LTXVGemmaTokenizer,
    PaddingSide,
)

_TOKENIZER_CFG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)


def _build_gemma4_tokenizer_compat(assets: GemmaAssets) -> LTXVGemmaTokenizer:
    """Build ``LTXVGemmaTokenizer`` compatible with both ``transformers 4.x`` and ``5.x``."""
    raw_cfg = assets.sidecar_json("tokenizer_config.json")
    kwargs = {k: v for k, v in raw_cfg.items() if k not in _TOKENIZER_CFG_SKIP}
    if isinstance(kwargs.get("extra_special_tokens"), list):
        extra = list(kwargs.pop("extra_special_tokens"))
        existing = list(kwargs.get("additional_special_tokens") or [])
        kwargs["additional_special_tokens"] = list(dict.fromkeys([*existing, *extra]))
    tpl = _chat_template_from_assets(assets)
    if tpl is not None:
        kwargs.setdefault("chat_template", tpl)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(assets.tokenizer_json),
        model_max_length=TOKENIZER_MAX_LENGTH,
        **kwargs,
    )
    return LTXVGemmaTokenizer(hf_tok, TOKENIZER_MAX_LENGTH, padding_side=PaddingSide.LEFT)


class Gemma4RMSNorm(nn.Module):
    """Exact Gemma-4 RMSNorm (direct ``weight`` scaling, optional unscaled ``v_norm``)."""

    def __init__(self, dim: int, eps: float = 1e-6, *, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.with_scale = bool(with_scale)
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_fp32 = hidden_states.float()
        normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(hidden_states.dtype)
        if self.with_scale and self.weight is not None:
            return normed * self.weight
        return normed


def _build_rope_inv_freq(
    *,
    head_dim: int,
    rope_theta: float,
    rope_type: str,
    partial_rotary_factor: float = 1.0,
) -> torch.Tensor:
    """Compute Gemma-4 ``default`` or ``proportional`` RoPE inverse frequencies."""
    half = head_dim // 2
    if rope_type == "proportional":
        rotary_dim = round(head_dim * float(partial_rotary_factor))
        rotary_pairs = rotary_dim // 2
        active = 1.0 / (float(rope_theta) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / float(head_dim)))
        if rotary_pairs < half:
            zeros = torch.zeros(half - rotary_pairs, dtype=torch.float32)
            return torch.cat((active, zeros), dim=0)
        return active
    return 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))


def _compute_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # position_ids: [B, T], inv_freq: [D/2] -> freqs: [B, 1, T, D]
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.to(device=position_ids.device, dtype=torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class Gemma4UnifiedAttention(nn.Module):
    """Hybrid sliding / global attention for Gemma-4 Unified with ``scale=1.0`` & ``v_norm``."""

    def __init__(self, text_cfg: Mapping[str, Any], *, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.is_global = layer_type == "full_attention"
        hidden_size = int(text_cfg["hidden_size"])
        self.num_heads = int(text_cfg["num_attention_heads"])
        self.head_dim = int(text_cfg["global_head_dim"] if self.is_global else text_cfg["head_dim"])
        self.num_kv_heads = int(
            text_cfg["num_global_key_value_heads"] if self.is_global else text_cfg["num_key_value_heads"]
        )
        self.k_eq_v = bool(self.is_global and text_cfg.get("attention_k_eq_v", False))
        eps = float(text_cfg.get("rms_norm_eps", 1e-6))

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj: nn.Linear | None = (
            None if self.k_eq_v else nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=True)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=True)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.v_proj is None:
            v_raw = k_raw
        else:
            v_raw = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k_raw)
        v = self.v_norm(v_raw)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Gemma-4 uses explicit scaling = 1.0 (Q/K/V are all RMS-normalized).
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class Gemma4UnifiedMLP(nn.Module):
    """GeGLU MLP using ``gelu_pytorch_tanh``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4UnifiedDecoderLayer(nn.Module):
    """Single Gemma-4 Unified text decoder block with 4 RMSNorms and ``layer_scalar``."""

    def __init__(self, text_cfg: Mapping[str, Any], *, layer_type: str) -> None:
        super().__init__()
        hidden_size = int(text_cfg["hidden_size"])
        intermediate_size = int(text_cfg["intermediate_size"])
        eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        self.layer_type = layer_type
        self.self_attn = Gemma4UnifiedAttention(text_cfg, layer_type=layer_type)
        self.mlp = Gemma4UnifiedMLP(hidden_size, intermediate_size)
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=True)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=True)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=True)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=True)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, cos=cos, sin=sin, attn_mask=attn_mask)
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        residual = hidden_states
        mlp_in = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(mlp_in)
        hidden_states = residual + self.post_feedforward_layernorm(mlp_out)

        scalar = self.layer_scalar.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states * scalar


class NativeGemma4UnifiedTextBackbone(nn.Module):
    """48-layer Gemma-4 12B Unified language backbone emitting all 49 hidden states."""

    def __init__(self, text_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.text_cfg = dict(text_cfg)
        self.hidden_size = int(text_cfg["hidden_size"])
        self.vocab_size = int(text_cfg["vocab_size"])
        self.sliding_window = int(text_cfg.get("sliding_window", 1024))
        eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        layer_types = list(text_cfg["layer_types"])

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([Gemma4UnifiedDecoderLayer(text_cfg, layer_type=lt) for lt in layer_types])
        self.norm = Gemma4RMSNorm(self.hidden_size, eps=eps, with_scale=True)

        rope_params = dict(text_cfg.get("rope_parameters") or {})
        sliding_p = dict(rope_params.get("sliding_attention") or {"rope_type": "default", "rope_theta": 10000.0})
        full_p = dict(
            rope_params.get("full_attention")
            or {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
        )
        self.register_buffer(
            "sliding_inv_freq",
            _build_rope_inv_freq(
                head_dim=int(text_cfg["head_dim"]),
                rope_theta=float(sliding_p["rope_theta"]),
                rope_type=str(sliding_p.get("rope_type", "default")),
                partial_rotary_factor=float(sliding_p.get("partial_rotary_factor", 1.0)),
            ),
            persistent=False,
        )
        self.register_buffer(
            "full_inv_freq",
            _build_rope_inv_freq(
                head_dim=int(text_cfg["global_head_dim"]),
                rope_theta=float(full_p["rope_theta"]),
                rope_type=str(full_p.get("rope_type", "proportional")),
                partial_rotary_factor=float(full_p.get("partial_rotary_factor", 0.25)),
            ),
            persistent=False,
        )
        self._embed_scale = math.sqrt(self.hidden_size)

    def _apply(self, fn: Any, recurse: bool = True) -> NativeGemma4UnifiedTextBackbone:
        """Preserve float32 precision on RoPE ``inv_freq`` buffers across ``.to(dtype=...)``."""
        prev_sliding = self.sliding_inv_freq.detach().float() if not self.sliding_inv_freq.is_meta else None
        prev_full = self.full_inv_freq.detach().float() if not self.full_inv_freq.is_meta else None
        result = super()._apply(fn, recurse=recurse)
        if prev_sliding is not None and not self.sliding_inv_freq.is_meta:
            self.sliding_inv_freq = prev_sliding.to(device=self.sliding_inv_freq.device, dtype=torch.float32)
        if prev_full is not None and not self.full_inv_freq.is_meta:
            self.full_inv_freq = prev_full.to(device=self.full_inv_freq.device, dtype=torch.float32)
        return result

    @staticmethod
    def _build_masks(
        attention_mask: torch.Tensor,
        *,
        sliding_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build boolean ``[B, 1, T, T]`` causal masks for full & sliding attention (True = attend)."""
        bsz, seq_len = attention_mask.shape
        device = attention_mask.device
        idx = torch.arange(seq_len, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # [T, T]: i - j
        causal_full = diff >= 0
        causal_sliding = causal_full & (diff < sliding_window)

        key_valid = attention_mask.bool().view(bsz, 1, 1, seq_len)
        full_mask = causal_full.view(1, 1, seq_len, seq_len) & key_valid
        sliding_mask = causal_sliding.view(1, 1, seq_len, seq_len) & key_valid

        # Prevent all-False rows on left-padded positions so SDPA never emits NaNs.
        eye = torch.eye(seq_len, device=device, dtype=torch.bool).view(1, 1, seq_len, seq_len)
        padded_q = ~attention_mask.bool().view(bsz, 1, seq_len, 1)
        full_mask = full_mask | (padded_q & eye)
        sliding_mask = sliding_mask | (padded_q & eye)
        return full_mask, sliding_mask

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.embed_tokens(input_ids) * self._embed_scale
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        cos_s, sin_s = _compute_cos_sin(self.sliding_inv_freq, position_ids, hidden_states.dtype)
        cos_f, sin_f = _compute_cos_sin(self.full_inv_freq, position_ids, hidden_states.dtype)
        full_mask, sliding_mask = self._build_masks(attention_mask, sliding_window=self.sliding_window)

        all_hidden: list[torch.Tensor] = [hidden_states]
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                hidden_states = layer(hidden_states, cos=cos_f, sin=sin_f, attn_mask=full_mask)
            else:
                hidden_states = layer(hidden_states, cos=cos_s, sin=sin_s, attn_mask=sliding_mask)
            all_hidden.append(hidden_states)

        all_hidden[-1] = self.norm(all_hidden[-1])
        return tuple(all_hidden)


class NativeGemma4UnifiedEncoderWrapper(nn.Module):
    """Drop-in ``LTXGemmaTextEncoder``-compatible wrapper for ``LTXGemmaPromptConditioner``."""

    def __init__(
        self,
        backbone: NativeGemma4UnifiedTextBackbone,
        tokenizer: Any,
        *,
        execution_device: torch.device,
        offload_to_cpu: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._execution_device = torch.device(execution_device)
        self._offload_to_cpu = bool(offload_to_cpu)

    @property
    def execution_device(self) -> torch.device:
        if self._offload_to_cpu:
            return self._execution_device
        param = next(self.backbone.parameters(), None)
        if param is not None and not param.is_meta:
            return param.device
        return self._execution_device

    @execution_device.setter
    def execution_device(self, value: str | torch.device) -> None:
        self._execution_device = torch.device(value)

    @property
    def encoder(self) -> NativeGemma4UnifiedEncoderWrapper:
        return self

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        if not prompts:
            return []
        target_device = self.execution_device
        tokenized = [self.tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
        input_ids = torch.tensor(
            [[tok for tok, _ in pairs] for pairs in tokenized],
            device=target_device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[w for _, w in pairs] for pairs in tokenized],
            device=target_device,
            dtype=torch.long,
        )
        if self._offload_to_cpu and target_device.type != "cpu":
            self.backbone.to(target_device)
            try:
                hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                self.backbone.to("cpu")
                if target_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return [(tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1]) for i in range(len(prompts))]


def load_native_gemma4_unified_encoder(
    text_encoder_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = False,
) -> tuple[NativeGemma4UnifiedEncoderWrapper, GemmaAssets]:
    """Load Gemma-4 12B Unified directly from ``gemma4-12b-with-proj-ltx-2.5-bf16.safetensors``."""
    path = Path(text_encoder_path).expanduser().resolve()
    assets = GemmaAssets.load(path)
    text_cfg = dict(assets.config_dict["text_config"])
    target_device = torch.device(device)

    with torch.device("meta"):
        backbone = NativeGemma4UnifiedTextBackbone(text_cfg)

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            if key.startswith("model."):
                dst = key.removeprefix("model.")
                state_dict[dst] = handle.get_tensor(key)

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False, assign=True)
    real_missing = [k for k in missing if k not in {"sliding_inv_freq", "full_inv_freq"}]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Native Gemma-4 Unified checkpoint mismatch: {len(real_missing)} missing ({real_missing[:5]}), "
            f"{len(unexpected)} unexpected ({unexpected[:5]})"
        )

    # Populate non-persistent RoPE buffers (meta init leaves them unmaterialized).
    rope_params = dict(text_cfg.get("rope_parameters") or {})
    sliding_p = dict(rope_params.get("sliding_attention") or {"rope_type": "default", "rope_theta": 10000.0})
    full_p = dict(
        rope_params.get("full_attention")
        or {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
    )
    backbone.register_buffer(
        "sliding_inv_freq",
        _build_rope_inv_freq(
            head_dim=int(text_cfg["head_dim"]),
            rope_theta=float(sliding_p["rope_theta"]),
            rope_type=str(sliding_p.get("rope_type", "default")),
            partial_rotary_factor=float(sliding_p.get("partial_rotary_factor", 1.0)),
        ),
        persistent=False,
    )
    backbone.register_buffer(
        "full_inv_freq",
        _build_rope_inv_freq(
            head_dim=int(text_cfg["global_head_dim"]),
            rope_theta=float(full_p["rope_theta"]),
            rope_type=str(full_p.get("rope_type", "proportional")),
            partial_rotary_factor=float(full_p.get("partial_rotary_factor", 0.25)),
        ),
        persistent=False,
    )

    load_device = torch.device("cpu") if offload_to_cpu else target_device
    backbone = backbone.to(device=load_device, dtype=dtype).eval()
    tokenizer = _build_gemma4_tokenizer_compat(assets)
    wrapper = NativeGemma4UnifiedEncoderWrapper(
        backbone,
        tokenizer,
        execution_device=target_device,
        offload_to_cpu=offload_to_cpu,
    )
    return wrapper, assets


__all__ = [
    "Gemma4RMSNorm",
    "Gemma4UnifiedAttention",
    "Gemma4UnifiedDecoderLayer",
    "Gemma4UnifiedMLP",
    "NativeGemma4UnifiedEncoderWrapper",
    "NativeGemma4UnifiedTextBackbone",
    "load_native_gemma4_unified_encoder",
]
