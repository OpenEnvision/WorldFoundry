"""Pluggable transformer execution callables shared by native architectures.

Pre-attention / gated-attention / post-SA hooks so a fused kernel can
replace a subgraph without rewriting the module tree. Default ops are
PyTorch; Triton swaps in via ``kernels.diffusion``.

Not this module:
    Kernel implementations live in :mod:`worldfoundry.core.kernels`.
    Attention backend enums live in
    :mod:`worldfoundry.core.attention.backends.model_backends`. Block shells
    that *call* these hooks live in :mod:`worldfoundry.core.nn.transformer.vit`.

Public surface:

- Protocols :class:`PreAttentionCallable`, :class:`AdaZeroCallable`,
  :class:`PostSACallable`, :class:`GatedAttentionCallable`.
- Portable impls :class:`PytorchPreAttention`,
  :class:`PytorchAdaZeroFunction`, :class:`PytorchPostSAFunction`,
  :class:`PytorchGatedAttention`.
- :class:`TransformerAttentionOps` / :class:`TransformerOpsConfig` /
  :data:`DEFAULT_TRANSFORMER_OPS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch
from torch import nn

from worldfoundry.core.attention.backends.model_backends import (
    AttentionCallable,
    AttentionFunction,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from worldfoundry.core.kernels import rms_norm_scale_shift
from worldfoundry.core.nn.blocks.normalization import rms_norm

_FUSED_OP_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MAX_FUSED_POST_SA_FEATURES = 8192
_MAX_FUSED_HEAD_GATE_DIM = 512
_TRITON_DIFFUSION_MOD: Any = None
_TRITON_DIFFUSION_UNAVAILABLE = False


def _get_triton_diffusion() -> Any | None:
    """Lazily load ``triton_diffusion`` on first CUDA inference call without importing Triton at import time."""
    global _TRITON_DIFFUSION_MOD, _TRITON_DIFFUSION_UNAVAILABLE
    if _TRITON_DIFFUSION_MOD is not None:
        return _TRITON_DIFFUSION_MOD
    if _TRITON_DIFFUSION_UNAVAILABLE:
        return None
    try:
        from worldfoundry.core.kernels import triton_diffusion as mod
    except ImportError:
        _TRITON_DIFFUSION_UNAVAILABLE = True
        return None
    _TRITON_DIFFUSION_MOD = mod
    return mod


@torch.library.custom_op(
    "worldfoundry::fused_residual_gate_add_rms_norm",
    mutates_args=(),
    device_types="cuda",
)
def _fused_residual_gate_add_rms_norm_op(
    residual: torch.Tensor,
    value: torch.Tensor,
    norm_weights: torch.Tensor | None,
    eps: float,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _get_triton_diffusion()
    if mod is not None:
        return mod.residual_gate_add_rms_norm(residual, value, norm_weights, eps, gate)
    output = residual + value * gate
    return output, rms_norm(output, norm_weights, eps=eps)


@_fused_residual_gate_add_rms_norm_op.register_fake
def _fused_residual_gate_add_rms_norm_op_fake(
    residual: torch.Tensor,
    value: torch.Tensor,
    norm_weights: torch.Tensor | None,
    eps: float,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del norm_weights, eps
    out_dtype = torch.promote_types(residual.dtype, torch.promote_types(value.dtype, gate.dtype))
    return torch.empty_like(residual, dtype=out_dtype), torch.empty_like(residual, dtype=out_dtype)


@torch.library.custom_op(
    "worldfoundry::fused_head_sigmoid_gate",
    mutates_args=(),
    device_types="cuda",
)
def _fused_head_sigmoid_gate_op(
    attention_output: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    mod = _get_triton_diffusion()
    if mod is not None:
        return mod.head_sigmoid_gate(attention_output, gate_logits, heads, dim_head)
    batch, tokens, _ = attention_output.shape
    output = attention_output.view(batch, tokens, heads, dim_head)
    gates = 2.0 * torch.sigmoid(gate_logits)
    return (output * gates.unsqueeze(-1)).view(batch, tokens, heads * dim_head)


@_fused_head_sigmoid_gate_op.register_fake
def _fused_head_sigmoid_gate_op_fake(
    attention_output: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    del heads, dim_head
    out_dtype = torch.promote_types(attention_output.dtype, gate_logits.dtype)
    return torch.empty_like(attention_output, dtype=out_dtype)


# ──────────────────────────────────────────────────────────────────────────
# Hook protocols — fused kernels must match these signatures
# ──────────────────────────────────────────────────────────────────────────


class PreAttentionCallable(Protocol):
    """Q/K RMSNorm + optional RoPE before the attention kernel."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_module: nn.Module,
        mask: torch.Tensor | None,
        query_position: torch.Tensor | None,
        key_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (and optionally rotated) ``(query, key)``.

        ``mask`` is part of the fused-op signature; portable impls may ignore it.
        """
        ...


class AdaZeroCallable(Protocol):
    """AdaLN-zero: RMSNorm then scale/shift from a timestep embedding."""

    def __call__(
        self,
        value: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RMSNorm + affine; ``scale`` / ``shift`` are already broadcastable."""
        ...


class PostSACallable(Protocol):
    """Residual + gate add after self-attention, then optional RMSNorm."""

    def __call__(
        self,
        residual: torch.Tensor,
        value: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gated_residual, normalized_residual)`` for the FFN path."""
        ...


class GatedAttentionCallable(Protocol):
    """Per-head gate applied to attention output (e.g. LTX ``to_gate_logits``)."""

    def __call__(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        attention_module: nn.Module,
    ) -> torch.Tensor:
        """Gate ``attention_output`` using logits projected from ``residual``."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Portable PyTorch defaults — Triton / fused ops replace these per config
# ──────────────────────────────────────────────────────────────────────────


class PytorchPreAttention:
    """Normalize Q/K and delegate model-specific rotary math through a narrow method."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_module: nn.Module,
        mask: torch.Tensor | None,
        query_position: torch.Tensor | None,
        key_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ``q_norm`` / ``k_norm``; RoPE only when a position tensor is present.

        Raises:
            TypeError: positions were supplied but the module has no
                ``apply_rotary_embedding``.
        """

        del mask
        # Attention modules contribute model-specific norms and rotary helpers beyond nn.Module.
        module = cast(Any, attention_module)
        query = module.q_norm(query)
        key = module.k_norm(key)
        if query_position is not None:
            apply_rotary = getattr(attention_module, "apply_rotary_embedding", None)
            if not callable(apply_rotary):
                raise TypeError("attention module must provide apply_rotary_embedding when positions are supplied")
            query = apply_rotary(query, query_position)
            key = apply_rotary(key, query_position if key_position is None else key_position)
        return query, key


class PytorchAdaZeroFunction:
    """Portable AdaLN-zero implemented with :func:`rms_norm_scale_shift`.

    ``backend`` pins kernel dispatch; ``"torch"`` keeps eager PyTorch rounding.
    """

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend

    def __call__(
        self,
        value: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Same math as fused AdaLN-zero; last-dim RMS then ``x*(1+scale)+shift``."""

        return rms_norm_scale_shift(value, scale, shift, eps=eps, backend=self.backend)


class PytorchPostSAFunction:
    """Residual-gate add plus RMSNorm after self-attention (fused single-pass Triton on CUDA inference)."""

    def __call__(
        self,
        residual: torch.Tensor,
        value: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``residual + value * gate``, then RMSNorm for the following FFN."""

        if (
            not torch.is_grad_enabled()
            and isinstance(residual, torch.Tensor)
            and residual.is_cuda
            and isinstance(value, torch.Tensor)
            and isinstance(gate, torch.Tensor)
            and value.device == residual.device
            and gate.device == residual.device
            and residual.shape == value.shape
            and 1 <= residual.ndim <= 5
            and residual.numel() > 0
            and residual.shape[-1] <= _MAX_FUSED_POST_SA_FEATURES
            and residual.is_contiguous()
            and value.is_contiguous()
            and residual.dtype in _FUSED_OP_DTYPES
            and value.dtype in _FUSED_OP_DTYPES
            and gate.dtype in _FUSED_OP_DTYPES
        ):
            out_dtype = torch.promote_types(residual.dtype, torch.promote_types(value.dtype, gate.dtype))
            weights_ok = norm_weights is None or (
                isinstance(norm_weights, torch.Tensor)
                and norm_weights.device == residual.device
                and norm_weights.ndim == 1
                and norm_weights.shape[0] == residual.shape[-1]
                and norm_weights.is_contiguous()
                and norm_weights.dtype == out_dtype
            )
            if weights_ok:
                return _fused_residual_gate_add_rms_norm_op(residual, value, norm_weights, float(eps), gate)

        output = residual + value * gate
        return output, rms_norm(output, norm_weights, eps=eps)


class PytorchGatedAttention:
    """Apply a sigmoid gate derived from the residual to each attention head (fused Triton on CUDA inference)."""

    def __call__(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        attention_module: nn.Module,
    ) -> torch.Tensor:
        """``2·σ(to_gate_logits(residual))`` per head; 2× keeps the gate mean near 1."""

        module = cast(Any, attention_module)
        gate_logits = module.to_gate_logits(residual)
        batch, tokens, hidden = attention_output.shape
        heads = int(module.heads)
        dim_head = int(module.dim_head)
        if (
            not torch.is_grad_enabled()
            and attention_output.is_cuda
            and gate_logits.device == attention_output.device
            and attention_output.numel() > 0
            and attention_output.is_contiguous()
            and gate_logits.is_contiguous()
            and attention_output.dtype in _FUSED_OP_DTYPES
            and gate_logits.dtype in _FUSED_OP_DTYPES
            and hidden == heads * dim_head
            and gate_logits.shape == (batch, tokens, heads)
            and 1 <= dim_head <= _MAX_FUSED_HEAD_GATE_DIM
        ):
            return _fused_head_sigmoid_gate_op(attention_output, gate_logits, heads, dim_head)

        output = attention_output.view(batch, tokens, heads, dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        return (output * gates.unsqueeze(-1)).view(
            batch,
            tokens,
            heads * dim_head,
        )


# ──────────────────────────────────────────────────────────────────────────
# Frozen configs — swap one hook without rewriting the module tree
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransformerAttentionOps:
    """Backend-neutral attention execution hooks for transformer architectures."""

    attention_function: AttentionCallable = field(default_factory=lambda: AttentionFunction.AUTOMATIC.to_callable())
    masked_attention_function: MaskedAttentionCallable = field(
        default_factory=lambda: MaskedAttentionFunction.AUTOMATIC.to_callable()
    )
    preattention_function: PreAttentionCallable = field(default_factory=PytorchPreAttention)
    gated_attention_function: GatedAttentionCallable = field(default_factory=PytorchGatedAttention)


@dataclass(frozen=True)
class TransformerOpsConfig:
    """Complete pluggable execution policy for transformer blocks."""

    attention_ops: TransformerAttentionOps = field(default_factory=TransformerAttentionOps)
    ada_zero_function: AdaZeroCallable = field(default_factory=PytorchAdaZeroFunction)
    post_sa_function: PostSACallable = field(default_factory=PytorchPostSAFunction)

    @classmethod
    def from_functions(
        cls,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.AUTOMATIC,
        masked_attention: MaskedAttentionFunction | MaskedAttentionCallable = MaskedAttentionFunction.AUTOMATIC,
        preattention: PreAttentionCallable | None = None,
        gated_attention: GatedAttentionCallable | None = None,
        ada_zero: AdaZeroCallable | None = None,
        post_sa: PostSACallable | None = None,
    ) -> TransformerOpsConfig:
        """Build a config from enums or already-resolved callables.

        ``None`` hook arguments keep the portable PyTorch defaults.
        """
        attention_callable = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        masked_callable = (
            masked_attention.to_callable()
            if isinstance(masked_attention, MaskedAttentionFunction)
            else masked_attention
        )
        return cls(
            attention_ops=TransformerAttentionOps(
                attention_function=attention_callable,
                masked_attention_function=masked_callable,
                preattention_function=preattention if preattention is not None else PytorchPreAttention(),
                gated_attention_function=(gated_attention if gated_attention is not None else PytorchGatedAttention()),
            ),
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


__all__ = [
    "DEFAULT_TRANSFORMER_OPS",
    "AdaZeroCallable",
    "GatedAttentionCallable",
    "PostSACallable",
    "PreAttentionCallable",
    "PytorchAdaZeroFunction",
    "PytorchGatedAttention",
    "PytorchPostSAFunction",
    "PytorchPreAttention",
    "TransformerAttentionOps",
    "TransformerOpsConfig",
]
