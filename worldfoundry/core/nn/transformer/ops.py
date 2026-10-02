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
from typing import Protocol

import torch
from torch import nn

from worldfoundry.core.attention.backends.model_backends import (
    AttentionCallable,
    AttentionFunction,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from worldfoundry.core.kernels import residual_gate_add, rms_norm_scale_shift
from worldfoundry.core.nn.blocks.normalization import rms_norm


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
        query = attention_module.q_norm(query)
        key = attention_module.k_norm(key)
        if query_position is not None:
            apply_rotary = getattr(attention_module, "apply_rotary_embedding", None)
            if not callable(apply_rotary):
                raise TypeError("attention module must provide apply_rotary_embedding when positions are supplied")
            query = apply_rotary(query, query_position)
            key = apply_rotary(key, query_position if key_position is None else key_position)
        return query, key


class PytorchAdaZeroFunction:
    """Portable AdaLN-zero implemented with :func:`rms_norm_scale_shift`."""

    def __call__(
        self,
        value: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Same math as fused AdaLN-zero; last-dim RMS then ``x*(1+scale)+shift``."""

        return rms_norm_scale_shift(value, scale, shift, eps=eps)


class PytorchPostSAFunction:
    """Portable residual-gate add plus RMSNorm after self-attention."""

    def __call__(
        self,
        residual: torch.Tensor,
        value: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``residual + value * gate``, then RMSNorm for the following FFN."""

        output = residual_gate_add(residual, value, gate)
        return output, rms_norm(output, norm_weights, eps=eps)


class PytorchGatedAttention:
    """Apply a sigmoid gate derived from the residual to each attention head."""

    def __call__(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        attention_module: nn.Module,
    ) -> torch.Tensor:
        """``2·σ(to_gate_logits(residual))`` per head; 2× keeps the gate mean near 1."""

        gate_logits = attention_module.to_gate_logits(residual)
        batch, tokens, _ = attention_output.shape
        output = attention_output.view(batch, tokens, attention_module.heads, attention_module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        return (output * gates.unsqueeze(-1)).view(
            batch,
            tokens,
            attention_module.heads * attention_module.dim_head,
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
                gated_attention_function=(
                    gated_attention if gated_attention is not None else PytorchGatedAttention()
                ),
            ),
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


__all__ = [
    "AdaZeroCallable",
    "DEFAULT_TRANSFORMER_OPS",
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
