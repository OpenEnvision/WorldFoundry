# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
# Copyright 2025 The MiniMax Team and The HuggingFace Team.
# Adapted from Video-Rebirth/hyperflow at 1dd2f342aba5ab51da02b62885939655e8e268da.
"""HyperFlow's step/layer Sol-Attn policy inside canonical H3 attention."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from worldfoundry.core.attention.backends.native import scaled_dot_product_attention
from worldfoundry.core.distributed.collectives.tensor import all_to_all_concat

_SDPA_BACKENDS = {
    "native": None,
    "_native_cudnn": "cudnn",
    "_native_efficient": "efficient",
    "_native_flash": "flash",
    "_native_math": "math",
}
_KERNEL_BACKENDS = {"flash": "flash_attention_2", "_flash_3": "flash_attention_3", "xformers": "xformers_attention"}


@dataclass
class HyperFlowAttentionState:
    step: int | None = None
    sequence_length: int = 0
    degree: int = 1
    rank: int = 0


def resolve_attention_backend(name: str, *, gpus: int) -> str:
    """Preserve the public aliases and the release's single-GPU backend passthrough."""
    backend = {"sdpa": "native", "fa3": "_flash_3_hub"}.get(name, name)
    if gpus > 1 and backend not in {"native", "_flash_3_hub"}:
        raise ValueError("HyperFlow multi-GPU attention requires sdpa or fa3.")
    if backend not in _SDPA_BACKENDS and backend not in _KERNEL_BACKENDS:
        from diffusers.models.attention_dispatch import AttentionBackendName

        AttentionBackendName(backend)
    return backend


def _sol_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    from sol_attn import sol_attn

    if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
        raise ValueError("HyperFlow Sol-Attn requires BF16 activations and head_dim=128.")
    splits = 1
    if query.is_cuda and torch.cuda.get_device_capability(query.device) == (9, 0) and query.shape[1] >= 65536:
        try:
            import cuda.bindings.driver  # noqa: F401 - Required by the upstream SM90 CuTe path.
            import cutlass.cute  # noqa: F401 - Required by the upstream SM90 CuTe path.
        except ImportError:
            pass
        else:
            splits = 4
    if splits == 1:
        from worldfoundry.core.attention.backends.sol import sol_attention

        return sol_attention(query, key, value)
    # Core's public Sol provider supports one split. The released SM90 policy
    # additionally uses four splits for long sequences through the same kernel.
    options: dict[str, Any] = {
        "tau": 1.0,
        "thresh_type": "diag",
        "sink_tokens": 0,
        "sink_start": None,
        "kv_splits": splits,
    }
    accepted = inspect.signature(sol_attn).parameters
    return sol_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        **{name: value for name, value in options.items() if name in accepted},
    )


class HyperFlowAttention:
    """Dense for steps 0–1 and blocks 0–1; optional sparse attention thereafter.

    Ulysses exchanges canonical Q/K/V with core collectives. The padded tail is
    removed before attention, including Sol-Attn, so it never becomes a key.
    """

    def __init__(
        self,
        state: HyperFlowAttentionState,
        *,
        layer: int | None,
        backend: str = "native",
        sol_attn: bool = False,
    ) -> None:
        self.state = state
        self.layer = layer
        self.backend = backend
        self.sol_attn = sol_attn

    def use_sparse(self) -> bool:
        return (
            self.sol_attn
            and self.state.step is not None
            and self.state.step >= 2
            and self.layer is not None
            and self.layer not in (0, 1)
        )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        del cu_seqlens  # HyperFlow has one unpadded attention document per request.
        parallel = self.layer is not None and self.state.degree > 1
        padded_length = query.shape[0] * self.state.degree
        if parallel:
            query, key, value = (
                all_to_all_concat(tensor, scatter_dim=1, gather_dim=0)[: self.state.sequence_length]
                for tensor in (query, key, value)
            )
        if self.use_sparse():
            result = _sol_attention(query[None], key[None], value[None])[0]
        elif self.backend in _SDPA_BACKENDS:
            heads = [tensor.transpose(0, 1)[None] for tensor in (query, key, value)]
            if self.backend == "_native_cudnn":
                heads = [tensor.contiguous() for tensor in heads]
            result = scaled_dot_product_attention(
                *heads, dropout_p=0.0, is_causal=False, scale=softmax_scale, backend=_SDPA_BACKENDS[self.backend]
            )[0].transpose(0, 1)
        elif self.backend in _KERNEL_BACKENDS:
            from worldfoundry.core.attention.backends import dispatch

            provider = getattr(dispatch, _KERNEL_BACKENDS[self.backend])
            result = provider(
                query[None],
                key[None],
                value[None],
                q_pattern="b s n d",
                k_pattern="b s n d",
                v_pattern="b s n d",
                out_pattern="b s n d",
                scale=softmax_scale,
            )[0]
        else:
            # Hub, strict varlen and NHD Sage backends require the library's
            # dispatch semantics; no Diffusers model or pipeline executes here.
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            result = dispatch_attention_fn(
                query[None], key[None], value[None], dropout_p=0.0, is_causal=False, backend=self.backend
            )[0]
        if parallel:
            result = F.pad(result, (0, 0, 0, 0, 0, padded_length - result.shape[0]))
            result = all_to_all_concat(result, scatter_dim=0, gather_dim=1)
        return result
