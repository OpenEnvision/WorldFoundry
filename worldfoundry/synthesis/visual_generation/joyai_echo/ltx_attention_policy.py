# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo's LTX backend selection over shared attention implementations.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

import importlib
from enum import Enum
from functools import lru_cache

import torch

from worldfoundry.core.attention.backends.model_backends import FlashAttention3, XFormersAttention
from worldfoundry.core.attention.backends.native import flattened_multihead_attention
from worldfoundry.core.nn import PytorchAdaZeroFunction, TransformerAttentionOps, TransformerOpsConfig


@lru_cache(maxsize=1)
def _xformers_available() -> bool:
    try:
        memory_efficient_attention = importlib.import_module("xformers.ops").memory_efficient_attention
    except ImportError:
        return False
    return memory_efficient_attention is not None


class AttentionFunction(Enum):
    """Keep Echo's published backend names and xFormers-first default."""

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = self
        if selected is AttentionFunction.DEFAULT:
            selected = AttentionFunction.XFORMERS if _xformers_available() else AttentionFunction.PYTORCH
        if selected is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        if selected is AttentionFunction.FLASH_ATTENTION_3:
            # Echo loads FA3 only when xFormers is absent.
            if _xformers_available():
                raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")
            if mask is not None:
                raise NotImplementedError("Mask is not supported for FlashAttention3")
            return FlashAttention3()(q, k, v, heads)
        # Do not pin a new priority: Echo honors the caller's live SDPA context.
        return flattened_multihead_attention(q, k, v, heads, attn_mask=mask)


def build_attention_ops(attention: AttentionFunction = AttentionFunction.DEFAULT) -> TransformerAttentionOps:
    """Use the same selected backend for masked and unmasked calls."""
    if attention is AttentionFunction.DEFAULT:
        attention = AttentionFunction.XFORMERS if _xformers_available() else AttentionFunction.PYTORCH
    return TransformerAttentionOps(attention_function=attention, masked_attention_function=attention)


def build_transformer_ops(attention: AttentionFunction = AttentionFunction.DEFAULT) -> TransformerOpsConfig:
    """Preserve eager torch AdaLN rounding and Echo's attention policy."""
    return TransformerOpsConfig(
        attention_ops=build_attention_ops(attention),
        ada_zero_function=PytorchAdaZeroFunction(backend="torch"),
    )
