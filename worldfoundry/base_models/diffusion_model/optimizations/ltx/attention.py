"""Execution policy for Lightricks LTX split checkpoints.

Derived from Lightricks/LTX-2@a95ab856 (LTX-2.x Community License; see
THIRD-PARTY-NOTICES). The released runtime selects installed FlashAttention
on Hopper/Blackwell and MPSGraph SDPA on Apple Silicon, and evaluates AdaLN
with eager PyTorch arithmetic. Other native recipes retain the core defaults;
the split-checkpoint loader opts into this policy.
"""

import importlib
import sys
from functools import lru_cache

import torch
from torch.nn.attention import SDPBackend

from worldfoundry.core.attention.backends.model_backends import (
    AttentionCallable,
    FlashAttention3,
    FlashAttention4,
    MaskedAttentionCallable,
    PytorchAttention,
)
from worldfoundry.core.nn import (
    PytorchAdaZeroFunction,
    TransformerAttentionOps,
    TransformerOpsConfig,
)


class MPSSdpaAttention:
    """Run the optional MPSGraph SDPA kernel on ``(batch, tokens, heads * head_dim)`` inputs."""

    label = "MPS-SDPA"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accept additive or boolean masks broadcastable to ``(batch, heads, queries, keys)``."""
        from mps_sdpa import sdpa_opt

        if q.device.type != "mps":
            raise RuntimeError("MPSSdpaAttention requires MPS tensors.")
        batch, _, width = q.shape
        head_dim = width // heads
        query, key, value = (tensor.view(batch, -1, heads, head_dim).transpose(1, 2) for tensor in (q, k, v))
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        output = sdpa_opt(query, key, value, attn_mask=mask)
        return output.transpose(1, 2).reshape(batch, -1, width)


@lru_cache(maxsize=1)
def split_checkpoint_attention_ops() -> TransformerAttentionOps:
    """Resolve the released unmasked/masked policy outside compiled forwards."""
    priority = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    primary: AttentionCallable = PytorchAttention(priority=priority)
    masked: MaskedAttentionCallable = PytorchAttention(priority=priority)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        try:
            flash3 = importlib.import_module("flash_attn_interface")
        except ImportError:
            flash3 = None
        try:
            flash4 = importlib.import_module("flash_attn.cute").flash_attn_func
        except Exception:
            # The supported FA4 extra can fail on a mismatched cutlass-dsl build.
            flash4 = None
        if major == 9 and flash3 is not None:
            primary = FlashAttention3()
        elif major in (9, 10) and flash4 is not None:
            primary = FlashAttention4()
    if sys.platform == "darwin":
        try:
            importlib.import_module("mps_sdpa")
        except ImportError:
            pass
        else:
            primary = MPSSdpaAttention()
            masked = MPSSdpaAttention()
    return TransformerAttentionOps(attention_function=primary, masked_attention_function=masked)


def split_checkpoint_transformer_ops() -> TransformerOpsConfig:
    """Combine the released attention selection with eager AdaLN arithmetic."""
    return TransformerOpsConfig(
        attention_ops=split_checkpoint_attention_ops(),
        ada_zero_function=PytorchAdaZeroFunction(backend="torch"),
    )
