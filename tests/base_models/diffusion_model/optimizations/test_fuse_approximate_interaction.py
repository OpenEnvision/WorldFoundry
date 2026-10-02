"""Regression: merged QKV must compose with approximate-attention dispatch.

Approximate processors explicitly advertise fused-QKV support.  Fusion then
merges only the projections while retaining ``SelfAttention.forward`` and its
processor dispatch, so both acceleration receipts come from real execution.
Unknown non-default processors remain fail-closed and are not fused.
"""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    ApproximateSelfAttentionProcessor,
    install_approximate_attention,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections


class _Wrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = SelfAttention(256, 8)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.a(x, freqs)


class _UnknownProcessor:
    def __call__(self, attention, x, freqs, **kwargs):
        del freqs, kwargs
        return attention.o(attention.q(x))


def test_fusion_composes_with_approximate_processor_blocks() -> None:
    m = _Wrap().eval()
    state = install_approximate_attention(m, ApproximateAttentionConfig(kind="vsa"))
    assert state.wrapped_blocks == 1
    original_forward = m.a.forward.__func__
    fused = fuse_qkv_projections(m)
    assert fused == 1
    assert hasattr(m.a, "qkv") and not hasattr(m.a, "q")
    assert isinstance(m.a.get_processor(), ApproximateSelfAttentionProcessor)
    assert m.a.forward.__func__ is original_forward


def test_fusion_alone_still_fuses() -> None:
    m = _Wrap().eval()
    fused = fuse_qkv_projections(m)
    assert fused == 1
    assert hasattr(m.a, "qkv") and not hasattr(m.a, "q")


def test_default_processor_block_is_fusible() -> None:
    # A block with the default SelfAttentionProcessor (unchanged) must fuse.
    m = _Wrap().eval()
    assert type(m.a.get_processor()).__name__ == "SelfAttentionProcessor"
    assert fuse_qkv_projections(m) == 1


def test_unknown_processor_without_fused_projection_contract_is_skipped() -> None:
    m = _Wrap().eval()
    m.a.set_processor(_UnknownProcessor())

    assert fuse_qkv_projections(m) == 0
    assert hasattr(m.a, "q") and not hasattr(m.a, "qkv")
