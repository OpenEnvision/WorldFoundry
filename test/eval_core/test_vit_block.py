"""Focused tests for the shared ViT residual scheduling helper."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from worldfoundry.core.nn.vit_block import apply_prenorm_transformer_residuals


class _RecordingScale(nn.Module):
    """Deterministic residual-path stand-in that records its invocations."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, value: Tensor) -> Tensor:
        self.calls += 1
        return value * self.scale


def test_moderate_stochastic_depth_uses_ffn_drop_path() -> None:
    """Attention and FFN use their respective paths in the per-sample branch."""

    x = torch.zeros(2, 3)
    attention_path = _RecordingScale(2.0)
    ffn_path = _RecordingScale(3.0)

    result = apply_prenorm_transformer_residuals(
        x,
        attn_residual=lambda value: torch.ones_like(value),
        ffn_residual=lambda value: torch.ones_like(value),
        sample_drop_ratio=0.05,
        drop_path1=attention_path,
        drop_path2=ffn_path,
        training=True,
    )

    torch.testing.assert_close(result, torch.full_like(x, 5.0))
    assert attention_path.calls == 1
    assert ffn_path.calls == 1


def test_missing_ffn_drop_path_preserves_legacy_fallback() -> None:
    """Callers that omit the optional second path retain the old behavior."""

    x = torch.zeros(1, 2)
    shared_path = _RecordingScale(4.0)

    result = apply_prenorm_transformer_residuals(
        x,
        attn_residual=lambda value: torch.ones_like(value),
        ffn_residual=lambda value: torch.ones_like(value),
        sample_drop_ratio=0.05,
        drop_path1=shared_path,
        training=True,
    )

    torch.testing.assert_close(result, torch.full_like(x, 8.0))
    assert shared_path.calls == 2
