"""Tests for the approximate-attention dense-boundary step schedule.

The lossy sparse lane runs dense on the first/last ``dense_steps`` denoise
steps (which set global structure) and sparse in between. This locks the
schedule logic and the per-step counter on CPU (no kernel needed: without
fastvideo_kernel every step falls back to exact, but the schedule decision —
is_dense_step — is what we verify).
"""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    advance_approximate_step,
    install_approximate_attention,
    reset_approximate_attention,
)


def _state(dense_steps: int, total: int):
    m = torch.nn.Module()
    m.blk = SelfAttention(64, 4)
    st = install_approximate_attention(m, ApproximateAttentionConfig(kind="vsa", dense_steps=dense_steps))
    st.total_steps = total
    return st


def test_no_dense_steps_never_dense() -> None:
    st = _state(0, 10)
    assert all(not (st.step == i and st.is_dense_step()) for i in range(10))


def test_boundary_steps_are_dense() -> None:
    st = _state(2, 10)
    dense_flags = []
    reset_approximate_attention(st)
    for _ in range(10):
        dense_flags.append(st.is_dense_step())
        advance_approximate_step(st, total_steps=10)
    # First 2 and last 2 steps dense; middle 6 sparse.
    assert dense_flags[0] and dense_flags[1]
    assert dense_flags[8] and dense_flags[9]
    assert not any(dense_flags[2:8])


def test_reset_restarts_schedule() -> None:
    st = _state(1, 5)
    for _ in range(5):
        advance_approximate_step(st, total_steps=5)
    assert st.step == 5
    reset_approximate_attention(st)
    assert st.step == 0
    assert st.is_dense_step()  # step 0 is a boundary
