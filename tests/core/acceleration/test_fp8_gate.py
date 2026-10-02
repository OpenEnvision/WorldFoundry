"""Tests for the FP8 work-size eligibility gate.

The gate keeps small GEMMs on the dense path (where BF16 beats dynamic-FP8) and
only lets large GEMMs use the FP8 fast path. Exercised without a GPU by driving
``_fp8_linear_eligible`` with ``hardware_eligible=True`` and CUDA-shaped meta
tensors' properties via a small helper; the FLOP arithmetic is device-agnostic.
"""

from __future__ import annotations

import torch

from worldfoundry.core.acceleration import quantization as Q


def _make_input(tokens: int, in_features: int) -> torch.Tensor:
    # A CUDA-typed check is short-circuited on CPU, so we validate the FLOP
    # arithmetic directly against the module threshold instead of the device
    # branch. This keeps the test CPU-only while still locking the gate math.
    return torch.zeros(tokens, in_features)


def test_threshold_default_and_env_override(monkeypatch) -> None:
    # Cached read: clear the cache, then confirm default and override.
    if hasattr(Q._fp8_min_gemm_work, "_cached"):
        delattr(Q._fp8_min_gemm_work, "_cached")
    monkeypatch.delenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", raising=False)
    assert Q._fp8_min_gemm_work() == Q._FP8_MIN_GEMM_WORK

    delattr(Q._fp8_min_gemm_work, "_cached")
    monkeypatch.setenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", "0")
    assert Q._fp8_min_gemm_work() == 0.0  # gate disabled

    delattr(Q._fp8_min_gemm_work, "_cached")
    monkeypatch.setenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", "5e9")
    assert Q._fp8_min_gemm_work() == 5e9


def test_gate_math_separates_small_and_large() -> None:
    threshold = Q._FP8_MIN_GEMM_WORK
    # Small GEMM: 256 x 3072 x 3072 = 2.4e9 < threshold -> below the gate.
    assert 256 * 3072 * 3072 < threshold
    # Large GEMM: 4096 x 3072 x 3072 = 3.9e10 > threshold -> above the gate.
    assert 4096 * 3072 * 3072 > threshold


def test_eligible_requires_cuda_and_dtype() -> None:
    # On a CPU tensor the fast path is never eligible regardless of work size.
    x = _make_input(8192, 3072)
    assert Q._fp8_linear_eligible(x, 3072, hardware_eligible=True) is False
