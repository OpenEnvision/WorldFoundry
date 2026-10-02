"""Safety contracts for inference-only alias-output diffusion kernels."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from worldfoundry.core.kernels import diffusion


def test_mutating_kernel_failure_never_reapplies_fallback(monkeypatch) -> None:
    from worldfoundry.core.kernels import triton_diffusion

    residual = torch.zeros(2, 4)
    update = torch.full_like(residual, 3)
    gate = torch.ones(1, 4)

    def partially_mutate_then_fail(
        residual: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        del update, gate
        residual[0].add_(1)
        raise RuntimeError("simulated post-launch failure")

    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    monkeypatch.setattr(diffusion, "_eligible_residual_gate_inplace", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        triton_diffusion,
        "residual_gate_inplace",
        partially_mutate_then_fail,
    )

    with pytest.raises(RuntimeError, match="post-launch"):
        diffusion.residual_gate_add_(residual, update, gate)

    # The first row contains only the simulated partial write. A generic
    # recover-and-fallback path would have added update*gate to every row.
    torch.testing.assert_close(residual[0], torch.ones(4))
    torch.testing.assert_close(residual[1], torch.zeros(4))


def test_mutating_contract_rejects_storage_aliases() -> None:
    residual = torch.randn(2, 4)
    assert diffusion._storage_overlaps(residual, residual)
    assert diffusion._storage_overlaps(residual, residual.view(-1))
    assert not diffusion._storage_overlaps(residual, residual.clone())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_mutating_kernel_preserves_alias_version_and_eager_bits(
    monkeypatch,
) -> None:
    from worldfoundry.core.kernels.registry import (
        kernel_dispatch_receipt_scope,
    )

    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    residual = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    expected = residual.clone().add_(update * gate)
    actual = residual.clone()
    pointer = actual.data_ptr()
    version = actual._version
    receipt: dict[str, object] = {}

    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        returned = diffusion.residual_gate_add_(actual, update, gate)
    torch.cuda.synchronize()

    assert returned is actual
    assert actual.data_ptr() == pointer
    assert actual._version > version
    assert torch.equal(actual, expected)
    dispatches = receipt["dispatches"]
    assert isinstance(dispatches, list)
    assert dispatches[-1]["implementation"] == (
        "triton_residual_gate_add_inplace"
    )
