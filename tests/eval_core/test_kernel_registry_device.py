from __future__ import annotations

import pytest
import torch

from worldfoundry.core.kernels.registry import KernelRegistry


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_kernel_candidate_uses_input_cuda_device_on_selection_and_cache_hit() -> None:
    registry = KernelRegistry()
    observed_devices: list[int] = []

    def implementation(value: torch.Tensor) -> torch.Tensor:
        observed_devices.append(torch.cuda.current_device())
        return value

    registry.register(
        "device_guard_fixture",
        backend="triton",
        name="fixture",
        implementation=implementation,
        predicate=lambda value: value.is_cuda,
    )

    original_device = torch.cuda.current_device()
    input_device = 1 if original_device == 0 else 0
    value = torch.empty(1, device=f"cuda:{input_device}")
    signature = (str(value.device), tuple(value.shape), str(value.dtype))

    try:
        torch.cuda.set_device(original_device)
        assert registry.dispatch("device_guard_fixture", lambda item: item, value, signature=signature) is value
        assert torch.cuda.current_device() == original_device

        # The second invocation exercises the cached-selection fast path.
        assert registry.dispatch("device_guard_fixture", lambda item: item, value, signature=signature) is value
        assert torch.cuda.current_device() == original_device
    finally:
        torch.cuda.set_device(original_device)

    assert observed_devices == [input_device, input_device]
