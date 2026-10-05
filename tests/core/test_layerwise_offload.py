from __future__ import annotations

import copy

import pytest
import torch

from worldfoundry.core.vram.layerwise_offload import enable_layerwise_cpu_offload


class _TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(32, 64),
                    torch.nn.SiLU(),
                    torch.nn.Linear(64, 32),
                )
                for _ in range(4)
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = value + block(value)
        return value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_layerwise_offload_matches_resident_and_proves_double_buffer() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    cpu_model = _TinyTransformer().eval()
    resident = copy.deepcopy(cpu_model).to(device).eval()
    offloaded = copy.deepcopy(cpu_model).eval()
    sample = torch.randn(2, 8, 32, device=device)

    with torch.inference_mode():
        expected = resident(sample)

    handle = enable_layerwise_cpu_offload(
        offloaded,
        layer_container="blocks",
        device=device,
        pin_memory=True,
    )
    assert handle.enabled is True
    # This mirrors NativeModuleLoader: only non-block tensors are moved after
    # the block parameters have become zero-sized CUDA placeholders.
    offloaded.to(device)
    handle.reset_request_window()

    with torch.inference_mode():
        actual = offloaded(sample)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    report = handle.report()
    assert report["effective"] is True
    assert report["mode"] == "async-double-buffer"
    assert report["request"]["forward_calls"] == len(offloaded.blocks)
    assert report["request"]["async_copy_tensors"] > 0
    assert report["request"]["synchronous_copy_tensors"] == 0
    assert report["request"]["pageable_cpu_tensors"] == 0
    assert report["request"]["peak_active_layers"] == 2

    assert handle.disable() is True
    assert all(parameter.device.type == "cpu" for parameter in offloaded.parameters())
