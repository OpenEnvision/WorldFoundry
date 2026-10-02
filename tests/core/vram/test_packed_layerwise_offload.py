from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from worldfoundry.core.vram import layerwise_offload
from worldfoundry.core.vram.layerwise_offload import (
    enable_layerwise_cpu_offload,
    layerwise_offload_mutation_scope,
)


class Stack(nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(count)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).tanh()
        return x


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("count", [1, 3, 4])
def test_cuda_slot_reuse_parity_and_disable(count):
    model = Stack(count).cuda().eval()
    baseline = copy.deepcopy(model)
    identities = [id(p) for p in model.parameters()]
    handle = enable_layerwise_cpu_offload(model, device="cuda", reuse_buffers=True)
    state = model.layers[0]._worldfoundry_layerwise_cpu_offload_state
    pointers = [[b.data_ptr() for b in slot.buffers.values()] for slot in state.pool.slots]
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(30):
                x = torch.randn(8, 32, device="cuda")
                torch.testing.assert_close(model(x), baseline(x))
        stream.synchronize()
        assert pointers == [[b.data_ptr() for b in slot.buffers.values()] for slot in state.pool.slots]
        assert identities == [id(p) for p in model.parameters()]
        report = handle.report()
        assert report["buffer_mode"] == "packed-two-slot"
        assert report["effective"]
        assert report["request"]["peak_active_layers"] <= min(2, count)
        assert report["request"]["synchronous_copy_tensors"] == 0
        with pytest.raises(RuntimeError, match="state_dict"):
            model.state_dict()
    finally:
        handle.disable()
    assert all(p.device.type == "cpu" for p in model.parameters())
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, baseline.state_dict()[name].cpu())


def test_mutation_and_failed_forward_restore_slots():
    model = Stack().cuda().eval()
    baseline = copy.deepcopy(model)
    handle = enable_layerwise_cpu_offload(model, reuse_buffers=True)
    try:
        with torch.no_grad(), layerwise_offload_mutation_scope(model.layers[1]):
            model.layers[1].weight.add_(0.25)
            baseline.layers[1].weight.add_(0.25)
        original = model.layers[1].forward

        def fail(x):
            raise ValueError("failed block")

        model.layers[1].forward = fail
        with torch.no_grad(), pytest.raises(ValueError, match="failed block"):
            model(torch.randn(8, 32, device="cuda"))
        model.layers[1].forward = original
        with torch.no_grad():
            x = torch.randn(8, 32, device="cuda")
            torch.testing.assert_close(model(x), baseline(x))
    finally:
        handle.disable()


def test_shared_weights_rejected_before_mutation():
    model = Stack().cuda()
    model.layers[1].weight = model.layers[0].weight
    shape = model.layers[0].weight.shape
    with pytest.raises(ValueError, match="shared parameter"):
        enable_layerwise_cpu_offload(model, reuse_buffers=True)
    assert model.layers[0].weight.shape == shape


def test_grad_mode_is_rejected_and_legacy_mode_remains_available():
    model = Stack().cuda().eval()
    handle = enable_layerwise_cpu_offload(model, reuse_buffers=True)
    try:
        with pytest.raises(RuntimeError, match="no_grad"):
            model(torch.randn(8, 32, device="cuda"))
    finally:
        handle.disable()
    legacy = enable_layerwise_cpu_offload(model.cuda(), reuse_buffers=False)
    try:
        with torch.no_grad():
            assert model(torch.randn(8, 32, device="cuda")).shape == (8, 32)
        assert legacy.report()["buffer_mode"] == "per-parameter"
    finally:
        legacy.disable()


def test_allocation_failure_restores_parameters_and_hooks(monkeypatch):
    model = Stack().cuda().eval()
    original = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}

    def fail(self, states):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(layerwise_offload._PackedBufferPool, "initialize", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        enable_layerwise_cpu_offload(model, reuse_buffers=True)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), original[name])
    assert not getattr(model, "_worldfoundry_layerwise_cpu_offload", False)
    assert all(not layer._forward_pre_hooks and not layer._forward_hooks for layer in model.layers)


def test_queued_forwards_with_unequal_layers_and_mixed_dtypes():
    class ScaledLinear(nn.Linear):
        def __init__(self, input_width, output_width):
            super().__init__(input_width, output_width, dtype=torch.float16, device="cuda")
            self.gain = nn.Parameter(torch.tensor(0.9, dtype=torch.float32, device="cuda"))

        def forward(self, x):
            return super().forward(x) * self.gain

    model = Stack(0)
    model.layers.extend([ScaledLinear(32, 48), ScaledLinear(48, 16), ScaledLinear(16, 32)])
    baseline = copy.deepcopy(model)
    handle = enable_layerwise_cpu_offload(model, reuse_buffers=True)
    try:
        results = []
        with torch.no_grad():
            for _ in range(20):
                x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
                results.append((model(x), baseline(x)))
        torch.cuda.synchronize()
        for actual, expected in results:
            torch.testing.assert_close(actual, expected)
    finally:
        handle.disable()
