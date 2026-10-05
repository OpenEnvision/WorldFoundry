"""Numerical and artifact contracts for the independent packed SVDQuant path."""

import pytest
import torch
from torch import nn

from worldfoundry.core.acceleration.quantization.calibration import (
    ChannelObserver,
    load_calibration,
    save_calibration,
)
from worldfoundry.core.acceleration.quantization.svdquant import (
    PackedSVDQuantLinear,
    calibrate_svdquant,
    validate_svdquant_state,
)


def _state(dtype=torch.float32, device="cpu"):
    source = nn.Linear(128, 80, dtype=dtype, device=device).eval()
    return source, calibrate_svdquant(source, torch.ones(128, device=device), rank=16)


def test_calibration_observes_then_unhooks_and_binds_weights(tmp_path):
    source, state = _state()
    with ChannelObserver({"linear": source}) as observer:
        source(torch.full((3, 128), 2.0))
        source(torch.ones(1, 128))
    observer.validate()
    assert observer.calls == {"linear": 2}
    torch.testing.assert_close(observer.maxima["linear"], torch.full((128,), 2.0))
    assert not source._forward_pre_hooks
    artifact = tmp_path / "calibration.pt"
    save_calibration(artifact, kind="svdquant", states={"linear": state}, metadata={"samples": [7, 8]})
    loaded = load_calibration(artifact, kind="svdquant")
    validate_svdquant_state(source, loaded["states"]["linear"])
    with pytest.raises(FileExistsError):
        save_calibration(artifact, kind="svdquant", states={"linear": state}, metadata={"seed": 1})
    with torch.no_grad():
        source.weight.add_(1)
    with pytest.raises(ValueError, match="source weights"):
        validate_svdquant_state(source, state)
    with pytest.raises(ValueError, match="weights changed"):
        observer.validate()


def test_observer_unhooks_on_failed_calibration():
    module = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="nonfinite"):
        with ChannelObserver({"linear": module}):
            module(torch.full((2, 4), float("nan")))
    assert not module._forward_pre_hooks


def test_deterministic_export_preserves_rng_and_rejects_bad_state():
    source, state = _state()
    before = torch.random.get_rng_state().clone()
    again = calibrate_svdquant(source, torch.ones(128), rank=16)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(state["qweight"], again["qweight"])
    broken = {**state, "smooth": state["smooth"].clone().zero_()}
    with pytest.raises(ValueError, match="positive"):
        validate_svdquant_state(source, broken)
    with pytest.raises(ValueError, match="rank"):
        calibrate_svdquant(source, torch.ones(128), rank=17)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_integer_kernel_matches_independent_groupwise_reference(dtype):
    source, state = _state(dtype, "cuda")
    quantized = PackedSVDQuantLinear(source, state)
    x = torch.randn(3, 25, 128, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual = quantized(x)
        normalized = x.float().reshape(-1, 128) / quantized.smooth
        groups = normalized.reshape(-1, 2, 64)
        scales = (groups.abs().amax(-1) / 7).clamp_min(1e-8)
        a = (groups / scales[..., None]).round().clamp(-7, 7)
        packed = quantized.qweight.to(torch.int32)
        unpacked = torch.stack((packed & 15, (packed >> 4) & 15), -1).flatten(-2)
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked).reshape(80, 2, 64).float()
        reference = sum(
            (a[:, group] @ unpacked[:, group].T) * scales[:, group, None] * quantized.scales[:, group][None]
            for group in range(2)
        )
        low = torch.nn.functional.linear(x.reshape(-1, 128), quantized.down)
        reference += low.float() @ quantized.up.float().T
        reference += quantized.bias.float()
        reference = reference.reshape_as(actual).to(dtype)
    torch.testing.assert_close(actual, reference, atol=0.008 if dtype == torch.bfloat16 else 0.001, rtol=0.008)
    report = quantized.runtime_report()
    assert report["native_packed_int4_calls"] == 1
    assert report["dense_compute_calls"] == report["dense_fallback_calls"] == 0
    quantized.reset_request_window()
    assert quantized.runtime_report()["low_precision_kernel_calls"] == 0
    with pytest.raises(RuntimeError, match="placement"):
        quantized.to("cpu")
    with pytest.raises(RuntimeError, match="training"):
        quantized.train()
