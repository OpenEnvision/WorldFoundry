import copy

import pytest
import torch

from worldfoundry.core.acceleration.quantization.fused_ffn import FusedFP8GELUFeedForward, fuse_fp8_gelu_feed_forwards
from worldfoundry.core.acceleration.quantization.linear import Float8Linear, quantization_runtime_report
from worldfoundry.core.model_loading.optimize import apply_quantization_policy
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy


def _model(device="cpu", dtype=torch.float32, approximate="tanh"):
    torch.manual_seed(11)
    return (
        torch.nn.ModuleDict(
            {
                "ffn": torch.nn.Sequential(
                    torch.nn.Linear(64, 128), torch.nn.GELU(approximate=approximate), torch.nn.Linear(128, 64)
                )
            }
        )
        .to(device=device, dtype=dtype)
        .eval()
    )


def _quantize(model, fused):
    return apply_quantization_policy(
        model, QuantizationPolicy(mode=QuantizationMode.FP8, options={"min_features": 16, "fuse_fp8_ffn": fused})
    )


def test_opt_in_fusion_preserves_checkpoint_keys_and_exact_cpu_fallback():
    reference = _model()
    optimized = copy.deepcopy(reference)
    assert _quantize(optimized, True).extra["fused_fp8_ffn_blocks"] == 1
    assert isinstance(optimized["ffn"], FusedFP8GELUFeedForward)
    assert isinstance(optimized["ffn"][0], Float8Linear)
    keys = list(optimized.state_dict())
    assert "ffn.0.weight" in keys and "ffn.2.weight" in keys
    assert fuse_fp8_gelu_feed_forwards(optimized) == 0
    with torch.no_grad():
        value = torch.randn(2, 3, 64)
        torch.testing.assert_close(reference["ffn"](value), optimized["ffn"](value), rtol=0, atol=0)
    assert optimized["ffn"].fused_calls == 0
    assert optimized["ffn"].fallback_calls == 1
    assert quantization_runtime_report(optimized)["low_precision_kernel_calls"] == 0


def test_fusion_is_opt_in_and_rejects_wrong_quantization_mode():
    model = _model()
    _quantize(model, False)
    assert type(model["ffn"]) is torch.nn.Sequential
    with pytest.raises(ValueError, match="requires FP8"):
        apply_quantization_policy(
            _model(), QuantizationPolicy(mode=QuantizationMode.INT8, options={"fuse_fp8_ffn": True})
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gelu_quantization_matches_materialized_activation(approximate, dtype):
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    torch.manual_seed(18)
    value = torch.randn(7, 130, device="cuda", dtype=dtype)[:, ::2]
    value[0].zero_()
    value[1].fill_(3)
    activation = torch.nn.functional.gelu(value, approximate=approximate)
    expected, expected_scale = quantize_rowwise_fp8_triton(activation)
    got, got_scale = quantize_rowwise_fp8_triton(value, activation="gelu" if approximate == "none" else "gelu_tanh")
    torch.testing.assert_close(got_scale, expected_scale, atol=1e-8, rtol=0.002)
    torch.testing.assert_close(got.float() * got_scale, expected.float() * expected_scale, atol=0.002, rtol=0.06)
    assert torch.isfinite(got.float()).all()
    assert torch.count_nonzero(got[0].float()) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fused_ffn_real_fp8_graph_and_policy_fallback(monkeypatch, approximate):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("real FP8 scaled-mm requires SM90+")
    from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

    monkeypatch.setenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", "0")
    from worldfoundry.core.acceleration.quantization.linear import _fp8_min_gemm_work

    monkeypatch.setattr(_fp8_min_gemm_work, "_cached", None, raising=False)
    baseline = _model("cuda", torch.bfloat16, approximate)
    candidate = copy.deepcopy(baseline)
    _quantize(baseline, False)
    _quantize(candidate, True)
    graph = InferenceCUDAGraphRunner(candidate["ffn"].forward)
    with torch.no_grad():
        for seed in range(3):
            torch.manual_seed(seed)
            value = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
            expected = baseline["ffn"](value)
            actual = graph(value)
            torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.06)
        assert graph.report()["capture_failed"] == graph.report()["eager"] == 0
        assert graph.report()["replay"] == 3
        assert candidate["ffn"].fused_calls > 0
        assert quantization_runtime_report(candidate)["low_precision_kernel_calls"] > 0
        for layer in (candidate["ffn"][0], candidate["ffn"][2]):
            layer.low_precision_enabled = False
        graph.invalidate()
        # The fused activation path must obey policy disablement on dense fallback.
        actual = graph(value)
        expected = candidate["ffn"](value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert candidate["ffn"].fallback_calls > 0
        assert quantization_runtime_report(candidate)["dense_fallback_calls"] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_ffn_empty_input_and_invalid_activation(monkeypatch):
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    value = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    codes, scales = quantize_rowwise_fp8_triton(value, activation="gelu")
    assert codes.shape == value.shape and scales.shape == (0, 1)
    with pytest.raises(ValueError, match="activation"):
        quantize_rowwise_fp8_triton(value, activation="other")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_ffn_inductor_preserves_real_fp8_math(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 requires SM90+")
    from worldfoundry.core.acceleration.quantization.linear import _fp8_min_gemm_work

    monkeypatch.setenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", "0")
    monkeypatch.setattr(_fp8_min_gemm_work, "_cached", None, raising=False)
    model = _model("cuda", torch.bfloat16)
    _quantize(model, True)
    compiled = torch.compile(model["ffn"], backend="inductor", fullgraph=True)
    with torch.no_grad():
        for index in range(2):
            value = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16) + index
            torch.testing.assert_close(compiled(value), model["ffn"](value), rtol=0.06, atol=0.004)
