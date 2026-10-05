"""CPU-only tests for the load-time quantization transform dispatch.

GPU parity/speedup for FP8 is covered by the H100 benchmarks; here we lock the
policy dispatch: none is a no-op, unwired modes report honestly, exclude is
honored, and the dense fallback keeps a transformed model runnable on CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.core.acceleration.quantization.linear import (
    _int8_linear_profitable,
    quantization_runtime_report,
    reset_quantization_runtime_window,
)
from worldfoundry.core.model_loading.optimize import apply_quantization_policy
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.big = nn.Linear(1024, 1024)
        self.small = nn.Linear(16, 16)
        self.keep_me = nn.Linear(1024, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.big(x)


def test_none_mode_is_noop() -> None:
    net = _Net()
    report = apply_quantization_policy(net, QuantizationPolicy(mode=QuantizationMode.NONE))
    assert report.applied is False
    assert report.replaced_modules == 0
    assert isinstance(net.big, nn.Linear)  # untouched


def test_int8_mode_installs_real_packed_weight_storage() -> None:
    net = _Net()
    report = apply_quantization_policy(
        net,
        QuantizationPolicy(
            mode=QuantizationMode.INT8,
            options={"min_features": 512, "group_size": 64},
        ),
    )
    assert report.applied is True
    assert type(net.big).__name__ == "WeightOnlyLinear"
    assert net.big.qweight.dtype == torch.int8
    assert net.big.weight is None
    out = net.big(torch.randn(2, 1024))
    assert out.shape == (2, 1024)
    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["effective"] == "int8-weight-only-dequantize+dense-gemm"
    assert runtime["packed_weight_calls"] == 1


def test_quantization_report_is_request_local_after_reset() -> None:
    net = _Net()
    apply_quantization_policy(
        net,
        QuantizationPolicy(
            mode=QuantizationMode.INT8,
            options={"min_features": 512, "group_size": 64},
        ),
    )
    net.big(torch.randn(2, 1024))
    assert reset_quantization_runtime_window(net) == 2

    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["effective"] == "pending"
    assert runtime["packed_weight_calls"] == 0
    assert runtime["lifetime_packed_weight_calls"] == 1

    net.big(torch.randn(2, 1024))
    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["packed_weight_calls"] == 1
    assert runtime["lifetime_packed_weight_calls"] == 2


def test_default_int8_retains_dense_weights_for_profitability_dispatch() -> None:
    net = _Net()

    report = apply_quantization_policy(
        net,
        QuantizationPolicy(
            mode=QuantizationMode.INT8,
            options={"min_features": 512},
        ),
    )

    assert report.extra["dense_fallback_retained"] is True
    assert net.big.weight is not None
    net.big.low_precision_kernel_calls = 2
    net.big.dense_policy_calls = 3
    net.big.last_policy_reason = "calibrated small GEMM"
    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["effective"] == "int8-kernel+calibrated-dense"
    assert runtime["dense_policy_calls"] == 3
    assert runtime["dense_fallback_calls"] == 0
    assert runtime["policy_reasons"] == ["calibrated small GEMM"]


def test_int8_exclude_matches_qualified_module_path() -> None:
    net = nn.ModuleDict(
        {
            "first": nn.ModuleDict({"projection": nn.Linear(1024, 1024)}),
            "second": nn.ModuleDict({"projection": nn.Linear(1024, 1024)}),
        }
    )

    apply_quantization_policy(
        net,
        QuantizationPolicy(
            mode=QuantizationMode.INT8,
            exclude=("first.projection",),
            options={"min_features": 512},
        ),
    )

    assert isinstance(net["first"]["projection"], nn.Linear)
    assert type(net["second"]["projection"]).__name__ == "WeightOnlyLinear"


def test_int8_profitability_gate_rejects_measured_small_shape() -> None:
    small = torch.empty(8192, 1024, device="meta")
    wan_sp4 = torch.empty(27280, 3072, device="meta")

    assert _int8_linear_profitable(small, 1024) is False
    assert _int8_linear_profitable(wan_sp4, 9216) is True


def test_int4_mode_packs_two_codes_per_byte() -> None:
    layer = nn.Sequential(nn.Linear(64, 64))
    report = apply_quantization_policy(
        layer,
        QuantizationPolicy(
            mode=QuantizationMode.INT4,
            options={"min_features": 1, "group_size": 32},
        ),
    )
    assert report.applied is True
    quantized = layer[0]
    assert type(quantized).__name__ == "WeightOnlyLinear"
    assert quantized.qweight.dtype == torch.uint8
    assert quantized.qweight.numel() == 64 * 64 // 2
    assert quantized(torch.randn(3, 64)).shape == (3, 64)


def test_gguf_mode_repacks_loaded_weights_for_compressed_runtime_storage() -> None:
    net = _Net()
    report = apply_quantization_policy(
        net,
        QuantizationPolicy(
            mode=QuantizationMode.GGUF,
            options={"min_features": 512, "runtime_bits": 4, "group_size": 64},
        ),
    )

    assert report.applied is True
    assert report.extra["storage"] == "gguf-loaded+groupwise-int4"
    assert type(net.big).__name__ == "WeightOnlyLinear"
    assert net.big.qweight.dtype == torch.uint8
    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["low_precision_kernel_calls"] == 0


def test_gguf_mode_rejects_an_unknown_runtime_width() -> None:
    with pytest.raises(ValueError, match="runtime_bits must be 4 or 8"):
        apply_quantization_policy(
            _Net(),
            QuantizationPolicy(
                mode=QuantizationMode.GGUF,
                options={"runtime_bits": 3},
            ),
        )


def test_fp8_replaces_eligible_and_honors_exclude() -> None:
    net = _Net()
    policy = QuantizationPolicy(
        mode=QuantizationMode.FP8,
        exclude=("keep_me",),
        options={"min_features": 512},
    )
    report = apply_quantization_policy(net, policy)
    assert report.applied is True
    # big is eligible (1024>=512); small is too narrow; keep_me is excluded.
    assert report.replaced_modules == 1
    assert type(net.big).__name__ == "Float8Linear"
    assert isinstance(net.small, nn.Linear)
    assert isinstance(net.keep_me, nn.Linear)


def test_fp8_dense_fallback_runs_on_cpu() -> None:
    # On CPU the FP8 fast path is ineligible, but the retained dense weight must
    # keep the transformed module numerically equal to the original linear.
    net = _Net()
    x = torch.randn(4, 1024)
    original_weight = net.big.weight.detach().clone()
    original_bias = net.big.bias.detach().clone()
    reference = torch.nn.functional.linear(x, original_weight, original_bias)
    apply_quantization_policy(net, QuantizationPolicy(mode=QuantizationMode.FP8, options={"min_features": 512}))
    out = net.big(x)
    assert out.shape == (4, 1024)
    torch.testing.assert_close(out, reference)
    runtime = quantization_runtime_report(net)
    assert runtime is not None
    assert runtime["effective"] == "dense"
    assert runtime["dense_fallback_calls"] == 1
    assert runtime["fallback_reasons"] == ["FP8 kernel requires CUDA, got cpu"]
