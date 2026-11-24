# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Explicit checkpoint quantizers preserve stored codes, scales and placement."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from worldfoundry.core.acceleration.quantization.cast_linear import Float8CastLinear
from worldfoundry.core.acceleration.quantization.checkpoint import construct_linear_like, prepare_quantized_linears
from worldfoundry.core.acceleration.quantization.linear import Float8Linear
from worldfoundry.core.acceleration.quantization.nvfp4 import NVFP4Linear
from worldfoundry.core.model_loading.optimize import AppliedOptimizations, QuantizationReport
from worldfoundry.core.model_loading.policy import (
    OffloadMode,
    OffloadPolicy,
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)
from worldfoundry.core.vram import AutoWrappedModule


@pytest.mark.parametrize("bias", [False, True])
def test_construct_linear_like_preserves_configuration_and_constructor_dtype(bias: bool) -> None:
    source = torch.nn.Linear(4, 8, bias=bias, dtype=torch.float16)
    result = construct_linear_like(source, linear_cls=torch.nn.Linear)
    assert type(result) is torch.nn.Linear
    assert (result.in_features, result.out_features) == (4, 8)
    assert (result.bias is not None) is bias
    assert result.weight.device == source.weight.device
    assert result.weight.dtype == torch.get_default_dtype()
    assert result.weight is not source.weight


class _OneLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(16, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _cast_adapter(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    converted = prepare_quantized_linears(
        model,
        state,
        mode=QuantizationMode.FP8_CAST,
        linear_names=("projection",),
        device=torch.device("cpu"),
    )
    cast(Any, model)._worldfoundry_checkpoint_quantization = QuantizationReport(
        requested_mode="fp8-cast",
        applied=True,
        replaced_modules=1,
        scaling="unscaled",
    )
    return converted


@pytest.mark.parametrize("offload", [OffloadMode.NONE, OffloadMode.DISK])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp8_cast_checkpoint_loading_and_disk_restore_keep_raw_storage_rounding(
    tmp_path: Path,
    offload: OffloadMode,
    dtype: torch.dtype,
) -> None:
    weight = torch.full((16, 16), 1.0626)
    bias = torch.linspace(-0.9, 0.9, 16)
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"projection.weight": weight, "projection.bias": bias}, str(checkpoint))
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=_OneLinear,
            lazy_state_dict=True,
            state_dict_adapter=_cast_adapter,
            buffer_aware_offload=True,
            vram_module_map={Float8CastLinear: AutoWrappedModule},
        ),
        CheckpointSpec(source=str(checkpoint)),
        RuntimePolicy(
            dtype=dtype,
            device="cpu",
            offload=OffloadPolicy(mode=offload, target="disk"),
            quantization=QuantizationPolicy(mode=QuantizationMode.FP8_CAST),
        ),
    )
    x = torch.linspace(-0.5, 0.5, 32, dtype=dtype).reshape(2, 16)
    expected = torch.nn.functional.linear(
        x,
        weight.to(torch.float8_e4m3fn).to(dtype),
        bias.to(torch.float8_e4m3fn).to(dtype),
    )
    for _ in range(2):
        torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)
        if offload is OffloadMode.DISK:
            wrapped = loaded.projection
            assert isinstance(wrapped, AutoWrappedModule)
            wrapped.offload()
            assert isinstance(wrapped.module, Float8CastLinear)
            assert wrapped.module.weight.device.type == "meta"
    applied = loaded._worldfoundry_applied_optimizations
    assert isinstance(applied, AppliedOptimizations)
    snapshot = applied.to_optimization_snapshot()
    assert snapshot.requested["quantization"] == "fp8-cast"


def test_fp8_cast_folds_calibrated_weight_and_bias_scales_before_casting() -> None:
    model = _OneLinear()
    codes = torch.full((16, 16), 2, dtype=torch.float8_e4m3fn)
    bias = torch.ones(16, dtype=torch.float8_e4m3fn)
    converted = prepare_quantized_linears(
        model,
        {
            "projection.weight": codes,
            "projection.weight_scale": torch.tensor(0.25),
            "projection.bias": bias,
            "projection.bias_scale": torch.tensor(0.5),
            "projection.input_scale": torch.tensor(0.1),
        },
        mode=QuantizationMode.FP8_CAST,
        linear_names=("projection",),
        device=torch.device("cpu"),
    )
    model.load_state_dict(converted, assign=True)
    model.to(dtype=torch.bfloat16)
    assert model.projection.weight.dtype is torch.float8_e4m3fn
    assert model.projection.bias.dtype is torch.float8_e4m3fn
    assert set(converted) == {"projection.weight", "projection.bias"}
    torch.testing.assert_close(model.projection.weight.float(), torch.full((16, 16), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(model.projection.bias.float(), torch.full((16,), 0.5), rtol=0, atol=0)


def test_prequantized_fp8_restore_keeps_calibration_and_does_not_enable_dense_fallback() -> None:
    model = _OneLinear()
    codes = torch.arange(256).reshape(16, 16).float().to(torch.float8_e4m3fn)
    converted = prepare_quantized_linears(
        model,
        {
            "projection.weight": codes,
            "projection.weight_scale": torch.tensor(0.125),
            "projection.input_scale": torch.tensor(0.0625),
            "projection.bias": torch.ones(16),
        },
        mode=QuantizationMode.FP8_SCALED_MM,
        linear_names=("projection",),
        device=torch.device("cpu"),
    )
    model.load_state_dict(converted, assign=True)
    model.to(dtype=torch.bfloat16)
    linear = model.projection
    assert isinstance(linear, Float8Linear)
    assert linear.weight is None
    assert linear.weight_scale.dtype is torch.float32
    assert linear.input_scale.dtype is torch.float32
    torch.testing.assert_close(linear.weight_fp8.view(torch.uint8), codes.view(torch.uint8), rtol=0, atol=0)
    assert linear.input_scale.item() == 0.0625
    with pytest.raises(RuntimeError, match="requires CUDA"):
        model(torch.ones(2, 16, dtype=torch.bfloat16))


def test_prequantized_fp8_requires_calibrated_input_scale() -> None:
    with pytest.raises(ValueError, match="calibrated input/weight scales"):
        prepare_quantized_linears(
            _OneLinear(),
            {"projection.weight": torch.ones(16, 16), "projection.weight_scale": torch.tensor(0.1)},
            mode=QuantizationMode.FP8_SCALED_MM,
            linear_names=("projection",),
            device=torch.device("cpu"),
        )


def test_prequantized_fp8_adds_bias_after_the_scaled_gemm(monkeypatch) -> None:
    linear = Float8Linear.from_prequantized(
        torch.ones(16, 16, dtype=torch.float8_e4m3fn),
        torch.tensor(0.5),
        torch.tensor(0.25),
        torch.arange(16, dtype=torch.bfloat16),
    )
    calls = []

    def scaled_mm(a, b, scale_a, scale_b, **kwargs):
        calls.append((scale_a, scale_b, kwargs))
        return torch.full((a.shape[0], b.shape[1]), 2, dtype=kwargs["out_dtype"])

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    result = linear._forward_quantized(
        torch.ones(2, 16, dtype=torch.float8_e4m3fn),
        linear.input_scale,
        (2, 16),
        torch.bfloat16,
    )
    assert calls[0][2]["bias"] is None
    assert calls[0][2]["use_fast_accum"] is True
    torch.testing.assert_close(result, (torch.arange(16, dtype=torch.bfloat16) + 2).expand(2, -1), rtol=0, atol=0)


@pytest.mark.parametrize("mode", [QuantizationMode.NVFP4_CAST, QuantizationMode.NVFP4_PREQUANT])
def test_nvfp4_optional_provider_is_lazy_and_has_an_install_error(monkeypatch, mode: QuantizationMode) -> None:
    from worldfoundry.core.acceleration.quantization.nvfp4 import ltx_nvfp4_kernels

    real_import = builtins.__import__

    def without_kernels(name, *args, **kwargs):
        if name == "ltx_kernels" or name.startswith("ltx_kernels."):
            raise ModuleNotFoundError("ltx_kernels")
        return real_import(name, *args, **kwargs)

    ltx_nvfp4_kernels.cache_clear()
    monkeypatch.setattr(builtins, "__import__", without_kernels)
    with pytest.raises(RuntimeError, match="install ltx-kernels"):
        prepare_quantized_linears(
            _OneLinear(),
            {},
            mode=mode,
            linear_names=("projection",),
            device=torch.device("cpu"),
        )
    assert Float8CastLinear(torch.ones(16, 16))(torch.ones(1, 16)).shape == (1, 16)
    ltx_nvfp4_kernels.cache_clear()


def test_ltx_packed_nvfp4_keeps_codes_scales_and_provider_arguments(monkeypatch) -> None:
    from worldfoundry.core.acceleration.quantization import nvfp4

    weight = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
    scales = torch.ones(128, 4, dtype=torch.uint8)
    bias = torch.arange(16, dtype=torch.bfloat16)
    linear = NVFP4Linear.from_ltx_packed(weight, scales, torch.tensor(0.5), bias, input_scale=torch.tensor(0.25))
    linear.to(dtype=torch.float16)
    assert linear.provider == "ltx_kernels"
    assert linear.weight is None
    assert linear.weight_fp4.dtype is torch.uint8
    assert linear.weight_scale.dtype is torch.uint8
    assert linear.weight_global_scale.dtype is torch.float32
    assert linear.input_scale.dtype is torch.float32
    torch.testing.assert_close(linear.weight_fp4, weight, rtol=0, atol=0)
    calls = []

    def kernel(value, codes, blocks, global_scale, **kwargs):
        calls.append((value, codes, blocks, global_scale, kwargs))
        return torch.zeros(value.shape[0], 16, dtype=value.dtype)

    monkeypatch.setattr(nvfp4, "ltx_nvfp4_kernels", lambda: SimpleNamespace(linear_nvfp4=kernel))
    assert linear(torch.ones(2, 3, 16)).shape == (2, 3, 16)
    assert calls[0][0].dtype is torch.bfloat16
    assert calls[0][1] is linear.weight_fp4
    assert calls[0][2] is linear.weight_scale
    assert calls[0][4]["act_scale"] is linear.input_scale
    assert calls[0][4]["bias"] is linear.bias
    assert linear.low_precision_kernel_calls == 1


def _prequantized_state(mode: QuantizationMode) -> dict[str, torch.Tensor]:
    state = {
        "projection.bias": torch.zeros(16, dtype=torch.bfloat16),
        "projection.input_scale": torch.tensor(0.25),
    }
    if mode is QuantizationMode.FP8_SCALED_MM:
        state.update(
            {
                "projection.weight": torch.ones(16, 16, dtype=torch.float8_e4m3fn),
                "projection.weight_scale": torch.tensor(0.5),
            }
        )
    else:
        state.update(
            {
                "projection.weight": torch.arange(128, dtype=torch.uint8).reshape(16, 8),
                "projection.weight_scale": torch.ones(128, 4, dtype=torch.float8_e4m3fn),
                "projection.weight_scale_2": torch.tensor(0.5),
            }
        )
    return state


@pytest.mark.parametrize("mode", [QuantizationMode.FP8_SCALED_MM, QuantizationMode.NVFP4_PREQUANT])
def test_ltx_prequantized_selection_and_restore_use_lazy_dtype_shape_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: QuantizationMode,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.ltx.quantization import prepare_ltx_quantized_state_dict
    from worldfoundry.core.acceleration.quantization import nvfp4
    from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict
    from worldfoundry.core.vram.disk_map import DiskMap

    monkeypatch.setattr(nvfp4, "ltx_nvfp4_kernels", lambda: object())
    state = _prequantized_state(mode)
    checkpoint = tmp_path / "prequantized.safetensors"
    save_file({"checkpoint." + key: value for key, value in state.items()}, str(checkpoint))
    source = DiskMap(str(checkpoint), "cpu")
    renamed = TransformedStateDict(source, {key: "checkpoint." + key for key in state})
    with torch.device("meta"):
        model = _OneLinear()
    converted = prepare_ltx_quantized_state_dict(
        model,
        renamed,
        policy=QuantizationPolicy(mode=mode),
        device=torch.device("cpu"),
    )
    assert source.num_params == 0
    model.load_state_dict(converted, assign=True)
    assert source.num_params > 0
    assert model.projection.input_scale.item() == 0.25
    if mode is QuantizationMode.FP8_SCALED_MM:
        assert isinstance(model.projection, Float8Linear)
        assert model.projection.weight_scale.item() == 0.5
        torch.testing.assert_close(
            model.projection.weight_fp8.view(torch.uint8),
            state["projection.weight"].view(torch.uint8),
            rtol=0,
            atol=0,
        )
    else:
        assert isinstance(model.projection, NVFP4Linear)
        assert model.projection.weight_global_scale.item() == 0.5
        assert model.projection.weight_scale.shape == (128, 4)
        torch.testing.assert_close(model.projection.weight_fp4, state["projection.weight"], rtol=0, atol=0)
        torch.testing.assert_close(
            model.projection.weight_scale,
            state["projection.weight_scale"].view(torch.uint8),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("mode", "key", "dtype"),
    [
        (QuantizationMode.FP8_SCALED_MM, "projection.weight", torch.bfloat16),
        (QuantizationMode.NVFP4_PREQUANT, "projection.weight", torch.bfloat16),
        (QuantizationMode.NVFP4_PREQUANT, "projection.weight_scale", torch.float32),
        (QuantizationMode.NVFP4_PREQUANT, "projection.weight_scale_2", torch.bfloat16),
    ],
)
def test_ltx_prequantized_selection_rejects_nonmatching_storage_dtypes(
    mode: QuantizationMode,
    key: str,
    dtype: torch.dtype,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.ltx.quantization import prepare_ltx_quantized_state_dict

    state = _prequantized_state(mode)
    state[key] = state[key].to(dtype)
    with pytest.raises(ValueError, match="did not find any matching checkpoint linears"):
        prepare_ltx_quantized_state_dict(
            _OneLinear(),
            state,
            policy=QuantizationPolicy(mode=mode),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("key", ["projection.weight_scale", "projection.weight_scale_2"])
def test_ltx_nvfp4_prequantized_selection_requires_paired_scales(key: str) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.ltx.quantization import prepare_ltx_quantized_state_dict

    state = _prequantized_state(QuantizationMode.NVFP4_PREQUANT)
    state.pop(key)
    with pytest.raises(ValueError, match="paired weight_scale/weight_scale_2"):
        prepare_ltx_quantized_state_dict(
            _OneLinear(),
            state,
            policy=QuantizationPolicy(mode=QuantizationMode.NVFP4_PREQUANT),
            device=torch.device("cpu"),
        )


def test_ltx_nvfp4_prequantized_selection_requires_fp32_static_input_scale() -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.ltx.quantization import prepare_ltx_quantized_state_dict

    state = _prequantized_state(QuantizationMode.NVFP4_PREQUANT)
    state["projection.input_scale"] = state["projection.input_scale"].bfloat16()
    with pytest.raises(ValueError, match="calibrated FP32 input_scale"):
        prepare_ltx_quantized_state_dict(
            _OneLinear(),
            state,
            policy=QuantizationPolicy(mode=QuantizationMode.NVFP4_PREQUANT),
            device=torch.device("cpu"),
        )
