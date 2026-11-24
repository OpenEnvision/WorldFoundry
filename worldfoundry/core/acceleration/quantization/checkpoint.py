# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Load-time plans for explicit cast and calibrated low-precision linears.

Model components select the qualified linear names. Core owns the storage,
packing, calibrated scales and execution classes. The checkpoint remains a
lazy mapping so the same plan supports resident, CPU and disk placement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TypeVar

import torch
from torch import nn

from worldfoundry.core.model_loading.checkpoints.transforms import (
    TensorGroupTransform,
    TensorTransform,
    TransformedStateDict,
    checkpoint_tensor_metadata,
)
from worldfoundry.core.model_loading.policy import QuantizationMode

LinearModule = TypeVar("LinearModule", bound=nn.Module)


def construct_linear_like(layer: nn.Linear, *, linear_cls: Callable[..., LinearModule]) -> LinearModule:
    """Construct the selected linear type with matching dimensions, bias presence and device."""
    return linear_cls(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
    )


def _cast_fp8(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float8_e4m3fn)


def _fold_fp8_scale(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim != 0:
        raise ValueError("cast FP8 checkpoint scales must be scalars")
    return (value.float() * scale.to(value.device)).to(torch.bfloat16)


def _fp8_codes(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.float8_e4m3fn:
        raise ValueError("fp8-scaled-mm requires prequantized E4M3 weights")
    return value


def _nvfp4_codes(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.uint8:
        raise ValueError("nvfp4-prequant requires packed uint8 weights")
    return value


def _scale_bytes(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.uint8).contiguous()


def _scalar_scale(value: torch.Tensor) -> torch.Tensor:
    return value.float().reshape(())


def prepare_quantized_linears(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    mode: QuantizationMode,
    linear_names: Sequence[str],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    """Swap selected linears before restore and return their lazy checkpoint plan.

    ``fp8-cast`` folds prequantized scalar scales before the unscaled cast.
    ``fp8-scaled-mm`` and ``nvfp4-prequant`` require calibrated input scales.
    ``nvfp4-cast`` uses fixed-1 activation scaling and the optional LTX kernels.
    Existing generic FP8 and NVFP4 policies use their normal post-load path.
    """
    from .cast_linear import Float8CastLinear
    from .linear import Float8Linear
    from .nvfp4 import NVFP4Linear, ltx_nvfp4_kernels, quantize_nvfp4_ltx

    if not linear_names:
        raise ValueError(f"{mode.value} did not find any matching checkpoint linears")
    if mode in {QuantizationMode.NVFP4_CAST, QuantizationMode.NVFP4_PREQUANT}:
        ltx_nvfp4_kernels()
    source = state_dict
    if mode is QuantizationMode.FP8_CAST:
        folded: dict[str, str | TensorTransform] = {}
        for key in source:
            if key.endswith((".weight_scale", ".bias_scale", ".input_scale")):
                continue
            scale_key = key + "_scale"
            folded[key] = TensorTransform((key, scale_key), _fold_fp8_scale) if scale_key in source else key
        source = TransformedStateDict(source, folded)
    keys: dict[str, str | TensorTransform | TensorGroupTransform] = {key: key for key in source}
    pack = partial(quantize_nvfp4_ltx, device=device)
    for name in linear_names:
        linear = module.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"quantization target {name!r} is not nn.Linear")
        weight_key = name + ".weight"
        if mode is QuantizationMode.FP8_CAST:
            replacement = Float8CastLinear(linear.weight, linear.bias)
            keys[weight_key] = TensorTransform((weight_key,), _cast_fp8)
            if linear.bias is not None:
                bias_key = name + ".bias"
                keys[bias_key] = TensorTransform((bias_key,), _cast_fp8)
        elif mode is QuantizationMode.FP8_SCALED_MM:
            input_key, scale_key = name + ".input_scale", name + ".weight_scale"
            if input_key not in source or scale_key not in source:
                raise ValueError(f"fp8-scaled-mm requires calibrated input/weight scales for {name}")
            scalar = torch.empty((), dtype=torch.float32, device=linear.weight.device)
            replacement = Float8Linear.from_prequantized(
                torch.empty_like(linear.weight, dtype=torch.float8_e4m3fn),
                scalar,
                scalar.clone(),
                linear.bias,
            )
            keys.pop(weight_key)
            keys[name + ".weight_fp8"] = TensorTransform((weight_key,), _fp8_codes)
            keys[scale_key] = TensorTransform((scale_key,), _scalar_scale)
            keys[input_key] = TensorTransform((input_key,), _scalar_scale)
        elif mode in {QuantizationMode.NVFP4_CAST, QuantizationMode.NVFP4_PREQUANT}:
            scale_key, global_key = name + ".weight_scale", name + ".weight_scale_2"
            input_key = name + ".input_scale"
            prequantized = mode is QuantizationMode.NVFP4_PREQUANT
            if prequantized and any(key not in source for key in (scale_key, global_key, input_key)):
                raise ValueError(f"nvfp4-prequant requires weight, block, global and input scales for {name}")
            scale_rows = (
                checkpoint_tensor_metadata(source, scale_key)[1][0]
                if prequantized
                else (linear.out_features + 127) // 128 * 128
            )
            scale_columns = (linear.in_features // 16 + 3) // 4 * 4
            scalar = torch.empty((), dtype=torch.float32, device=linear.weight.device)
            replacement = NVFP4Linear.from_ltx_packed(
                torch.empty(
                    linear.out_features, linear.in_features // 2, dtype=torch.uint8, device=linear.weight.device
                ),
                torch.empty(scale_rows, scale_columns, dtype=torch.uint8, device=linear.weight.device),
                scalar,
                linear.bias,
                input_scale=scalar.clone() if prequantized else None,
            )
            for key in (weight_key, scale_key, global_key, input_key):
                keys.pop(key, None)
            if prequantized:
                keys[name + ".weight_fp4"] = TensorTransform((weight_key,), _nvfp4_codes)
                keys[scale_key] = TensorTransform((scale_key,), _scale_bytes)
                keys[name + ".weight_global_scale"] = TensorTransform((global_key,), _scalar_scale)
                keys[input_key] = TensorTransform((input_key,), _scalar_scale)
            else:
                for index, suffix in enumerate((".weight_fp4", ".weight_scale", ".weight_global_scale")):
                    keys[name + suffix] = TensorGroupTransform((weight_key,), pack, index)
        else:
            raise ValueError(f"{mode.value} is not a checkpoint-layout quantization mode")
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, replacement)
    return TransformedStateDict(source, keys)
