# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""LTX checkpoint selection for the shared cast and prequantized linear modes.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Layer selection is checkpoint/model-specific;
all storage transforms and kernels belong to core quantization.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.core.model_loading.checkpoints.transforms import checkpoint_tensor_metadata
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy

CHECKPOINT_QUANTIZATION_MODES = frozenset(
    {
        QuantizationMode.FP8_CAST,
        QuantizationMode.FP8_SCALED_MM,
        QuantizationMode.NVFP4_CAST,
        QuantizationMode.NVFP4_PREQUANT,
    }
)

_FP8_CAST_SUFFIXES = (".to_q", ".to_k", ".to_v", ".to_out.0", "ff.net.0.proj", "ff.net.2")
_NVFP4_CAST_SUFFIXES = frozenset(
    {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_out.0",
        "audio_to_video_attn.to_q",
        "audio_to_video_attn.to_out.0",
        "video_to_audio_attn.to_k",
        "video_to_audio_attn.to_v",
        "ff.net.0.proj",
        "ff.net.2",
    }
)


def prepare_ltx_quantized_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    policy: QuantizationPolicy,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    """Select cast linears or discover calibrated layers, then delegate to core."""
    from worldfoundry.core.acceleration.quantization.checkpoint import prepare_quantized_linears
    from worldfoundry.core.model_loading.optimize import QuantizationReport

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(pattern and pattern in name for pattern in policy.exclude):
            continue
        if policy.mode is QuantizationMode.FP8_CAST:
            selected = "transformer_blocks." in name and name.endswith(_FP8_CAST_SUFFIXES)
        elif policy.mode is QuantizationMode.NVFP4_CAST:
            suffix = name.removeprefix("velocity_model.transformer_blocks.").partition(".")
            selected = suffix[0].isdigit() and suffix[2] in _NVFP4_CAST_SUFFIXES
        elif policy.mode is QuantizationMode.FP8_SCALED_MM:
            selected = (
                name + ".weight_scale" in state_dict
                and checkpoint_tensor_metadata(state_dict, name + ".weight")[0] == torch.float8_e4m3fn
            )
        elif policy.mode is QuantizationMode.NVFP4_PREQUANT:
            scale_key, global_key = name + ".weight_scale", name + ".weight_scale_2"
            has_scale, has_global = scale_key in state_dict, global_key in state_dict
            if has_scale != has_global:
                raise ValueError(f"nvfp4-prequant requires paired weight_scale/weight_scale_2 tensors for {name}")
            selected = (
                has_scale
                and has_global
                and checkpoint_tensor_metadata(state_dict, name + ".weight")[0] == torch.uint8
                and checkpoint_tensor_metadata(state_dict, scale_key)[0] == torch.float8_e4m3fn
                and checkpoint_tensor_metadata(state_dict, global_key)[0] == torch.float32
            )
            if selected and (
                name + ".input_scale" not in state_dict
                or checkpoint_tensor_metadata(state_dict, name + ".input_scale")[0] != torch.float32
            ):
                raise ValueError(f"nvfp4-prequant requires a calibrated FP32 input_scale for {name}")
        else:
            raise ValueError(f"unsupported LTX checkpoint quantization mode: {policy.mode.value}")
        if selected:
            names.append(name)
    converted = prepare_quantized_linears(model, state_dict, mode=policy.mode, linear_names=names, device=device)
    model._worldfoundry_checkpoint_quantization = QuantizationReport(
        requested_mode=policy.mode.value,
        applied=True,
        replaced_modules=len(names),
        scaling="unscaled" if policy.mode is QuantizationMode.FP8_CAST else "checkpoint-specific",
        extra={
            "storage": policy.mode.value,
            "execution": "checkpoint-wrapper-installed (runtime-pending)",
            "dense_fallback_retained": False,
        },
    )
    return converted
