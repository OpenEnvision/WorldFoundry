# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Released IC-LoRA key mapping and lazy fusion for the canonical LTX transformer.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Fusion scales B before the matrix product and
accumulates in BF16 (FP32 on MPS), matching the released inference loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import torch

from worldfoundry.core.model_loading.checkpoints.transforms import (
    TensorTransform,
    TransformedStateDict,
    checkpoint_tensor_metadata,
)


def convert_ltx_ic_lora_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Map every released ``diffusion_model.*.lora_[AB].weight`` key to its native wrapper."""
    keys = {}
    for key in state_dict:
        if not key.startswith("diffusion_model.") or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unsupported LTX IC-LoRA key: {key}")
        keys[f"velocity_model.{key.removeprefix('diffusion_model.')}"] = key
    return TransformedStateDict(state_dict, keys)


def _fuse_ltx_ic_lora_weight(
    weight: torch.Tensor,
    *,
    lora: Mapping[str, torch.Tensor],
    stem: str,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    aggregation_dtype = torch.float32 if device.type == "mps" else torch.bfloat16
    a = lora[f"{stem}.lora_A.weight"].to(device=device, dtype=aggregation_dtype)
    b = lora[f"{stem}.lora_B.weight"].to(device=device, dtype=aggregation_dtype)
    delta = torch.matmul(b * strength, a).to(aggregation_dtype)
    delta.add_(weight.to(device))
    return delta.to(device=weight.device, dtype=weight.dtype)


def prepare_ltx_ic_lora_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    lora: Mapping[str, torch.Tensor],
    strength: float,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    """Validate adapter pairs against the graph, then fuse each weight only when read.

    The base checkpoint and its module are not mutated. CPU/disk offload and
    resident loading consume the same transform; a second stage can load the
    clean checkpoint independently.
    """
    parameters = dict(module.named_parameters())
    keys: dict[str, str | TensorTransform] = {key: key for key in state_dict}
    stems = {key.rsplit(".lora_", 1)[0] for key in lora}
    if not stems:
        raise ValueError("LTX IC-LoRA checkpoint contains no adapter pairs")
    for stem in sorted(stems):
        key_a, key_b, weight_key = f"{stem}.lora_A.weight", f"{stem}.lora_B.weight", f"{stem}.weight"
        if key_a not in lora or key_b not in lora:
            raise ValueError(f"LTX IC-LoRA requires both A and B for {stem}")
        if weight_key not in parameters or weight_key not in state_dict:
            raise ValueError(f"LTX IC-LoRA target is absent from the transformer: {weight_key}")
        _, shape_a = checkpoint_tensor_metadata(lora, key_a)
        _, shape_b = checkpoint_tensor_metadata(lora, key_b)
        shape = tuple(parameters[weight_key].shape)
        if len(shape_a) != 2 or len(shape_b) != 2 or shape_b[1] != shape_a[0] or (shape_b[0], shape_a[1]) != shape:
            raise ValueError(f"LTX IC-LoRA pair shapes {shape_a}, {shape_b} do not match {weight_key}: {shape}")
        keys[weight_key] = TensorTransform(
            (weight_key,),
            partial(_fuse_ltx_ic_lora_weight, lora=lora, stem=stem, strength=strength, device=device),
        )
    return TransformedStateDict(state_dict, keys)
