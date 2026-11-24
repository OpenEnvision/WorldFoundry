# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Resolve the class names serialized in released DINO-WM model checkpoints."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

# Torch's legacy serialization reads its scalar headers through pickle_module.load.
from pickle import load as load
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig

MODEL_MODULES = {
    f"models.{name}": f"worldfoundry.synthesis.visual_generation.dino_wm.models.{name}"
    for name in (
        "dino",
        "dummy",
        "proprio",
        "vit",
        "vqvae",
        "visual_world_model",
        "decoder.transposed_conv",
        "encoder.resnet",
        "encoder.r3m.models.models_r3m",
    )
}
DINOV2_MODULES = {
    f"dinov2.{name}": f"worldfoundry.base_models.perception_core.general_perception.dinov2.{name}"
    for name in ("models.vision_transformer", "layers.attention", "layers.block")
}
DINOV2_MODULES.update(
    {
        f"dinov2.layers.{name}": "worldfoundry.core.nn.blocks.layers"
        for name in ("mlp", "patch_embed", "layer_scale", "drop_path", "swiglu_ffn")
    }
)


class Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        module = MODEL_MODULES.get(module, DINOV2_MODULES.get(module, module))
        return super().find_class(module, name)


def load_checkpoint(path: Path | str, *, map_location: torch.device | str | None = None) -> Any:
    """Load released module pickles without installing top-level package aliases."""
    import torch

    from . import checkpoint
    from .models.dino import DinoV2Encoder, prepare_dinov2

    payload = torch.load(path, map_location=map_location, pickle_module=checkpoint, weights_only=False)
    components = payload.values() if isinstance(payload, dict) else (payload,)
    for component in components:
        if isinstance(component, DinoV2Encoder):
            prepare_dinov2(component.base_model)
    return payload


def model_config(config: DictConfig) -> DictConfig:
    """Qualify released model targets in a copy, preserving the checkpoint config."""
    from omegaconf import OmegaConf

    resolved = copy.deepcopy(config)
    for key in ("model", "encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"):
        target = OmegaConf.select(resolved, f"{key}._target_")
        if isinstance(target, str):
            module, _, name = target.rpartition(".")
            if module in MODEL_MODULES:
                OmegaConf.update(resolved, f"{key}._target_", f"{MODEL_MODULES[module]}.{name}")
    return resolved
