"""Strict checkpoint restoration for StarVLA inference models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import read_model_config


class StarVLAInferenceModel(nn.Module):
    """Small common interface shared by the released StarVLA policy variants."""

    norm_stats: dict[str, Any]

    def predict_action(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


def _validate_tensor_state_dict(payload: Any, checkpoint: Path) -> dict[str, torch.Tensor]:
    if (
        not isinstance(payload, Mapping)
        or not payload
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items())
    ):
        raise TypeError(f"StarVLA checkpoint must contain a non-empty string-to-tensor state dict: {checkpoint}")
    return dict(payload)


def _load_policy_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        return _validate_tensor_state_dict(load_file(str(checkpoint), device="cpu"), checkpoint)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    return _validate_tensor_state_dict(payload, checkpoint)


def load_starvla_model(
    checkpoint_file: str | Path,
    *,
    base_vlm: str | None = None,
    base_world_model: str | None = None,
    attn_implementation: str = "auto",
    device: str = "cuda",
    torch_dtype: str = "auto",
) -> StarVLAInferenceModel:
    """Construct the checkpoint-declared variant and restore every tensor strictly."""

    checkpoint = Path(checkpoint_file).expanduser().resolve()
    config, norm_stats = read_model_config(checkpoint)
    framework_name = str(config.framework.name)

    if base_vlm:
        config.framework.qwenvl.base_vlm = base_vlm
    if hasattr(config.framework, "qwenvl"):
        config.framework.qwenvl.attn_implementation = attn_implementation
        config.framework.qwenvl.device = device
        config.framework.qwenvl.torch_dtype = torch_dtype
    if base_world_model:
        if not hasattr(config.framework, "world_model"):
            config.framework.world_model = {}
        config.framework.world_model.base_wm = base_world_model
    if hasattr(config.framework, "world_model"):
        config.framework.world_model.device = device
        config.framework.world_model.torch_dtype = torch_dtype

    if framework_name == "QwenOFT":
        from .qwen_oft import QwenvlOFT

        model: StarVLAInferenceModel = QwenvlOFT(config)
    elif framework_name == "WanOFT":
        from .wan_oft import WanOFT

        model = WanOFT(config)
    else:
        raise NotImplementedError(
            f"StarVLA inference variant {framework_name!r} is not integrated; supported: QwenOFT, WanOFT."
        )

    state_dict = _load_policy_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.norm_stats = norm_stats
    return model


__all__ = ["StarVLAInferenceModel", "load_starvla_model"]
