"""Checkpoint-compatible SmolVLA inference configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyFeature:
    """A tensor feature declared by the official policy config."""

    type: str
    shape: tuple[int, ...]

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> PolicyFeature:
        return cls(
            type=str(value.get("type", "")).upper(),
            shape=tuple(int(dimension) for dimension in value.get("shape", ())),
        )


@dataclass
class SmolVLAConfig:
    """Inference subset of the official configuration schema."""

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    device: str | None = "cuda"
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] | None = (512, 512)
    empty_cameras: int = 0
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False
    tokenizer_max_length: int = 48
    num_steps: int = 10
    use_cache: bool = True
    vlm_model_name: str = ""
    load_vlm_weights: bool = True
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    min_period: float = 4e-3
    max_period: float = 4.0
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    @classmethod
    def from_json(cls, path: str | Path) -> SmolVLAConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in payload.items() if key in allowed}
        values["input_features"] = {
            key: PolicyFeature.from_value(value) for key, value in payload.get("input_features", {}).items()
        }
        values["output_features"] = {
            key: PolicyFeature.from_value(value) for key, value in payload.get("output_features", {}).items()
        }
        values["normalization_mapping"] = {
            str(key).upper(): str(value).upper() for key, value in payload.get("normalization_mapping", {}).items()
        }
        resize = values.get("resize_imgs_with_padding")
        if resize is not None:
            values["resize_imgs_with_padding"] = tuple(int(item) for item in resize)
        return cls(**values)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: feature for key, feature in self.input_features.items() if feature.type in {"VISUAL", "IMAGE"}}

    @property
    def state_feature(self) -> PolicyFeature:
        for key, feature in self.input_features.items():
            if key == "observation.state" or feature.type == "STATE":
                return feature
        raise ValueError("SmolVLA config does not declare an observation state feature")

    @property
    def action_feature(self) -> PolicyFeature:
        for key, feature in self.output_features.items():
            if key == "action" or feature.type == "ACTION":
                return feature
        raise ValueError("SmolVLA config does not declare an action feature")

    def validate_features(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if not self.normalization_mapping:
            raise ValueError("SmolVLA checkpoint config requires normalization_mapping")
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError("delta-joint Aloha conversion is unavailable for inference")
        if self.state_feature.shape[-1] > self.max_state_dim:
            raise ValueError("state feature exceeds max_state_dim")
        if self.action_feature.shape[-1] > self.max_action_dim:
            raise ValueError("action feature exceeds max_action_dim")


__all__ = ["PolicyFeature", "SmolVLAConfig"]
