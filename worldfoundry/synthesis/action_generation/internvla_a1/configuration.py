"""Executable checkpoint configuration for InternVLA-A1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class InternVLAA1Config:
    """Inference fields consumed by the official InternVLA-A1 model graph."""

    qwen3_vl_variant: str
    action_expert_variant: str
    dtype: str
    chunk_size: int
    max_state_dim: int
    max_action_dim: int
    num_inference_steps: int
    time_sampling_beta_alpha: float
    time_sampling_beta_beta: float
    time_sampling_scale: float
    time_sampling_offset: float
    min_period: float
    max_period: float
    scale_factor: int
    lambda_gen: float
    compile_model: bool
    compile_mode: str
    freeze_vision_encoder: bool
    train_expert_only: bool
    train_vlm_only: bool

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any],
        dtype: str,
        num_inference_steps: int | None = None,
    ) -> InternVLAA1Config:
        values = {**dict(defaults), **dict(payload)}
        required = (
            "qwen3_vl_variant",
            "action_expert_variant",
            "chunk_size",
            "max_state_dim",
            "max_action_dim",
            "num_inference_steps",
            "time_sampling_beta_alpha",
            "time_sampling_beta_beta",
            "time_sampling_scale",
            "time_sampling_offset",
            "min_period",
            "max_period",
            "scale_factor",
            "lambda_gen",
        )
        missing = [name for name in required if values.get(name) in (None, "")]
        if missing:
            raise ValueError(f"InternVLA-A1 checkpoint config is missing fields: {missing}")
        return cls(
            qwen3_vl_variant=str(values["qwen3_vl_variant"]),
            action_expert_variant=str(values["action_expert_variant"]),
            dtype=dtype,
            chunk_size=int(values["chunk_size"]),
            max_state_dim=int(values["max_state_dim"]),
            max_action_dim=int(values["max_action_dim"]),
            num_inference_steps=int(
                num_inference_steps if num_inference_steps is not None else values["num_inference_steps"]
            ),
            time_sampling_beta_alpha=float(values["time_sampling_beta_alpha"]),
            time_sampling_beta_beta=float(values["time_sampling_beta_beta"]),
            time_sampling_scale=float(values["time_sampling_scale"]),
            time_sampling_offset=float(values["time_sampling_offset"]),
            min_period=float(values["min_period"]),
            max_period=float(values["max_period"]),
            scale_factor=int(values["scale_factor"]),
            lambda_gen=float(values["lambda_gen"]),
            compile_model=False,
            compile_mode=str(values.get("compile_mode") or "default"),
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_vlm_only=False,
        )


__all__ = ["InternVLAA1Config"]
