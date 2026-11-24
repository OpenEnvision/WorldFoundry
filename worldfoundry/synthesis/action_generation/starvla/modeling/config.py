"""Inference configuration helpers for the in-tree StarVLA runtimes."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def apply_config_compat(config: DictConfig) -> DictConfig:
    """Normalize released pre-0.21 action-window fields for inference."""

    action_model = OmegaConf.select(config, "framework.action_model", default=None)
    if action_model is None:
        raise ValueError("StarVLA config is missing framework.action_model.")

    action_horizon = OmegaConf.select(action_model, "action_horizon", default=None)
    future_window = OmegaConf.select(action_model, "future_action_window_size", default=None)
    if action_horizon is None and future_window is None:
        raise ValueError("StarVLA config must define action_horizon or future_action_window_size.")
    if action_horizon is None:
        action_horizon = int(future_window) + 1
        OmegaConf.update(config, "framework.action_model.action_horizon", action_horizon, force_add=True)
    if future_window is None or int(future_window) + 1 != int(action_horizon):
        OmegaConf.update(
            config,
            "framework.action_model.future_action_window_size",
            int(action_horizon) - 1,
            force_add=True,
        )
    if OmegaConf.select(action_model, "past_action_window_size", default=None) is None:
        OmegaConf.update(config, "framework.action_model.past_action_window_size", 0, force_add=True)
    OmegaConf.update(config, "version_id", "0.21", force_add=True)
    return config


def merge_framework_config(default_config_cls: type[Any], config: Any) -> DictConfig:
    """Merge framework defaults with the checkpoint YAML, preserving extra fields."""

    cfg = config if isinstance(config, DictConfig) else OmegaConf.create(config or {})
    defaults = OmegaConf.create(dataclasses.asdict(default_config_cls()))
    checkpoint_framework = OmegaConf.select(cfg, "framework", default={})
    cfg.framework = OmegaConf.merge(defaults, checkpoint_framework)
    return apply_config_compat(cfg)


def read_model_config(checkpoint_file: str | Path) -> tuple[DictConfig, dict[str, Any]]:
    """Load the config and normalization statistics adjacent to a policy weight file."""

    checkpoint = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"StarVLA checkpoint does not exist: {checkpoint}")
    if checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError(f"Unsupported StarVLA checkpoint format: {checkpoint.suffix}")

    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"StarVLA config is missing: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"StarVLA dataset statistics are missing: {stats_path}")

    config = apply_config_compat(OmegaConf.load(config_path))
    norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return config, norm_stats


def state_to_token_string(state: Any, num_bins: int = 256) -> str:
    """Quantize one proprioceptive state vector to StarVLA's text-token format."""

    array = np.asarray(state, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("StarVLA state must contain at least one value.")
    if array.ndim > 1:
        array = array.reshape(-1, array.shape[-1])[0]
    bins = np.linspace(-1.0, 1.0, num_bins + 1, dtype=np.float32)[:-1]
    indices = np.clip(np.digitize(array, bins=bins) - 1, 0, num_bins - 1)
    return " ".join(map(str, indices.tolist()))


def add_discretized_state_to_instruction(
    instructions: Sequence[str], states: Sequence[Any], num_bins: int = 256
) -> list[str]:
    if len(instructions) != len(states):
        raise ValueError("StarVLA instructions and states must have equal batch size.")
    return [
        f"{instruction} [STATE] {state_to_token_string(state, num_bins=num_bins)} [ACTION]"
        for instruction, state in zip(instructions, states, strict=False)
    ]


__all__ = [
    "add_discretized_state_to_instruction",
    "apply_config_compat",
    "merge_framework_config",
    "read_model_config",
    "state_to_token_string",
]
