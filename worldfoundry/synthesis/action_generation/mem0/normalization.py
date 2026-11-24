"""Checkpoint-stat normalization for Mem-0 inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def load_stats(path: str | Path) -> dict[str, np.ndarray | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, np.ndarray | None] = dict.fromkeys(
        (
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
            "state_min",
            "state_max",
            "action_min",
            "action_max",
            "state_q01",
            "state_q99",
            "action_q01",
            "action_q99",
        )
    )
    for key in tuple(result):
        if key in data:
            result[key] = np.asarray(data[key], dtype=np.float32)
    if isinstance(data.get("observation.state"), Mapping):
        state = data["observation.state"]
        for quantile in ("q01", "q99"):
            if quantile in state:
                result[f"state_{quantile}"] = np.asarray(state[quantile], dtype=np.float32)
    if isinstance(data.get("action"), Mapping):
        action = data["action"]
        for quantile in ("q01", "q99"):
            if quantile in action:
                result[f"action_{quantile}"] = np.asarray(action[quantile], dtype=np.float32)
    return result


def _stat(stats: Mapping[str, np.ndarray | None], key: str, dimensions: int) -> np.ndarray:
    value = stats.get(key)
    if value is None:
        raise KeyError(f"Mem-0 normalization statistics are missing {key!r}")
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size < dimensions:
        raise ValueError(f"Mem-0 statistic {key!r} has {value.size} values; expected at least {dimensions}")
    return value[:dimensions]


def normalize(
    values: np.ndarray,
    *,
    prefix: str,
    mode: str,
    dimensions: int,
    stats: Mapping[str, np.ndarray | None],
    inverse: bool,
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    mode = mode.lower()
    if mode == "meanstd":
        mean = _stat(stats, f"{prefix}_mean", dimensions)
        std = _stat(stats, f"{prefix}_std", dimensions)
        std = np.where(np.abs(std) > 1e-6, std, 1.0)
        if inverse:
            output[..., :dimensions] = output[..., :dimensions] * std + mean
        else:
            output[..., :dimensions] = (output[..., :dimensions] - mean) / std
        return output
    if mode == "minmax":
        lower = _stat(stats, f"{prefix}_min", dimensions)
        upper = _stat(stats, f"{prefix}_max", dimensions)
    elif mode == "quantile":
        lower = _stat(stats, f"{prefix}_q01", dimensions)
        upper = _stat(stats, f"{prefix}_q99", dimensions)
    else:
        raise ValueError(f"Unsupported Mem-0 normalization mode: {mode}")
    raw_width = upper - lower
    valid = np.abs(raw_width) > 1e-8
    width = np.where(valid, raw_width, 1.0)
    if inverse:
        output[..., :dimensions] = 0.5 * (output[..., :dimensions] + 1.0) * width + lower
        output[..., :dimensions] = np.clip(output[..., :dimensions], lower, upper)
    else:
        clipped = np.clip(output[..., :dimensions], lower, upper)
        output[..., :dimensions] = 2.0 * (clipped - lower) / width - 1.0
        output[..., :dimensions] = np.where(valid, output[..., :dimensions], 0.0)
    return output


__all__ = ["load_stats", "normalize"]
