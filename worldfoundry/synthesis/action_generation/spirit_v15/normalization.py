# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team. Modified by WorldFoundry for inference-only use.
# ==============================================================================

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    stats_buffers = {}
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue
        assert isinstance(norm_mode, NormalizationMode)
        if norm_mode is not NormalizationMode.MIN_MAX:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)
        min_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        max_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {"min": nn.Parameter(min_v, requires_grad=False), "max": nn.Parameter(max_v, requires_grad=False)}
        )
        if stats is not None:
            if key not in stats:
                raise ValueError(f"Missing stats for feature `{key}` (expected `min`/`max`).")
            if "min" not in stats[key] or "max" not in stats[key]:
                raise ValueError(f"Stats for `{key}` must contain `min` and `max` for MIN_MAX normalization.")
            min_src, max_src = stats[key]["min"], stats[key]["max"]
            if isinstance(min_src, np.ndarray) and isinstance(max_src, np.ndarray):
                buffer["min"].data = torch.from_numpy(min_src).to(dtype=torch.float32)
                buffer["max"].data = torch.from_numpy(max_src).to(dtype=torch.float32)
            elif isinstance(min_src, torch.Tensor) and isinstance(max_src, torch.Tensor):
                buffer["min"].data = min_src.clone().to(dtype=torch.float32)
                buffer["max"].data = max_src.clone().to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected stats type for `{key}`: min={type(min_src)}, max={type(max_src)}")
        stats_buffers[key] = buffer
    return stats_buffers


def no_stats_error_str(name: str) -> str:
    return f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model."


def build_norm_state(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[FeatureType, NormalizationMode], dict[str, nn.ParameterDict]]:
    norm_mode_map: dict[FeatureType, NormalizationMode] = {}
    for k, v in (norm_map or {}).items():
        ft = k if isinstance(k, FeatureType) else FeatureType(k)
        mode = v if isinstance(v, NormalizationMode) else NormalizationMode(v)
        if mode not in (NormalizationMode.IDENTITY, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {mode}")
        norm_mode_map[ft] = mode
    stats_buffers = create_stats_buffers(features, norm_mode_map, stats)
    return norm_mode_map, stats_buffers
