"""Pure AnyFlow forward-map v1.5 contracts used by the Wan Stage1 route."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, NamedTuple

import numpy as np

from ...errors import BackendContractError
from .objectives import apply_timestep_shift

SAMPLE_TYPE_DIFFUSION = 0
SAMPLE_TYPE_CONSISTENCY = 1
SAMPLE_TYPE_FLOW_MAP = 2


class AnyFlowTimePairs(NamedTuple):
    t: Any
    r: Any
    is_diffusion: Any
    is_consistency: Any
    sample_type: Any


def build_flowmap_schedule_array(
    num_inference_steps: int, shift: float, num_train_timesteps: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    if num_inference_steps <= 0 or num_train_timesteps <= 0:
        raise ValueError("step counts must be positive")
    normalized = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)
    shifted = float(shift) * normalized / (1.0 + (float(shift) - 1.0) * normalized)
    raw = shifted * num_train_timesteps
    return raw[:-1].copy(), raw[1:].copy()


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("AnyFlow math requires the 'wan' dependency extra") from exc
    return torch


def validate_anyflow_config(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    """Validate the supported AnyFlow route."""

    removed = "anyflow_negative_embedding_digest"
    if removed in train:
        raise BackendContractError(
            f"train.{removed} is not supported; provide the embedding path and SolarWM will validate the loaded tensor"
        )
    checks = {
        "train.objective_variant": (str(train.get("objective_variant", "")) == "v1_5"),
        "train.anyflow_variant": (str(train.get("anyflow_variant", "")) == "v1_5"),
        "train.anyflow_weight_type": (str(train.get("anyflow_weight_type", "")) == "gaussian"),
        "train.anyflow_deltatime_type": (str(train.get("anyflow_deltatime_type", "")) == "r"),
    }
    failed = [field for field, accepted in checks.items() if not accepted]
    if failed:
        raise BackendContractError(f"AnyFlow v1.5 contract mismatch: {failed}")
    gate = float(train.get("anyflow_gate", -1.0))
    epsilon = float(train.get("anyflow_epsilon", 0.0))
    diffusion = float(train.get("anyflow_diffusion_ratio", -1.0))
    consistency = float(train.get("anyflow_consistency_ratio", -1.0))
    guidance = float(train.get("anyflow_fuse_guidance_scale", 0.0))
    if not all(math.isfinite(value) for value in (gate, epsilon, diffusion, consistency, guidance)):
        raise BackendContractError("AnyFlow numeric settings must be finite")
    if not 0.0 <= gate <= 1.0 or epsilon <= 0:
        raise BackendContractError("AnyFlow gate must be in [0,1] and epsilon must be positive")
    if diffusion < 0 or consistency < 0 or diffusion + consistency > 1.0:
        raise BackendContractError("AnyFlow sample ratios must be non-negative and sum to at most one")
    if guidance <= 0:
        raise BackendContractError("AnyFlow guidance must be positive")
    if guidance != 1.0:
        embedding = str(train.get("anyflow_negative_embedding", "")).strip()
        path = PurePosixPath(embedding)
        if (
            not embedding
            or embedding.startswith("/")
            or "://" in embedding
            or "\\" in embedding
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise BackendContractError(
                "guided AnyFlow train.anyflow_negative_embedding must be a base-model-relative POSIX path"
            )
    passes = validation.get("passes", [])
    if not isinstance(passes, list) or not passes:
        raise BackendContractError("AnyFlow validation.passes must be a non-empty list")
    nfes = {int(item.get("num_inference_steps", 0)) for item in passes if isinstance(item, Mapping)}
    solvers = {str(item.get("solver", "")) for item in passes if isinstance(item, Mapping)}
    if nfes != {4, 50} or solvers != {"flowmap"}:
        raise BackendContractError("AnyFlow validation must contain only flowmap NFE4 and NFE50 passes")


def build_flowmap_schedule(
    num_inference_steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
    *,
    device: Any = "cpu",
) -> tuple[Any, Any]:
    """Return shifted raw (t,r) pairs used by validation and inference."""

    torch = _torch()
    if num_inference_steps <= 0 or num_train_timesteps <= 0:
        raise ValueError("step counts must be positive")
    normalized = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)
    raw = apply_timestep_shift(normalized, shift) * num_train_timesteps
    return raw[:-1].contiguous(), raw[1:].contiguous()


__all__ = [
    "SAMPLE_TYPE_CONSISTENCY",
    "SAMPLE_TYPE_DIFFUSION",
    "SAMPLE_TYPE_FLOW_MAP",
    "AnyFlowTimePairs",
    "build_flowmap_schedule",
    "build_flowmap_schedule_array",
    "validate_anyflow_config",
]
