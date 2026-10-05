"""Model-agnostic normalization helpers for continuous robot actions.

Policy checkpoints ship a statistics mapping (min/max, quantiles, or
mean/std) that describes how actions were scaled during training.
:func:`normalize_action_values` applies that transform along the last
axis; :func:`unnormalize_action_values` is the inverse used when a
policy's output must be sent to an environment. Both are NumPy-only so
control-plane code can share the contract without importing a training
stack or a specific robot adapter.

Accepted statistic layouts:

- Direct: ``{"min": ..., "max": ...}`` (or ``q01``/``q99``, ``mean``/``std``).
- Dataset-keyed: ``{"robot_name": {"action": {...}}}``. When more than
  one dataset key is present, :func:`select_modality_statistics` requires
  an explicit ``key``.

Modes: ``min_max`` (and aliases ``q99`` / ``quantile`` / ``bounds``)
map a low/high pair onto ``[-1, 1]``; ``mean_std`` / ``standard`` /
``zscore`` center by mean and divide by std; ``identity`` / ``none``
leave values unchanged. An optional per-dimension ``mask`` skips
unnormalized channels (for example discrete gripper bits). Degenerate
ranges (``high == low`` or ``std == 0``) are left untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_BOUNDED_MODES = {
    "min_max": ("min", "max"),
    "q99": ("q01", "q99"),
    "quantile": ("q01", "q99"),
    "bounds": ("q01", "q99"),
}


def select_modality_statistics(
    dataset_statistics: Mapping[str, Any],
    *,
    modality: str = "action",
    key: str | None = None,
) -> tuple[str | None, Mapping[str, Any]]:
    """Select one modality's statistics from a checkpoint statistics mapping.

    Both common layouts are accepted: a direct ``{"min": ..., "max": ...}``
    mapping and a dataset-keyed ``{"robot": {"action": {...}}}`` mapping.
    Dataset-key selection is strict when more than one key is available.

    Args:
        dataset_statistics: Checkpoint statistics mapping.
        modality: Nested key under a dataset entry (default ``"action"``).
        key: Dataset name when *dataset_statistics* is multi-key. Omit
            only when a single key (or a direct stats mapping) is present.

    Returns:
        ``(resolved_key, stats_mapping)``. *resolved_key* is ``None`` when
        the input was already a direct statistics mapping.

    Raises:
        ValueError: Empty input, or multiple dataset keys with no *key*.
        KeyError: *key* is missing or has no *modality* mapping.
    """

    if not isinstance(dataset_statistics, Mapping) or not dataset_statistics:
        raise ValueError("Dataset normalization statistics are empty or invalid.")

    if modality in dataset_statistics and isinstance(dataset_statistics[modality], Mapping):
        return key, dataset_statistics[modality]
    if any(name in dataset_statistics for name in ("min", "max", "q01", "q99", "mean", "std")):
        return key, dataset_statistics

    available = tuple(str(item) for item in dataset_statistics)
    if key is None:
        if len(available) != 1:
            raise ValueError(
                f"Multiple normalization keys are available: {list(available)}; "
                "select one explicitly."
            )
        key = available[0]
    if key not in dataset_statistics:
        raise KeyError(f"Normalization key {key!r} is unavailable; choices: {list(available)}.")

    selected = dataset_statistics[key]
    if not isinstance(selected, Mapping) or not isinstance(selected.get(modality), Mapping):
        raise KeyError(f"Normalization key {key!r} has no {modality!r} statistics.")
    return key, selected[modality]


def _coerce_values_and_mask(
    values: Any,
    statistics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Copy *values* to a writable float array and align the optional per-dim mask."""
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("Action values must have at least one dimension.")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    else:
        array = array.copy()

    width = array.shape[-1]
    raw_mask = statistics.get("mask")
    mask = np.ones(width, dtype=bool) if raw_mask is None else np.asarray(raw_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != width:
        raise ValueError(f"Normalization mask has shape {mask.shape}; expected ({width},).")
    return array, mask


def _stat_vector(statistics: Mapping[str, Any], name: str, width: int, dtype: np.dtype) -> np.ndarray:
    """Load one 1-D statistic; width must match the action last-axis, or the scale is wrong."""
    if name not in statistics:
        raise KeyError(f"Normalization statistics are missing {name!r}.")
    vector = np.asarray(statistics[name], dtype=dtype)
    if vector.ndim != 1 or vector.shape[0] != width:
        raise ValueError(f"Normalization statistic {name!r} has shape {vector.shape}; expected ({width},).")
    return vector


def normalize_action_values(
    values: Any,
    statistics: Mapping[str, Any],
    *,
    mode: str = "min_max",
    clip: float | None = None,
) -> np.ndarray:
    """Normalize action values using checkpoint statistics along the last axis.

    Args:
        values: Array-like with at least one dimension. The last axis is
            the action width and must match each statistic vector.
        statistics: Selected modality mapping from
            :func:`select_modality_statistics`.
        mode: ``min_max`` (default), ``q99`` / ``quantile`` / ``bounds``,
            ``mean_std`` / ``standard`` / ``zscore``, or ``identity``.
        clip: When set, clamp masked channels to ``[-clip, clip]`` after
            normalization.

    Returns:
        A writable ``float`` array with the same shape as *values*.
    """

    normalized, mask = _coerce_values_and_mask(values, statistics)
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode in _BOUNDED_MODES:
        low_name, high_name = _BOUNDED_MODES[normalized_mode]
        low = _stat_vector(statistics, low_name, normalized.shape[-1], normalized.dtype)
        high = _stat_vector(statistics, high_name, normalized.shape[-1], normalized.dtype)
        valid = mask & (high != low)
        normalized[..., valid] = 2.0 * (
            (normalized[..., valid] - low[valid]) / (high[valid] - low[valid])
        ) - 1.0
    elif normalized_mode in {"mean_std", "standard", "zscore"}:
        mean = _stat_vector(statistics, "mean", normalized.shape[-1], normalized.dtype)
        std = _stat_vector(statistics, "std", normalized.shape[-1], normalized.dtype)
        valid = mask & (std != 0)
        normalized[..., valid] = (normalized[..., valid] - mean[valid]) / std[valid]
    elif normalized_mode in {"identity", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported action normalization mode: {mode!r}.")
    if clip is not None:
        normalized[..., mask] = np.clip(normalized[..., mask], -float(clip), float(clip))
    return normalized


def unnormalize_action_values(
    normalized_values: Any,
    statistics: Mapping[str, Any],
    *,
    mode: str = "min_max",
) -> np.ndarray:
    """Convert normalized policy outputs back to environment-space actions.

    Inverse of :func:`normalize_action_values` for the same *mode* and
    *statistics*. Masked-off channels are copied through unchanged.
    Unlike the forward path, this function does not clip.

    Args:
        normalized_values: Policy outputs in the normalized space.
        statistics: Same modality mapping used at normalize time.
        mode: Must match the mode used to produce *normalized_values*.

    Returns:
        Environment-space actions with the same shape as the input.
    """

    actions, mask = _coerce_values_and_mask(normalized_values, statistics)
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode in _BOUNDED_MODES:
        low_name, high_name = _BOUNDED_MODES[normalized_mode]
        low = _stat_vector(statistics, low_name, actions.shape[-1], actions.dtype)
        high = _stat_vector(statistics, high_name, actions.shape[-1], actions.dtype)
        actions[..., mask] = (
            (actions[..., mask] + 1.0) * 0.5 * (high[mask] - low[mask]) + low[mask]
        )
    elif normalized_mode in {"mean_std", "standard", "zscore"}:
        mean = _stat_vector(statistics, "mean", actions.shape[-1], actions.dtype)
        std = _stat_vector(statistics, "std", actions.shape[-1], actions.dtype)
        actions[..., mask] = actions[..., mask] * std[mask] + mean[mask]
    elif normalized_mode in {"identity", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported action normalization mode: {mode!r}.")
    return actions


__all__ = [
    "normalize_action_values",
    "select_modality_statistics",
    "unnormalize_action_values",
]
