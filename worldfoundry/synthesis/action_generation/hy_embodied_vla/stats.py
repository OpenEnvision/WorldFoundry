"""Safe loading and shared normalization for Hy-VLA checkpoint statistics."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.core.action_normalization import (
    normalize_action_values,
    unnormalize_action_values,
)

_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _NumpyStatsUnpickler(pickle.Unpickler):
    """Unpickle only the NumPy globals used by official ``norm_stats.pkl`` files."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"disallowed global in Hy-VLA normalization statistics: {module}.{name}")
        return super().find_class(module, name)


@dataclass(frozen=True)
class HyVLANormalizationStats:
    """Validated statistics shipped with a released Hy-VLA checkpoint."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    action_mean_abs: np.ndarray | None = None
    action_std_abs: np.ndarray | None = None

    @property
    def has_absolute_actions(self) -> bool:
        return self.action_mean_abs is not None and self.action_std_abs is not None

    @property
    def action_horizon(self) -> int:
        return int(self.action_mean.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> HyVLANormalizationStats:
        """Load an official stats pickle without allowing arbitrary Python globals."""

        source = Path(path).expanduser().resolve()
        payload = _NumpyStatsUnpickler(io.BytesIO(source.read_bytes())).load()
        if not isinstance(payload, dict):
            raise TypeError(f"Hy-VLA normalization root must be a mapping: {source}")

        required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Hy-VLA normalization statistics are missing {missing}: {source}")

        def array(key: str) -> np.ndarray:
            value = np.asarray(payload[key], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"Hy-VLA normalization statistic {key!r} contains non-finite values")
            return value

        state_mean = array("qpos_mean")
        state_std = array("qpos_std")
        action_mean = array("action_mean")
        action_std = array("action_std")
        if state_mean.ndim != 1 or state_mean.shape != state_std.shape:
            raise ValueError("Hy-VLA qpos_mean/qpos_std must be matching one-dimensional arrays")
        if action_mean.ndim != 2 or action_mean.shape != action_std.shape:
            raise ValueError("Hy-VLA action_mean/action_std must be matching (T, D) arrays")

        action_mean_abs = None
        action_std_abs = None
        if "action_mean_abs" in payload or "action_std_abs" in payload:
            if not {"action_mean_abs", "action_std_abs"}.issubset(payload):
                raise KeyError("Hy-VLA absolute action statistics must include both mean and std")
            action_mean_abs = array("action_mean_abs")
            action_std_abs = array("action_std_abs")
            if action_mean_abs.shape != action_mean.shape or action_std_abs.shape != action_mean.shape:
                raise ValueError("Hy-VLA absolute action statistics must match relative action shape")

        return cls(
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            action_mean_abs=action_mean_abs,
            action_std_abs=action_std_abs,
        )

    def normalize_state(self, state: Any) -> np.ndarray:
        """Normalize a state vector through WorldFoundry's shared mean/std helper."""

        return normalize_action_values(
            state,
            {"mean": self.state_mean, "std": self.state_std},
            mode="mean_std",
        ).astype(np.float32, copy=False)

    @staticmethod
    def _temporal_unnormalize(values: Any, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Hy-VLA action chunk must have shape (T, D), got {actions.shape}")
        if actions.shape != mean.shape:
            raise ValueError(f"Hy-VLA action chunk shape {actions.shape} does not match statistics {mean.shape}")
        rows = [
            unnormalize_action_values(
                actions[index],
                {"mean": mean[index], "std": std[index]},
                mode="mean_std",
            )
            for index in range(actions.shape[0])
        ]
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    def unnormalize_relative(self, actions: Any) -> np.ndarray:
        return self._temporal_unnormalize(actions, self.action_mean, self.action_std)

    def unnormalize_absolute(self, actions: Any) -> np.ndarray:
        if self.action_mean_abs is None or self.action_std_abs is None:
            raise ValueError("This Hy-VLA checkpoint does not provide absolute-action statistics")
        return self._temporal_unnormalize(actions, self.action_mean_abs, self.action_std_abs)


__all__ = ["HyVLANormalizationStats"]
