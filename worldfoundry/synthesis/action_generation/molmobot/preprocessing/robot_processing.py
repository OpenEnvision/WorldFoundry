from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from ..config import BaseConfig


def _to_array(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _normalize_array(
    x: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    min_val: np.ndarray | None = None,
    max_val: np.ndarray | None = None,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
    mode: str = "mean_std",
) -> np.ndarray:
    eps = 1e-6
    if mode == "mean_std":
        assert mean is not None and std is not None
        return (x - mean) / np.maximum(std, eps)
    if mode == "min_max":
        assert min_val is not None and max_val is not None
        denom = np.maximum(max_val - min_val, eps)
        return 2.0 * (x - min_val) / denom - 1.0
    if mode == "quantiles":
        assert q_low is not None and q_high is not None
        denom = np.maximum(q_high - q_low, eps)
        return 2.0 * (x - q_low) / denom - 1.0
    if mode == "quantile10":
        assert q_low is not None and q_high is not None
        denom = np.maximum(q_high - q_low, eps)
        return 2.0 * (x - q_low) / denom - 1.0
    return x


def _unnormalize_array(
    x: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    min_val: np.ndarray | None = None,
    max_val: np.ndarray | None = None,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
    mode: str = "mean_std",
) -> np.ndarray:
    if mode == "mean_std":
        assert mean is not None and std is not None
        return x * std + mean
    if mode == "min_max":
        assert min_val is not None and max_val is not None
        return (x + 1.0) * (max_val - min_val) / 2.0 + min_val
    if mode in {"quantiles", "quantile10"}:
        assert q_low is not None and q_high is not None
        return (x + 1.0) * (q_high - q_low) / 2.0 + q_low
    return x


@dataclass
class _FeatureNormalizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    min_val: np.ndarray | None = None
    max_val: np.ndarray | None = None
    q_low: np.ndarray | None = None
    q_high: np.ndarray | None = None
    mode: str = "min_max"

    @classmethod
    def from_stats(
        cls,
        stats: Mapping[str, Sequence[float]],
        mode: str,
    ) -> Optional["_FeatureNormalizer"]:
        if stats is None:
            return None
        if mode == "mean_std":
            mean = _to_array(stats.get("mean"))
            std = _to_array(stats.get("std"))
            if mean is None or std is None:
                return None
            return cls(mean=mean, std=std, mode=mode)
        if mode == "min_max":
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            if min_val is None or max_val is None:
                return None
            return cls(min_val=min_val, max_val=max_val, mode=mode)
        if mode == "quantiles":
            q_low = _to_array(stats.get("q01"))
            q_high = _to_array(stats.get("q99"))
            if q_low is None or q_high is None:
                return None
            return cls(q_low=q_low, q_high=q_high, mode=mode)
        if mode == "quantile10":
            q_low = _to_array(stats.get("q10"))
            q_high = _to_array(stats.get("q90"))
            if q_low is None or q_high is None:
                return None
            return cls(q_low=q_low, q_high=q_high, mode=mode)
        return None

    def normalize(self, x):
        arr = _to_array(x)
        if arr is None:
            return None
        normed = _normalize_array(
            arr,
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            q_low=self.q_low,
            q_high=self.q_high,
            mode=self.mode,
        )
        if torch.is_tensor(x):
            return torch.as_tensor(normed, device=x.device, dtype=x.dtype)
        return normed

    def unnormalize(self, x):
        arr = _to_array(x)
        if arr is None:
            return None
        unnorm = _unnormalize_array(
            arr,
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            q_low=self.q_low,
            q_high=self.q_high,
            mode=self.mode,
        )
        if torch.is_tensor(x):
            return torch.as_tensor(unnorm, device=x.device, dtype=x.dtype)
        return unnorm


@dataclass
class RobotPreprocessor:
    """Normalizes robot state for policy inference."""

    state_normalizers: dict[str, _FeatureNormalizer] = field(default_factory=dict)
    default_repo_id: str = "default"

    def normalize_state(self, state, repo_id: str | None) -> object | None:
        normalizer = self._get_normalizer(self.state_normalizers, repo_id)
        if normalizer is None:
            return state
        return normalizer.normalize(state)

    def _get_normalizer(
        self, mapping: Mapping[str, _FeatureNormalizer], repo_id: str | None
    ) -> _FeatureNormalizer | None:
        if repo_id and repo_id in mapping:
            return mapping[repo_id]
        return mapping.get(self.default_repo_id)


@dataclass
class RobotPostprocessor:
    """Unnormalizes model actions back to the robot's raw scale."""

    action_normalizers: dict[str, _FeatureNormalizer] = field(default_factory=dict)
    default_repo_id: str = "default"

    def unnormalize_action(self, action, repo_id: str | None) -> object | None:
        normalizer = self._get_normalizer(self.action_normalizers, repo_id)
        if normalizer is None:
            return action
        return normalizer.unnormalize(action)

    def _get_normalizer(
        self, mapping: Mapping[str, _FeatureNormalizer], repo_id: str | None
    ) -> _FeatureNormalizer | None:
        if repo_id and repo_id in mapping:
            return mapping[repo_id]
        return mapping.get(self.default_repo_id)


@dataclass
class RobotProcessorConfig(BaseConfig):
    """Configuration container for robot pre/post processing."""

    stats_by_repo: dict[str, dict[str, Any]] = field(default_factory=dict, metadata={"allow_objects": True})
    default_repo_id: str = "default"
    action_key: str = "action"
    state_keys: list[str] = field(default_factory=lambda: ["observation.state"])
    action_norm_mode: str = "min_max"
    state_norm_mode: str = "min_max"

    def build_preprocessor(self) -> RobotPreprocessor:
        _, state_norms = self._build_normalizers()
        return RobotPreprocessor(
            state_normalizers=state_norms,
            default_repo_id=self.default_repo_id,
        )

    def build_postprocessor(self) -> RobotPostprocessor:
        action_norms, _ = self._build_normalizers()
        return RobotPostprocessor(
            action_normalizers=action_norms,
            default_repo_id=self.default_repo_id,
        )

    def _build_normalizers(self):
        action_norms: dict[str, _FeatureNormalizer] = {}
        state_norms: dict[str, _FeatureNormalizer] = {}
        for repo_id, stats in self.stats_by_repo.items():
            if self.action_key in stats:
                norm = _FeatureNormalizer.from_stats(stats[self.action_key], mode=self.action_norm_mode)
                if norm is not None:
                    action_norms[repo_id] = norm
            for key in self.state_keys:
                if key in stats:
                    norm = _FeatureNormalizer.from_stats(stats[key], mode=self.state_norm_mode)
                    if norm is not None:
                        state_norms[repo_id] = norm
        # Provide a default normalizer if there's exactly one repo
        if len(action_norms) == 1 and self.default_repo_id not in action_norms:
            only = next(iter(action_norms.values()))
            action_norms[self.default_repo_id] = only
        if len(state_norms) == 1 and self.default_repo_id not in state_norms:
            only = next(iter(state_norms.values()))
            state_norms[self.default_repo_id] = only
        return action_norms, state_norms
