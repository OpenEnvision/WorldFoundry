"""Local backend for Appearance Consistency."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..io import value_digest
from ..skills.skill_appearance_consistency import BACKEND_OUTPUT_SCHEMA, SAMPLING
from .common import (
    _decode_frames,
    _raw_score,
    _sample_frames,
    _validate_video,
)


BACKEND_ID = "harnesseval.appearance_consistency"
BACKEND_VERSION = "harnesseval.appearance_consistency.clip_b32"
CONFIG_DIGEST = value_digest(
    {
        "implementation": "harnesseval",
        "metric": "background_consistency",
        "model": "clip-vit-b32",
        "sampling": SAMPLING,
    }
)

__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "CONFIG_DIGEST",
    "LocalBackend",
]


class LocalBackend:
    backend_id = BACKEND_ID
    version = BACKEND_VERSION
    config_digest = CONFIG_DIGEST
    execution_mode = "local"

    def __init__(self, weights_root: Path, device: str = "cuda") -> None:
        self.weights_root = weights_root.resolve()
        self.device = device
        self.load_count = 0
        self._evaluator: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _create_evaluator(self) -> Any:
        if not self.weights_root.is_dir():
            raise FileNotFoundError(self.weights_root)
        os.environ["HARNESSEVAL_WEIGHTS_ROOT"] = str(self.weights_root)
        from ..metrics.background_consistency import BackgroundConsistencyMetric

        return BackgroundConsistencyMetric(device=self.device)

    def ensure_ready(self) -> None:
        if self._evaluator is not None:
            return
        with self._load_lock:
            if self._evaluator is None:
                self._evaluator = self._create_evaluator()
                self.load_count += 1

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> dict[str, Any]:
        _validate_video(video_path, video_fingerprint)
        self.ensure_ready()
        with self._inference_lock:
            frames, source_fps = _decode_frames(video_path)
            sampled = _sample_frames(frames, source_fps, {"fps": 2.0})
            if len(sampled) < 2:
                sampled = list(frames)
                sampling_policy = "all_frames_short_video"
            else:
                sampling_policy = "2fps"
            result = self._evaluator.compute(sampled)
        metric_name = "background_consistency"
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "metrics": {
                metric_name: {
                    "raw_score": _raw_score(metric_name, result),
                    "sampled_frames": len(sampled),
                    "sampling_policy": sampling_policy,
                }
            },
            "source_fps": source_fps,
            "decoded_frames": len(frames),
        }
