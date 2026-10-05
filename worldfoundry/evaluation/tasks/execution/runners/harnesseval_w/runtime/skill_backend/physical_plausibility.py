"""Local PAVRM backend for Physical Plausibility."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..aggregate import numeric
from ..io import value_digest
from ..skills.skill_physical_plausibility import BACKEND_OUTPUT_SCHEMA, SAMPLING
from .common import _validate_video


BACKEND_ID = "harnesseval.physical_plausibility.pavrm"
BACKEND_VERSION = "harnesseval.pavrm_qwen3vl_a3b"
CONFIG_DIGEST = value_digest(
    {
        "implementation": "harnesseval",
        "metric": "pavrm_qwen3vl_a3b",
        "normalization": "raw_score/5",
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

    def __init__(
        self,
        weights_root: Path,
        model_path: Path,
        device: str = "cuda",
    ) -> None:
        self.weights_root = weights_root.resolve()
        self.model_path = model_path.resolve()
        self.device = device
        self.load_count = 0
        self._evaluator: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _create_evaluator(self) -> Any:
        if not self.weights_root.is_dir():
            raise FileNotFoundError(self.weights_root)
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        os.environ["HARNESSEVAL_WEIGHTS_ROOT"] = str(self.weights_root)
        from ..metrics.physical_plausibility import PhysicalPlausibilityEvaluator

        return PhysicalPlausibilityEvaluator(
            model_path=str(self.model_path),
            device=self.device,
        )

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
        try:
            prepare = getattr(self._evaluator, "prepare_video", None)
            score_prepared = getattr(self._evaluator, "score_prepared", None)
            if callable(prepare) and callable(score_prepared):
                prepared = prepare(str(video_path), fps=SAMPLING["fps"])
                with self._inference_lock:
                    result = score_prepared(prepared)
            else:
                with self._inference_lock:
                    result = self._evaluator.score_video(
                        str(video_path), fps=SAMPLING["fps"]
                    )
        except Exception as exc:  # Match the official evaluator's failure payload.
            result = {"raw_score": None, "score": None, "error": str(exc)}
        raw = None if isinstance(result.get("raw_score"), bool) else numeric(result.get("raw_score"))
        reported = None if isinstance(result.get("score"), bool) else numeric(result.get("score"))
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "metrics": {
                "pavrm": {
                    "raw_score": raw,
                    "reported_score": reported,
                    "error": result.get("error"),
                }
            },
        }
