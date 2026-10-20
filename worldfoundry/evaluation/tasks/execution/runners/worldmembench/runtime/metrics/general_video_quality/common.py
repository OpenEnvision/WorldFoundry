# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Reference stride-4 frame scoring and equal-weight video aggregation."""

import math

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.aggregation import (
    aggregate_video_summary,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import ItemResult, UnitResult
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import (
    VideoReader,
    fixed_stride_indices,
)


class FrameQualityMetric(Metric):
    raw_key: str
    normalizer: float
    unit_type = "frame"

    def __init__(self, *, backend=None, reader=None) -> None:
        self.backend = backend
        self.reader = reader or VideoReader()

    def _backend(self, context: EvaluationContext):
        raise NotImplementedError

    def evaluate(self, sample: dict, context: EvaluationContext) -> list[UnitResult]:
        path = sample["video"]
        info = self.reader.probe(path)
        indices = fixed_stride_indices(info.total_frames, 4)
        decoded = context.get_cached(
            ("rgb_frames", path, tuple(indices)),
            lambda: self.reader.read_indices(path, indices),
        )
        backend = self.backend if self.backend is not None else self._backend(context)
        raw_scores = [float(value) for value in backend.score([frame.rgb for frame in decoded])]
        if len(raw_scores) != len(decoded):
            raise EvaluationError(f"{self.name} backend returned {len(raw_scores)} scores for {len(decoded)} frames")
        if not all(math.isfinite(value) for value in raw_scores):
            raise EvaluationError(f"{self.name} backend returned a non-finite score")
        return [
            UnitResult(
                unit_id=f"frame_{frame.frame_idx:08d}",
                unit_type="frame",
                scores={"score": max(0.0, min(1.0, raw / self.normalizer)), self.raw_key: raw},
                metadata={"frame_idx": frame.frame_idx, "video_path": path},
            )
            for frame, raw in zip(decoded, raw_scores)
        ]

    def aggregate_item(self, sample: dict, units: list[UnitResult]) -> ItemResult:
        if not units:
            raise EvaluationError(f"{self.name} produced no frame units")
        identity = {key: sample.get(key) for key in ("item_id", "model_id", "category", "sample_id", "condition_id")}
        non_scored = [unit for unit in units if not unit.status.is_scored]
        if non_scored:
            first = non_scored[0]
            return ItemResult(**identity, num_units=len(units), status=first.status, error=first.error)
        return ItemResult(
            **identity,
            scores={key: sum(unit.scores[key] for unit in units) / len(units) for key in ("score", self.raw_key)},
            num_units=len(units),
            metadata={"sampled_frame_indices": [unit.metadata["frame_idx"] for unit in units]},
        )

    aggregate_summary = staticmethod(aggregate_video_summary)
