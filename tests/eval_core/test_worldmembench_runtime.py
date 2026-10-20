# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Weight-free checks of the native WorldMemBench protocol and runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.evaluate import evaluate
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics import registry
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.dynamic.checklist import CATEGORIES
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.dynamic.scoring import (
    aggregate_scores,
    score_answers,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.general_video_quality.aesthetic_quality import (
    AestheticQualityMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.general_video_quality.imaging_quality import (
    ImagingQualityMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.human.evaluator import (
    MaskedDinoV2Features as HumanFeatures,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.object.evaluator import (
    MaskedDinoV2Features as ObjectFeatures,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.registry import MetricRegistry
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import DecodedFrame, VideoInfo
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.worldmembench_metrics import (
    METRIC_SPECS,
    load_official_results,
)

pytestmark = pytest.mark.unit


class _Reader:
    def probe(self, path: str) -> VideoInfo:
        return VideoInfo(9 if Path(path).stem == "long" else 5, 24.0, 2, 2)

    def read_indices(self, path: str, indices: list[int]) -> list[DecodedFrame]:
        value = 80 if Path(path).stem == "long" else 20
        return [DecodedFrame(index, np.full((2, 2, 3), value, dtype=np.uint8)) for index in indices]


class _Backend:
    def score(self, frames: list[np.ndarray]) -> list[float]:
        return [float(frame[0, 0, 0]) for frame in frames]


def test_all_official_metrics_are_available_without_upstream_package() -> None:
    assert set(registry.names()) == set(METRIC_SPECS)
    completed = subprocess.run(
        [sys.executable, "-m", "worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime", "list-metrics"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(completed.stdout.splitlines()) == set(METRIC_SPECS)


def test_human_and_object_share_masked_feature_extraction() -> None:
    assert HumanFeatures is ObjectFeatures
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    mask[1, 2] = False
    crop = HumanFeatures._masked_crop(frame, mask)
    assert crop is not None
    expected = frame[1:3, 1:4].copy()
    expected[0, 1] = 0
    np.testing.assert_array_equal(crop, expected)
    assert HumanFeatures._masked_crop(frame, np.zeros_like(mask)) is None


@pytest.mark.parametrize("metric_type,normalizer", [(ImagingQualityMetric, 100.0), (AestheticQualityMetric, 10.0)])
def test_quality_uses_stride_four_clipping_and_equal_video_weights(
    tmp_path: Path, metric_type: Any, normalizer: float
) -> None:
    manifest = tmp_path / "inputs.jsonl"
    rows = []
    for name in ("long", "short"):
        (tmp_path / f"{name}.mp4").touch()
        rows.append({"item_id": name, "model_id": "model_a", "video": f"{name}.mp4"})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = MetricRegistry()
    metric = metric_type(backend=_Backend(), reader=_Reader())
    metrics.register(metric)
    output, success = evaluate(
        manifest,
        metrics=[metric.name],
        metric_registry=metrics,
        device="cpu",
        weights_dir=tmp_path / "assets",
        output=tmp_path / "native",
    )
    assert success
    model, extracted, cases = load_official_results(output)
    assert model == "model_a"
    expected = (min(1.0, 80.0 / normalizer) + min(1.0, 20.0 / normalizer)) / 2
    assert extracted[metric.name]["raw_score"] == pytest.approx(expected)
    assert [row["metadata"]["sampled_frame_indices"] for row in cases] == [[0, 4, 8], [0, 4]]
    assert [row["num_units"] for row in cases] == [3, 2]


@pytest.mark.parametrize("started,expected", [("yes", 1.0), ("no", 0.0)])
def test_dynamic_memory_event_gate_applies_to_all_eight_questions(started: str, expected: float) -> None:
    categories = [category for category in CATEGORIES for _ in range(8 // len(CATEGORIES))]
    assert len(categories) == 8
    checklist = SimpleNamespace(
        questions=[
            SimpleNamespace(
                id=f"q{index}", expected_answer="yes", category=category, question="question", evidence_window="return"
            )
            for index, category in enumerate(categories)
        ]
    )
    response = SimpleNamespace(
        event_started=SimpleNamespace(answer=started),
        answers=[SimpleNamespace(answer="yes", reason="visible", evidence_image_numbers=(1,)) for _ in range(8)],
    )
    units = score_answers(checklist, response, [4])
    assert aggregate_scores(units)["score"] == expected
    assert all(unit.metadata["ungated_score"] == 1.0 for unit in units)
    assert all(unit.metadata["evidence_frame_indices"] == [4] for unit in units)
