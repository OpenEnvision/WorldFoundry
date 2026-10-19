"""WorldScore fallback follows the official instance/metric/aspect reduction order."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from worldfoundry.evaluation.tasks.execution.runners.worldscore import run_worldscore_official_runner as runner

pytestmark = pytest.mark.unit


def test_tree_aggregation_matches_official_normalized_reference(tmp_path: Path) -> None:
    reference_path = Path(runner.__file__).parent / "runtime/worldscore/worldscore/run_evaluate.py"
    tree = ast.parse(reference_path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_mean_scores"
    )
    namespace = {"np": np, "worldscore_list": {"static": [*runner.CONTROLLABILITY_ASPECTS, *runner.QUALITY_ASPECTS]}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(reference_path), "exec"), namespace)
    payloads = [
        {
            "subjective_quality": {
                "clip_aesthetic": {"score": 20.0, "score_normalized": 0.8},
                "clip_iqa+": {"score": 999.0, "score_normalized": 0.2},
            }
        },
        {"subjective_quality": {"clip_aesthetic": {"score": 50.0, "score_normalized": 1.0}}},
    ]
    official_rows = defaultdict(lambda: defaultdict(list))
    for index, payload in enumerate(payloads):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "evaluation.json").write_text(json.dumps(payload))
        for aspect, metrics in payload.items():
            for metric_id, value in metrics.items():
                official_rows[aspect][metric_id].append(value)
    expected = namespace["calculate_mean_scores"](official_rows, "static", {})
    actual = runner.aggregate_worldscore_evaluation_tree(tmp_path)
    assert actual["subjective_quality"] == expected["subjective_quality"] == 55.0
    assert "worldscore_average" not in actual
    assert runner.sample_scores_from_evaluation_payload(payloads[1]) == {"subjective_quality": 100.0}
    assert runner.normalize_worldscore_value(0.5, percentage=True) == 0.005


def test_worldscore_overall_weights_aspects_not_families() -> None:
    scores = {aspect: 10.0 for aspect in runner.CONTROLLABILITY_ASPECTS}
    scores.update({aspect: 80.0 for aspect in runner.QUALITY_ASPECTS})
    assert runner.extract_scores(scores)["worldscore_average"]["raw_score"] == 60.0
    scores.update({aspect: 20.0 for aspect in runner.DYNAMICS_ASPECTS})
    assert runner.extract_scores(scores)["worldscore_average"]["raw_score"] == 48.0
    assert runner.extract_scores({"controllability": 10, "quality": 80})["worldscore_average"]["raw_score"] == 60.0


def test_worldscore_ignores_raw_numbers_and_diagnostics() -> None:
    assert runner.sample_scores_from_evaluation_payload(
        {
            "subjective_quality": {"clip_aesthetic": {"score": 20, "score_normalized": 0.8}},
            "timings": {"seconds": 999},
        }
    ) == {"subjective_quality": 80.0}
