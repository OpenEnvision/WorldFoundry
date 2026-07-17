from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.evaluation.reporting.validation import validate_contract_file
from worldfoundry.evaluation.tasks.execution.runners.four_d_worldbench.runtime.four_d_worldbench.runner import (
    validate_video_inputs,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_metrics import (
    COMPONENT_METRIC_ORDER,
    compute_iworldbench_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
    CANONICAL_PROMPT_COUNT,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.run_iworldbench_official_runner import (
    _metric_rows as iworldbench_metric_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.run_iworldbench_official_runner import (
    _scorecard as iworldbench_scorecard,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.run_iworldbench_official_runner import (
    main as iworldbench_main,
)
from worldfoundry.evaluation.tasks.execution.runners.memobench.run_memobench_official_runner import (
    _scorecard as memobench_scorecard,
)
from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
    PLUS_VARIANT_AVERAGE,
    aggregate_rows,
    canonical_declared_dimensions,
    raw_dimension_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_metrics import (
    compute_videophy2_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.videoscience_bench.runtime.videoscience_bench.judge.vlm_as_a_judge import (
    WEIGHTS,
    _compute_overall_1to4,
)
from worldfoundry.evaluation.tasks.execution.runners.videoscience_bench.runtime.videoscience_bench.videoscience_batch import (
    _metrics_from_rubric,
)
from worldfoundry.evaluation.tasks.execution.runners.videoverse.run_videoverse_official_runner import (
    normalize_videoverse_results,
)
from worldfoundry.evaluation.tasks.execution.runners.vmbench.vmbench_prompts import (
    materialize_vmbench_meta_info,
)


def test_memobench_step1_is_not_full_benchmark_verification(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_dir=tmp_path,
        benchmark_id="memobench",
        generated_artifact_dir=tmp_path / "videos",
    )
    metrics = {
        "visual_quality": {
            "raw_score": 0.5,
            "normalized_score": 0.5,
            "source": "step1.csv",
            "sample_count": 1,
        }
    }

    scorecard = memobench_scorecard(
        args=args,
        results_paths=[tmp_path / "step1.csv"],
        metrics=metrics,
        official_runtime={"returncode": 0},
    )

    assert scorecard["evaluation"]["step1_runtime_verified"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["evaluation"]["full_benchmark_verified"] is False


def test_iworldbench_partial_component_does_not_emit_full_suite_average() -> None:
    computed = compute_iworldbench_metrics(
        rows=[
            {"metric_id": "memory_symmetry", "score": 1.0},
            {"metric_id": "iworldbench_average", "score": 1.0},
        ]
    )

    assert computed["metrics"] == {"memory_symmetry": 1.0}
    assert "iworldbench_average" not in computed["components"]["sample_counts"]


def test_iworldbench_average_requires_complete_metric_suite() -> None:
    rows = [
        {"metric_id": metric_id, "score": index / 10}
        for index, metric_id in enumerate(COMPONENT_METRIC_ORDER, start=1)
    ]

    computed = compute_iworldbench_metrics(rows=rows)

    assert computed["metrics"]["iworldbench_average"] == pytest.approx(0.5)
    assert computed["components"]["sample_counts"]["iworldbench_average"] == len(
        COMPONENT_METRIC_ORDER
    )


def test_iworldbench_partial_official_run_is_not_full_benchmark(tmp_path: Path) -> None:
    source_path = tmp_path / "reports"
    computed = compute_iworldbench_metrics(
        rows=[{"metric_id": "memory_symmetry", "score": 1.0}]
    )
    metric_rows = iworldbench_metric_rows(
        computed=computed,
        source_path=source_path,
        official_runtime_executed=True,
    )

    scorecard = iworldbench_scorecard(
        benchmark_id="iworld-bench",
        output_dir=tmp_path,
        official_results_path=source_path,
        prompt_manifest_path=None,
        metric_rows=metric_rows,
        video_coverage={"expected_count": 1, "actual_count": 1, "complete": True},
        runtime_summary={"backend": "official", "metric": "memory"},
        official_runtime_executed=True,
        prompt_count=CANONICAL_PROMPT_COUNT,
        split="mem",
    )

    assert scorecard["integration_evidence"] is True
    assert scorecard["eligibility"]["official_component_verified"] is True
    assert scorecard["eligibility"]["full_suite_valid"] is False
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["metrics"]["leaderboard"] == {"memory_symmetry": 1.0}
    assert scorecard["dataset"] == {
        "dataset_id": "iworld-bench",
        "split": "mem",
        "prompt_manifest": None,
        "sample_count": CANONICAL_PROMPT_COUNT,
        "canonical_sample_count": CANONICAL_PROMPT_COUNT,
        "generated_video_count": 1,
        "coverage_complete": True,
    }
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(json.dumps(scorecard), encoding="utf-8")
    assert validate_contract_file(scorecard_path)["ok"] is True


def test_iworldbench_failure_scorecard_has_valid_dataset_mapping(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed"
    exit_code = iworldbench_main(
        [
            "--official-results-path",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(output_dir),
            "--split",
            "mem",
        ]
    )

    assert exit_code == 1
    scorecard_path = output_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert scorecard["dataset"]["dataset_id"] == "iworld-bench"
    assert scorecard["dataset"]["split"] == "mem"
    assert scorecard["dataset"]["sample_count"] == 0
    assert validate_contract_file(scorecard_path)["ok"] is True


def test_vbench2_aggregates_require_complete_declared_dimensions() -> None:
    expected_dimensions = canonical_declared_dimensions("vbench2")
    partial_rows = aggregate_rows(
        "vbench2",
        {"diversity": 0.75},
        expected_dimensions=expected_dimensions,
    )
    assert all(row["available"] is False for row in partial_rows)

    complete_category_rows = aggregate_rows(
        "vbench2",
        {"composition": 0.5, "diversity": 0.7},
        expected_dimensions=expected_dimensions,
    )
    by_metric = {row["metric_id"]: row for row in complete_category_rows}
    assert by_metric["vbench2_creativity"]["raw_score"] == pytest.approx(0.6)
    assert by_metric["vbench2_total"]["available"] is False

    full_rows = aggregate_rows(
        "vbench2",
        {metric_id: 0.5 for metric_id in expected_dimensions},
        expected_dimensions=expected_dimensions,
    )
    assert {row["metric_id"] for row in full_rows if row["available"]} == {
        "vbench2_creativity",
        "vbench2_commonsense",
        "vbench2_controllability",
        "vbench2_human_fidelity",
        "vbench2_physics",
        "vbench2_total",
    }


@pytest.mark.parametrize("variant", sorted(PLUS_VARIANT_AVERAGE))
def test_vbench_plus_plus_aggregates_require_complete_variant(
    variant: str,
) -> None:
    expected_dimensions = canonical_declared_dimensions(variant)
    one_dimension = next(iter(expected_dimensions))
    partial_rows = aggregate_rows(
        variant,
        {one_dimension: 0.8},
        expected_dimensions=expected_dimensions,
    )
    assert all(row["available"] is False for row in partial_rows)

    complete_rows = aggregate_rows(
        variant,
        {metric_id: 0.8 for metric_id in expected_dimensions},
        expected_dimensions=expected_dimensions,
    )
    by_metric = {row["metric_id"]: row for row in complete_rows}
    assert by_metric[PLUS_VARIANT_AVERAGE[variant]]["raw_score"] == pytest.approx(0.8)
    assert by_metric["vbench_plus_plus_average"]["available"] is False


def test_vbench_series_does_not_trust_imported_aggregate_scores() -> None:
    rows, scores = raw_dimension_rows(
        {
            "temporal_flickering": 0.9,
            "vbench_plus_plus_long_average": 0.9,
            "vbench_plus_plus_average": 0.9,
        }
    )

    assert scores == {"temporal_flickering": 0.9}
    aggregates = [row for row in rows if row["metric_id"] != "temporal_flickering"]
    assert all(row["available"] is False for row in aggregates)


def test_vmbench_rejects_zero_matching_generated_videos(tmp_path: Path) -> None:
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(
        json.dumps([{"index": "1", "prompt": "an object moves"}]),
        encoding="utf-8",
    )
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No VMBench generated videos matched"):
        materialize_vmbench_meta_info(
            video_dir=video_dir,
            output_path=tmp_path / "meta.json",
            prompt_suite_path=prompt_manifest,
        )


def test_videoverse_bounded_run_scores_only_the_selected_subset(tmp_path: Path) -> None:
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(
        json.dumps(
            {
                "1": {
                    "t2v_eval_event_info": {"verification_plan": [{"event": "first"}]},
                    "verification_checks": [],
                },
                "2": {
                    "t2v_eval_event_info": {
                        "verification_plan": [{"event": "a"}, {"event": "b"}, {"event": "c"}]
                    },
                    "verification_checks": [],
                },
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "1": {
                    "t2v_eval_event_info": {
                        "verification_plan": [{"event": "first"}],
                        "overall_event_processed_res": "A",
                    },
                    "verification_checks": [],
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmark_id="videoverse",
        official_results_path=results_path,
        output_dir=tmp_path / "out",
        generated_artifact_dir=None,
        prompt_manifest=prompt_manifest,
        decomposed_prompt_manifest=None,
        limit=1,
        strict=False,
    )

    scorecard = normalize_videoverse_results(
        args,
        official_runtime_executed=True,
        judge_summary={"processed_count": 1, "judge_backend": "test"},
    )

    assert scorecard["dataset"]["manifest_stats"]["prompt_count"] == 1
    assert scorecard["dataset"]["result_coverage"]["expected_count"] == 1
    assert scorecard["metrics"]["per_metric"]["event_coverage"]["score"] == 1.0
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False


def test_4dworldbench_rejects_an_undecodable_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "broken.mp4"
    video_path.write_bytes(b"not a video")

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API compatibility
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=FakeCapture))

    with pytest.raises(ValueError, match="could not decode a frame"):
        validate_video_inputs([{"video_list": [str(video_path)]}])


def test_videophy2_deprecated_aggregates_do_not_enter_metrics() -> None:
    computed = compute_videophy2_metrics(
        rows=[
            {"metric_id": "joint_score", "score": "0.8"},
            {"metric_id": "rule_classification_accuracy", "score": "0.7"},
            {"metric_id": "videophy2_average", "score": "0.75"},
        ]
    )

    assert computed["metrics"] == {"joint_score": 0.8}
    assert computed["components"]["deprecated_metrics"] == {
        "rule_classification_accuracy": 0.7,
        "videophy2_average": 0.75,
    }

    official_rows = compute_videophy2_metrics(
        rows=[
            {
                "sa": 5,
                "pc": 5,
                "physics_rules_followed": ["rule-a", "rule-b"],
                "physics_rules_unfollowed": ["rule-c"],
                "rule_classification_accuracy": 0.9,
            }
        ]
    )
    assert official_rows["metrics"]["rule_followed_rate"] == pytest.approx(2 / 3)
    assert "videophy2_average" not in official_rows["metrics"]
    assert official_rows["components"]["deprecated_metrics"] == {
        "rule_classification_accuracy": 0.9
    }


def test_videoscience_average_uses_official_weights() -> None:
    rubric = {
        "prompt_consistency": 1.0,
        "expected_phenomenon": 4.0,
        "coherence": 1.0,
        "immutability": 1.0,
        "dynamism": 1.0,
    }
    overall = _compute_overall_1to4(rubric, WEIGHTS)

    metrics = _metrics_from_rubric(rubric, overall_1to4=overall)

    assert metrics["videoscience_average"] == pytest.approx(0.3)
    assert metrics["videoscience_average"] != pytest.approx(
        sum(metrics[metric] for metric in metrics if metric != "videoscience_average") / 5
    )
