from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.tasks.execution.framework.video_quality_contract import (
    LOCAL_METRIC_IDS,
    evaluate_video_quality_contract_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
BENCHMARKS = ("aigcbench", "mirabench", "genai-bench", "fetv")


@pytest.fixture()
def dummy_manifest(tmp_path: Path) -> Path:
    """Build a one-sample generated-video manifest.

    Parameters:
        tmp_path: Pytest temporary directory for the manifest and artifact.
    """

    video_path = tmp_path / "sample-001.mp4"
    video_path.write_bytes(b"dummy video bytes")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "prompt_id": "sample-001",
                        "prompt": "A calm ocean wave in slow motion.",
                        "generated_video": str(video_path),
                        "metadata": {"model_id": "dummy-video-model"},
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize("benchmark_id", BENCHMARKS)
def test_video_quality_local_evaluator_scores_dummy_manifest(benchmark_id: str, dummy_manifest: Path) -> None:
    """Verify local checks run without official scorers.

    Parameters:
        benchmark_id: Target AIGC/video benchmark contract id.
        dummy_manifest: One-sample manifest with a readable video artifact.
    """

    result = evaluate_video_quality_contract_file(benchmark_id, dummy_manifest)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert result["benchmark_id"] == benchmark_id
    assert result["summary"]["sample_count"] == 1
    assert result["summary"]["blocked"] > 0
    for metric_id in LOCAL_METRIC_IDS:
        assert rows[metric_id]["score"] == 1.0
        assert rows[metric_id]["value"] == 1.0
        assert rows[metric_id]["status"] == "passed"
        assert rows[metric_id]["evidence"]["protocol"] == "local_manifest_metadata_contract"
        assert rows[metric_id]["blocked_reason"] is None
        assert {"metric_id", "score", "value", "status", "evidence", "blocked_reason"} <= set(rows[metric_id])


@pytest.mark.parametrize(
    ("benchmark_id", "metric_id", "blocked_reason"),
    (
        ("aigcbench", "aigcbench_average", "asset_required"),
        ("mirabench", "mirabench_average", "asset_required"),
        ("genai-bench", "genai_bench_average", "judge_required"),
        ("ipv-bench", "ipv_bench_average", "judge_required"),
        ("fetv", "fetv_average", "asset_required"),
    ),
)
def test_video_quality_local_evaluator_blocks_official_metrics(
    benchmark_id: str,
    metric_id: str,
    blocked_reason: str,
    dummy_manifest: Path,
) -> None:
    """Verify official metrics are not fabricated.

    Parameters:
        benchmark_id: Target AIGC/video benchmark contract id.
        metric_id: Official benchmark metric expected to be blocked.
        blocked_reason: Required blocked reason for this benchmark family.
        dummy_manifest: One-sample manifest with a readable video artifact.
    """

    result = evaluate_video_quality_contract_file(benchmark_id, dummy_manifest)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert rows[metric_id]["status"] == "blocked"
    assert rows[metric_id]["score"] is None
    assert rows[metric_id]["value"] is None
    assert rows[metric_id]["blocked_reason"] == blocked_reason
    assert rows[metric_id]["evidence"]["required_input_fields"]


def test_video_quality_evaluator_normalizes_toy_aigcbench_official_results(tmp_path: Path) -> None:
    """Verify caller-provided official AIGCBench rows are normalized.

    Parameters:
        tmp_path: Pytest temporary directory for toy official result files.
    """

    video_path = tmp_path / "sample-001.mp4"
    video_path.write_bytes(b"dummy video bytes")
    official_path = tmp_path / "official.csv"
    official_path.write_text(
        "\n".join(
            [
                "metric_id,score,prompt_type",
                "DOVER,0.8,ours",
                "DOVER,0.6,webvid",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "prompt_id": "sample-001",
                        "prompt": "A calm ocean wave in slow motion.",
                        "generated_video": str(video_path),
                    }
                ],
                "official_results_path": str(official_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = evaluate_video_quality_contract_file("aigcbench", manifest_path)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert rows["dover"]["status"] == "scored"
    assert rows["dover"]["score"] == 0.7
    assert rows["dover"]["evidence"]["category_scores"] == {"ours": 0.8, "webvid": 0.6}
    assert rows["mse_first"]["status"] == "blocked"


def test_video_quality_evaluator_scores_genai_pairwise_preferences(tmp_path: Path) -> None:
    """Verify GenAI-Bench preference accuracy is deterministic with labels.

    Parameters:
        tmp_path: Pytest temporary directory for toy preference data.
    """

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [{"prompt_id": "sample-001", "prompt": "A wave.", "generated_video": "sample.mp4"}],
                "pairwise_preferences": [
                    {"task": "video_generation", "human_label": "A>B", "prediction": "A>B"},
                    {"task": "video_generation", "human_label": "B>A", "prediction": "A>B"},
                    {"task": "image_generation", "human_label": "A=B=Good", "prediction": "A=B=Good"},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = evaluate_video_quality_contract_file("genai-bench", manifest_path)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert rows["pairwise_accuracy"]["status"] == "scored"
    assert rows["pairwise_accuracy"]["score"] == pytest.approx(2 / 3)
    assert rows["video_preference_accuracy"]["score"] == 0.5
    assert rows["image_generation_preference_accuracy"]["score"] == 1.0
    assert rows["image_editing_preference_accuracy"]["status"] == "blocked"
    assert rows["genai_bench_average"]["score"] == 0.75


def test_video_quality_evaluator_normalizes_fetv_official_results(tmp_path: Path) -> None:
    """Verify FETV official metric files are normalized and aggregated.

    Parameters:
        tmp_path: Pytest temporary directory for toy official result files.
    """

    official_path = tmp_path / "fetv.csv"
    official_path.write_text(
        "\n".join(
            [
                "metric_id,score",
                "static_quality,0.80",
                "temporal_quality,0.70",
                "overall_alignment,0.90",
                "fine_grained_alignment,0.60",
                "clip_score,0.50",
                "blip_score,0.40",
                "fid,0.30",
                "fvd,0.20",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [{"prompt_id": "sample-001", "prompt": "A wave.", "generated_video": "sample.mp4"}],
                "official_results_path": str(official_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = evaluate_video_quality_contract_file("fetv", manifest_path)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert rows["static_quality"]["status"] == "scored"
    assert rows["static_quality"]["score"] == 0.8
    assert rows["fetv_average"]["status"] == "scored"
    assert rows["fetv_average"]["score"] == pytest.approx(0.55)


def test_video_quality_evaluator_blocks_genai_without_labels(dummy_manifest: Path) -> None:
    """Verify GenAI-Bench remains blocked when official labels are absent.

    Parameters:
        dummy_manifest: One-sample manifest without official preference labels.
    """

    result = evaluate_video_quality_contract_file("genai-bench", dummy_manifest)
    rows = {item["metric_id"]: item for item in result["results"]}

    assert rows["pairwise_accuracy"]["status"] == "blocked"
    assert rows["pairwise_accuracy"]["blocked_reason"] == "judge_required"


@pytest.mark.parametrize(
    ("benchmark_id", "blocked_reason"),
    (
        ("aigcbench", "asset_required"),
        ("mirabench", "asset_required"),
        ("genai-bench", "judge_required"),
        ("ipv-bench", "judge_required"),
        ("fetv", "asset_required"),
    ),
)
def test_video_quality_task_yaml_declares_local_evaluator(benchmark_id: str, blocked_reason: str) -> None:
    """Verify task metadata keeps the in-tree evaluator planned.

    Parameters:
        benchmark_id: Target AIGC/video benchmark task yaml id.
        blocked_reason: Required blocked reason for official metrics.
    """

    task_yaml = yaml.safe_load((TASK_DIR / f"{benchmark_id}.yaml").read_text(encoding="utf-8"))
    task_name = next(iter(task_yaml["tasks"]))
    protocols = task_yaml["tasks"][task_name]["evaluation_protocol"]

    assert task_yaml["metadata"]["local_evaluator"]["status"] == "implemented"
    assert task_yaml["metadata"]["local_evaluator"]["official_metric_blocked_reason"] == blocked_reason
    assert task_yaml["metadata"]["local_evaluator"]["official_result_normalization"]["status"] == "implemented"
    local_protocol = next(item for item in protocols if item.get("name") == "in_tree_local_quality_evaluator")
    assert local_protocol["status"] == "implemented"
    assert "video_quality_contract_file" in local_protocol["evaluator_target"]
    blocked_evaluators = []
    for metric in task_yaml["metrics"].values():
        evaluator = metric["evaluator"]
        assert "video_quality_contract_file" in evaluator["target"]
        if evaluator["status"] == "blocked":
            assert evaluator["blocked_reason"] == blocked_reason
            blocked_evaluators.append(evaluator)
        else:
            assert evaluator["status"] == "bounded_official_runtime_verified"
    assert blocked_evaluators


@pytest.mark.parametrize("benchmark_id", BENCHMARKS)
def test_video_quality_checked_in_sample_fixture_runs(benchmark_id: str, tmp_path: Path) -> None:
    import importlib.util

    from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
    from worldfoundry.evaluation.utils import REPO_ROOT, benchmark_task_sample_path

    assert benchmark_task_sample_path(benchmark_id) is not None
    fixture_path = benchmark_task_sample_path(benchmark_id)
    assert fixture_path is not None
    script = REPO_ROOT / VIDEO_RUNNER_REGISTRY[benchmark_id].script
    spec = importlib.util.spec_from_file_location(f"{benchmark_id}_runner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_dir = tmp_path / benchmark_id
    exit_code = module.main(
        [
            "--official-results-path",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert scorecard["normalization_ok"] is True
    assert scorecard["metrics"]["leaderboard"]
