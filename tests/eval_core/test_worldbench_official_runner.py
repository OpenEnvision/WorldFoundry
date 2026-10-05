from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

def _load_script(name: str) -> ModuleType:
    from worldfoundry.evaluation.tasks.execution.framework.script_paths import resolve_benchmark_script

    path = resolve_benchmark_script(name)
    spec = importlib.util.spec_from_file_location(f"test_benchmark_zoo_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_worldbench_official_runner_normalizes_summary_and_sample_rows(tmp_path: Path) -> None:
    run_worldbench_official_runner = _load_script("run_worldbench_official_runner")
    upstream_results = tmp_path / "worldbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "summary": {
                    "video_based": {"accuracy": "62.5%"},
                    "text_based_accuracy": 0.75,
                    "multiple_choice_accuracy": 0.6,
                    "binary_accuracy": "80%",
                },
                "per_sample_scores": [
                    {"scene_id": "scene-001", "component": "video_based", "correct": True},
                    {"question_id": "q-001", "question_type": "multiple_choice", "prediction": "A", "answer": "B"},
                    {"question_id": "q-002", "question_type": "binary", "score": "100%"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = _jsonl(output_dir / "raw_metric_table.jsonl")
    sample_rows = _jsonl(output_dir / "per_sample_scores.jsonl")
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "official_worldbench_result_normalizer"
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["per_metric"]["video_based_accuracy"]["raw_score"] == 62.5
    assert scorecard["metrics"]["per_metric"]["video_based_accuracy"]["normalized_score"] == 0.625
    assert scorecard["metrics"]["per_metric"]["binary_accuracy"]["normalized_score"] == 0.8
    assert scorecard["metrics"]["per_metric"]["worldbench_average"]["normalized_score"] == pytest.approx(0.6875)
    assert raw_rows[-1]["metric_id"] == "worldbench_average"
    assert len(sample_rows) == 3
    assert sample_rows[1]["correct"] is False


def test_worldbench_official_runner_reads_results_path_env_and_computes_sample_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_worldbench_official_runner = _load_script("run_worldbench_official_runner")
    upstream_results = tmp_path / "worldbench_samples.csv"
    upstream_results.write_text(
        "\n".join(
            [
                "sample_id,component,question_type,correct",
                "v-001,video_based,,true",
                "v-002,video_based,,false",
                "q-001,text_based,multiple_choice,true",
                "q-002,text_based,multiple_choice,false",
                "q-003,text_based,binary,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDFOUNDRY_WORLDBENCH_RESULTS_PATH", str(upstream_results))
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(["--output-dir", str(output_dir), "--json"])

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    sample_rows = _jsonl(output_dir / "per_sample_scores.jsonl")
    assert exit_code == 0
    assert scorecard["validation"]["official_result_shape"]["formats"] == ["csv"]
    assert scorecard["metrics"]["per_metric"]["video_based_accuracy"]["normalized_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["multiple_choice_accuracy"]["normalized_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["binary_accuracy"]["normalized_score"] == 1.0
    assert scorecard["metrics"]["per_metric"]["text_based_accuracy"]["normalized_score"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["per_metric"]["worldbench_average"]["normalized_score"] == pytest.approx(7 / 12)
    assert scorecard["integration_evidence"] is False
    assert len(sample_rows) == 5


def test_worldbench_official_runner_materializes_in_tree_artifact_scores(tmp_path: Path) -> None:
    run_worldbench_official_runner = _load_script("run_worldbench_official_runner")
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "metrics.json").write_text(
        json.dumps(
            {
                "video_based_accuracy": 0.7,
                "multiple_choice_accuracy": 0.8,
                "binary_accuracy": 0.6,
            }
        ),
        encoding="utf-8",
    )
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "0001.mp4").write_bytes(b"")
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(
        [
            "--run-official",
            "--artifact-score-dir",
            str(score_dir),
            "--generated-video-dir",
            str(video_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "worldbench_official_in_tree"
    assert scorecard["validation"]["normalizer_only"] is False
    assert scorecard["validation"]["official_runtime_executed"] is True
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_results_imported"] is False
    assert scorecard["metrics"]["per_metric"]["worldbench_average"]["normalized_score"] == pytest.approx(0.7)
