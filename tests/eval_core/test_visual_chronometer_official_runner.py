from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_predict import (
    mock_predict_directory,
    parse_results_csv,
    write_results_csv,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.visual_chronometer_metrics import (
    compute_visual_chronometer_metrics,
)


def test_visual_chronometer_metrics_from_mock_records(tmp_path: Path) -> None:
    tmp_path.joinpath("0001.mp4").write_bytes(b"fake")
    records = mock_predict_directory(tmp_path)
    computed = compute_visual_chronometer_metrics(video_records=records)
    metrics = computed["metrics"]
    assert metrics["mean_phyfps"] is not None
    assert metrics["inter_video_cv"] is not None
    assert metrics["intra_video_cv"] is not None
    assert metrics["visual_chronometer_average"] is not None


def test_visual_chronometer_official_run_with_mock_backend_writes_scorecard(tmp_path: Path) -> None:
    import os

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-a.mp4").write_bytes(b"fake")
    (generated_dir / "sample-b.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_VISUAL_CHRONOMETER_PREDICT_BACKEND"] = "mock"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyfps_bench_gen/run_visual_chronometer_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert (output_dir / "results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "visual_chronometer_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 4


def test_visual_chronometer_results_csv_roundtrip(tmp_path: Path) -> None:
    tmp_path.joinpath("0001.mp4").write_bytes(b"fake")
    records = mock_predict_directory(tmp_path)
    csv_path = tmp_path / "results.csv"
    write_results_csv(csv_path, records)
    parsed = parse_results_csv(csv_path)
    assert parsed
    assert parsed[0].video == "0001.mp4"
    assert parsed[0].avg_phyfps == records[0].avg_phyfps
