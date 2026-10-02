from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_metrics import compute_phyfps_metrics
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_predict import (
    VideoPhyFPSRecord,
    mock_predict_directory,
    parse_results_csv,
    write_results_csv,
)


def test_phyfps_metrics_match_official_formulas() -> None:
    records = [
        VideoPhyFPSRecord("0001.mp4", 30.0, (28.0, 30.0, 32.0)),
        VideoPhyFPSRecord("0002.mp4", 20.0, (19.0, 20.0, 21.0)),
    ]
    metrics = compute_phyfps_metrics(
        video_records=records,
        meta_fps_by_video={"0001.mp4": 24.0, "0002.mp4": 24.0},
    )["metrics"]
    assert metrics["avg_error_fps"] == 5.0
    assert metrics["pct_error"] == pytest.approx((25.0 + 16.666666666666664) / 2)
    assert metrics["inter_video_cv"] == pytest.approx(0.2)
    assert metrics["intra_video_cv"] is not None
    assert metrics["phyfps_bench_gen_average"] is not None


def test_phyfps_official_run_with_mock_backend_writes_scorecard(tmp_path: Path, monkeypatch) -> None:
    import os
    import subprocess
    import sys

    prompt_manifest = tmp_path / "prompts.txt"
    prompt_manifest.write_text("A dog runs across a lawn.\nA cat jumps onto a table.\n", encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0001.mp4").write_bytes(b"fake")
    (generated_dir / "0002.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_PHYFPS_PREDICT_BACKEND"] = "mock"
    env["WORLDFOUNDRY_PHYFPS_META_FPS"] = "24"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyfps_bench_gen/run_phyfps_bench_gen_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--prompt-manifest",
            str(prompt_manifest),
            "--meta-fps",
            "24",
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
    assert scorecard["evaluation"]["kind"] == "phyfps_bench_gen_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 5


def test_phyfps_prompt_materialization_uses_zero_padded_ids(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_prompts import (
        materialize_phyfps_generation_requests,
    )

    prompt_manifest = tmp_path / "prompts.txt"
    prompt_manifest.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    requests = materialize_phyfps_generation_requests(prompt_manifest_path=prompt_manifest)
    assert [request.sample_id for request in requests] == ["0001", "0002"]


def test_phyfps_results_csv_roundtrip(tmp_path: Path) -> None:
    records = mock_predict_directory(tmp_path)
    tmp_path.joinpath("0001.mp4").write_bytes(b"fake")
    csv_path = tmp_path / "results.csv"
    write_results_csv(csv_path, records)
    parsed = parse_results_csv(csv_path)
    assert parsed
    assert parsed[0].video == "0001.mp4"
    assert parsed[0].avg_phyfps == records[0].avg_phyfps
