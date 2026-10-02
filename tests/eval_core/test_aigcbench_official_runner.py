from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_metrics import compute_aigcbench_metrics
from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_aigcbench_generation_requests,
    unique_prompt_records,
    load_prompt_records,
)


def test_aigcbench_prompt_materialization_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompt_suite.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "265",
                    "prompt": "A playful panda is brewing a potion.",
                    "prompt_type": "ours",
                    "reference_image": "265_panda.png",
                },
                {
                    "prompt_id": "1010439683",
                    "prompt": "",
                    "prompt_type": "webvid",
                    "reference_video": "1010439683.mp4",
                },
            ]
        ),
        encoding="utf-8",
    )
    requests = materialize_aigcbench_generation_requests(prompt_manifest_path=manifest)
    assert [request.sample_id for request in requests] == ["265", "1010439683"]


def test_aigcbench_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in (
        "mse_first",
        "ssim_first",
        "image_genvideo_clip",
        "genvideo_text_clip",
        "genvideo_refvideo_clip_keyframes",
        "flow_square_mean",
        "genvideo_refvideo_clip_corresponding_frames",
        "genvideo_clip_adjacent_frames",
        "frame_count",
        "dover",
        "genvideo_refvideo_ssim",
        "aigcbench_average",
    )]
    metrics = compute_aigcbench_metrics(rows=rows)["metrics"]
    assert metrics["dover"] == 0.75
    assert metrics["aigcbench_average"] == 0.75


def test_aigcbench_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    from worldfoundry.evaluation.utils import benchmark_task_sample_path

    sample_path = benchmark_task_sample_path("aigcbench")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/aigcbench/run_aigcbench_official_runner.py",
            "--official-results-path",
            str(sample_path),
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] >= 1


def test_aigcbench_official_run_imports_workspace_artifact(tmp_path: Path) -> None:
    from worldfoundry.evaluation.utils import benchmark_task_sample_path

    manifest = tmp_path / "prompt_suite.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "265",
                    "prompt": "A playful panda is brewing a potion.",
                    "prompt_type": "ours",
                }
            ]
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "265.mp4").write_bytes(b"fake")
    sample_path = benchmark_task_sample_path("aigcbench")
    assert sample_path is not None
    generated_results_path = generated_dir / "aigcbench_results.csv"
    generated_results_path.write_bytes(sample_path.read_bytes())
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_AIGCBENCH_PROMPT_MANIFEST"] = str(manifest)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/aigcbench/run_aigcbench_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--prompt-manifest",
            str(manifest),
            "--limit",
            "1",
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
    assert (output_dir / "aigcbench_results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "aigcbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] >= 1
    assert scorecard["run"]["scorer_summary"]["source_results_path"] == str(generated_results_path.resolve())


def test_aigcbench_canonical_prompt_count_matches_documentation() -> None:
    assert CANONICAL_PROMPT_COUNT == 3927


def test_aigcbench_unique_prompt_records_from_inline_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompt_suite.json"
    manifest.write_text(
        json.dumps(
            [
                {"prompt_id": "a", "prompt": "first", "prompt_type": "ours"},
                {"prompt_id": "a", "prompt": "duplicate", "prompt_type": "ours"},
                {"prompt_id": "b", "prompt": "second", "prompt_type": "laion"},
            ]
        ),
        encoding="utf-8",
    )
    records = unique_prompt_records(load_prompt_records(prompt_manifest_path=manifest))
    assert [record["prompt_id"] for record in records] == ["b", "a"]
