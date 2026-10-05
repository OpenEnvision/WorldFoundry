import json
import runpy
import subprocess
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.wbench.run_wbench_official_runner import (
    METRIC_ORDER,
    build_parser,
    run_official_wbench,
    stage_generated_videos,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import CLI_RUNNERS


def test_wbench_dataset_root_reaches_runtime(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "WBench"
    (dataset_root / "cases").mkdir(parents=True)
    args = build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--dataset-root", str(dataset_root)]
    )
    monkeypatch.setenv("WBENCH_DATA_DIR", str(args.dataset_root))
    runtime = runpy.run_path(
        "worldfoundry/evaluation/tasks/execution/runners/wbench/runtime/wbench/main.py",
        run_name="wbench_runtime_test",
    )

    assert CLI_RUNNERS["wbench"].dataset_root_arg == "--dataset-root"
    assert Path(runtime["DATA_DIR"]) == dataset_root


def test_wbench_stages_only_case_identified_videos(tmp_path: Path) -> None:
    dataset_root = tmp_path / "WBench"
    cases_dir = dataset_root / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "case_2.json").write_text('{"id": "2"}', encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    source = generated_dir / "case_2__generated_video.mp4"
    source.write_bytes(b"video")
    videos_dir = tmp_path / "out" / "wbench_work" / "model" / "videos"
    manifest_path = tmp_path / "out" / "wbench_staging.json"

    rows = stage_generated_videos(
        generated_artifact_dir=generated_dir,
        videos_dir=videos_dir,
        dataset_root=dataset_root,
        staging_manifest_path=manifest_path,
    )

    staged = videos_dir / "case_2_combined.mp4"
    assert staged.is_file()
    assert staged.read_bytes() == b"video"
    assert rows[0]["case_id"] == "2"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["video_count"] == 1

    (videos_dir.parent / "evaluation").mkdir()
    source.write_bytes(b"different video")
    with pytest.raises(ValueError, match="inputs changed"):
        stage_generated_videos(
            generated_artifact_dir=generated_dir,
            videos_dir=videos_dir,
            dataset_root=dataset_root,
            staging_manifest_path=manifest_path,
        )

    unmatched_dir = tmp_path / "unmatched"
    unmatched_dir.mkdir()
    (unmatched_dir / "sample-0000.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="no identifiable WBench videos"):
        stage_generated_videos(
            generated_artifact_dir=unmatched_dir,
            videos_dir=tmp_path / "unmatched-out" / "videos",
            dataset_root=dataset_root,
            staging_manifest_path=tmp_path / "unmatched-out" / "staging.json",
        )


def test_wbench_official_command_uses_isolated_staged_work_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "WBench"
    cases_dir = dataset_root / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "case_1.json").write_text('{"id": "1"}', encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    source = generated_dir / "case_1_combined.mp4"
    source.write_bytes(b"video")
    output_dir = tmp_path / "output"
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, text, capture_output, check, timeout):
        captured.update(command=command, cwd=cwd, env=env)
        model = command[command.index("--model") + 1]
        work_dir = Path(command[command.index("--work_dir") + 1])
        evaluation_dir = work_dir / model / "evaluation"
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "report.json").write_text(
            json.dumps({"full": {"temporal_flickering": {"mean": 0.99}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setenv("WORLDFOUNDRY_MODEL_ID", "org/bernini")
    monkeypatch.setattr(subprocess, "run", fake_run)
    args = build_parser().parse_args(
        [
            "--run-official",
            "--dataset-root",
            str(dataset_root),
            "--generated-artifact-dir",
            str(generated_dir),
            "--phase",
            "gpu",
            "--metrics",
            "temporal_flickering",
            "--output-dir",
            str(output_dir),
        ]
    )

    scorecard = run_official_wbench(args)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--model") + 1] == "org_bernini"
    assert command[command.index("--phase") + 1] == "gpu"
    assert command[command.index("--metrics") + 1] == "temporal_flickering"
    work_dir = Path(command[command.index("--work_dir") + 1])
    assert work_dir == output_dir / "wbench_work"
    assert (work_dir / "org_bernini" / "videos" / "case_1_combined.mp4").is_file()
    assert captured["env"]["WBENCH_DATA_DIR"] == str(dataset_root)
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["metrics"]["leaderboard"] == {"temporal_flickering": 0.99}
    assert scorecard["metrics"]["summary"] == {
        "sample_count": 1,
        "metric_count": len(METRIC_ORDER),
        "available_metrics": 1,
        "failed_metrics": len(METRIC_ORDER) - 1,
    }
    assert scorecard["model"] == {
        "model_id": "org/bernini",
        "submission_protocol": "unverified",
    }
    assert scorecard["evaluation"]["benchmark_comparable"] is False
    assert "did not declare" in " ".join(scorecard["evaluation"]["comparability_blockers"])
