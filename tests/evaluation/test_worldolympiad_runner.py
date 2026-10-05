import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import CLI_RUNNERS
from worldfoundry.evaluation.tasks.execution.runners.worldolympiad import (
    run_worldolympiad_official_runner as runner,
)
from worldfoundry.evaluation.tasks.execution.runners.worldolympiad.worldolympiad_metrics import (
    METRIC_ORDER,
    WorldOlympiadResultError,
    normalize_results,
)

JUDGE_TEMPLATE = {
    "video": "cases/general/case_a/gen_case_a.mp4",
    "physical": {
        "score": 0.6,
        "dimension_scores": {"mechanics": 0.6, "thermotics": None, "material": None},
        "results": [
            {"question_id": "q1", "dimension": "mechanics", "related": True, "compliant": True, "confidence": 0.6},
            {"question_id": "q2", "dimension": "mechanics", "related": True, "compliant": False, "confidence": 0.9},
            {"question_id": "q3", "dimension": "material", "related": False, "compliant": None, "confidence": 0.0},
        ],
    },
    "interaction": {
        "summary": {"chunk_mean": 4.0, "transition_mean": 3.0, "global_score": 2.0, "overall": 3.0},
        "overall_raw": 3.0,
        "score": 0.6,
    },
    "clip_interaction": {"summary": {"semantic_adherence": 0.25}},
    "three_d": {
        "gs_score": 0.6,
        "meta_score": 0.5,
        "camera_motion_score": 0.4,
        "final_score": 1.5,
        "final_score_raw": 1.5,
        "final_score_normalized": 0.5,
    },
    "combined_score": 0.5667,
}


def write_case(case_root: Path, domain: str, case_id: str, payload: dict) -> Path:
    case_dir = case_root / domain / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"gen_judge_{case_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fixture_run_normalizes_every_declared_metric(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    assert runner.main(["--run-fixture", "--output-dir", str(output_dir)]) == 0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "worldolympiad_result_normalizer"
    assert scorecard["evaluation"]["full_triathlon"] is True
    assert scorecard["evaluation"]["covered_tracks"] == ["physical", "geometry", "interaction"]
    assert scorecard["metrics"]["summary"]["sample_count"] == 3
    assert set(scorecard["metrics"]["leaderboard"]) == set(METRIC_ORDER)
    # A normalizer-only run never claims official verification or leaderboard validity.
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["normalizer_only"] is True

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["metric_id"] for row in rows] == list(METRIC_ORDER)
    interaction_raw = next(row for row in rows if row["metric_id"] == "interaction_raw")
    assert interaction_raw["scale"] == "0-5"
    assert interaction_raw["normalized_score"] == pytest.approx(interaction_raw["raw_score"] / 5.0)

    cases = [
        json.loads(line)
        for line in (output_dir / "per_case_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(cases) == 3


def test_scores_average_per_case_and_compliance_accumulates_corpus_wide(tmp_path: Path) -> None:
    case_root = tmp_path / "cases"
    write_case(case_root, "general", "case_a", JUDGE_TEMPLATE)
    second = json.loads(json.dumps(JUDGE_TEMPLATE))
    second["three_d"]["final_score_normalized"] = 0.9
    second["physical"]["results"] = [
        {"question_id": "q1", "dimension": "mechanics", "related": True, "compliant": True, "confidence": 1.0}
    ]
    write_case(case_root, "gaming", "case_b", second)

    normalized = normalize_results(case_root)

    assert normalized["kind"] == "per_case_judge_files"
    assert normalized["case_count"] == 2
    # Score fields average per case; question verdicts accumulate as corpus counts (2 of 3 related).
    assert normalized["scores"]["three_d_score"] == pytest.approx(0.7)
    assert normalized["scores"]["physical_compliance_rate"] == pytest.approx(2 / 3)


def test_missing_geometry_track_is_reported_and_strict_mode_fails(tmp_path: Path) -> None:
    case_root = tmp_path / "cases"
    payload = json.loads(json.dumps(JUDGE_TEMPLATE))
    payload["three_d"] = None
    write_case(case_root, "embodied", "case_c", payload)
    output_dir = tmp_path / "out"

    assert runner.main(["--official-results-path", str(case_root), "--output-dir", str(output_dir)]) == 0
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["evaluation"]["full_triathlon"] is False
    assert scorecard["evaluation"]["covered_tracks"] == ["physical", "interaction"]
    assert any("missing tracks: geometry" in blocker for blocker in scorecard["evaluation"]["comparability_blockers"])

    strict_dir = tmp_path / "strict"
    exit_code = runner.main(
        ["--official-results-path", str(case_root), "--output-dir", str(strict_dir), "--strict"]
    )
    assert exit_code == 1
    strict_scorecard = json.loads((strict_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert strict_scorecard["normalization_ok"] is False


def test_run_official_uses_the_checked_in_runtime_and_isolates_batch_state(tmp_path: Path, monkeypatch) -> None:
    case_root = tmp_path / "outputs_batch"
    write_case(case_root, "general", "case_a", JUDGE_TEMPLATE)
    output_dir = tmp_path / "out"
    weights_dir = tmp_path / "weights"
    da3_src = tmp_path / "Depth-Anything-3" / "src"
    da3_src.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run_bounded_command(command, *, cwd, env=None, timeout, **kwargs):
        captured.update(command=list(command), cwd=cwd, env=env, timeout=timeout)
        return {"returncode": 0, "stdout": "ok", "stderr": "", "timed_out": False}

    monkeypatch.setattr(runner, "run_bounded_command", fake_run_bounded_command)
    # CLIP has no staged base_models asset: the runner always resolves its cache
    # from CLIP_DOWNLOAD_ROOT (or the managed checkpoint dir), never --weights-dir.
    monkeypatch.setenv("CLIP_DOWNLOAD_ROOT", str(weights_dir / "clip"))

    exit_code = runner.main(
        [
            "--run-official",
            "--generated-artifact-dir",
            str(case_root),
            "--pipelines",
            "cosmos-predict",
            "--weights-dir",
            str(weights_dir),
            "--da3-src",
            str(da3_src),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    # No external checkout: --run-official defaults to the vendored runtime.
    runtime_root = runner.DEFAULT_RUNTIME_ROOT
    assert (runtime_root / runner.RUNTIME_ENTRYPOINT).is_file()
    assert command[1] == str(runtime_root / runner.RUNTIME_ENTRYPOINT)
    assert captured["cwd"] == runtime_root

    # Mutable batch state must live under --output-dir, never inside the runtime.
    assert command[command.index("--manifest-dir") + 1] == str(output_dir / "batch_manifests")
    assert command[command.index("--log-root") + 1] == str(output_dir / "batch_logs")
    assert command[command.index("--output-root") + 1] == str(case_root)
    assert command[command.index("--domains") + 1] == "general"
    assert command[command.index("--pipelines") + 1] == "cosmos-predict"

    # Weights are passed explicitly so upstream never defaults them into the tree.
    assert command[command.index("--sam3-model") + 1] == str(weights_dir / "sam3" / "sam3.pt")
    assert command[command.index("--three-d-model-name") + 1] == str(weights_dir / "da3")
    assert command[command.index("--clip-download-root") + 1] == str(weights_dir / "clip")
    for flag, _relative in runner.WEIGHT_ARGS:
        assert not Path(command[command.index(flag) + 1]).is_relative_to(runtime_root)

    env = captured["env"]
    assert isinstance(env, dict)
    assert str(da3_src) in env["PYTHONPATH"].split(":")

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["evaluation"]["kind"] == "worldolympiad_official_in_tree"
    assert scorecard["evaluation"]["official_runtime_executed"] is True
    assert scorecard["run"]["official_runtime"]["in_tree_runtime"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["run"]["official_runtime"]["case_root"] == str(case_root)
    assert (output_dir / "official_runtime.log").is_file()
    # Verification and blockers must never disagree; optional per-dimension
    # metrics may be absent without blocking comparability.
    assert scorecard["evaluation"]["comparability_blockers"] == []
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["evaluation"]["missing_metrics"] == ["physical_thermotics", "physical_material"]


def test_checked_in_runtime_is_never_used_as_mutable_work_storage(tmp_path: Path) -> None:
    runtime_root = runner.DEFAULT_RUNTIME_ROOT
    args = runner.build_parser().parse_args(["--output-dir", str(tmp_path)])

    with pytest.raises(ValueError, match="must not write into the checked-in runtime"):
        runner.resolve_isolated_output_dir(runtime_root / "batch_logs", runtime_root=runtime_root)

    args.weights_dir = runtime_root / "weights"
    with pytest.raises(ValueError, match="must not write into the checked-in runtime"):
        runner.resolve_weights_dir(args, runtime_root=runtime_root)


def test_missing_runtime_entrypoint_is_reported_as_a_failure_scorecard(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--run-official",
            "--worldolympiad-root",
            str(tmp_path / "not-a-checkout"),
            "--generated-artifact-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["run"]["status"] == "failed"
    assert "evaluate_pipelines.py" in scorecard["run"]["error"]


def test_unreadable_results_raise_a_typed_error(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    with pytest.raises(WorldOlympiadResultError, match="no WorldOlympiad judge JSON files"):
        normalize_results(empty_root)


def test_worldolympiad_is_wired_into_both_runner_registries() -> None:
    spec = CLI_RUNNERS["worldolympiad"]
    video_spec = VIDEO_RUNNER_REGISTRY["worldolympiad"]

    assert spec.module == video_spec.script.removesuffix(".py").replace("/", ".")
    assert spec.supports_official_runtime is True
    assert spec.accepts_generated_artifacts is True
    assert spec.supports_fixture is True
    assert set(get_external_benchmark_contract("worldolympiad").metric_ids) == set(METRIC_ORDER)
