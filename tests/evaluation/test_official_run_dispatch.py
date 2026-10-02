from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation import framework
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark
from worldfoundry.evaluation.tasks.execution.orchestration.interfaces import OfficialRunResult
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import (
    CLI_RUNNERS,
    build_workspace_benchmark_command,
    validate_workspace_registry,
)
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR


def test_every_video_benchmark_has_a_registered_route_with_explicit_capability() -> None:
    catalog = load_benchmark_zoo_registry(BENCHMARK_ZOO_DIR)

    assert validate_workspace_registry() == []
    assert set(VIDEO_RUNNER_REGISTRY) == set(CLI_RUNNERS) | {
        "vbench",
        "vbench-2.0",
        "vbench-plus-plus",
    }
    for benchmark_id in VIDEO_RUNNER_REGISTRY:
        entry = catalog.get(benchmark_id)
        assert benchmark_id in CLI_RUNNERS or entry.run_command is not None
    assert CLI_RUNNERS["worldbench"].supports_official_runtime is True
    assert CLI_RUNNERS["worldbench"].accepts_generated_artifacts is True
    assert CLI_RUNNERS["physics-iq"].supports_official_runtime is True
    assert CLI_RUNNERS["physics-iq"].accepts_generated_artifacts is True


def test_bounded_execution_success_is_distinct_from_full_official_evidence(tmp_path: Path) -> None:
    result = OfficialRunResult(
        benchmark_id="physics-iq",
        output_dir=tmp_path,
        scorecard_path=tmp_path / "scorecard.json",
        official_benchmark_verified=False,
        integration_evidence=True,
    )

    assert result.execution_ok is True
    assert result.operation_ok is True
    assert result.ok is True
    assert result.full_official_ok is False


def test_result_import_success_is_distinct_from_integration_evidence(tmp_path: Path) -> None:
    result = OfficialRunResult(
        benchmark_id="evalcrafter",
        output_dir=tmp_path,
        scorecard_path=tmp_path / "scorecard.json",
        official_benchmark_verified=False,
        integration_evidence=False,
        metadata={
            "mode": "official-validation",
            "normalizer_only": True,
            "normalization_ok": True,
            "returncode": 0,
        },
    )

    assert result.execution_ok is False
    assert result.operation_ok is True
    assert result.ok is True
    assert result.full_official_ok is False
    assert result.to_dict()["operation_ok"] is True


def test_official_run_does_not_promote_normalizer_success_to_evidence(tmp_path: Path) -> None:
    result = OfficialRunResult(
        benchmark_id="evalcrafter",
        output_dir=tmp_path,
        scorecard_path=tmp_path / "scorecard.json",
        official_benchmark_verified=False,
        integration_evidence=False,
        metadata={"mode": "official-run", "normalization_ok": True, "returncode": 0},
    )

    assert result.execution_ok is False
    assert result.operation_ok is False
    assert result.ok is False
    assert result.full_official_ok is False


def test_specialized_result_import_refreshes_runtime_report(tmp_path: Path) -> None:
    import json

    from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
        run_benchmark_execution,
    )
    from worldfoundry.evaluation.utils import benchmark_task_sample_path

    sample_results = benchmark_task_sample_path("evalcrafter")
    assert sample_results is not None
    result = run_benchmark_execution(
        "evalcrafter",
        output_dir=tmp_path,
        mode="official-validation",
        official_results_path=sample_results,
    )

    assert result.execution_ok is False
    assert result.operation_ok is True
    assert result.ok is True
    assert result.full_official_ok is False
    report = json.loads((tmp_path / "runner_runtime_report.json").read_text(encoding="utf-8"))
    assert all(item["ok"] is True for item in report["expected_artifact_checks"])
    assert report["scorecard_runtime_flags"]["normalization_ok"] is True
    assert report["official_benchmark_verified"] is False
    assert report["integration_evidence"] is False


def test_canonical_command_routes_generated_data_and_bounded_subset() -> None:
    command = build_workspace_benchmark_command(
        {
            "benchmark_id": "physics-iq-verified",
            "dataset_root": "/benchmark-data",
            "params": {
                "generated_artifact_dir": "/generated-videos",
                "run_official": True,
                "limit": 3,
            },
        },
        "/output",
    )

    assert command[command.index("--benchmark-id") + 1] == "physics-iq-verified"
    assert command[command.index("--dataset-root") + 1] == "/benchmark-data"
    assert command[command.index("--generated-artifact-dir") + 1] == "/generated-videos"
    assert command[command.index("--limit") + 1] == "3"
    assert "--run-official" in command

    evalcrafter_command = build_workspace_benchmark_command(
        {
            "benchmark_id": "evalcrafter",
            "params": {
                "generated_artifact_dir": "/evalcrafter-videos",
                "prompt_manifest": "/custom-prompts.txt",
                "metrics": ["clip_score", "clip_temp_score"],
                "limit": 1,
                "run_official": True,
            },
        },
        "/output",
    )
    assert evalcrafter_command[evalcrafter_command.index("--generated-video-dir") + 1] == "/evalcrafter-videos"
    assert evalcrafter_command[evalcrafter_command.index("--prompt-manifest") + 1] == "/custom-prompts.txt"
    assert evalcrafter_command[evalcrafter_command.index("--metrics") + 1] == "clip_score,clip_temp_score"
    assert evalcrafter_command[evalcrafter_command.index("--limit") + 1] == "1"
    assert "--run-official" in evalcrafter_command

    iworld_command = build_workspace_benchmark_command(
        {
            "benchmark_id": "iworld-bench",
            "dataset_root": "/iworld-data",
            "params": {
                "generated_artifact_dir": "/iworld-videos",
                "run_official": True,
                "limit": 2,
                "split": "mem",
            },
        },
        "/output",
    )
    assert iworld_command[iworld_command.index("--dataset-root") + 1] == "/iworld-data"
    assert "--iworld-root" not in iworld_command
    assert iworld_command[iworld_command.index("--split") + 1] == "mem"
    assert iworld_command[iworld_command.index("--generated-video-dir") + 1] == "/iworld-videos"


def test_canonical_commands_match_runner_capabilities() -> None:
    video_bench = build_workspace_benchmark_command(
        {
            "benchmark_id": "video-bench",
            "params": {"generated_artifact_dir": "/videos", "run_official": True},
        },
        "/output",
    )
    assert "--generated-video-dir" in video_bench
    assert "--run-official" not in video_bench

    wbench = build_workspace_benchmark_command(
        {
            "benchmark_id": "wbench",
            "params": {
                "generated_artifact_dir": "/videos",
                "model_id": "new-model",
                "run_official": True,
            },
        },
        "/output",
    )
    assert wbench[wbench.index("--model-name") + 1] == "new-model"

    worldscore = build_workspace_benchmark_command(
        {
            "benchmark_id": "worldscore",
            "dataset_root": "/worldscore-data",
            "params": {
                "generated_artifact_dir": "/videos",
                "model_id": "new-model",
            },
        },
        "/output",
    )
    assert worldscore[worldscore.index("--stage-dynamic-source") + 1] == "/videos"
    assert worldscore[worldscore.index("--model-name") + 1] == "new-model"

    worldbench = build_workspace_benchmark_command(
        {
            "benchmark_id": "worldbench",
            "dataset_root": "/worldbench-data",
            "params": {"generated_artifact_dir": "/videos", "run_official": True, "limit": 2},
        },
        "/output",
    )
    assert worldbench[worldbench.index("--dataset-root") + 1] == "/worldbench-data"
    assert worldbench[worldbench.index("--generated-video-dir") + 1] == "/videos"
    assert worldbench[worldbench.index("--limit") + 1] == "2"
    assert "--run-official" in worldbench


def test_model_benchmark_propagates_generation_subset_to_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "sample.mp4").write_bytes(b"video")
    captured: dict[str, object] = {}

    class StopAfterDispatch(RuntimeError):
        pass

    def capture_dispatch(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise StopAfterDispatch

    monkeypatch.setattr(model_benchmark, "run_benchmark_execution", capture_dispatch)
    with pytest.raises(StopAfterDispatch):
        model_benchmark.run_model_benchmark(
            benchmark_id="vbench",
            benchmark_manifest_path=BENCHMARK_ZOO_DIR,
            benchmark_mode="official-run",
            model_id="bernini",
            dataset_root="/benchmark-data",
            num_samples=3,
            split="test",
            generated_artifact_dir=generated,
            output_dir=tmp_path / "run",
        )

    assert captured["dataset_root"] == "/benchmark-data"
    assert captured["limit"] == 3
    assert captured["model_id"] == "bernini"
    assert captured["split"] == "test"


def test_empty_generated_artifact_dir_does_not_start_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "empty"
    generated.mkdir()

    def unexpected_dispatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluator must not start without generated artifacts")

    monkeypatch.setattr(model_benchmark, "run_benchmark_execution", unexpected_dispatch)
    with pytest.raises(RuntimeError, match="no generated artifacts"):
        model_benchmark.run_model_benchmark(
            benchmark_id="vbench",
            benchmark_manifest_path=BENCHMARK_ZOO_DIR,
            benchmark_mode="official-run",
            model_id="bernini",
            generated_artifact_dir=generated,
            output_dir=tmp_path / "run",
        )


def test_public_run_forwards_custom_benchmark_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class StopAfterDispatch(RuntimeError):
        pass

    def capture(request: object) -> None:
        captured["request"] = request
        raise StopAfterDispatch

    monkeypatch.setattr(framework, "run_model_benchmark", capture)
    with pytest.raises(StopAfterDispatch):
        framework.run_worldfoundry(
            output_dir=tmp_path,
            model_ids=("bernini",),
            benchmark_ids=("wbench",),
            benchmark_parameters={"metrics": ["temporal_flickering"], "phase": "gpu"},
        )

    request = captured["request"]
    assert isinstance(request, model_benchmark.ModelBenchmarkRunRequest)
    assert request.benchmark_parameters == {
        "metrics": ["temporal_flickering"],
        "phase": "gpu",
    }


def test_single_video_vbench_runtime_is_bounded_evidence(tmp_path: Path) -> None:
    import argparse
    import json

    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        DEFAULT_VBENCH_ROOT,
        normalize_vbench_results,
        validate_prompt_suite_materialization,
    )

    full_info = bundled_benchmark_asset("vbench", "VBench_full_info.json")
    rows = json.loads(full_info.read_text(encoding="utf-8"))
    prompt = next(row["prompt_en"] for row in rows if "temporal_flickering" in row["dimension"])
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / f"{prompt}-0.mp4").touch()
    split_clip = videos / "split_clip"
    split_clip.mkdir()
    (split_clip / "runtime-generated-fragment.mp4").touch()
    validation = validate_prompt_suite_materialization(
        argparse.Namespace(
            mode="vbench_standard",
            preset=[],
            full_json_dir=full_info,
            vbench_root=DEFAULT_VBENCH_ROOT,
            videos_path=videos,
            dimension=["temporal_flickering"],
        )
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"temporal_flickering": 0.99}', encoding="utf-8")
    scorecard = normalize_vbench_results(
        {"temporal_flickering": 0.99},
        benchmark_id="vbench",
        dimensions=["temporal_flickering"],
        output_dir=tmp_path / "output",
        upstream_results_path=upstream,
        videos_path=videos,
        command=["official-vbench"],
        duration_seconds=1.0,
        returncode=0,
        prompt_suite_validation=validation,
    )

    assert validation["covered_video_count"] == 1
    assert validation["full_suite_complete"] is False
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["dataset"]["generated_file_count"] == 1


def test_vbench_non_finite_results_fail_closed(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        normalize_vbench_results,
    )

    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"temporal_flickering": NaN}', encoding="utf-8")
    scorecard = normalize_vbench_results(
        {"temporal_flickering": float("nan")},
        benchmark_id="vbench",
        dimensions=["temporal_flickering"],
        output_dir=tmp_path / "output",
        upstream_results_path=upstream,
        videos_path=None,
        command=["official-vbench"],
        duration_seconds=1.0,
        returncode=0,
    )

    assert scorecard["normalization_ok"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["leaderboard"] == {}
    assert scorecard["metrics"]["per_metric"]["temporal_flickering"]["available"] is False


def test_vbench_series_non_finite_results_fail_closed() -> None:
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        raw_dimension_rows,
    )

    rows, scores = raw_dimension_rows(
        {"temporal_flickering": float("nan"), "motion_smoothness": "inf"}
    )

    assert scores == {}
    assert all(row["available"] is False for row in rows)


@pytest.mark.parametrize(
    ("variant", "asset_parts", "dimension", "mode", "custom_input"),
    [
        ("vbench2", ("vbench-2.0", "VBench2_full_info.json"), "diversity", "custom_input", False),
        (
            "i2v",
            ("vbench-plus-plus", "i2v", "vbench2_i2v_full_info.json"),
            "temporal_flickering",
            "custom_input",
            False,
        ),
        (
            "long",
            ("vbench-plus-plus", "long", "VBench_full_info.json"),
            "temporal_flickering",
            "long_custom_input",
            False,
        ),
        (
            "trustworthiness",
            ("vbench-plus-plus", "trustworthiness", "vbench2_trustworthy.json"),
            "temporal_flickering",
            "vbench_standard",
            True,
        ),
    ],
)
def test_single_video_vbench_series_runtime_is_never_full_suite(
    tmp_path: Path,
    variant: str,
    asset_parts: tuple[str, ...],
    dimension: str,
    mode: str,
    custom_input: bool,
) -> None:
    import argparse
    import json

    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        build_vbench2_video_coverage,
        canonical_suite_coverage,
        normalize_results,
    )

    videos = tmp_path / variant / "videos"
    videos.mkdir(parents=True)
    (videos / "bounded-0.mp4").touch()
    coverage = canonical_suite_coverage(
        argparse.Namespace(
            variant=variant,
            full_json_dir=bundled_benchmark_asset(*asset_parts),
            dimension=[dimension],
            mode=mode,
            custom_input=custom_input,
            videos_path=videos,
        )
    )
    upstream = tmp_path / variant / "upstream.json"
    upstream.write_text(json.dumps({dimension: 0.75}), encoding="utf-8")
    dataset_manifest = None
    video_coverage = None
    if variant == "vbench2":
        dataset_manifest = {
            "ready": False,
            "prompt_count": 0,
            "reference_video_names": [],
            "prompt_manifest": None,
        }
        video_coverage = build_vbench2_video_coverage(videos, dataset_manifest)
    scorecard = normalize_results(
        {dimension: 0.75},
        benchmark_id="vbench-2.0" if variant == "vbench2" else "vbench-plus-plus",
        variant=variant,
        output_dir=tmp_path / variant / "output",
        upstream_results_path=upstream,
        videos_path=videos,
        command=["official-vbench-series"],
        duration_seconds=1.0,
        returncode=0,
        stdout_path=None,
        stderr_path=None,
        dataset_manifest=dataset_manifest,
        video_coverage=video_coverage,
        suite_coverage=coverage,
    )

    assert coverage["full_suite_complete"] is False
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["leaderboard_valid"] is False


def test_imported_vbench_results_are_not_runtime_evidence(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        normalize_vbench_results,
    )
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        normalize_results,
    )

    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"temporal_flickering": 0.5}', encoding="utf-8")
    classic = normalize_vbench_results(
        {"temporal_flickering": 0.5},
        benchmark_id="vbench",
        dimensions=["temporal_flickering"],
        output_dir=tmp_path / "classic",
        upstream_results_path=upstream,
        videos_path=None,
        command=None,
        duration_seconds=None,
        returncode=0,
    )
    plus = normalize_results(
        {"temporal_flickering": 0.5},
        benchmark_id="vbench-plus-plus",
        variant="i2v",
        output_dir=tmp_path / "plus",
        upstream_results_path=upstream,
        videos_path=None,
        command=None,
        duration_seconds=None,
        returncode=0,
        stdout_path=None,
        stderr_path=None,
    )

    for scorecard in (classic, plus):
        assert scorecard["normalization_ok"] is True
        assert scorecard["official_results_imported"] is True
        assert scorecard["integration_evidence"] is False
        assert scorecard["official_benchmark_verified"] is False
        assert scorecard["leaderboard_valid"] is False
