from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldfoundry.cli import main
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark as model_benchmark_module
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import (
    ModelBenchmarkRunRequest,
    run_model_benchmark,
)
from worldfoundry.evaluation.tasks.execution.orchestration.interfaces import OfficialRunResult


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
FORMAL_BENCHMARK_COUNT = len(formal_benchmark_ids())
REAL_PLAN_MODEL_ID = "vchitect-2-t2v"


def _write_task_materialization_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_root = tmp_path / "tasks"
    data_root = tmp_path / "data"
    task_root.mkdir()
    data_root.mkdir()
    (task_root / "task.yaml").write_text(
        """
name: model-benchmark-task
benchmark_name: vbench
protocol: external_official_benchmark
evaluation_protocol: external_official_runner
input_keys: [prompt, prompt_id]
output_keys: [generated_video]
data:
  metadata_path: samples.jsonl
generation_defaults:
  height: 256
  width: 256
""",
        encoding="utf-8",
    )
    (data_root / "samples.jsonl").write_text(
        "\n".join(
            [
                '{"sample_id": "sample-a", "prompt_id": "a", "prompt": "camera pan across a city"}',
                '{"sample_id": "sample-b", "prompt_id": "b", "prompt": "slow dolly through a forest"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return task_root, data_root


def test_model_benchmark_runner_feeds_model_outputs_to_benchmark_runner(tmp_path: Path) -> None:
    result = run_model_benchmark(
        ModelBenchmarkRunRequest(
            output_dir=tmp_path / "model-benchmark",
            benchmark_id="vbench",
            benchmark_manifest_path=MANIFEST_PATH,
            benchmark_mode="contract",
            model_id=CONTRACT_FIXTURE_MODEL_ID,
            model_parameters={
                "artifact_kind": "generated_video",
                "artifact_uri_template": "memory://{sample_id}/{artifact_name}.mp4",
            },
            contract_fixture=True,
        )
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    standard_manifest = json.loads((result.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    benchmark_scorecard = json.loads((result.output_dir / "benchmark" / "scorecard.json").read_text(encoding="utf-8"))
    artifact_rows = [
        json.loads(line)
        for line in result.artifact_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.exit_code == 0
    assert result.status == "succeeded"
    assert manifest["schema_version"] == "worldfoundry-model-benchmark-run"
    assert manifest["materialized_artifact_count"] == 1
    assert manifest["artifacts"]["standard_run_manifest"].endswith("run_manifest.json")
    assert manifest["artifacts"]["environment"].endswith("environment.json")
    assert manifest["artifacts"]["env_requirements"].endswith("env_requirements.json")
    assert standard_manifest["schema_version"] == "worldfoundry-run-manifest"
    assert standard_manifest["runner"] == "model_benchmark_runner"
    assert standard_manifest["environment"]["schema_version"] == "worldfoundry-environment"
    assert standard_manifest["env_requirements"]["schema_version"] == "worldfoundry-env-requirements"
    assert manifest["artifacts"]["run_summary"].endswith("summary.json")
    assert manifest["artifacts"]["generated_artifact_dir"].endswith("generated_artifacts")
    assert run_summary["schema_version"] == "worldfoundry-run-summary"
    assert run_summary["run"]["started_at"].endswith("Z")
    assert run_summary["run"]["finished_at"].endswith("Z")
    assert run_summary["run"]["worldfoundry_version"]
    assert len(run_summary["run"]["run_fingerprint"]["hash"]) == 64
    assert standard_manifest["run_fingerprint"] == run_summary["run"]["run_fingerprint"]
    assert run_summary["artifacts"]["generated_artifact_manifest"].endswith("generated_artifacts.jsonl")
    assert run_summary["artifacts"]["benchmark_scorecard"].endswith("benchmark/scorecard.json")
    assert run_summary["leaderboard"]["materialized_artifact_count"] == 1.0
    assert run_summary["leaderboard"]["placeholder_artifact_count"] == 1.0
    assert run_summary["leaderboard"]["real_artifact_count"] == 0.0
    assert run_summary["leaderboard"]["benchmark_ok"] == 0.0
    assert run_summary["eligibility"]["score_valid"] is False
    assert run_summary["eligibility"]["leaderboard_valid"] is False
    assert run_summary["eligibility"]["leaderboard_eligible"] is False
    assert run_summary["eligibility"]["blocking_reasons"] == [
        "benchmark runner ran in contract-only mode",
        "model-benchmark run used contract fixture",
        "generated artifacts include placeholders",
    ]
    assert artifact_rows[0]["placeholder"] is True
    assert artifact_rows[0]["destination"].endswith(".mp4")
    placeholder_video = Path(artifact_rows[0]["destination"])
    placeholder_video_bytes = placeholder_video.read_bytes()
    assert b"ftyp" in placeholder_video_bytes[:32]
    assert b"moov" in placeholder_video_bytes
    assert benchmark_scorecard["benchmark"]["benchmark_id"] == "vbench"
    assert benchmark_scorecard["dataset"]["generated_file_count"] == 1


def test_model_benchmark_runner_materializes_task_yaml_requests(tmp_path: Path) -> None:
    task_root, data_root = _write_task_materialization_fixture(tmp_path)

    result = run_model_benchmark(
        ModelBenchmarkRunRequest(
            output_dir=tmp_path / "task-model-benchmark",
            benchmark_id="vbench",
            benchmark_manifest_path=MANIFEST_PATH,
            benchmark_mode="contract",
            model_id=CONTRACT_FIXTURE_MODEL_ID,
            model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
            model_parameters={"artifact_kind": "generated_video"},
            task_name="model-benchmark-task",
            task_roots=(task_root,),
            dataset_root=data_root,
            num_samples=2,
            contract_fixture=True,
        )
    )
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    task_plan = json.loads((result.output_dir / "task_run_plan.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert result.generation_result is not None
    assert result.generation_result.sample_count == 2
    assert manifest["task"]["task_name"] == "model-benchmark-task"
    assert manifest["artifacts"]["task_run_plan"].endswith("task_run_plan.json")
    assert task_plan["requests"][0]["inputs"]["prompt_id"] == "a"
    assert task_plan["requests"][0]["generation_kwargs"]["height"] == 256
    assert manifest["materialized_artifact_count"] == 2


def test_unified_run_cli_can_materialize_task_yaml(tmp_path: Path, capsys) -> None:
    task_root, data_root = _write_task_materialization_fixture(tmp_path)
    output_dir = tmp_path / "top-level-task-model-benchmark"

    exit_code = main(
        [
            "run",
            "--benchmark",
            "vbench",
            "--benchmark-manifest-dir",
            str(MANIFEST_PATH),
            "--model",
            CONTRACT_FIXTURE_MODEL_ID,
            "--model-runner",
            CONTRACT_FIXTURE_RUNNER_TARGET,
            "--model-parameter",
            "artifact_kind=generated_video",
            "--task-name",
            "model-benchmark-task",
            "--task-root",
            str(task_root),
            "--data-path",
            str(data_root),
            "--num-samples",
            "1",
            "--output-dir",
            str(output_dir),
            "--mode",
            "contract",
            "--contract-fixture",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["kind"] == "model-benchmark"
    assert payload["status"] == "succeeded"
    assert payload["delegate"]["generation_result"]["sample_count"] == 1
    assert Path(payload["delegate"]["run_manifest_path"]).is_file()
    assert (output_dir / "task_run_plan.json").is_file()


def test_unified_run_cli_contract_fixture(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--benchmark",
            "vbench",
            "--benchmark-manifest-dir",
            str(MANIFEST_PATH),
            "--model",
            CONTRACT_FIXTURE_MODEL_ID,
            "--model-parameter",
            "artifact_kind=generated_video",
            "--output-dir",
            str(tmp_path / "top-level-model-benchmark"),
            "--mode",
            "contract",
            "--contract-fixture",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-result"
    assert payload["kind"] == "model-benchmark"
    assert payload["status"] == "succeeded"
    assert payload["delegate"]["schema_version"] == "worldfoundry-model-benchmark-result"
    assert payload["delegate"]["benchmark_result"]["benchmark_id"] == "vbench"


def test_unified_run_cli_routes_model_and_benchmark(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            CONTRACT_FIXTURE_MODEL_ID,
            "--benchmark",
            "vbench",
            "--benchmark-manifest-dir",
            str(MANIFEST_PATH),
            "--model-parameter",
            "artifact_kind=generated_video",
            "--output-dir",
            str(tmp_path / "unified-run-model-benchmark"),
            "--mode",
            "contract",
            "--contract-fixture",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-result"
    assert payload["kind"] == "model-benchmark"
    assert Path(payload["run_manifest_path"]).is_file()
    assert Path(payload["scorecard_path"]).is_file()
    assert Path(payload["benchmark_scorecard_path"]).is_file()
    assert "generation_scorecard_path" not in payload
    assert Path(payload["artifact_manifest_path"]).is_file()
    assert payload["artifacts"]["benchmark_scorecard"] == payload["benchmark_scorecard_path"]
    assert payload["delegate"]["schema_version"] == "worldfoundry-model-benchmark-result"
    assert payload["delegate"]["status"] == "succeeded"
    assert payload["delegate"]["benchmark_result"]["benchmark_id"] == "vbench"


def test_model_benchmark_runner_rejects_implicit_contract_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require generated inputs"):
        run_model_benchmark(
            ModelBenchmarkRunRequest(
                output_dir=tmp_path / "implicit-contract",
                benchmark_id="vbench",
                benchmark_manifest_path=MANIFEST_PATH,
                benchmark_mode="contract",
                model_id=CONTRACT_FIXTURE_MODEL_ID,
                model_parameters={"artifact_kind": "generated_video"},
            )
        )


def test_model_benchmark_runner_requires_generation_source_without_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require generated inputs"):
        run_model_benchmark(
            ModelBenchmarkRunRequest(
                output_dir=tmp_path / "missing-generation-source",
                benchmark_id="vbench",
                benchmark_manifest_path=MANIFEST_PATH,
                benchmark_mode="contract",
                model_id="demo-model",
                model_runner="tests.eval_core.contract_fixture:ContractFixtureRunner",
                model_parameters={"artifact_kind": "generated_video"},
            )
        )


def test_model_benchmark_official_run_fails_preflight_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "preflight-benchmark.yaml").write_text(
        f"""
id: preflight-benchmark
python_path: {json.dumps(sys.executable)}
pythonpath_roots: []
required_env: []
required_imports: [json]
required_paths:
  - id: missing-runtime-asset
    path: missing-runtime-asset
requires_cuda_visibility: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_benchmark_module, "OFFICIAL_RUNTIME_PROFILE_DIR", profile_dir)
    generation_called = False

    def unexpected_generation(*args, **kwargs):
        nonlocal generation_called
        generation_called = True
        raise AssertionError("generation must not start when runtime preflight is not ready")

    monkeypatch.setattr(model_benchmark_module, "_run_generation", unexpected_generation)
    output_dir = tmp_path / "preflight-failure"

    with pytest.raises(RuntimeError, match="preflight failed before model generation"):
        run_model_benchmark(
            ModelBenchmarkRunRequest(
                output_dir=output_dir,
                benchmark_id="preflight-benchmark",
                benchmark_manifest_path=tmp_path / "unused-catalog",
                benchmark_mode="official-run",
                model_id="test-model",
            )
        )

    report_path = output_dir / "preflight_report.json"
    scorecard = json.loads((output_dir / "benchmark" / "scorecard.json").read_text(encoding="utf-8"))
    assert generation_called is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["ok"] is False
    assert scorecard["run"]["status"] == "runtime_preflight_failed"
    assert scorecard["artifacts"]["preflight_report"] == str(report_path.resolve())


def test_model_benchmark_attaches_successful_preflight_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "vbench.yaml").write_text(
        f"""
id: vbench
python_path: {json.dumps(sys.executable)}
pythonpath_roots: []
required_env: []
required_imports: [json]
required_paths: []
requires_cuda_visibility: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_benchmark_module, "OFFICIAL_RUNTIME_PROFILE_DIR", profile_dir)
    output_dir = tmp_path / "preflight-success"
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample.mp4").write_bytes(b"fixture")
    preflight_preceded_generation = False

    def generation_after_preflight(*args, **kwargs):
        nonlocal preflight_preceded_generation
        preflight_preceded_generation = (output_dir / "preflight_report.json").is_file()
        return None

    def successful_benchmark(benchmark_id, *, output_dir, **kwargs):
        benchmark_dir = Path(output_dir)
        benchmark_dir.mkdir(parents=True)
        scorecard_path = benchmark_dir / "scorecard.json"
        scorecard_path.write_text(
            json.dumps(
                {
                    "schema_version": "worldfoundry-scorecard",
                    "official_benchmark_verified": True,
                    "integration_evidence": True,
                    "artifacts": {"scorecard": str(scorecard_path)},
                }
            ),
            encoding="utf-8",
        )
        return OfficialRunResult(
            benchmark_id=benchmark_id,
            output_dir=benchmark_dir,
            scorecard_path=scorecard_path,
            official_benchmark_verified=True,
            integration_evidence=True,
            artifacts={"scorecard": str(scorecard_path)},
            metadata={"mode": "official-run"},
        )

    monkeypatch.setattr(model_benchmark_module, "_run_generation", generation_after_preflight)
    monkeypatch.setattr(model_benchmark_module, "run_benchmark_execution", successful_benchmark)

    result = run_model_benchmark(
        ModelBenchmarkRunRequest(
            output_dir=output_dir,
            benchmark_id="vbench",
            benchmark_manifest_path=MANIFEST_PATH,
            benchmark_mode="official-run",
            model_id="test-model",
            generated_artifact_dir=generated_dir,
        )
    )

    report_path = output_dir / "preflight_report.json"
    scorecard = json.loads((output_dir / "benchmark" / "scorecard.json").read_text(encoding="utf-8"))
    assert preflight_preceded_generation is True
    assert result.artifacts["preflight_report"] == str(report_path.resolve())
    assert result.benchmark_result["artifacts"]["preflight_report"] == str(report_path.resolve())
    assert scorecard["artifacts"]["preflight_report"] == str(report_path.resolve())


@pytest.mark.parametrize(
    ("mode", "install_profile", "skip_preflight"),
    [
        ("contract", True, False),
        ("normalizer", True, False),
        ("official-run", False, False),
        ("official-run", True, True),
    ],
)
def test_model_benchmark_preflight_does_not_change_unscoped_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    install_profile: bool,
    skip_preflight: bool,
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    if install_profile:
        (profile_dir / "preflight-benchmark.yaml").write_text(
            """
id: preflight-benchmark
python_path: missing-python-executable
pythonpath_roots: []
required_env: []
required_imports: []
required_paths: []
requires_cuda_visibility: false
""",
            encoding="utf-8",
        )
    monkeypatch.setattr(model_benchmark_module, "OFFICIAL_RUNTIME_PROFILE_DIR", profile_dir)
    if skip_preflight:
        monkeypatch.setenv("WORLDFOUNDRY_SKIP_PREFLIGHT", "1")

    class GenerationReached(Exception):
        pass

    def generation_reached(*args, **kwargs):
        raise GenerationReached

    monkeypatch.setattr(model_benchmark_module, "_run_generation", generation_reached)

    with pytest.raises(GenerationReached):
        run_model_benchmark(
            ModelBenchmarkRunRequest(
                output_dir=tmp_path / f"unscoped-{mode}-{install_profile}-{skip_preflight}",
                benchmark_id="preflight-benchmark",
                benchmark_manifest_path=tmp_path / "unused-catalog",
                benchmark_mode=mode,
                model_id="test-model",
            )
        )


def test_unified_run_cli_all_benchmarks_plans_formal_inventory(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--all-benchmarks",
            "--model",
            REAL_PLAN_MODEL_ID,
            "--output-dir",
            str(tmp_path / "all-benchmarks-plan"),
            "--plan-only",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-result"
    assert payload["kind"] == "model-benchmark-suite"
    assert payload["status"] == "planned"
    assert Path(payload["suite_manifest_path"]).is_file()
    assert Path(payload["suite_report_path"]).is_file()
    assert len(payload["delegate"]["cells"]) == FORMAL_BENCHMARK_COUNT
    assert {cell["benchmark_id"] for cell in payload["delegate"]["cells"]} >= {"vbench", "worldarena", "robotwin"}


def test_unified_run_cli_all_benchmarks_requires_model_without_contract_fixture(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--all-benchmarks",
            "--output-dir",
            str(tmp_path / "all-benchmarks-plan"),
            "--plan-only",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "require --model" in captured.err


def test_unified_run_cli_all_benchmarks_contract_fixture_executes_every_cell(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--all-benchmarks",
            "--mode",
            "contract",
            "--contract-fixture",
            "--output-dir",
            str(tmp_path / "all-benchmarks-contract-run"),
            "--generation-cache-dir",
            str(tmp_path / "generation-cache"),
            "--generation-cache-mode",
            "read-write",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-result"
    assert payload["kind"] == "model-benchmark-suite"
    assert payload["status"] == "succeeded"
    assert payload["delegate"]["summary"]["total"] == FORMAL_BENCHMARK_COUNT
    assert payload["delegate"]["summary"]["succeeded"] == FORMAL_BENCHMARK_COUNT
    assert payload["delegate"]["summary"]["failed"] == 0
    assert payload["delegate"]["summary"]["skipped"] == 0
    assert payload["artifacts"]["scorecard_count"] == FORMAL_BENCHMARK_COUNT
    assert payload["artifacts"]["indexed_run_count"] == FORMAL_BENCHMARK_COUNT
    assert payload["artifacts"]["comparison_run_count"] == 0
    assert Path(payload["suite_manifest_path"]).is_file()
    assert Path(payload["suite_report_path"]).is_file()
    assert Path(payload["artifacts"]["scorecards_jsonl"]).is_file()
    assert all(Path(cell["benchmark_scorecard_path"]).is_file() for cell in payload["delegate"]["cells"])
