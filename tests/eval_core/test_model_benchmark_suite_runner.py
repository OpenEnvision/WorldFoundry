from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.cli import main
from worldfoundry.evaluation.runner import (
    ModelBenchmarkSuiteRequest,
    list_model_benchmark_suite_presets,
    run_model_benchmark_suite,
)
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import CONTRACT_VALIDATION_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
MODEL_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
REAL_PLAN_MODEL_ID = "vchitect-2-t2v"


def test_model_benchmark_suite_runs_contract_cells_before_official_verification(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench", "libero"),
            mode="contract",
            contract_fixture=True,
        )
    )
    payload = json.loads(result.suite_manifest_path.read_text(encoding="utf-8"))
    cells = {(cell["model_id"], cell["benchmark_id"]): cell for cell in payload["cells"]}

    assert result.exit_code == 0
    assert result.status == "succeeded"
    assert payload["schema_version"] == "worldfoundry-model-benchmark-suite"
    assert payload["run_fingerprint"] == result.run_fingerprint
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["succeeded"] == 2
    assert payload["summary"]["skipped"] == 0
    assert cells[(CONTRACT_FIXTURE_MODEL_ID, "vbench")]["output_artifact"] == "generated_video"
    assert cells[(CONTRACT_FIXTURE_MODEL_ID, "vbench")]["cell_fingerprint"]
    assert cells[(CONTRACT_FIXTURE_MODEL_ID, "libero")]["output_artifact"] == "action_trace"
    assert cells[(CONTRACT_FIXTURE_MODEL_ID, "libero")]["status"] == "succeeded"
    assert cells[(CONTRACT_FIXTURE_MODEL_ID, "libero")]["compatibility"] == "unknown"
    assert Path(cells[(CONTRACT_FIXTURE_MODEL_ID, "vbench")]["run_manifest_path"]).is_file()
    assert Path(cells[(CONTRACT_FIXTURE_MODEL_ID, "libero")]["run_manifest_path"]).is_file()
    assert result.suite_report_path.is_file()
    comparison = json.loads(Path(result.artifacts["comparison_json"]).read_text(encoding="utf-8"))
    assert comparison["run_count"] == 0
    assert comparison["issues"] == [
        "suite spans multiple benchmarks; select one benchmark from index.json before comparing runs"
    ]


def test_model_benchmark_suite_skips_incompatible_cells(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=("splatt3r",),
            benchmark_ids=("vbench",),
            mode="contract",
        )
    )
    cell = result.cells[0]

    assert result.exit_code == 0
    assert result.status == "skipped"
    assert cell["status"] == "skipped"
    assert cell["compatibility"] == "incompatible"
    assert "required artifact" in cell["reason"] or "do not satisfy" in cell["reason"]


def test_model_benchmark_suite_plan_only_writes_matrix_without_running_cells(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite-plan",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(REAL_PLAN_MODEL_ID,),
            benchmark_ids=("vbench",),
            mode="contract",
            execute=False,
        )
    )

    assert result.exit_code == 0
    assert result.status == "planned"
    assert result.cells[0]["status"] == "planned"
    assert "run_manifest_path" not in result.cells[0]


def test_model_benchmark_suite_requires_model_without_contract_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require at least one model id"):
        run_model_benchmark_suite(
            ModelBenchmarkSuiteRequest(
                output_dir=tmp_path / "suite-missing-model",
                benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
                model_manifest_dir=None,
                benchmark_ids=("vbench",),
                mode="contract",
            )
        )


def test_model_benchmark_suite_contract_model_without_fixture_fails_cell(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite-implicit-contract",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            mode="contract",
        )
    )

    assert result.exit_code == 1
    assert result.status == "failed"
    assert result.cells[0]["status"] == "failed"
    assert "require generated inputs" in result.cells[0]["reason"]


def test_unified_run_cli_contract_suite_fixture(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--benchmark-manifest-dir",
            str(BENCHMARK_MANIFEST_DIR),
            "--model-manifest-dir",
            str(MODEL_MANIFEST_DIR),
            "--model",
            CONTRACT_FIXTURE_MODEL_ID,
            "--benchmark",
            "vbench",
            "--benchmark",
            "libero",
            "--output-dir",
            str(tmp_path / "cli-suite"),
            "--mode",
            "contract",
            "--contract-fixture",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-result"
    assert payload["kind"] == "model-benchmark-suite"
    assert payload["status"] == "succeeded"
    assert payload["delegate"]["schema_version"] == "worldfoundry-model-benchmark-suite-result"
    assert payload["delegate"]["run_fingerprint"]
    assert payload["delegate"]["summary"]["total"] == 2
    assert payload["delegate"]["summary"]["succeeded"] == 2
    assert payload["delegate"]["summary"]["skipped"] == 0


def test_model_benchmark_suite_fingerprint_is_output_location_independent(tmp_path: Path) -> None:
    left = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "left",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(REAL_PLAN_MODEL_ID,),
            benchmark_ids=("vbench",),
            execute=False,
        )
    )
    right = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "right",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(REAL_PLAN_MODEL_ID,),
            benchmark_ids=("vbench",),
            execute=False,
        )
    )

    assert left.run_fingerprint == right.run_fingerprint
    assert left.cells[0]["cell_fingerprint"] == right.cells[0]["cell_fingerprint"]


def test_model_benchmark_suite_resume_reuses_successful_cells(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    first = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            mode="contract",
            contract_fixture=True,
        )
    )
    second = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            mode="contract",
            resume=True,
            contract_fixture=True,
        )
    )

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert second.cells[0]["resumed"] is True
    assert second.cells[0]["run_manifest_path"] == first.cells[0]["run_manifest_path"]


def test_model_benchmark_suite_resume_requires_matching_cell_fingerprint(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    first = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            output_artifact="generated_video",
            contract_fixture=True,
        )
    )
    second = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            output_artifact="predicted_video",
            resume=True,
            contract_fixture=True,
        )
    )

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert second.cells[0]["resumed"] is False
    assert second.cells[0]["cell_fingerprint"] != first.cells[0]["cell_fingerprint"]


def test_model_benchmark_suite_preset_expands_named_suite(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "preset-suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_VALIDATION_ID,),
            benchmark_ids=("vbench",),
            execute=False,
        )
    )

    assert result.status == "planned"
    assert result.summary["total"] == 1
    assert result.cells[0]["model_id"] == CONTRACT_VALIDATION_ID
    assert result.cells[0]["benchmark_id"] == "vbench"


def test_formal_benchmark_inventory_comes_from_catalog() -> None:
    from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids

    benchmark_ids = set(formal_benchmark_ids(BENCHMARK_MANIFEST_DIR))
    assert {"vbench", "libero", "robotwin"} <= benchmark_ids


def test_model_benchmark_suite_presets_are_discoverable() -> None:
    assert list_model_benchmark_suite_presets() == ()


def test_suites_cli_lists_and_shows_presets(capsys) -> None:
    assert main(["suites", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == []

    assert main(["suites", "show", "missing-suite", "--json"]) != 0


def test_unified_run_cli_accepts_explicit_model_benchmark_ids(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--benchmark-manifest-dir",
            str(BENCHMARK_MANIFEST_DIR),
            "--model-manifest-dir",
            str(MODEL_MANIFEST_DIR),
            "--model-id",
            CONTRACT_VALIDATION_ID,
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(tmp_path / "cli-explicit-suite"),
            "--plan-only",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["kind"] == "model-benchmark-suite"
    assert payload["delegate"]["summary"]["total"] == 1
    assert payload["delegate"]["cells"][0]["benchmark_id"] == "vbench"


def test_model_benchmark_suite_plan_writes_rollup_artifacts(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "vchitect-suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=("vchitect-2-t2v",),
            benchmark_ids=("vbench-2.0",),
            execute=False,
        )
    )
    payload = json.loads(result.suite_manifest_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]

    assert result.status == "planned"
    assert result.summary["total"] == 1
    assert {cell["model_id"] for cell in result.cells} == {"vchitect-2-t2v"}
    assert {cell["benchmark_id"] for cell in result.cells} == {"vbench-2.0"}
    assert result.cells[0]["output_artifact"] == "generated_video"
    assert result.cells[0]["compatibility"] == "compatible"
    assert result.cells[0]["reason"] is None
    provenance = result.cells[0]["provenance"]
    assert provenance["claim"]["level"] == "diagnostic"
    assert provenance["claim"]["leaderboard_candidate"] is False
    assert "benchmark catalog does not establish full official-suite evidence" in provenance["reasons"]
    assert result.cells[0]["provenance"]["fidelity"]["data"] == "official"
    assert payload["request"]["model_ids"] == ["vchitect-2-t2v"]
    assert payload["request"]["benchmark_ids"] == ["vbench-2.0"]
    assert Path(artifacts["scorecards_json"]).is_file()
    assert Path(artifacts["scorecards_jsonl"]).is_file()
    assert Path(artifacts["index_json"]).is_file()
    assert Path(artifacts["index_jsonl"]).is_file()
    assert Path(artifacts["comparison_json"]).is_file()
    assert Path(artifacts["comparison_markdown"]).is_file()
    assert result.artifacts["comparison_run_count"] == 0


def test_unified_run_cli_returns_rollup_artifact_paths(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--benchmark-manifest-dir",
            str(BENCHMARK_MANIFEST_DIR),
            "--model-manifest-dir",
            str(MODEL_MANIFEST_DIR),
            "--model-id",
            "vchitect-2-t2v",
            "--benchmark-id",
            "vbench-2.0",
            "--output-dir",
            str(tmp_path / "cli-vchitect-suite"),
            "--plan-only",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["delegate"]["summary"]["total"] == 1
    assert Path(payload["artifacts"]["scorecards_json"]).is_file()
    assert Path(payload["artifacts"]["index_json"]).is_file()
    assert Path(payload["artifacts"]["comparison_json"]).is_file()
