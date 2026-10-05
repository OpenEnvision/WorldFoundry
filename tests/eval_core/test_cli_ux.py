from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from worldfoundry import cli
from worldfoundry.cli import models as cli_models
from worldfoundry.cli import zoo as cli_zoo
from worldfoundry.core.io.paths import hfd_root_path, resolve_worldfoundry_path
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "worldfoundry.cli", *args],
        cwd=REPO_ROOT,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_no_args_prints_first_run_banner(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    output = capsys.readouterr().out

    assert "WorldFoundry evaluation CLI" in output
    assert "worldfoundry-eval zoo benchmarks" in output
    assert "worldfoundry-eval zoo models" in output
    assert "worldfoundry-eval tasks list" in output
    legacy_command = " ".join(("worldfoundry-eval", "validation"))
    assert legacy_command not in output
    assert "worldfoundry-eval contract run" not in output
    assert "worldfoundry-eval <command> --help" in output


def test_help_prints_command_areas_without_heavy_runtime_imports() -> None:
    code = """
import sys
from worldfoundry.cli import main
try:
    main(["--help"])
except SystemExit as exc:
    exit_code = int(exc.code or 0)
else:
    exit_code = 0
print("HEAVY_IMPORTS=" + repr([name for name in ("torch", "diffusers", "tensorflow") if name in sys.modules]))
raise SystemExit(exit_code)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Command areas:" in result.stdout
    assert "Discovery:" in result.stdout
    assert "run-benchmark" not in result.stdout
    assert "run-suite" not in result.stdout
    assert "HEAVY_IMPORTS=[]" in result.stdout


def test_package_module_entrypoint_matches_public_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "worldfoundry", "--help"],
        cwd=REPO_ROOT,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WorldFoundry benchmark evaluation subsystem" in result.stdout
    assert "worldfoundry-eval" in result.stdout


def test_package_module_no_args_prints_first_run_banner() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "worldfoundry"],
        cwd=REPO_ROOT,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WorldFoundry evaluation CLI" in result.stdout
    legacy_command = " ".join(("worldfoundry-eval", "validation"))
    assert legacy_command not in result.stdout


def test_unknown_command_still_returns_argparse_error() -> None:
    result = _run_cli("not-a-real-command")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_run_help_exposes_all_benchmarks_shortcut() -> None:
    result = _run_cli("run", "--help")

    assert result.returncode == 0
    assert "--all-benchmarks" in result.stdout
    assert "benchmark inventory suite" in result.stdout


def test_zoo_embodied_assets_help_is_public() -> None:
    result = _run_cli("zoo", "embodied-assets", "--help")

    assert result.returncode == 0
    assert "Download or check official embodied action model assets" in result.stdout
    assert "--report-jsonl" in result.stdout
    assert "--plan-only" in result.stdout


def test_models_assets_help_exposes_base_model_stacks() -> None:
    result = _run_cli("models", "assets", "--help")

    assert result.returncode == 0
    assert "Plan or download reusable base-model assets" in result.stdout
    assert "--capability" in result.stdout
    assert "--execute-downloads" in result.stdout
    assert "depth_stack" in result.stdout
    assert "segmentation_stack" in result.stdout
    assert "detection_segmentation_stack" in result.stdout
    assert "spatial_perception_core_stack" in result.stdout
    assert "spatial_perception_heavy_stack" in result.stdout
    assert "spatial_perception_data_stack" in result.stdout


def test_models_assets_json_reports_stack_plan_without_runtime_imports() -> None:
    result = _run_cli("models", "assets", "--capability", "depth_stack", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "worldfoundry-base-model-assets-v1"
    assert payload["stack_ids"] == ["depth_stack"]
    assert payload["capability_ids"][:2] == ["depth_anything_v3", "moge_vitl"]
    assert "download_command_argvs" in payload


def test_models_assets_list_outputs_inventory() -> None:
    result = _run_cli("models", "assets", "--list", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "worldfoundry-base-model-inventory-v1"
    assert "segmentation_stack" in {item["id"] for item in payload["stacks"]}
    assert "grounding_dino" in {item["id"] for item in payload["capabilities"]}


def test_models_assets_handler_writes_report_and_executes_argvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.base_models import capabilities

    def fake_plan(capability: list[str] | None = None) -> dict[str, object]:
        return {
            "schema_version": "worldfoundry-base-model-assets-v1",
            "ok": False,
            "requested_ids": capability or [],
            "stack_ids": ["depth_stack"],
            "stacks": [],
            "capability_ids": ["depth_anything_v3"],
            "checks": [],
            "download_commands": ["hf download depth-anything/DA3-LARGE-1.1 --local-dir /tmp/depth"],
            "download_command_argvs": [["hf", "download", "depth-anything/DA3-LARGE-1.1", "--local-dir", "/tmp/depth"]],
            "export_commands": ["export WORLDFOUNDRY_DEPTH_ANYTHING_MODEL_DIR=/tmp/depth"],
            "pip_install_packages": [],
            "manual_actions": [],
        }

    monkeypatch.setattr(capabilities, "base_model_materialization_plan", fake_plan)
    monkeypatch.setattr(
        cli_models,
        "_execute_download_commands",
        lambda commands: [{"command": commands[0], "returncode": 0, "stdout": "", "stderr": ""}],
    )

    report_path = tmp_path / "base-model-assets.json"
    args = argparse.Namespace(
        capability=["depth_stack"],
        list=False,
        execute_downloads=True,
        report_path=report_path,
        json=False,
    )

    assert cli_models._handle_models_assets(args) == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["executed_downloads"][0]["command"] == ["hf", "download", "depth-anything/DA3-LARGE-1.1", "--local-dir", "/tmp/depth"]


def test_models_assets_downloads_use_hf_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(cli_models.subprocess, "run", fake_run)

    result = cli_models._execute_download_commands([["hf", "download", "org/repo"]])

    assert result[0]["returncode"] == 0
    assert calls[0]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[0]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_zoo_benchmark_run_help_exposes_official_results_path() -> None:
    result = _run_cli("zoo", "benchmark-run", "--help")

    assert result.returncode == 0
    assert "--official-results-path" in result.stdout
    assert "normalizer" in result.stdout


def test_zoo_benchmark_show_json_exposes_complete_runnable_commands() -> None:
    result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "video-bench", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    discovery = payload["discovery"]

    assert payload["declared_maturity"] == discovery["declared_surface"]
    assert payload["maturity"] == discovery["surface"]
    assert payload["verification_status"] == payload["runner"]["verification_status"]
    assert payload["commands"] == discovery["commands"]
    assert payload["needs"] == discovery["needs"]
    assert payload["next_action"] == discovery["next_action"]
    assert discovery["contract_command_ready"] is True
    assert discovery["normalizer_command_ready"] is True
    assert discovery["official_runner_ready"] is False
    assert discovery["ready_now_command_ready"] is False
    assert discovery["contract_ready"] is True
    assert discovery["normalizer_ready"] is True
    assert discovery["official_ready"] is False
    assert discovery["validation_or_normalizer_ready"] is True
    assert discovery["one_command_ready"] is False
    for key in [
        "contract_command_ready",
        "normalizer_command_ready",
        "official_runner_ready",
        "ready_now_command_ready",
        "contract_ready",
        "normalizer_ready",
        "official_ready",
        "validation_or_normalizer_ready",
        "one_command_ready",
    ]:
        assert payload[key] is discovery[key]
    assert discovery["commands"]["contract_run"] == (
        "worldfoundry-eval zoo benchmark-run --benchmark-id video-bench "
        "--mode contract --output-dir tmp/benchmark_zoo/benchmark_contract/video-bench --json"
    )
    assert discovery["contract_validation_command"] == discovery["commands"]["contract_run"]
    assert discovery["ready_now_command"] is None
    assert discovery["one_click_command"] is None
    assert discovery["commands"]["normalizer_run"] == (
        "worldfoundry-eval zoo benchmark-run --benchmark-id video-bench "
        "--mode official-validation --official-results-path '<official_results.json>' "
        "--generated-artifact-dir '<generated_videos>' --output-dir '<out>' --json"
    )


def test_zoo_benchmark_show_exposes_user_prepared_official_run_commands() -> None:
    videoscore_result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "videoscore", "--json")
    iworld_result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "iworld-bench", "--json")
    genai_result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "genai-bench", "--json")

    assert videoscore_result.returncode == 0
    assert iworld_result.returncode == 0
    assert genai_result.returncode == 0

    videoscore_commands = json.loads(videoscore_result.stdout)["commands"]
    iworld_commands = json.loads(iworld_result.stdout)["commands"]
    genai_commands = json.loads(genai_result.stdout)["commands"]

    assert "--frames-dir" in videoscore_commands["official_run"]
    assert "--dataset-root" in videoscore_commands["official_run"]
    assert "--bounded-sample-count" not in videoscore_commands["official_run"]
    assert "--bounded-sample-count" in videoscore_commands["official_validation"]
    assert "--official-results-path" in videoscore_commands["normalizer_run"]
    assert "--bounded-sample-count" not in videoscore_commands["normalizer_run"]

    assert '--metric "${WORLDFOUNDRY_IWORLD_BENCH_METRIC:-all}"' in iworld_commands["official_run"]
    assert '--metric "${WORLDFOUNDRY_IWORLD_BENCH_METRIC:-memory}"' in iworld_commands["official_validation"]

    assert "--official-results-path" in genai_commands["normalizer_run"]
    assert "--official-results-path" in genai_commands["official_validation"]


def test_zoo_benchmark_show_json_keeps_validation_separate_from_normalizer() -> None:
    result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "vbench", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    discovery = payload["discovery"]

    assert discovery["contract_command_ready"] is True
    assert discovery["normalizer_command_ready"] is False
    assert discovery["official_runtime_declared"] is True
    assert discovery["official_runtime_command_ready"] is True
    assert discovery["bounded_official_validation_ready"] is True
    assert discovery["official_evidence_ready"] is True
    assert discovery["leaderboard_ready"] is False
    assert discovery["full_leaderboard_ready"] is False
    assert payload["next_action"].startswith("run official validation with ")
    assert "official_validation" in discovery["commands"] or "normalizer_run" in discovery["commands"]
    assert "normalizer_run" not in discovery["commands"]


def test_zoo_benchmark_show_json_separates_verified_evidence_from_integration_status() -> None:
    result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "worldscore", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    discovery = payload["discovery"]

    assert payload["verification_status"] == "verified"
    assert payload["integration_status"] == "planned"
    assert discovery["official_runtime_declared"] is True
    assert discovery["bounded_official_validation_ready"] is True
    assert discovery["official_evidence_ready"] is True
    assert discovery["official_runner_ready"] is False
    assert discovery["one_command_ready"] is False
    assert discovery["leaderboard_ready"] is False
    assert discovery["full_leaderboard_ready"] is False


def test_zoo_benchmarks_json_exposes_complete_runnable_commands_for_every_inventory_entry() -> None:
    result = _run_cli("zoo", "benchmarks", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    expected_ids = set(formal_benchmark_ids())
    by_id = {entry["benchmark_id"]: entry for entry in payload}

    assert set(by_id) == expected_ids
    for benchmark_id, entry in by_id.items():
        commands = entry["commands"]
        expected_contract = (
            f"worldfoundry-eval zoo benchmark-run --benchmark-id {benchmark_id} "
            f"--mode contract --output-dir tmp/benchmark_zoo/benchmark_contract/{benchmark_id} --json"
        )
        assert entry["contract_command_ready"] is True, benchmark_id
        assert entry["contract_ready"] is True, benchmark_id
        assert entry["validation_or_normalizer_ready"] is True, benchmark_id
        assert entry["one_command_ready"] is entry["official_runner_ready"], benchmark_id
        assert entry["ready_now_command_ready"] is entry["official_runner_ready"], benchmark_id
        assert entry["contract_validation_command"] == expected_contract
        if entry["official_runner_ready"]:
            assert entry["ready_now_command"], benchmark_id
            assert entry["one_click_command"] == entry["ready_now_command"], benchmark_id
        else:
            assert entry["ready_now_command"] is None, benchmark_id
            assert entry["one_click_command"] is None, benchmark_id
        assert commands["contract_run"] == expected_contract
        assert "<" not in commands["contract_run"], benchmark_id

        if entry["normalizer_command_ready"]:
            normalizer_command = commands["normalizer_run"]
            assert normalizer_command
            assert "--json" in normalizer_command, benchmark_id


def test_zoo_embodied_assets_forwards_script_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[str] = []

    class FakeModule:
        @staticmethod
        def main(argv: list[str]) -> int:
            recorded.extend(argv)
            return 0

    monkeypatch.setattr(cli_zoo, "_load_repo_script", lambda relative_path: FakeModule)

    args = argparse.Namespace(
        models=["giga-brain-0", "gr00t"],
        hf_root=tmp_path / "hfd",
        asset_root=tmp_path / "assets",
        openpi_root=tmp_path / "openpi",
        repos_root=tmp_path / "repos",
        report_jsonl=tmp_path / "report.jsonl",
        summary_json=tmp_path / "summary.json",
        log_dir=tmp_path / "logs",
        max_workers=4,
        url_timeout_seconds=30,
        skip_existing=False,
        plan_only=True,
        list=False,
    )

    assert cli_zoo._handle_zoo_embodied_assets(args) == 0
    assert recorded == [
        "giga-brain-0",
        "gr00t",
        "--hf-root",
        str(tmp_path / "hfd"),
        "--asset-root",
        str(tmp_path / "assets"),
        "--openpi-root",
        str(tmp_path / "openpi"),
        "--repos-root",
        str(tmp_path / "repos"),
        "--report-jsonl",
        str(tmp_path / "report.jsonl"),
        "--summary-json",
        str(tmp_path / "summary.json"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--max-workers",
        "4",
        "--url-timeout-seconds",
        "30",
        "--no-skip-existing",
        "--plan-only",
    ]
