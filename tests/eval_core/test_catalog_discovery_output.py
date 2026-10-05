from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "worldfoundry.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_zoo_models_human_output_is_decision_table() -> None:
    result = _run_cli("zoo", "models", "--integration-status", "planned")

    assert result.returncode == 0
    header = result.stdout.splitlines()[0]
    assert "id" in header
    assert "status" in header
    assert "source" in header
    assert "runnable" in header
    assert "needs" in header
    assert "runner" in header
    assert "aliases" in header
    assert "manifest" in header
    assert "pixelsplat" in result.stdout


def test_zoo_benchmarks_human_output_is_decision_table() -> None:
    result = _run_cli("zoo", "benchmarks", "--integration-status", "planned", "--show-manifest")

    assert result.returncode == 0
    header = result.stdout.splitlines()[0]
    assert "id" in header
    assert "leaderboard" in header
    assert "surface" in header
    assert "needs" in header
    assert "runner" in header
    assert "manifest" in header
    assert "vbench" in result.stdout


def test_zoo_models_json_promotes_readiness_fields() -> None:
    result = _run_cli("zoo", "models", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
    zeroscope = next(row for row in payload if row["model_id"] == "zeroscope")
    assert zeroscope["verification_status"] == zeroscope["runner_parity"]["status"]
    assert zeroscope["runner_entry_kind"] == "runnable_runner"
    assert zeroscope["one_command_ready"] is True
    assert zeroscope["runner_ready"] is True
    assert "needs" in zeroscope
    assert "next_action" in zeroscope
    assert zeroscope["commands"]["run"] == (
        "worldfoundry-eval run --model zeroscope --benchmark <benchmark-id> "
        "--mode contract --output-dir tmp/model_benchmark/zeroscope/<benchmark-id> --json"
    )


def test_zoo_model_show_json_exposes_discovery_commands() -> None:
    result = _run_cli("zoo", "model-show", "--model-id", "pixelsplat", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    discovery = payload["discovery"]
    assert discovery["manifest_path"]
    assert discovery["next_action"]
    assert discovery["commands"]["checkpoint_check"] == (
        "worldfoundry-eval zoo model-download --model-id pixelsplat --check-local --json"
    )
    assert discovery["commands"]["plan"] == (
        "worldfoundry-eval run --model pixelsplat --benchmark <benchmark-id> "
        "--mode contract --plan-only --output-dir tmp/model_benchmark/pixelsplat/<benchmark-id> --json"
    )
    assert discovery["runner_target_declared"] is True
    assert discovery["runner_ready"] is False


def test_run_help_exposes_model_benchmark_contract_fixture_mode() -> None:
    result = _run_cli("run", "--help")

    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "contract" in result.stdout
    assert "--contract-fixture" in result.stdout


def test_contract_fixture_mode_executes_without_real_model(tmp_path: Path) -> None:
    result = _run_cli(
        "run",
        "--benchmark",
        "vbench",
        "--mode",
        "contract",
        "--contract-fixture",
        "--output-dir",
        str(tmp_path / "contract-fixture"),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["kind"] == "model-benchmark"
    assert payload["status"] == "succeeded"
    assert payload["delegate"]["benchmark_result"]["metadata"]["contract_only"] is True


def test_zoo_benchmarks_json_promotes_readiness_fields() -> None:
    result = _run_cli("zoo", "benchmarks", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    expected_ids = set(formal_benchmark_ids())

    assert {row["benchmark_id"] for row in payload} == expected_ids
    for row in payload:
        benchmark_id = row["benchmark_id"]
        contract_command = (
            f"worldfoundry-eval zoo benchmark-run --benchmark-id {benchmark_id} "
            f"--mode contract --output-dir tmp/benchmark_zoo/benchmark_contract/{benchmark_id} --json"
        )

        assert row["verification_status"] == row["runner"]["verification_status"]
        assert row["contract_validation_command"] == contract_command
        assert row["ready_now_command"] is None
        assert row["one_click_command"] is None
        assert row["commands"]["contract_run"] == contract_command
        assert row["contract_command_ready"] is True
        assert row["contract_ready"] is True
        if row["normalizer_command_ready"]:
            assert "normalizer_run" in row["commands"]
        else:
            assert "normalizer_run" not in row["commands"]
        assert "one_command_ready" in row
        assert isinstance(row["needs"], list)
        assert row["next_action"]


def test_zoo_show_json_includes_discovery_commands_and_machine_readable_spec() -> None:
    result = _run_cli("zoo", "benchmark-show", "--benchmark-id", "vbench", "--include-spec", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["benchmark_id"] == "vbench"
    assert payload["discovery"]["manifest_path"]
    assert payload["discovery"]["next_action"]
    assert payload["discovery"]["commands"]["contract_run"] == payload["contract_validation_command"]
    assert payload["discovery"]["contract_validation_command"] == payload["contract_validation_command"]
    assert payload["discovery"]["ready_now_command"] is None
    assert payload["one_click_command"] is None
    assert isinstance(payload["discovery"]["needs"], list)
    assert payload["benchmark_spec"]["name"] == "vbench"
    assert payload["benchmark_spec"]["metrics"]
    assert payload["benchmark_spec"]["tasks"]
