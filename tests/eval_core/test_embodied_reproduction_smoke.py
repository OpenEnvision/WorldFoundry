from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "evaluation" / "embodied_fixtures"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "benchmark_zoo" / "verify_embodied_reproduction.sh"
PYTHON = sys.executable


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["WORLDFOUNDRY_PYTHON"] = PYTHON
    return env


def _load_fixture_manifest() -> dict:
    payload = yaml.safe_load((FIXTURES_ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize("benchmark_id", ["libero", "robomme", "robotwin"])
def test_embodied_contract_scorecard(tmp_path: Path, benchmark_id: str) -> None:
    output_dir = tmp_path / "contract" / benchmark_id
    completed = subprocess.run(
        [
            PYTHON,
            "-m",
            "worldfoundry.cli.main",
            "zoo",
            "benchmark-run",
            "--benchmark-id",
            benchmark_id,
            "--mode",
            "contract",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, completed.stderr
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert (output_dir / "benchmark_contract.json").is_file()
    assert scorecard["benchmark"]["benchmark_id"] == benchmark_id
    assert scorecard.get("official_benchmark_verified") is False


@pytest.mark.parametrize(
    ("benchmark_id", "fixture_rel"),
    [
        ("libero", "libero/libero_official_results.json"),
        ("robomme", "robomme/robomme_official_results.jsonl"),
        ("robotwin", "robotwin/eval_result"),
    ],
)
def test_embodied_official_validation_fixtures(tmp_path: Path, benchmark_id: str, fixture_rel: str) -> None:
    fixture_path = FIXTURES_ROOT / fixture_rel
    assert fixture_path.exists(), fixture_path
    output_dir = tmp_path / "validation" / benchmark_id
    completed = subprocess.run(
        [
            PYTHON,
            "-m",
            "worldfoundry.cli.main",
            "zoo",
            "benchmark-run",
            "--benchmark-id",
            benchmark_id,
            "--mode",
            "official-validation",
            "--official-results-path",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, completed.stderr
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    evaluation = scorecard.get("evaluation") if isinstance(scorecard.get("evaluation"), dict) else {}
    assert scorecard.get("normalization_ok") is True or evaluation.get("available") is True
    assert scorecard["metrics"]["leaderboard"]
    assert scorecard.get("official_benchmark_verified") is False


def test_embodied_fixture_manifest_matches_parametrize() -> None:
    manifest = _load_fixture_manifest()
    fixture_ids = {item["benchmark_id"] for item in manifest["fixtures"]}
    contract_ids = set(manifest["contract_benchmark_ids"])
    assert {"libero", "robomme", "robotwin"}.issubset(fixture_ids)
    assert {"libero", "robomme", "robotwin"}.issubset(contract_ids)


def test_verify_embodied_reproduction_smoke_script(tmp_path: Path) -> None:
    assert VERIFY_SCRIPT.is_file()
    completed = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "smoke", str(tmp_path / "verify")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
