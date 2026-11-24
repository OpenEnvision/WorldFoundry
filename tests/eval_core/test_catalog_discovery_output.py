from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
