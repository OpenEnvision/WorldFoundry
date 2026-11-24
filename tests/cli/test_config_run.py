"""Environment expansion contracts for ``worldfoundry config run``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.cli.config import CONFIG_SCHEMA_VERSION
from worldfoundry.cli.main import main


def test_config_run_env_overrides_expand_as_literal_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    template = tmp_path / "workflow.yaml"
    template.write_text(
        f"schema_version: {CONFIG_SCHEMA_VERSION}\n"
        "command: [wf, run, --seed, '${SEED}', --cache, '$CACHE', --scale, '${SCALE:-1.0}']\n",
        encoding="utf-8",
    )

    argv = ["config", "run", str(template), "--env", "SEED=42", "--env", "CACHE=true", "--env", "SCALE=1.5e3"]
    assert main([*argv, "--plan-only", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"] == ["run", "--seed", "42", "--cache", "true", "--scale", "1.5e3"]
