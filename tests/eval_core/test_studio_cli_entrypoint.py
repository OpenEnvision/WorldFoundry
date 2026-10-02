from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_studio_cli_entrypoint_import_is_lightweight() -> None:
    from worldfoundry.studio import cli

    assert callable(cli.main)


def test_studio_entrypoint_parses_world_realtime_session() -> None:
    from worldfoundry.studio.ui.launcher import parse_launch_config

    config = parse_launch_config(["lingbot-world", "--frontend", "world", "--variant", "fast", "--device", "cuda:0"])

    assert config.model_id == "lingbot-world"
    assert config.frontend == "world"
    assert config.variant_id == "fast"
    assert config.device == "cuda:0"


def test_studio_cli_help_uses_standalone_parser() -> None:
    code = """
import sys
from worldfoundry.studio import cli
try:
    cli.main(["--help"])
except SystemExit as exc:
    exit_code = int(exc.code or 0)
else:
    exit_code = 0
print("GRADIO_IMPORTED=" + str("gradio" in sys.modules))
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
    assert "Launch a WorldFoundry Studio frontend" in result.stdout
    assert "GRADIO_IMPORTED=False" in result.stdout


def test_studio_cli_routes_to_standalone_launcher(monkeypatch) -> None:
    from worldfoundry.studio import cli
    from worldfoundry.studio.ui import launcher as native_app

    recorded: list[list[str]] = []
    monkeypatch.setattr(native_app, "main", lambda argv=None: recorded.append(list(argv or ())))

    cli.main(["lingbot-world", "--frontend", "world"])

    assert recorded == [["lingbot-world", "--frontend", "world"]]


@pytest.mark.parametrize("frontend_arg", [["--frontend", "unified"], ["--frontend=unified"]])
def test_studio_cli_rejects_removed_frontend(frontend_arg, capsys) -> None:
    from worldfoundry.studio import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["lingbot-world", *frontend_arg])

    assert exc_info.value.code == 2
    assert "invalid choice: 'unified'" in capsys.readouterr().err


def test_studio_cli_rejects_removed_frontend_from_environment(monkeypatch, capsys) -> None:
    from worldfoundry.studio import cli

    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_FRONTEND", "unified")
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["lingbot-world"])

    assert exc_info.value.code == 2
    assert "Unsupported frontend `unified`" in capsys.readouterr().err
