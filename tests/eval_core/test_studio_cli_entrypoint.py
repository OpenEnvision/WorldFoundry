from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_studio_cli_entrypoint_import_is_lightweight() -> None:
    from worldfoundry.studio import cli

    assert callable(cli.main)


def test_native_studio_entrypoint_parses_without_gradio_stack() -> None:
    from worldfoundry.studio.ui.launcher import parse_launch_config

    config = parse_launch_config(["lingbot-world", "--frontend", "world", "--variant", "fast", "--device", "cuda:0"])

    assert config.model_id == "lingbot-world"
    assert config.frontend == "world"
    assert config.variant_id == "fast"
    assert config.device == "cuda:0"


def test_studio_cli_help_uses_gradio_free_native_parser() -> None:
    code = """
import sys
from worldfoundry.studio import cli
try:
    cli.main(["--help"])
except SystemExit as exc:
    exit_code = int(exc.code or 0)
else:
    exit_code = 0
print("APP_IMPORTED=" + str("worldfoundry.studio.ui.gradio_app" in sys.modules))
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
    assert "APP_IMPORTED=False" in result.stdout
    assert "GRADIO_IMPORTED=False" in result.stdout


def test_studio_cli_native_frontend_routes_without_importing_gradio_app(monkeypatch) -> None:
    from worldfoundry.studio import cli
    from worldfoundry.studio.ui import launcher as native_app

    recorded: list[list[str]] = []
    monkeypatch.setattr(native_app, "main", lambda argv=None: recorded.append(list(argv or ())))

    cli.main(["lingbot-world", "--frontend", "world"])

    assert recorded == [["lingbot-world", "--frontend", "world"]]
