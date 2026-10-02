from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_model_env_installer_avoids_bash_namerefs() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")

    assert "local -n" not in source
    assert "WORLDFOUNDRY_PIP_INDEX_ARGS" in source
