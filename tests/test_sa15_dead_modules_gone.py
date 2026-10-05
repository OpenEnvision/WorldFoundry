"""SA-15: confirmed-dead modules are gone; attention package still imports."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DELETED_MODULE_PATHS = (
    "worldfoundry/core/attention/avatar_context_parallel.py",
    "worldfoundry/core/attention/extension_context_parallel.py",
    "worldfoundry/core/attention/reference_context_parallel.py",
    "worldfoundry/studio/visualization/plugins/scene3d/pixelsplat_full/"
    "encoder_visualization/encoder_visualizer_epipolar.py",
)


def test_sa15_deleted_module_paths_do_not_exist() -> None:
    missing = [rel for rel in DELETED_MODULE_PATHS if (REPO_ROOT / rel).exists()]
    assert missing == [], missing


def test_attention_package_init_still_imports() -> None:
    module = importlib.import_module("worldfoundry.core.attention")
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == (REPO_ROOT / "worldfoundry/core/attention/__init__.py").resolve()
    assert "attention_forward" in module.__all__
    assert "avatar_context_parallel" not in module.__all__
    assert "extension_context_parallel" not in module.__all__
    assert "reference_context_parallel" not in module.__all__
