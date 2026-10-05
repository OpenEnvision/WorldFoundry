from __future__ import annotations

import sys
import types
from pathlib import Path

from worldfoundry.base_models.three_dimensions.three_d_four_d import pytorch3d_compat


def test_configure_pytorch3d_extension_prepends_overlay(
    monkeypatch, tmp_path: Path
) -> None:
    overlay = tmp_path / "pytorch3d"
    overlay.mkdir()
    (overlay / "_C.test.so").touch()

    package = types.ModuleType("pytorch3d")
    package.__path__ = ["site-packages/pytorch3d"]
    monkeypatch.setitem(sys.modules, "pytorch3d", package)
    monkeypatch.setenv("WORLDFOUNDRY_PYTORCH3D_EXTENSION_DIR", str(overlay))

    configured = pytorch3d_compat.configure_pytorch3d_extension()

    assert configured == overlay
    assert package.__path__[0] == str(overlay)


def test_configure_pytorch3d_extension_is_noop_without_binary(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_PYTORCH3D_EXTENSION_DIR", str(tmp_path))
    assert pytorch3d_compat.configure_pytorch3d_extension() is None


def test_versecrafter_configures_overlay_before_importing_renderer() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry"
        / "synthesis"
        / "visual_generation"
        / "versecrafter"
        / "versecrafter_runtime"
        / "inference"
        / "rendering_4D_control_maps.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert source.index("configure_pytorch3d_extension()") < source.index(
        "from pytorch3d.renderer import"
    )
