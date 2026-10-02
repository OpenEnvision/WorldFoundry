"""Vendored-integration leftovers: packaging ghosts (VI-20) and collision docs (VI-14).

Does not merge DUSt3R / YOLO-World trees, rewrite git history, or switch
setuptools to find_namespace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict:
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _package_data() -> dict[str, list[str]]:
    return _load_pyproject()["tool"]["setuptools"]["package-data"]


def _manifest_text() -> str:
    return (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")


LICENSE_GATED_EXCLUDES = (
    "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
    "worldfoundry.base_models.three_dimensions.general_3d.dust3r.*",
    "worldfoundry.base_models.three_dimensions.general_3d.monst3r",
    "worldfoundry.base_models.three_dimensions.general_3d.monst3r.*",
    "worldfoundry.base_models.three_dimensions.general_3d.mast3r",
    "worldfoundry.base_models.three_dimensions.general_3d.mast3r.*",
    "worldfoundry.base_models.three_dimensions.point_clouds.gaussian_splatting",
    "worldfoundry.base_models.three_dimensions.point_clouds.gaussian_splatting.*",
    "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft",
    "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.*",
)


def test_license_gated_find_excludes_remain() -> None:
    find = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    excludes = set(find["exclude"])
    missing = set(LICENSE_GATED_EXCLUDES) - excludes
    assert missing == set()


def test_setuptools_still_uses_find_not_find_namespace() -> None:
    packages = _load_pyproject()["tool"]["setuptools"]["packages"]
    assert "find" in packages
    assert "find_namespace" not in packages


def test_ghost_package_data_paths_are_gone() -> None:
    blob = str(_package_data())
    for ghost in (
        "diffsynth/tokenizer_configs",
        "vipe/LICENSE",
        "vipe/UPSTREAM.md",
        "rolling_forcing",
        "moverse/UPSTREAM.md",
        "bpe_simple_vocab_16e6.txt.gz",
        "openenvision_logo.png",
        "wbench/runtime/wbench/LICENSE",
    ):
        assert ghost not in blob, ghost


def test_live_package_data_assets_remain_and_exist() -> None:
    data = _package_data()
    base_models = data["worldfoundry.base_models"]
    assert _load_pyproject()["project"]["license-files"] == ["THIRD-PARTY-NOTICES"]
    assert "three_dimensions/general_3d/vipe/csrc/**/*" in base_models
    assert "perception_core/video_text/fetv_blip/**/*" in base_models
    assert "moverse_runtime/**/*" in data["worldfoundry.synthesis.visual_generation.moverse"]
    assert "tui_app.tcss" in data["worldfoundry.cli"]

    root = REPO_ROOT / "worldfoundry"
    assert (REPO_ROOT / "THIRD-PARTY-NOTICES").is_file()
    assert (root / "cli/tui_app.tcss").is_file()
    assert (root / "synthesis/visual_generation/moverse/moverse_runtime").is_dir()
    assert any((root / "base_models/three_dimensions/general_3d/vipe/csrc").rglob("*.cu"))


def test_ghost_manifest_paths_are_gone() -> None:
    manifest = _manifest_text()
    for ghost in (
        "rolling_forcing LICENSE",
        "stable_video_infinity LICENSE",
        "moverse/UPSTREAM.md",
        "vipe/LICENSE",
        "vipe/UPSTREAM.md",
        "diffsynth/tokenizer_configs",
        ".worldfoundry_upstream_commit",
        "bpe_simple_vocab_16e6.txt.gz",
    ):
        assert ghost not in manifest, ghost


def test_live_manifest_license_gated_prunes_remain() -> None:
    manifest = _manifest_text()
    for live in (
        "include THIRD-PARTY-NOTICES",
        "recursive-include worldfoundry/synthesis/visual_generation/moverse/moverse_runtime",
        "prune worldfoundry/base_models/three_dimensions/general_3d/dust3r",
        "prune worldfoundry/base_models/three_dimensions/general_3d/monst3r",
        "prune worldfoundry/base_models/three_dimensions/point_clouds/gaussian_splatting",
        "prune worldfoundry/synthesis/visual_generation/hunyuan_world",
    ):
        assert live in manifest, live


def test_vi14_unresolved_collisions_are_documented() -> None:
    from worldfoundry.base_models._vendor_imports import UNRESOLVED_TOP_LEVEL_COLLISIONS

    assert UNRESOLVED_TOP_LEVEL_COLLISIONS == (
        "dust3r",
        "croco",
        "opensora",
        "utils",
        "models",
    )
    source = (
        REPO_ROOT / "worldfoundry/base_models/_vendor_imports.py"
    ).read_text(encoding="utf-8")
    assert "assert_top_level_not_shadowed" in source
    assert "Do not wire these into assert_top_level_not_shadowed" in source
