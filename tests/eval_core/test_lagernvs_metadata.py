from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import worldfoundry.synthesis.visual_generation as visual_generation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_lagernvs_runnable_pipeline_was_removed() -> None:
    assert _module_exists("worldfoundry.pipelines.lagernvs.pipeline_lagernvs") is False


def test_lagernvs_metadata_only_facades_are_removed() -> None:
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation" / "lagernvs"
    runtime_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d" / "lagernvs"
    assert not (synthesis_root / ("lagernvs" + "_synthesis.py")).exists()
    assert not (runtime_root / "runtime" / "runtime.py").exists()

    module_name = ".".join(("worldfoundry", "synthesis", "visual_generation", "lagernvs", "lagernvs" + "_synthesis"))
    assert _module_exists(module_name) is False
    assert all("lagernvs" not in export.lower() for export in visual_generation.__all__)
