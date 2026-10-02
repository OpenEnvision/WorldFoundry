from pathlib import Path

import pytest

# worldfoundry.synthesis.visual_generation.worldfm requires the optional
# imageio dependency at import time; skip in environments without it.
pytest.importorskip("imageio")

from worldfoundry.representations.point_clouds_generation.worldfm.panogen import ensure_hy3dworld
from worldfoundry.representations.point_clouds_generation.worldfm.worldfm_representation import (
    WorldFMRepresentation,
)
from worldfoundry.synthesis.visual_generation.worldfm.worldfm_synthesis import WorldFMSynthesis


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHESIS_ROOT = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldfm"
VIDEO_RUNTIME_ROOT = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldfm/worldfm_runtime"
VIDEO_RUNTIME_ADAPTER = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldfm/runtime.py"
VIDEO_RUNTIME_PACKAGE = "worldfoundry.synthesis.visual_generation.worldfm.worldfm_runtime"
VIDEO_RUNTIME_ADAPTER_PACKAGE = "worldfoundry.synthesis.visual_generation.worldfm.runtime"


def test_worldfm_runtime_lives_under_video_base_models() -> None:
    assert not (SYNTHESIS_ROOT / "worldfm").exists()
    assert (VIDEO_RUNTIME_ROOT / "diffusion/model/nets/PixArtWorldFM.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "inference.py").is_file()


def test_worldfm_synthesis_missing_assets_fail_fast() -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        WorldFMSynthesis.from_pretrained(pretrained_model_path="/missing/worldfm-assets", device="cpu")

    message = str(exc_info.value)
    assert "WorldFM synthesis is configured for strict in-tree execution" in message
    assert str(REPO_ROOT / "cache/hfd/worldfm") in message


def test_worldfm_panogen_does_not_resolve_external_hunyuanworld_checkouts() -> None:
    source = (REPO_ROOT / "worldfoundry/representations/point_clouds_generation/worldfm/panogen.py").read_text(
        encoding="utf-8"
    )

    assert "github_repos" not in source
    assert "WORLDFOUNDRY_MODEL_SOURCE_DIR" not in source
    assert "WORLDFOUNDRY_GITHUB_REPOS" not in source
    assert "WORLDFM_HY3DWORLD_PATH" not in source

    with pytest.raises(RuntimeError, match="no longer accepts HunyuanWorld source checkout paths"):
        ensure_hy3dworld("/tmp/HunyuanWorld-1.0")

    with pytest.raises(RuntimeError, match="WorldFM no longer accepts `hw_path`"):
        WorldFMRepresentation(hw_path="/tmp/HunyuanWorld-1.0")


def test_worldfm_synthesis_wrapper_imports_video_runtime() -> None:
    assert not (SYNTHESIS_ROOT / "worldfm_infer.py").exists()

    source = (SYNTHESIS_ROOT / "worldfm_synthesis.py").read_text(encoding="utf-8")
    assert VIDEO_RUNTIME_PACKAGE in source or VIDEO_RUNTIME_ADAPTER_PACKAGE in source
    assert "three_dimensions.point_clouds.worldfm" not in source


def test_worldfm_runtime_logic_lives_in_base_models_adapter() -> None:
    runtime_source = VIDEO_RUNTIME_ADAPTER.read_text(encoding="utf-8")
    synthesis_source = (SYNTHESIS_ROOT / "worldfm_synthesis.py").read_text(encoding="utf-8")

    assert "class WorldFMRuntime" in runtime_source
    assert "worldfoundry.synthesis" not in runtime_source
    assert "WorldFMTriConditionInprocess" in runtime_source
    assert "infer_from_render_u8" in runtime_source
    assert "imageio" in runtime_source

    assert "load_worldfm_runtime" in synthesis_source
    assert "WorldFMPlan" not in synthesis_source
    assert "WorldFMTriConditionInprocess" not in synthesis_source
    assert "infer_from_render_u8" not in synthesis_source
    assert "imageio" not in synthesis_source
