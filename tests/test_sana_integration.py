
import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")
import importlib
import pkgutil
from pathlib import Path

from PIL import Image

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
import worldfoundry.pipelines.sana as sana_pipelines
from worldfoundry.pipelines.sana.pipeline_sana import LongsanaVideo2b480pPipeline
from worldfoundry.pipelines.sana.pipeline_sana import Sana1600M1024pxBf16Pipeline
from worldfoundry.pipelines.sana.pipeline_sana import SanaControlnet600M1024pxPipeline
from worldfoundry.pipelines.sana.pipeline_sana import SanaVideo2b480pPipeline
from worldfoundry.studio.inference.catalog import discover_catalog
from worldfoundry.base_models.diffusion_model.image.sana.variants import (
    SANA_VARIANTS,
    config_root,
    get_sana_variant,
    runtime_root,
)


def test_sana_variant_configs_are_vendored():
    root = config_root()
    assert not (runtime_root() / "configs").exists()
    for variant in SANA_VARIANTS.values():
        assert (root / variant.config_path).is_file(), variant.model_id


def test_sana_runtime_logic_lives_under_base_models():
    repo_root = Path(__file__).resolve().parents[1]
    synthesis_root = repo_root / "worldfoundry/synthesis/visual_generation/sana"
    base_root = repo_root / "worldfoundry/base_models/diffusion_model/image/sana"
    synthesis_text = (synthesis_root / "sana_synthesis.py").read_text(encoding="utf-8")
    runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")

    assert (base_root / "variants.py").is_file()
    assert "class SanaRuntime" in runtime_text
    assert "subprocess.run" in runtime_text
    assert "def _image_command" in runtime_text
    assert "worldfoundry.base_models.diffusion_model.image.sana.worldfoundry_runtime" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "def _image_command" not in synthesis_text
    assert "def _subprocess_env" not in synthesis_text


def test_all_sana_pipeline_modules_import():
    for module in pkgutil.iter_modules(sana_pipelines.__path__):
        if module.name.startswith("pipeline_"):
            importlib.import_module(f"{sana_pipelines.__name__}.{module.name}")


def test_sana_pipeline_plan_only_uses_official_image_runner(tmp_path):
    pipe = Sana1600M1024pxBf16Pipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a precise test cube",
        output_path=tmp_path / "image.png",
        plan_only=True,
        return_dict=True,
    )

    command = " ".join(result["command"])
    assert result["model_id"] == "sana-1600m-1024px-bf16"
    assert result["backend_quality"] == "execution_plan"
    assert result["artifact_path"].endswith("image.json")
    assert str((runtime_root() / "scripts/inference.py").resolve()) in command
    assert str((config_root() / "sana_config/1024ms/Sana_1600M_img1024.yaml").resolve()) in command


def test_sana_controlnet_plan_only_materializes_control_input(tmp_path):
    pipe = SanaControlnet600M1024pxPipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a controlnet test image",
        images=Image.new("RGB", (8, 8), color="white"),
        output_path=tmp_path / "control.png",
        plan_only=True,
        return_dict=True,
    )

    command = result["command"]
    assert result["model_id"] == "sana-controlnet-600m-1024px"
    assert "inference_controlnet.py" in " ".join(command)
    assert "--json_file" in command


def test_sana_video_plan_only_uses_official_video_runner(tmp_path):
    pipe = SanaVideo2b480pPipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a short motion test",
        output_path=tmp_path / "video.mp4",
        plan_only=True,
        num_frames=8,
        return_dict=True,
    )

    command = " ".join(result["command"])
    assert result["model_id"] == "sana-video-2b-480p"
    assert result["artifact_kind"] == "generated_video"
    assert "inference_video_scripts/inference_sana_video.py" in command
    assert "--num_frames 8" in command


def test_longsana_plan_only_uses_official_longlive_runner(tmp_path):
    pipe = LongsanaVideo2b480pPipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a long-context motion test",
        output_path=tmp_path / "longsana.mp4",
        plan_only=True,
        return_dict=True,
    )

    command = " ".join(result["command"])
    assert result["model_id"] == "longsana-video-2b-480p"
    assert result["backend_quality"] == "execution_plan"
    assert "Sana_2000M_480px_adamW_fsdp_longsana.yaml" in command
    assert "SANA_Video_2B_480p_LongLive.pth" in command
    assert result["artifact_path"].endswith("longsana.json")


def test_longsana_training_stage_variants_are_not_runnable_models():
    registry = load_model_zoo_registry()

    for key in ("longsana-video-2b-480p-ode", "longsana-video-2b-480p-self-forcing"):
        try:
            registry.get(key)
        except KeyError:
            pass
        else:
            raise AssertionError(f"{key} should not be registered as a runnable model")


def test_sana_catalog_entries_include_variants_and_aliases():
    registry = load_model_zoo_registry()
    for key in [
        "sana",
        "sana-1600m-1024px-bf16",
        "sana-controlnet-600m-1024px",
        "sana-video-2b-720p",
        "longsana-video-2b-480p",
    ]:
        entry = registry.get(key)
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.sana.pipeline_sana:")

    assert "sana-video" in registry.get("sana").aliases
    assert get_sana_variant("sana-video-720p").model_id == "sana-video-2b-720p"


def test_sana_variants_are_discoverable_by_studio_catalog():
    ids = {entry.model_id for entry in discover_catalog()}
    assert {
        "sana",
        "sana-1600m-1024px-bf16",
        "sana-video-2b-480p",
        "longsana-video-2b-480p",
    } <= ids
