from pathlib import Path

import pytest

from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec, import_pipeline_target
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile_manifest
from worldfoundry.evaluation.models.runtime.validate import validate_runtime_profile_references
from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline

MODEL_DATA = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models"
PIPELINE_TARGET = "worldfoundry.pipelines.worldplay2.pipeline_worldplay2:WorldPlay2Pipeline"
BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B"


def test_worldplay2_catalog_resolves_released_variants_to_native_pipeline():
    registry = load_model_zoo_registry()
    entry = registry.get("aejion/WorldPlay2-Fast")
    assert entry.model_id == "worldplay2"
    assert entry.integration_status == "integrated"
    assert entry.source.license == "cc-by-nc-4.0"
    assert entry.runner_parity.status == "pending"
    assert {variant.variant_id for variant in entry.variants} == {
        "worldplay2-fast", "worldplay2-ar", "worldplay2-bi",
    }
    for variant in entry.variants:
        assert variant.pipeline_target == PIPELINE_TARGET
        assert variant.integration_status == "integrated"
        assert variant.runner_parity.status == "pending"
    assert issubclass(import_pipeline_target(PIPELINE_TARGET), NativeVisualDiffusionPipeline)


@pytest.mark.parametrize(
    "model_id,recipe_id,repo_id,mode,steps,chunk_length",
    [
        ("worldplay2", "worldplay2", "aejion/WorldPlay2-Fast", "few_step", 4, 4),
        ("worldplay2-fast", "worldplay2", "aejion/WorldPlay2-Fast", "few_step", 4, 4),
        ("worldplay2-ar", "worldplay2-ar", "aejion/WorldPlay2-AR", "ar", 40, 4),
        ("worldplay2-bi", "worldplay2-bi", "aejion/WorldPlay2-BI", "bi", 40, 32),
    ],
)
def test_worldplay2_bindings_profiles_and_expert_recipes_match(
    model_id, recipe_id, repo_id, mode, steps, chunk_length,
):
    spec = build_pipeline_runner_spec(WorldModelConfig(model_id=model_id, runner="worldfoundry.pipeline"))
    assert spec.pipeline_target == PIPELINE_TARGET
    assert spec.runtime_profile_id == model_id
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime" / "profiles" / f"{model_id}.yaml")
    assert not validate_runtime_profile_references(profile)
    defaults = profile.execution["defaults"]
    assert (defaults["mode"], defaults["num_inference_steps"], defaults["chunk_length"]) == (
        mode, steps, chunk_length,
    )
    assert defaults["num_frames"] == 125
    assert (defaults["width"], defaults["height"], defaults["fps"]) == (832, 448, 16)
    recipe = default_native_diffusion_registry().resolve(model_id)
    assert recipe.model_id == recipe_id
    assert recipe.checkpoints["high"].repo_id == recipe.checkpoints["low"].repo_id == repo_id
    assert recipe.checkpoints["high"].files == ("high_noise_model/diffusion_pytorch_model.safetensors",)
    assert recipe.checkpoints["low"].files == ("low_noise_model/diffusion_pytorch_model.safetensors",)
    assert {recipe.checkpoints[role].repo_id for role in ("vae", "t5", "tokenizer")} == {BASE_REPO}
