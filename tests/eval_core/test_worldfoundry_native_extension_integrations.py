from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
UNIFIED_MODEL_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"

FORMER_NATIVE_TEMPLATE_MODEL_IDS = (
    "ai2thor",
    "emu3.5",
    "mmaudio",
    "omnivinci",
    "pi0-worldfoundry",
    "qwen2.5-omni",
    "simworld",
    "spatial-ladder",
    "spatial-reasoner",
    "spirit-v1.5",
    "thinksound",
    "wall-oss",
    "wonderworld",
)

REMOVED_NATIVE_PIPELINE_MODULES = (
    "worldfoundry.pipelines.emu.pipeline_emu3p5",
    "worldfoundry.pipelines.mmaudio.pipeline_mmaudio",
    "worldfoundry.pipelines.omnivinci.pipeline_omnivinci",
    "worldfoundry.pipelines.pi0.pipeline_pi0",
    "worldfoundry.pipelines.qwen.pipeline_qwen2p5_omni",
    "worldfoundry.pipelines.simworld.pipeline_simworld",
    "worldfoundry.pipelines.spatial_ladder.pipeline_spatial_ladder",
    "worldfoundry.pipelines.spatial_reasoner.pipeline_spatial_reasoner",
    "worldfoundry.pipelines.spirit_ai.pipeline_spirit_v1p5",
    "worldfoundry.pipelines.thinksound.pipeline_thinksound",
    "worldfoundry.pipelines.thor.pipeline_ai2thor",
    "worldfoundry.pipelines.wall_oss.pipeline_wall_oss",
    "worldfoundry.pipelines.wonder_journey.pipeline_wonder_world",
)

REMOVED_NATIVE_OPERATOR_MODULES = (
    "worldfoundry.operators.ai2thor_operator",
    "worldfoundry.operators.emu3p5_operator",
    "worldfoundry.operators.mmaudio_operator",
    "worldfoundry.operators.omnivinci_operator",
    "worldfoundry.operators.pi0_operator",
    "worldfoundry.operators.qwen2p5_omni_operator",
    "worldfoundry.operators.simworld_operator",
    "worldfoundry.operators.spatial_ladder_operator",
    "worldfoundry.operators.spatial_reasoner_operator",
    "worldfoundry.operators.spirit_v1p5_operator",
    "worldfoundry.operators.thinksound_operator",
    "worldfoundry.operators.wall_oss_operator",
    "worldfoundry.operators.wonder_world_operator",
)


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_native_extension_entries_use_real_pipeline_routes_not_template_modules() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id in FORMER_NATIVE_TEMPLATE_MODEL_IDS:
        entry = registry.get(model_id)

        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.integration_status == "integrated"
        assert entry.runner_target == UNIFIED_MODEL_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
        assert entry.runner_entry_kind == "runnable_runner"
        assert entry.is_runnable_runner_entry is True


def test_native_extension_template_modules_are_not_importable() -> None:
    for module_name in (*REMOVED_NATIVE_PIPELINE_MODULES, *REMOVED_NATIVE_OPERATOR_MODULES):
        assert _module_exists(module_name) is False
