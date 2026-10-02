from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# torch is an optional heavy dependency; skip this module on
# torch-less environments (e.g. the minimal CI runner).
torch = pytest.importorskip("torch")

from worldfoundry.evaluation.models.catalog import UnknownModelZooKeyError, load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile, load_runtime_profiles
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
VISUAL_GENERATION_DIR = REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation"

NEW_WORLDARENA_MODEL_SPECS = {
    "ctrl-world": {
        "repo": "https://github.com/Robert-gyj/Ctrl-World",
        "runtime_fragment": "ctrl_world_runtime",
        "runtime_root": VISUAL_GENERATION_DIR / "ctrl_world" / "ctrl_world_runtime",
    },
    "giga-world-0": {
        "repo": "https://github.com/open-gigaai/giga-world-0",
        "runtime_fragment": "giga_world_0_runtime",
        "runtime_root": VISUAL_GENERATION_DIR / "giga_world_0" / "giga_world_0_runtime",
    },
    "genie-envisioner": {
        "repo": "https://github.com/AgibotTech/Genie-Envisioner",
        "runtime_fragment": "genie_envisioner_runtime",
        "runtime_root": VISUAL_GENERATION_DIR / "genie_envisioner" / "genie_envisioner_runtime",
    },
    "tesseract": {
        "repo": "https://github.com/UMass-Embodied-AGI/TesserAct",
        "runtime_fragment": "tesseract_runtime",
        "runtime_root": VISUAL_GENERATION_DIR / "tesseract" / "tesseract_runtime",
    },
    "wow": {
        "repo": "https://github.com/wow-world-model/wow-world-model",
        "runtime_fragment": "wow_runtime",
        "runtime_root": VISUAL_GENERATION_DIR / "wow" / "wow_runtime",
    },
}

EXISTING_WORLDARENA_MODEL_IDS = {
    "cogvideox",
    "wan2.2",
    "cosmos-predict-2.5",
}


def test_worldarena_models_are_in_tree_runtime_manifest_entries() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id, spec in NEW_WORLDARENA_MODEL_SPECS.items():
        entry = registry.get(model_id)

        assert entry.source_status == "open_source"
        assert entry.official_repo_url == spec["repo"]
        assert entry.integration_status == "integrated"
        assert entry.runner_entry_kind == "runnable_runner"
        assert entry.output_artifacts == ()
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")


def test_worldarena_official_lives_in_benchmark_zoo_registry() -> None:
    registry = load_benchmark_zoo_registry()
    entry = registry.get("worldarena")

    assert entry.benchmark_id == "worldarena"
    assert entry.runner_target == "worldfoundry.evaluation.tasks.contracts.external:WorldArenaContract"
    assert entry.runner_availability["surface"] == "official_runner"
    assert entry.data_refs["task_yaml"].endswith("worldarena.yaml")


def test_worldarena_runtime_profiles_track_sources_with_in_tree_runners() -> None:
    for model_id, spec in NEW_WORLDARENA_MODEL_SPECS.items():
        profile = load_runtime_profile(model_id, check_conda_env_exists=False)
        runtime_root = spec["runtime_root"]

        assert profile.backend_stage in {
            "in_tree_runtime_manifest",
            "official_in_tree_runtime",
            # wow now runs on the shared native WAN/Cosmos2 backends.
            "native_wan_and_native_cosmos2",
        }
        assert profile.runtime_status in {
            "in_tree_runtime_manifest_checkpoint_env_pending",
            "official_wan_14b_workspace_verified",
            "native_wan_structurally_validated_dit2b_checkpoint_pending",
        }
        assert profile.artifact_kind in {"generated_world", "generated_video"}
        assert profile.artifact_filename.endswith((".mp4", ".json"))
        # Profile integration statuses moved to the registered catalog
        # vocabulary; ``runtime_ported`` is the canonical "in-tree runtime
        # surface exists" state for these ported models.
        assert profile.integration_status in {
            "runtime_ported",
            "planned",
            "integrated",
            "verified_official_wan_14b_workspace",
        }
        assert profile.source_repos[0]["url"] == spec["repo"]
        assert profile.source_repos[0]["revision"]
        assert runtime_root.is_dir()
        assert runtime_root.name == spec["runtime_fragment"]


def test_worldarena_model_request_plan_wrapper_was_removed() -> None:
    assert not (REPO_ROOT / "worldfoundry/evaluation/tasks/official/worldarena.py").exists()


def test_worldarena_existing_models_are_not_revendored() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    profiles = load_runtime_profiles(check_conda_env_exists=False)

    for model_id in EXISTING_WORLDARENA_MODEL_IDS:
        entry = registry.get(model_id)
        assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
        assert entry.pipeline_target is not None
        assert entry.runtime_profile is not None
        assert entry.output_artifacts == ()
        assert entry.runtime_profile.removeprefix("runtime-profile:") in profiles

    with pytest.raises(UnknownModelZooKeyError):
        registry.get("diffsynth-studio")

    duplicate_runtime_dirs = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/cogvideox/cogvideo_official_runtime",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan2p2_official_runtime",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/cosmos/cosmos_predict2p5_official_runtime",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/diffsynth_studio",
    )
    assert all(not path.exists() for path in duplicate_runtime_dirs)


def test_ctrl_world_runtime_stays_inference_only() -> None:
    runtime_root = NEW_WORLDARENA_MODEL_SPECS["ctrl-world"]["runtime_root"]

    # Dataset fixtures are no longer shipped under worldfoundry/data/test_cases
    # (MANIFEST.in prunes that tree). Eval paths are resolved at runtime.
    assert not (REPO_ROOT / "worldfoundry/data/test_cases/ctrl_world").exists()
    assert not (runtime_root / "dataset").exists()
    assert not (runtime_root / "dataset_example").exists()
    assert not (runtime_root / "dataset_meta_info").exists()
    assert not (runtime_root / "scripts/train_wm.py").exists()
    assert not (runtime_root / "models/action_adapter/train2.py").exists()
    assert not (runtime_root / "readme.md").exists()
    assert not (runtime_root / "requirements.txt").exists()

    dynamics_module = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.ctrl_world.ctrl_world_runtime.models.action_adapter.dynamics"
    )
    dynamics_source = Path(dynamics_module.__file__).read_text(encoding="utf-8")
    assert "accelerate" not in dynamics_source
    assert "wandb" not in dynamics_source
    assert "pandas" not in dynamics_source
    assert "decord" not in dynamics_source

    dynamics = dynamics_module.Dynamics(action_dim=7, action_num=15, hidden_size=16)
    dynamics.device = "cpu"
    dynamics.to("cpu")
    pred = dynamics(
        np.zeros((1, 7), dtype=np.float32),
        np.zeros((15, 7), dtype=np.float32),
        None,
        training=False,
    )
    assert pred.shape == (15, 7)

    utils_module = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.ctrl_world.ctrl_world_runtime.models.utils"
    )
    latents = torch.zeros((2, 4, 8, 9, 5))
    split = utils_module.split_ctrl_world_latents(latents, rows=3, cols=1)
    assert split.shape == (6, 4, 8, 3, 5)

    for script_name in ("rollout_replay_traj.py",):
        script_text = (runtime_root / "scripts" / script_name).read_text(encoding="utf-8")
        assert "einops" not in script_text
        assert "wandb" not in script_text
        assert "swanlab" not in script_text
