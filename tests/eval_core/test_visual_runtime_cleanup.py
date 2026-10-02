from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lyra1_checkpoint_helper_is_not_named_training() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime"
    sample_text = (runtime_root / "sample.py").read_text(encoding="utf-8")

    assert not (runtime_root / "src/models/utils/train.py").exists()
    assert (runtime_root / "src/models/utils/checkpoint.py").is_file()
    assert "src.models.utils.checkpoint" in sample_text
    assert "src.models.utils.train" not in sample_text


def test_warp_as_history_training_config_is_not_packaged() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime"

    assert not (runtime_root / "helios/utils/train_config.py").exists()


def test_wow_idm_runtime_is_inference_only() -> None:
    idm_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/wow/wow_runtime/idm"

    assert (idm_root / "infer.py").is_file()
    assert (idm_root / "models/idm_models.py").is_file()
    assert not (idm_root / "README.md").exists()
    assert not (idm_root / "train.py").exists()
    assert not (idm_root / "datasets").exists()


def test_wow_dit_runtime_does_not_package_webdataset_training_loaders() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/wow/wow_runtime"
    dit_root = runtime_root / "dit_models/wow-dit-2b"
    video2world_text = (dit_root / "video2world.py").read_text(encoding="utf-8")

    assert not (dit_root / "imaginaire/datasets").exists()
    assert "imaginaire.datasets" not in video2world_text
    assert "webdataset" not in video2world_text


def test_inspatio_world_runtime_does_not_package_training_datasets() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime"
    )
    runtime_text = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")

    assert not (runtime_root / "datasets").exists()
    assert "inspatio_world_runtime/datasets" not in runtime_text
    assert "from .datasets" not in runtime_text


def test_cameractrl_runtime_keeps_camera_geometry_without_training_datasets() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/cameractrl"
    upstream_root = runtime_root / "cameractrl_runtime"
    inference_text = (upstream_root / "inference.py").read_text(encoding="utf-8")
    runtime_text = (runtime_root / "runtime.py").read_text(encoding="utf-8")

    assert (upstream_root / "camera_geometry.py").is_file()
    assert not (upstream_root / "data").exists()
    assert ".data.dataset" not in inference_text
    assert ".data.dataset" not in runtime_text
    assert "RealEstate10K" not in inference_text
    assert "torch.utils.data" not in inference_text


def test_lyra2_runtime_keeps_inference_helpers_without_training_datasets() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2"
    upstream_manifest = runtime_root.parent / "UPSTREAM.md"
    source_root = runtime_root / "_src"
    checked_files = [
        source_root / "models/lyra2_model.py",
        source_root / "models/wan_t2v_model.py",
        source_root / "inference/lyra2_ar_inference.py",
        source_root / "modules/conditioner.py",
        source_root / "configs/config.py",
        source_root / "configs/experiment.py",
    ]

    assert not (source_root / "datasets").exists()
    assert not upstream_manifest.exists()
    assert not (source_root / "configs/defaults/dataloader.py").exists()
    assert (source_root / "utils/forward_warp_utils_pytorch.py").is_file()
    assert (source_root / "utils/plucker_embed_corrupter.py").is_file()
    assert (source_root / "utils/resolution.py").is_file()
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "lyra_2._src.datasets" not in text
        assert "lyra_register_dataloaders" not in text
        assert "get_gen3c_multiple_video_dataloader" not in text
        assert "get_itemdataset_option_local" not in text


def test_irasim_runtime_does_not_package_training_eval_or_dataset_tools() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/irasim/irasim_runtime"
    profile = (
        REPO_ROOT
        / "worldfoundry/data/models/runtime/profiles/irasim.yaml"
    )
    init_text = (runtime_root / "__init__.py").read_text(encoding="utf-8")
    profile_text = profile.read_text(encoding="utf-8")

    assert (runtime_root / "models/irasim.py").is_file()
    assert (runtime_root / "diffusion/mask_gaussian_diffusion.py").is_file()
    assert (runtime_root / "sample/pipeline_trajectory2videogen.py").is_file()
    assert not (runtime_root / "dataset").exists()
    assert not (runtime_root / "evaluate").exists()
    assert not (runtime_root / "application").exists()
    assert not (runtime_root / "main.py").exists()
    assert not (runtime_root / "sample/sample_autoregressive.py").exists()
    assert "setdefault(\"dataset\"" not in init_text
    assert "setdefault(\"evaluate\"" not in init_text
    assert "setdefault(\"application\"" not in init_text
    assert "from . import models" not in init_text
    assert "ensure_legacy_import_paths" in init_text
    assert "dataset, sample, evaluation, and application" not in profile_text
    assert "training dataloaders, evaluation scripts, and interactive application demos are pruned" in profile_text


def test_irasim_plan_wrapper_does_not_require_top_level_runtime_import(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.irasim.irasim_synthesis import IRASimSynthesis

    pipe = IRASimSynthesis.from_pretrained({"model_id": "irasim"}, device="cpu")
    result = pipe.predict(prompt="move the block", output_path=tmp_path / "irasim_plan.json")

    assert result["status"] == "prepared"
    assert result["model_id"] == "irasim"
    assert result["runtime"] == "worldfoundry.irasim.in_tree_runtime"
    assert (tmp_path / "irasim_plan.json").is_file()


def test_octo_runtime_keeps_inference_support_without_rlds_dataloader() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/octo/octo_runtime/octo"
    model_text = (runtime_root / "model/octo_model.py").read_text(encoding="utf-8")

    assert not (runtime_root / "data").exists()
    assert not (runtime_root / "inference_support/oxe").exists()
    assert (runtime_root / "inference_support/utils/data_utils.py").is_file()
    assert "octo.data.dataset" not in model_text
    assert "octo.inference_support.utils.data_utils" in model_text


def test_sana_runtime_keeps_inference_data_helpers_without_training_datasets() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/sana/sana_runtime"
    if not runtime_root.exists():
        assert not runtime_root.exists()
        return
    checked_files = [
        runtime_root / "scripts/inference.py",
        runtime_root / "scripts/inference_sana_sprint.py",
        runtime_root / "inference_video_scripts/inference_sana_video.py",
        runtime_root / "tools/controlnet/inference_controlnet.py",
        runtime_root / "diffusion/model/builder.py",
        runtime_root / "diffusion/data/__init__.py",
    ]

    assert not (runtime_root / "diffusion/data/datasets").exists()
    assert not (runtime_root / "diffusion/longsana/utils/dataset.py").exists()
    assert not (runtime_root / "diffusion/longsana/trainer").exists()
    assert not (runtime_root / "diffusion/data/runtime_utils.py").exists()
    assert not (runtime_root / "diffusion/data/zip_cache.py").exists()

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "diffusion.data.datasets" not in text
        assert "SanaZipDataset" not in text
        assert "diffusion.longsana.utils.dataset" not in text


def test_dreamdojo_runtime_uses_named_lerobot_sequence_reader_not_dataset_py() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/groot_dreams"
    loader_text = (runtime_root / "eval_inputs/loader.py").read_text(encoding="utf-8")
    config_text = (runtime_root / "groot_configs.py").read_text(encoding="utf-8")

    assert not (runtime_root / "data").exists()
    assert not (runtime_root / "dataloader.py").exists()
    assert (runtime_root / "eval_inputs/lerobot_sequences.py").is_file()
    assert "groot_dreams.eval_inputs.lerobot_sequences" in loader_text
    assert "groot_dreams.eval_inputs.lerobot_sequences" in config_text
    assert "from groot_dreams.data" not in loader_text
    assert "from groot_dreams.data" not in config_text


def test_worldcam_model_loader_config_lives_in_data_yaml() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/worldcam_runtime"
    data_config = REPO_ROOT / "worldfoundry/data/models/runtime/configs/worldcam/model_loader.yaml"
    manager_text = (runtime_root / "models/model_manager.py").read_text(encoding="utf-8")
    registry_text = (runtime_root / "model_registry.py").read_text(encoding="utf-8")
    config_text = data_config.read_text(encoding="utf-8")

    assert not (runtime_root / "configs").exists()
    assert "from ..configs.model_config" not in manager_text
    assert "from ..model_registry import" in manager_text
    assert "load_model_loader_registry" in registry_text
    assert "models/runtime/configs/worldcam/model_loader.yaml" in registry_text
    assert "model_loader_configs:" in config_text
    assert "keys_hash_with_shape: 9269f8db9040a9d860eaca435be61814" in config_text
    assert "model_classes: [WanModel]" in config_text


def test_infinite_world_bucket_config_lives_in_data_yaml() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/infinite_world/infinite_world_runtime"
    )
    data_config = REPO_ROOT / "worldfoundry/data/models/runtime/configs/infinite_world/buckets.yaml"
    inference_text = (runtime_root / "inference.py").read_text(encoding="utf-8")
    registry_text = (runtime_root / "bucket_registry.py").read_text(encoding="utf-8")
    config_text = data_config.read_text(encoding="utf-8")

    assert not (runtime_root / "configs").exists()
    assert "from .configs import bucket_config" not in inference_text
    assert "get_bucket_config(bucket_config_name)" in inference_text
    assert "worldfoundry.data" in registry_text
    assert "models/runtime/configs/infinite_world/buckets.yaml" in registry_text
    assert "yaml.safe_load" in registry_text
    assert "bucket_configs:" in config_text
    assert "ASPECT_RATIO_627_F64:" in config_text
    assert "ASPECT_RATIO_1440_F64:" in config_text


def test_kling_astra_model_loader_config_lives_in_data_yaml() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra_runtime"
    data_config = REPO_ROOT / "worldfoundry/data/models/runtime/configs/kling/astra/model_loader.yaml"
    manager_text = (runtime_root / "models/model_manager.py").read_text(encoding="utf-8")
    utils_text = (runtime_root / "astra_utils.py").read_text(encoding="utf-8")
    registry_text = (runtime_root / "model_registry.py").read_text(encoding="utf-8")
    config_text = data_config.read_text(encoding="utf-8")

    assert not (runtime_root / "configs").exists()
    assert "configs.model_config" not in manager_text
    assert "configs.model_config" not in utils_text
    assert "from ..model_registry import" in manager_text
    assert "from .model_registry import" in utils_text
    assert "load_model_loader_registry" in registry_text
    assert "models/runtime/configs/kling/astra/model_loader.yaml" in registry_text
    assert "model_loader_configs:" in config_text
    assert "keys_hash_with_shape: 9269f8db9040a9d860eaca435be61814" in config_text
