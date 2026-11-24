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


def test_wow_idm_runtime_is_not_packaged() -> None:
    assert not (REPO_ROOT / "worldfoundry/synthesis/visual_generation/wow/wow_runtime/idm").exists()


def test_inspatio_world_runtime_does_not_package_training_datasets() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime"
    runtime_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py"
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


def test_infinite_world_bucket_config_lives_in_data_yaml() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/infinite_world_runtime"
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
