
import pytest

# This test module imports worldfoundry code that requires the optional
# "hydra" dependency at import time; skip when it is unavailable.
pytest.importorskip("hydra")
from pathlib import Path

from hydra import compose
from omegaconf import OmegaConf

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES, BASE_MODEL_STACKS
from worldfoundry.base_models.perception_core.segment.sam2 import config_name
from worldfoundry.base_models.perception_core.segment.sam2.build_sam import (
    _canonicalize_in_tree_targets,
)


def test_sam2_official_config_composes_at_global_package():
    config = compose(config_name=config_name())

    assert config.model._target_ == "sam2.modeling.sam2_base.SAM2Base"


def test_sam2_targets_are_rewritten_recursively():
    config = OmegaConf.create(
        {
            "_target_": "sam2.modeling.sam2_base.SAM2Base",
            "children": [
                {"_target_": "sam2.modeling.backbones.hieradet.Hiera"},
                {"_target_": "unrelated.Package"},
            ],
        }
    )

    _canonicalize_in_tree_targets(config)

    prefix = "worldfoundry.base_models.perception_core.segment.sam2"
    assert config._target_ == f"{prefix}.modeling.sam2_base.SAM2Base"
    assert config.children[0]._target_ == f"{prefix}.modeling.backbones.hieradet.Hiera"
    assert config.children[1]._target_ == "unrelated.Package"


def test_perception_assets_resolve_from_configurable_checkpoint_root(tmp_path: Path):
    expected = {
        "mobile_sam": tmp_path / "MobileSAM" / "mobile_sam.pt",
        "repvit_sam": tmp_path / "RepViT" / "repvit_sam.pt",
        "midas_v21_small_256": tmp_path / "midas_v21_small_256.pt",
    }
    for path in expected.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")

    env = {"WORLDFOUNDRY_CKPT_DIR": str(tmp_path)}
    for capability_id, path in expected.items():
        asset = BASE_MODEL_CAPABILITIES[capability_id].assets[0]
        assert asset.local_dir(env) == path.parent
        assert asset.check(env)["local_path"] == str(path)


def test_perception_assets_are_in_aggregate_stacks():
    depth = BASE_MODEL_STACKS["depth_stack"].capability_ids
    segmentation = BASE_MODEL_STACKS["segmentation_stack"].capability_ids
    heavy = BASE_MODEL_STACKS["spatial_perception_heavy_stack"].capability_ids

    assert {"midas_v21_small_256", "unik3d_vitl"} <= set(depth)
    assert {"mobile_sam", "repvit_sam"} <= set(segmentation)
    assert {"midas_v21_small_256", "mobile_sam", "repvit_sam"} <= set(heavy)
