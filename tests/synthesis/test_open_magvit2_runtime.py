# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from worldfoundry.core.configuration.factory import instantiate_from_config
from worldfoundry.synthesis.visual_generation.open_magvit2 import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime import inference

pytestmark = pytest.mark.unit


def _small_config(variant: str) -> DictConfig:
    config = cast(
        DictConfig,
        OmegaConf.load(runtime.DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT / f"imagenet_conditional_llama_{variant}.yaml"),
    )
    args = config.model.init_args
    args.transformer_config.params.update(
        {
            "vocab_size": 4,
            "block_size": 16,
            "spatial_n_layer": 1,
            "factorized_n_layer": 1,
            "factorized_bits": [2, 2],
            "n_head": 4,
            "dim": 32,
            "cond_dim": 32,
        }
    )
    args.first_stage_config.params.update(
        {
            "n_embed": 16,
            "embed_dim": 4,
            "factorized_bits": [2, 2],
            "ddconfig": {
                "double_z": False,
                "z_channels": 4,
                "resolution": 4,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1],
                "num_res_blocks": 1,
            },
        }
    )
    return config


@pytest.mark.parametrize("variant", ["B", "L", "XL"])
@pytest.mark.parametrize("released_targets", [False, True])
def test_public_model_load_preserves_state_and_forward_without_import_aliases(
    variant: str, released_targets: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(variant)
    reference = instantiate_from_config(config.model).eval()
    image = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
    label = torch.tensor([207])
    with torch.no_grad():
        expected = reference(image, label)
        expected_decode = reference.first_stage_model(image)[0]
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": reference.state_dict()}, checkpoint)
    config_text = OmegaConf.to_yaml(config)
    if released_targets:
        config_text = config_text.replace(
            "worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.", ""
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)
    unrelated_src = ModuleType("src")
    unrelated_runtime = ModuleType("open_magvit2_runtime")
    monkeypatch.setitem(sys.modules, "src", unrelated_src)
    monkeypatch.setitem(sys.modules, "open_magvit2_runtime", unrelated_runtime)
    old_path = list(sys.path)
    wrapper = runtime.OpenMAGVIT2Runtime.from_pretrained(checkpoint, config_path=config_path, device="cpu")
    loaded, _ = wrapper._ensure_model()
    assert wrapper._ensure_model()[0] is loaded
    assert sys.path == old_path
    assert sys.modules["src"] is unrelated_src
    assert sys.modules["open_magvit2_runtime"] is unrelated_runtime
    assert config_path.read_text() == config_text
    assert type(loaded) is type(reference)
    assert loaded.training is False
    assert loaded.state_dict().keys() == reference.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        actual = loaded(image, label)
        actual_decode = loaded.first_stage_model(image)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=0, atol=0)
    result = wrapper.predict(output_path=tmp_path / "result.png", steps=16, top_k=(2, 2), cfg_scale=(1.0, 1.0))
    assert result["artifact_kind"] == "generated_image"
    assert np.asarray(Image.open(result["artifact_path"])).shape == (4, 4, 3)


def test_config_target_mapping_preserves_source_and_non_target_values() -> None:
    config = OmegaConf.create(
        {
            "model": {"class_path": "src.Open_MAGVIT2.models.lfqgan.VQModel"},
            "items": [{"target": "src.Open_MAGVIT2.modules.util.Labelator"}, {"target": "torch.nn.Identity"}],
            "checkpoint_path": "src.Open_MAGVIT2/model.ckpt",
            "other": {"target": "src.other.Model"},
        }
    )
    before = OmegaConf.to_container(config)
    resolved = inference._qualify_open_magvit2_config(config)
    prefix = "worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime."
    assert resolved.model.class_path == prefix + config.model.class_path
    assert resolved["items"][0].target == prefix + config["items"][0].target
    assert resolved["items"][1].target == "torch.nn.Identity"
    assert resolved.checkpoint_path == config.checkpoint_path
    assert resolved.other.target == config.other.target
    assert OmegaConf.to_container(config) == before
