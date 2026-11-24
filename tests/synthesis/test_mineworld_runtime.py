# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.visual_generation.mineworld import utils, worldfoundry_runtime
from worldfoundry.synthesis.visual_generation.mineworld.lvm import (
    LlamaForCausalLM,
    LlamaLVM,
)
from worldfoundry.synthesis.visual_generation.mineworld.vae import VAE

pytestmark = pytest.mark.unit
CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "mineworld")


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_packaged_config_resolves_owned_model_classes(config_path: Path) -> None:
    config = OmegaConf.load(config_path)

    assert get_obj_from_str(config.model.target) is LlamaLVM
    assert get_obj_from_str(config.model.params.model_class) is LlamaForCausalLM
    assert get_obj_from_str(config.model.params.tokenizer_config.target) is VAE


def test_model_import_keeps_the_package_search_path() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformers.modeling_utils import PreTrainedModel; before = sys.path.copy(); "
            "import worldfoundry.synthesis.visual_generation.mineworld.lvm; "
            "assert sys.path == before; assert 'lvm' not in sys.modules; assert 'utils' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("packaged", [False, True])
def test_checkpoint_config_targets_resolve_without_mutating_the_input(
    packaged: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "worldfoundry.synthesis.visual_generation.mineworld." if packaged else ""
    config = OmegaConf.create(
        {
            "target": f"{prefix}lvm.LlamaLVM",
            "params": {
                "model_class": f"{prefix}lvm.LlamaForCausalLM",
                "tokenizer_config": {"target": f"{prefix}vae.VAE", "params": {"ckpt_path": "vq.ckpt"}},
                "transformer_config": {"target": "transformers.LlamaConfig", "params": {"vocab_size": 8260}},
            },
        }
    )
    original = OmegaConf.to_container(config)
    captured = []
    model = torch.nn.Linear(2, 2)
    state = {"weight": torch.full((2, 2), 0.25), "bias": torch.tensor([-1.0, 1.0])}

    def instantiate(resolved: object) -> torch.nn.Module:
        captured.append(OmegaConf.to_container(resolved))
        return model

    monkeypatch.setattr(utils, "instantiate_from_config", instantiate)
    loaded = utils.load_model_from_config(config, state, gpu=False, eval_mode=True)["model"]

    expected = OmegaConf.to_container(config)
    assert isinstance(expected, dict)
    expected["target"] = "worldfoundry.synthesis.visual_generation.mineworld.lvm.LlamaLVM"
    expected["params"]["model_class"] = "worldfoundry.synthesis.visual_generation.mineworld.lvm.LlamaForCausalLM"
    expected["params"]["tokenizer_config"]["target"] = "worldfoundry.synthesis.visual_generation.mineworld.vae.VAE"
    assert captured == [expected]
    assert OmegaConf.to_container(config) == original
    assert loaded is model and not loaded.training
    assert loaded.weight.dtype == torch.float32
    torch.testing.assert_close(loaded.state_dict(), state, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [16, 32])
@pytest.mark.parametrize("algorithm", ["naive", "img_diagd", "video_diagd"])
def test_module_launch_keeps_decoding_and_asset_options(frames: int, algorithm: str) -> None:
    context = {
        "python": "/env/bin/python",
        "entrypoint": "/unused/inference.py",
        "output_dir": "/outputs",
        "options": {
            "input_dir": "/data",
            "checkpoint_path": "/weights/model.bin",
            "frames": frames,
            "accelerate_algo": algorithm,
            "demo_num": 3,
            "window_size": 4,
            "fps": 8,
            "val_data_num": 2,
            "top_p": 0.8,
            "top_k": 40,
        },
    }

    command = worldfoundry_runtime.build_command(context)

    assert command == [
        "/env/bin/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.mineworld.inference",
        "--data_root",
        "/data",
        "--model_ckpt",
        "/weights/model.bin",
        "--config",
        worldfoundry_runtime._default_config(frames),
        "--output_dir",
        "/outputs",
        "--demo_num",
        "3",
        "--frames",
        str(frames),
        "--window_size",
        "4",
        "--accelerate-algo",
        algorithm,
        "--fps",
        "8",
        "--val_data_num",
        "2",
        "--top_p",
        "0.8",
    ]
    context["options"].pop("top_p")
    assert worldfoundry_runtime.build_command(context)[-2:] == ["--top_k", "40"]
