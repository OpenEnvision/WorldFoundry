# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.core.io.paths import resolve_data_path

pytestmark = pytest.mark.unit
PACKAGE = "worldfoundry.synthesis.visual_generation.adaworld.worldmodel"


def _targets(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "target":
                yield item
            else:
                yield from _targets(item)
    elif isinstance(value, list):
        for item in value:
            yield from _targets(item)


@pytest.mark.parametrize("name", ["adaworld", "adaworld_adapt_discrete_action", "adaworld_adapt_continuous_action"])
def test_packaged_model_targets_resolve_to_owned_classes(name: str) -> None:
    path = resolve_data_path("models", "runtime", "configs", "adaworld", f"worldmodel/inference/{name}.yaml")
    config = OmegaConf.to_container(OmegaConf.load(path))
    targets = list(_targets(config))
    assert targets
    for target in targets:
        assert target.startswith("worldfoundry.synthesis.visual_generation.adaworld.")
        assert callable(get_obj_from_str(target))


def test_sampler_resolves_its_guider_and_discretization_through_the_package() -> None:
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel import sample_utils
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules.discretizer import (
        EDMShiftDiscretization,
    )
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules.guiders import (
        VanillaCFG,
    )

    sampler = sample_utils.init_sampling(steps=5, cfg_scale=1.1, n_context_frames=6)
    assert type(sampler.discretization) is EDMShiftDiscretization
    assert type(sampler.guider) is VanillaCFG
    assert sampler.num_steps == 5
    assert sampler.num_frames == 7


def test_staged_config_targets_resolve_to_the_same_classes_without_mutating_input() -> None:
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.sample_utils import qualify_vwm_config

    path = resolve_data_path("models", "runtime", "configs", "adaworld", "worldmodel/inference/adaworld.yaml")
    packaged = OmegaConf.load(path)
    staged = OmegaConf.create(OmegaConf.to_yaml(packaged).replace(f"{PACKAGE}.vwm.", "vwm."))
    assert isinstance(staged, DictConfig)
    resolved = qualify_vwm_config(staged)
    assert OmegaConf.to_container(resolved) == OmegaConf.to_container(packaged)
    assert staged.model.target == "vwm.models.diffusion.DiffusionEngine"
    assert resolved is not staged
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.models.diffusion import DiffusionEngine

    assert get_obj_from_str(resolved.model.target) is DiffusionEngine


def test_model_loading_resolves_staged_targets_at_the_load_site(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel import sample_utils

    captured = []

    class Model:
        def cuda(self) -> Model:
            captured.append("cuda")
            return self

        def eval(self) -> None:
            captured.append("eval")

    model = Model()

    def construct(config: Any) -> Model:
        captured.append(OmegaConf.to_container(config))
        return model

    monkeypatch.setattr(sample_utils, "instantiate_from_config", construct)
    config = OmegaConf.create(
        {"model": {"target": "vwm.models.diffusion.DiffusionEngine", "params": {"scale_factor": 0.18215}}}
    )
    assert sample_utils.load_model_from_config(config) is model
    assert captured == [
        {"target": f"{PACKAGE}.vwm.models.diffusion.DiffusionEngine", "params": {"scale_factor": 0.18215}},
        "cuda",
        "eval",
    ]
    assert config.model.target == "vwm.models.diffusion.DiffusionEngine"


def test_build_command_uses_the_installed_module_and_preserves_video_options() -> None:
    from worldfoundry.synthesis.visual_generation.adaworld import worldfoundry_runtime as runtime

    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": "sample.py",
            "output_path": "/out.mp4",
            "options": {
                "checkpoint": "/model.safetensors",
                "source_video": "/source.mp4",
                "target_video": "/target.mp4",
                "start_index": 0,
                "video_len": 4,
                "num_steps": 2,
            },
        }
    )
    assert command[:3] == ["/env/python", "-m", f"{PACKAGE}.sample"]
    assert command[command.index("--checkpoint") + 1] == "/model.safetensors"
    assert command[command.index("--source-video") + 1] == "/source.mp4"
    assert command[command.index("--target-video") + 1] == "/target.mp4"
    assert command[command.index("--start-index") + 1] == "0"
    assert command[command.index("--video-len") + 1] == "4"
    assert command[command.index("--num-steps") + 1] == "2"
