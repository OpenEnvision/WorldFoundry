# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from worldfoundry.synthesis.visual_generation.diamond import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.diamond.agent import AgentConfig
from worldfoundry.synthesis.visual_generation.diamond.models.actor_critic import ActorCriticConfig
from worldfoundry.synthesis.visual_generation.diamond.models.diffusion import DenoiserConfig, DiffusionSamplerConfig
from worldfoundry.synthesis.visual_generation.diamond.models.rew_end_model import RewEndModelConfig
from worldfoundry.synthesis.visual_generation.diamond.worldfoundry_runtime import CONFIG_DIR

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_hydra() -> Iterator[None]:
    state = GlobalHydra.instance()
    previous = state.hydra
    state.clear()
    try:
        yield
    finally:
        state.hydra = previous
        GlobalHydra.set_instance(state)


@pytest.mark.parametrize("model_id", ["diamond", "diamond-atari"])
def test_public_model_path_selects_the_pretrained_bundle(tmp_path: Path, model_id: str) -> None:
    from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import DIAMONDPipeline

    checkpoint = tmp_path / "diamond"
    pipeline = DIAMONDPipeline.from_pretrained(
        model_path=str(checkpoint), model_id=model_id, pretrained=True, pretrained_game="Pong", device="cpu"
    )
    synthesis = pipeline.synthesis_model
    assert synthesis.options["pretrained_dir"] == str(checkpoint)
    assert "repo_root" not in synthesis.options
    assert synthesis.device == "cpu"
    assert checkpoint in synthesis._checkpoint_paths()
    command = synthesis._command(output_path=tmp_path / "video.mp4", prompt="", plan_path=tmp_path / "plan.json")
    assert "--pretrained" in command
    assert command[command.index("--pretrained-game") + 1] == "Pong"
    assert command[command.index("--pretrained-dir") + 1] == str(checkpoint)


def test_packaged_agent_config_constructs_owned_dataclasses() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="trainer")
    agent = instantiate(config.agent, num_actions=6)

    assert isinstance(agent, AgentConfig)
    assert isinstance(agent.denoiser, DenoiserConfig)
    assert isinstance(agent.rew_end_model, RewEndModelConfig)
    assert isinstance(agent.actor_critic, ActorCriticConfig)
    assert agent.denoiser.inner_model.num_actions == 6
    assert agent.rew_end_model.num_actions == 6
    assert agent.actor_critic.num_actions == 6
    assert agent.denoiser.inner_model.num_steps_conditioning == 4
    assert agent.actor_critic.img_size == agent.rew_end_model.img_size == 64


def test_packaged_diffusion_sampler_config_constructs_owned_dataclass() -> None:
    config = OmegaConf.load(CONFIG_DIR / "trainer.yaml")
    config.world_model_env.diffusion_sampler.s_tmax = float("inf")
    sampler = instantiate(config.world_model_env.diffusion_sampler)

    assert (
        config.world_model_env._target_ == "worldfoundry.synthesis.visual_generation.diamond.envs.WorldModelEnvConfig"
    )
    assert isinstance(sampler, DiffusionSamplerConfig)
    assert config.world_model_env.horizon == 15
    assert sampler.num_steps_denoising == 3
    assert sampler.order == 1


def test_released_config_targets_resolve_without_global_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="trainer")
    expected = OmegaConf.to_container(config, resolve=False)
    for node in (
        "agent",
        "agent.denoiser",
        "agent.denoiser.inner_model",
        "agent.rew_end_model",
        "agent.actor_critic",
        "world_model_env",
        "world_model_env.diffusion_sampler",
    ):
        key = f"{node}._target_"
        target = OmegaConf.select(config, key)
        OmegaConf.update(config, key, target.removeprefix("worldfoundry.synthesis.visual_generation.diamond."))
    foreign = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)
    search_path = sys.path.copy()

    runtime.use_package_config_targets(config)

    assert OmegaConf.to_container(config, resolve=False) == expected
    assert sys.path == search_path
    assert sys.modules["models"] is foreign
    assert isinstance(instantiate(config.agent, num_actions=6), AgentConfig)
    runtime.use_package_config_targets(config)
    assert OmegaConf.to_container(config, resolve=False) == expected


def test_unused_training_config_values_remain_without_dangling_targets() -> None:
    config = OmegaConf.load(CONFIG_DIR / "trainer.yaml")

    assert OmegaConf.to_container(config.denoiser.sigma_distribution) == {
        "loc": -0.4,
        "scale": 1.2,
        "sigma_min": 0.002,
        "sigma_max": 20,
    }
    assert OmegaConf.to_container(config.actor_critic.actor_critic_loss) == {
        "backup_every": 15,
        "gamma": 0.985,
        "lambda_": 0.95,
        "weight_value_loss": 1.0,
        "weight_entropy_loss": 0.001,
    }


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("seed", [None, 0, 42])
def test_atari_module_launch_keeps_runtime_options(pretrained: bool, seed: int | None) -> None:
    context = {
        "python": "/env/python",
        "entrypoint": "/unused/play.py",
        "output_dir": "/output",
        "output_path": "/output/video.mp4",
        "options": {
            "pretrained": pretrained,
            "pretrained_dir": "/assets",
            "pretrained_game": "Pong",
            "model_path": "/model.pt",
            "config_dir": "/config",
            "fps": 12,
            "size": 320,
            "num_steps_initial_collect": 20,
            "frames": 4,
            "record": True,
            "store_denoising_trajectory": True,
            "store_original_obs": True,
            "seed": seed,
        },
    }
    checkpoint_args = (
        ["--pretrained", "--pretrained-game", "Pong", "--pretrained-dir", "/assets"]
        if pretrained
        else [
            "--checkpoint",
            "/model.pt",
        ]
    )

    assert runtime.build_command(context) == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.diamond.play",
        "--config-dir",
        "/config",
        "--fps",
        "12",
        "--size",
        "320",
        "--num-steps-initial-collect",
        "20",
        "--no-header",
        *checkpoint_args,
        "--record",
        "--store-denoising-trajectory",
        "--store-original-obs",
        "--headless-steps",
        "4",
        "--output-path",
        "/output/video.mp4",
        "--dataset-dir",
        "/output/initialization_dataset",
        *([] if seed is None else ["--seed", str(seed)]),
    ]


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_atari_optional_seed_reaches_worker_rngs_before_loading(seed: int | None) -> None:
    tree = ast.parse((runtime.RUNTIME_DIR / "play.py").read_text())
    argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_argument"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "--seed"
    )
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    guard = next(
        node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args.seed is not None"
    )
    loading = next(node for node in main.body if "prepare_play_mode" in ast.unparse(node))
    assert guard.lineno < loading.lineno
    code = f"""\
import random
from argparse import ArgumentParser
import numpy as np
import torch
parser = ArgumentParser()
{ast.unparse(argument)}
args = parser.parse_args()
def draw():
    return [random.random(), float(np.random.random()), torch.randn(()).item()]
expected_seed = 91 if args.seed is None else args.seed
random.seed(expected_seed)
np.random.seed(expected_seed)
torch.manual_seed(expected_seed)
expected = draw()
random.seed(91)
np.random.seed(91)
torch.manual_seed(91)
flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
{ast.unparse(guard)}
assert draw() == expected
if args.seed is None:
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
"""
    flags = [] if seed is None else ["--seed", str(seed)]
    result = subprocess.run([sys.executable, "-c", code, *flags], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("sampler", ["fast", "higher_quality"])
def test_csgo_module_launch_keeps_runtime_options(sampler: str) -> None:
    context = {
        "python": "/env/python",
        "output_path": "/output/video.mp4",
        "options": {
            "variant": "diamond-csgo",
            "pretrained_dir": "/assets",
            "spawn_id": 2,
            "action": "backward",
            "sampler": sampler,
            "headless_steps": 7,
            "frames": 99,
            "fps": 10,
            "seed": 19,
            "device": "cuda:1",
            "checkpoint_path": "/model.pt",
            "spawn_dir": "/spawn",
            "actions_path": "/actions.npy",
        },
    }

    assert runtime.build_command(context) == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.diamond.csgo_inference",
        "--pretrained-dir",
        "/assets",
        "--spawn-id",
        "2",
        "--action",
        "backward",
        "--sampler",
        sampler,
        "--headless-steps",
        "7",
        "--fps",
        "10",
        "--seed",
        "19",
        "--device",
        "cuda:1",
        "--output-path",
        "/output/video.mp4",
        "--checkpoint",
        "/model.pt",
        "--spawn-dir",
        "/spawn",
        "--actions-path",
        "/actions.npy",
    ]


def test_csgo_requirements_remain_variant_specific(tmp_path: Path) -> None:
    for filename in (
        "csgo/config/agent/csgo.yaml",
        "csgo/model/csgo.pt",
        "csgo/spawn/0/low_res.npy",
        "csgo/spawn/0/full_res.npy",
        "csgo/spawn/0/act.npy",
    ):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    options = {"variant": "csgo", "pretrained_dir": str(tmp_path)}
    kwargs = {"options": options, "runtime_root": None, "entrypoint": None, "profile": None}

    assert runtime.missing_requirements(**kwargs) == []
    options["actions_path"] = str(tmp_path / "missing.npy")
    missing = runtime.missing_requirements(**kwargs)
    assert len(missing) == 1
    assert missing[0]["path"] == options["actions_path"]


def test_dataset_save_and_load_round_trip_with_numpy_state(tmp_path: Path) -> None:
    import numpy as np

    from worldfoundry.synthesis.visual_generation.diamond.eval_inputs.dataset import Dataset

    dataset = Dataset(tmp_path / "pong_1000")
    dataset.num_episodes = 2
    dataset.num_steps = 10
    dataset.start_idx = np.array([0, 5], dtype=np.int64)
    dataset.lengths = np.array([5, 5], dtype=np.int64)
    dataset.save_to_default_path()

    reloaded = Dataset(tmp_path / "pong_1000")
    reloaded.load_from_default_path()

    assert reloaded.num_episodes == 2
    assert reloaded.num_steps == 10
    assert reloaded.start_idx is not None
    assert reloaded.lengths is not None
    np.testing.assert_array_equal(reloaded.start_idx, np.array([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(reloaded.lengths, np.array([5, 5], dtype=np.int64))
