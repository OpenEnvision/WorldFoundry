from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import (
    SEED_ENV,
    initialize_reproducibility,
    reproducibility_env,
    seed_for_rank,
)
from worldfoundry.synthesis.visual_generation.inspatio_world.worldfoundry_runtime import InspatioWorldRuntime

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime"


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "7"])
def test_invalid_seed_is_rejected_before_launch(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        reproducibility_env({}, seed)


def test_reproducibility_environment_is_scoped():
    env = {"CUDA_VISIBLE_DEVICES": "3", "PYTHONHASHSEED": "17"}
    assert reproducibility_env(env) == env
    child = reproducibility_env(env, seed=7, deterministic=True)
    assert env == {"CUDA_VISIBLE_DEVICES": "3", "PYTHONHASHSEED": "17"}
    assert child[SEED_ENV] == "7" and child["PYTHONHASHSEED"] == "7"
    assert child["WORLDFOUNDRY_DETERMINISTIC"] == "1"
    assert child["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("seed, rank, expected", [(0, 0, 0), (42, 3, 45), (2**32 - 1, 0, 2**32 - 1), (2**32 - 1, 1, 0)])
def test_rank_seed_stays_in_numpy_range(seed, rank, expected):
    resolved = seed_for_rank(seed, rank)
    assert resolved == expected
    assert np.random.RandomState(resolved).rand(4).shape == (4,)


@pytest.mark.parametrize("rank", [-1, 1.5, True])
def test_invalid_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rank must be a non-negative integer"):
        seed_for_rank(0, rank)


def test_public_pipeline_forwards_seed_and_deterministic():
    from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world import InspatioWorldPipeline

    class Synthesis:
        def predict(self, **kwargs):
            return kwargs

    result = InspatioWorldPipeline(synthesis_model=Synthesis())(
        videos="input.mp4", seed=1730, deterministic=True, return_dict=True,
    )
    assert result["seed"] == 1730 and result["deterministic"] is True


@pytest.mark.parametrize("seed, deterministic, expected", [(None, True, 0), (1730, True, 1730), (42, False, 42)])
def test_public_predict_propagates_reproducibility(monkeypatch, tmp_path, seed, deterministic, expected):
    monkeypatch.delenv(SEED_ENV, raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_DETERMINISTIC", raising=False)
    video = tmp_path / "video.mp4"
    video.touch()
    trajectory = tmp_path / "trajectory.txt"
    trajectory.touch()
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    runtime = InspatioWorldRuntime()
    monkeypatch.setattr(runtime, "_resolve_snapshot_dir", lambda source: str(tmp_path))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    result = runtime.predict(video, traj_txt_path=str(trajectory), config_path=str(config), prompt="scene",
                             output_root=str(tmp_path / "output"), skip_step3=True,
                             seed=seed, deterministic=deterministic, return_dict=True)
    command, options = calls[0]
    assert options["env"][SEED_ENV] == str(expected)
    assert options["env"]["PYTHONHASHSEED"] == str(expected)
    assert result["seed"] == expected and result["deterministic"] is deterministic
    assert "--deterministic" in command if deterministic else "--deterministic" not in command
    if seed is not None:
        assert command[command.index("--seed") + 1] == str(seed)
    assert SEED_ENV not in os.environ


def test_leaf_rng_initialization_is_reproducible_across_processes():
    probe = '''
import json, random
import numpy as np
import torch
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import initialize_reproducibility
initialize_reproducibility()
print(json.dumps({"python": [random.random() for _ in range(4)],
                  "numpy": np.random.choice(100, 10, replace=False).tolist(),
                  "torch": torch.randn(12).tolist(),
                  "strict": torch.are_deterministic_algorithms_enabled()}))
'''
    env = reproducibility_env(os.environ, seed=1730, deterministic=True)
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("WORLDFOUNDRY_REPLAY_BOOTSTRAP", None)
    values = [json.loads(subprocess.check_output([sys.executable, "-c", probe], env=env, text=True))
              for _ in range(2)]
    assert values[0] == values[1] and values[0]["strict"] is True
    env = reproducibility_env(env, seed=42)
    different = json.loads(subprocess.check_output([sys.executable, "-c", probe], env=env, text=True))
    assert different["numpy"] != values[0]["numpy"]
    assert different["torch"] != values[0]["torch"]


def test_unrequested_leaf_initialization_preserves_rng(monkeypatch):
    monkeypatch.delenv(SEED_ENV, raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_DETERMINISTIC", raising=False)
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state()
    initialize_reproducibility()
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())


@pytest.mark.parametrize("leaf", ["caption", "depth", "render"])
def test_leaf_entrypoints_initialize_before_work(monkeypatch, tmp_path, leaf):
    monkeypatch.setattr(sys, "path", list(sys.path))
    module_name = f"inspatio_leaf_{leaf}"
    calls = []
    if leaf == "caption":
        source = RUNTIME / "scripts/gen_json.py"
        transformers = types.ModuleType("transformers")
        for name in ("AutoProcessor", "AutoConfig", "AutoModelForCausalLM"):
            setattr(transformers, name, None)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        argv = [str(source), "--root_dir", str(tmp_path)]
    elif leaf == "depth":
        source = RUNTIME / "depth/depth_predict_da3_cli.py"
        depth = types.ModuleType("depth.depth_predict_da3")

        class FakeDepth:
            def __init__(self, config):
                calls.append("work")

            def run(self, files, output):
                return True

        depth.DepthPredictDA3 = FakeDepth
        monkeypatch.setitem(sys.modules, "depth.depth_predict_da3", depth)
        video = tmp_path / "input.mp4"
        video.touch()
        argv = [str(source), "--input", str(video), "--output", str(tmp_path / "output")]
    else:
        source = RUNTIME / "scripts/render_point_cloud.py"
        monkeypatch.setitem(sys.modules, "open3d", types.ModuleType("open3d"))
        trajectory_name = "worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.trajectory"
        trajectory = types.ModuleType(trajectory_name)
        trajectory.generate_traj_txt = None
        monkeypatch.setitem(sys.modules, trajectory_name, trajectory)
        argv = [str(source), "--da3_dir", str(tmp_path), "--traj_txt_path", "camera.txt", "--output_dir", str(tmp_path)]
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "initialize_reproducibility", lambda: calls.append("initialize"))
    if leaf == "caption":
        monkeypatch.setattr(module, "process_videos", lambda *args: calls.append("work") or [])
    elif leaf == "render":
        monkeypatch.setattr(module, "render_point_cloud", lambda **kwargs: calls.append("work"))
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert calls == ["initialize", "work"]
