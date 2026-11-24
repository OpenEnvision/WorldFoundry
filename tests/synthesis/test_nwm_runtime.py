# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.synthesis.visual_generation.nwm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config_option", ["exp", "config", "config_path"])
@pytest.mark.parametrize("eval_type", ["rollout", "time"])
def test_nwm_module_command_preserves_evaluation_args(config_option: str, eval_type: str) -> None:
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_dir": "/out",
            "options": {
                config_option: "nwm.yaml",
                "checkpoint_step": "0200000",
                "dataset": "recon,scand",
                "eval_type": eval_type,
                "num_sec_eval": 2,
                "input_fps": 8,
                "num_workers": 0,
                "batch_size": 2,
                "rollout_fps_values": "1,2,4",
                "gt": 1,
            },
        }
    )
    assert command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.nwm.isolated_nwm_infer"]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--output_dir": "/out",
        "--exp": "nwm.yaml",
        "--ckp": "0200000",
        "--datasets": "recon,scand",
        "--eval_type": eval_type,
        "--num_sec_eval": "2",
        "--input_fps": "8",
        "--num_workers": "0",
        "--batch_size": "2",
        "--rollout_fps_values": "1,2,4",
        "--gt": "1",
    }


@pytest.mark.parametrize("model_id", ["nwm", "mineworld"])
@pytest.mark.parametrize("seed", [None, 0, 42])
def test_nwm_and_mineworld_optional_seed_reaches_subprocess_rngs(model_id: str, seed: int | None) -> None:
    adapter = importlib.import_module(f"worldfoundry.synthesis.visual_generation.{model_id}.worldfoundry_runtime")
    command = adapter.build_command(
        {
            "python": sys.executable,
            "output_dir": "/out",
            "options": {"config": "nwm.yaml", "checkpoint_path": "/weights", "input_dir": "/data", "seed": seed},
        }
    )
    script = "isolated_nwm_infer" if model_id == "nwm" else "inference"
    assert command[:3] == [sys.executable, "-m", f"worldfoundry.synthesis.visual_generation.{model_id}.{script}"]
    if seed is None:
        assert "--seed" not in command
        flags = []
    else:
        assert command[-2:] == ["--seed", str(seed)]
        flags = command[-2:]
    source = (
        Path(__file__).resolve().parents[2] / "worldfoundry/synthesis/visual_generation" / model_id / f"{script}.py"
    )
    tree = ast.parse(source.read_text())
    seed_argument = next(
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
    seed_guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.seed is not None"
    )
    code = f"""\
import random
from argparse import ArgumentParser
import numpy as np
import torch
parser = ArgumentParser()
{ast.unparse(seed_argument)}
args = parser.parse_args()
def draw() -> list[float]:
    return [random.random(), float(np.random.random()), torch.rand(()).item()]
expected_seed = 91 if args.seed is None else args.seed
random.seed(expected_seed)
np.random.seed(expected_seed)
torch.manual_seed(expected_seed)
expected = draw()
random.seed(91)
np.random.seed(91)
torch.manual_seed(91)
flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
{ast.unparse(seed_guard)}
assert draw() == expected
if args.seed is None:
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
"""
    completed = subprocess.run([sys.executable, "-c", code, *flags], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("suffix", [".mp4", ".png"])
def test_nwm_collects_nested_prediction_pngs_before_the_plan_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    from worldfoundry.synthesis.visual_generation import runtime_manifest

    synthesis = runtime_manifest.WorldModelRuntimeSynthesis.from_pretrained(model_id="nwm", device="cpu")
    monkeypatch.setattr(synthesis, "plan", lambda **kwargs: {"missing_assets": []})
    monkeypatch.setattr(synthesis, "_command", lambda **kwargs: ["unused"])
    produced = tmp_path / "native/dataset/id_0/1.png"

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        from PIL import Image

        produced.parent.mkdir(parents=True)
        Image.new("RGB", (3, 2), color=(19, 20, 21)).save(produced)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(runtime_manifest.subprocess, "run", run)
    result = synthesis.predict(output_path=tmp_path / ("prediction" + suffix))
    assert result["status"] == "succeeded"
    assert result["artifact_path"] == str(tmp_path / "prediction.png")
    assert Path(result["artifact_path"]).read_bytes() == produced.read_bytes()


def test_generic_runtime_artifact_collection_does_not_include_nwm_pngs(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.runtime_manifest import WorldModelRuntimeSynthesis

    produced = tmp_path / "native/id_0/1.png"
    produced.parent.mkdir(parents=True)
    produced.write_bytes(b"png")
    assert WorldModelRuntimeSynthesis._resolve_produced_artifact(tmp_path / "prediction.mp4") is None


def test_nwm_build_command_forwards_vae_path_and_resolves_local_sd_vae(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worldfoundry.synthesis.visual_generation.nwm.isolated_nwm_infer as infer_mod

    command = runtime.build_command(
        {
            "python": "/env/python",
            "output_dir": "/out",
            "options": {"exp": "nwm.yaml", "datasets": "recon", "eval_type": "time", "vae_path": "/local/sd-vae"},
        }
    )
    assert command[-2:] == ["--vae_path", "/local/sd-vae"]

    vae_dir = tmp_path / "stabilityai--sd-vae-ft-ema"
    vae_dir.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "empty_hfd"))
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "empty_ckpt"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty_hf"))

    assert infer_mod._resolve_vae_source(None) == str(vae_dir.resolve())
    assert infer_mod._resolve_vae_source(str(tmp_path / "explicit-vae")) == str((tmp_path / "explicit-vae").resolve())
