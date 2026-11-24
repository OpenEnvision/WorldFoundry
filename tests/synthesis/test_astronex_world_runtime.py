# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the vendored Astronex-World inference runtime."""

from __future__ import annotations

import inspect
import sys

import pytest
from omegaconf import OmegaConf
from PIL import Image

from worldfoundry.core.io import resolve_data_path
from worldfoundry.synthesis.visual_generation.astronex_world import runtime as astronex_runtime


def test_plan_launches_the_vendored_generator_as_a_package_module(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    weights = tmp_path / "weights"
    for relative in ("transformer/config.json", "vae", "text_encoder", "tokenizer", "model-00001-of-00001.safetensors"):
        (weights / relative).parent.mkdir(parents=True, exist_ok=True)
        (weights / relative).write_bytes(b"fixture")
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(image)
    runtime = astronex_runtime.AstronexWorldRuntime(checkpoint_dir=weights, python_executable=sys.executable)

    plan = runtime.build_plan(
        request={"prompt": "walk forward", "images": str(image), "latent_frames": 7, "trajectory": "w*7"},
        output_dir=tmp_path / "run",
    )

    assert runtime.preflight()["status"] == "ready"
    assert plan.command[1:3] == ("-m", astronex_runtime.ENTRYPOINT_MODULE)
    assert plan.workdir == str(tmp_path / "run")
    assert plan.env.get("PYTHONPATH", "") == ""
    assert plan.command[plan.command.index("--weights") + 1] == str(weights.resolve())


@pytest.mark.parametrize("config_name", ["causal", "bidirectional", "causal_consumer"])
def test_shipped_sampling_configs_build_the_denoiser_against_the_weights(tmp_path, config_name):
    from worldfoundry.synthesis.visual_generation.astronex_world.astronex_world_runtime.inference.generate import (
        resolve_config,
    )
    from worldfoundry.synthesis.visual_generation.astronex_world.astronex_world_runtime.models.denoiser import (
        AstronexDenoiser,
    )

    source = resolve_data_path(f"models/runtime/configs/astronex_world/{config_name}.yaml")
    resolved = OmegaConf.load(resolve_config(str(source), str(tmp_path / "resolved.yaml"), str(tmp_path / "weights")))

    assert resolved.model_kwargs.model_root == str(tmp_path / "weights")
    inspect.signature(AstronexDenoiser).bind(**dict(resolved.model_kwargs), is_causal=bool(resolved.causal))
