from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.magi2 import (
    create_model_from_config,
)
from worldfoundry.base_models.diffusion_model.models.encoders import magi2_qwen35
from worldfoundry.base_models.diffusion_model.models.encoders.magi2_qwen35 import (
    Magi2Qwen35SubprocessTextEncoder,
)
from worldfoundry.pipelines.magi2.pipeline_magi2 import NativeMagi2Pipeline


def test_magi2_plan_matches_turbo_vae_temporal_ratio() -> None:
    pipe = NativeMagi2Pipeline(device="cpu")

    plan = pipe._plan(short_edge=64, aspect_ratio="1:1", duration_seconds=10.0)

    assert plan["frames"] == 125
    assert plan["video_latent_t"] == 32
    assert plan["audio_latent_t"] == 250
    assert 1 + (plan["video_latent_t"] - 1) * 4 == plan["frames"]


def test_audio_vae_factory_accepts_published_stable_audio_config() -> None:
    autoencoder = {
        "encoder": {
            "type": "oobleck",
            "config": {
                "in_channels": 2,
                "channels": 2,
                "c_mults": [1],
                "strides": [2],
                "latent_dim": 4,
            },
        },
        "decoder": {
            "type": "oobleck",
            "config": {
                "out_channels": 2,
                "channels": 2,
                "c_mults": [1],
                "strides": [2],
                "latent_dim": 2,
            },
        },
        "bottleneck": {"type": "vae"},
        "latent_dim": 2,
        "downsampling_ratio": 2,
        "io_channels": 2,
    }
    published_config = {
        "model_type": "diffusion_cond",
        "sample_rate": 44100,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "config": autoencoder,
            }
        },
    }

    audio_vae = create_model_from_config(published_config)

    assert audio_vae.sample_rate == 44100
    assert audio_vae.latent_dim == 2
    assert audio_vae.downsampling_ratio == 2


def test_qwen35_subprocess_uses_same_python_and_local_overlay(
    monkeypatch, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay"
    (overlay / "transformers").mkdir(parents=True)
    (overlay / "huggingface_hub").mkdir()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, env, text, capture_output, check):
        calls.append((command, env))
        output = Path(command[command.index("--output") + 1])
        torch.save(torch.ones(1, 3, 5120, dtype=torch.bfloat16), output)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(magi2_qwen35.subprocess, "run", fake_run)
    encoder = Magi2Qwen35SubprocessTextEncoder(
        str(tmp_path / "text_encoder"),
        device="cpu",
        python_executable="/shared/worldfoundry/bin/python",
        transformers_overlay=overlay,
        skip_layer=2,
    )

    context = encoder.encode("a red fox")

    command, env = calls[0]
    assert command[0] == "/shared/worldfoundry/bin/python"
    assert env["PYTHONPATH"].split(":")[0] == str(overlay)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert context.shape == (1, 3, 5120)
    assert context.dtype == torch.bfloat16
