from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
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


def test_qwen35_subprocess_uses_environment_python_and_package_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/caller/module-path")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(
        command: list[str], *, env: dict[str, str], text: bool, capture_output: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert text and capture_output and not check
        calls.append((command, env))
        request = Path(command[command.index("--request") + 1])
        assert json.loads(request.read_text()) == {"prompt": "a red fox"}
        output = Path(command[command.index("--output") + 1])
        torch.save(torch.ones(1, 3, 5120, dtype=torch.bfloat16), output)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(magi2_qwen35.subprocess, "run", fake_run)
    encoder = Magi2Qwen35SubprocessTextEncoder(
        str(tmp_path / "text_encoder"),
        device="cpu",
        python_executable="/envs/magi2-preview/bin/python",
        skip_layer=2,
    )

    context = encoder.encode("a red fox")

    command, env = calls[0]
    assert command[:3] == [
        "/envs/magi2-preview/bin/python",
        "-m",
        "worldfoundry.base_models.diffusion_model.models.encoders.magi2_qwen35",
    ]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--precision") + 1] == "bfloat16"
    assert command[command.index("--max-length") + 1] == "7000"
    assert command[command.index("--skip-layer") + 1] == "2"
    assert env["PYTHONPATH"] == "/caller/module-path"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert context.shape == (1, 3, 5120)
    assert context.dtype == torch.bfloat16


def test_qwen35_subprocess_reports_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "text encoder load failed")

    monkeypatch.setattr(magi2_qwen35.subprocess, "run", fail)
    encoder = Magi2Qwen35SubprocessTextEncoder("/weights/text_encoder", device="cpu")
    with pytest.raises(RuntimeError, match="text encoder load failed"):
        encoder.encode("a red fox")


def test_magi2_loader_forwards_text_encoder_python(tmp_path: Path) -> None:
    pipe = NativeMagi2Pipeline.from_pretrained(
        str(tmp_path),
        device="cpu",
        required_components={"python_executable": "/envs/magi2-preview/bin/python"},
        preview_transformer=object(),
    )
    pipe._ensure_components(load_decoders=False)
    assert isinstance(pipe.text_encoder, Magi2Qwen35SubprocessTextEncoder)
    assert pipe.text_encoder.python_executable == "/envs/magi2-preview/bin/python"
    assert pipe.text_encoder.model_path == str(tmp_path / "text_encoder")
    assert pipe.text_encoder.skip_layer == 2
