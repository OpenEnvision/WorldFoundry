"""Real-weight verification for MiniMax H3 VAEs.

These tests load the actual ``MiniMaxAI/MiniMax-H3`` VAE checkpoints and run a
GPU encode->decode round-trip. They are skipped automatically when the weights
are not staged locally, so the suite stays green without a 498 GB download.

Point ``MINIMAX_H3_CKPT_DIR`` at a directory containing ``audio_vae/`` and/or
``vae/`` (top-level diffusers layout) and/or ``FL2VA/video_vae/source/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

_CKPT = Path(
    os.environ.get(
        "MINIMAX_H3_CKPT_DIR",
        str(Path.home() / ".cache" / "worldfoundry" / "checkpoints" / "minimax-h3-components"),
    )
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-weight VAE tests need CUDA")


def _audio_dir() -> Path | None:
    d = _CKPT / "audio_vae"
    return d if (d / "diffusion_pytorch_model.safetensors").is_file() else None


def _video_source() -> Path | None:
    d = _CKPT / "FL2VA" / "video_vae" / "source"
    return d if any(d.glob("*.safetensors")) else None


@pytest.mark.skipif(_audio_dir() is None, reason="audio_vae weights not staged")
def test_real_audio_vae_load_and_roundtrip() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio import (
        MiniMaxH3AudioVAE,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio.config import (
        MiniMaxH3AudioVAEArchConfig,
        MiniMaxH3AudioVAEConfig,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_loading import (
        load_minimax_h3_vae_weights,
    )

    d = _audio_dir()
    cfg_json = json.loads((d / "config.json").read_text())
    arch = MiniMaxH3AudioVAEArchConfig(
        latent_channels=32,
        latents_mean=cfg_json["latents_mean"],
        latents_std=cfg_json["latents_std"],
    )
    model = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=arch))
    missing, unexpected = load_minimax_h3_vae_weights(model, d, strict=True)
    assert not missing and not unexpected
    model = model.cuda().eval()
    with torch.no_grad():
        wav = torch.randn(1, 1, 16000, device="cuda")
        z = model.encode(wav)
        assert z.shape[1] == 32
        out = model.decode(z)
    assert out.ndim == 3
    assert torch.isfinite(out).all()


@pytest.mark.skipif(_video_source() is None, reason="FL2VA/video_vae/source weights not staged")
def test_real_video_vae_load() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video import (
        MiniMaxH3VideoVAE,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.config import (
        MiniMaxH3VideoVAEArchConfig,
        MiniMaxH3VideoVAEConfig,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_loading import (
        load_minimax_h3_vae_weights,
    )

    d = _video_source()
    # latent stats live in the sibling config.json (FL2VA/video_vae/config.json).
    cfg_path = d.parent / "config.json"
    cfg_json = json.loads(cfg_path.read_text())
    arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=24,
        latents_mean=cfg_json["latents_mean"],
        latents_std=cfg_json["latents_std"],
    )
    model = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig(arch_config=arch))
    missing, unexpected = load_minimax_h3_vae_weights(model, d, strict=False)
    # The fused FL2VA layout should match the port with at most trivial gaps.
    assert len(missing) == 0, f"missing keys: {missing[:10]}"
    assert len(unexpected) == 0, f"unexpected keys: {unexpected[:10]}"
