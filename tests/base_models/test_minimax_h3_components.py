"""Stage 2/3 tests: MiniMax H3 VAE + encoder configs, contract, and imports.

These validate the ported component surfaces (constructors, shared latent-stats
contract, exported symbols) without materializing the multi-billion-parameter
VAE/encoder weights — full encode/decode round-trips were verified during the
port and are exercised at the GPU end-to-end stage.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_common import (
    MiniMaxH3VAEContractError,
    validate_minimax_h3_vae_latent_stats,
)


class _StatsConfig:
    def __init__(self, latent_channels, latents_mean, latents_std):
        self.latent_channels = latent_channels
        self.latents_mean = latents_mean
        self.latents_std = latents_std


def test_contract_accepts_valid_video_and_audio_stats() -> None:
    validate_minimax_h3_vae_latent_stats(
        _StatsConfig(24, [0.0] * 24, [1.0] * 24), "video_vae", 24
    )
    validate_minimax_h3_vae_latent_stats(
        _StatsConfig(32, [0.0] * 32, [1.0] * 32), "audio_vae", 32
    )


def test_contract_rejects_wrong_channel_count() -> None:
    with pytest.raises(MiniMaxH3VAEContractError):
        validate_minimax_h3_vae_latent_stats(
            _StatsConfig(16, [0.0] * 16, [1.0] * 16), "video_vae", 24
        )


def test_contract_rejects_missing_and_nonpositive_std() -> None:
    with pytest.raises(MiniMaxH3VAEContractError):
        validate_minimax_h3_vae_latent_stats(_StatsConfig(24, None, [1.0] * 24), "video_vae", 24)
    with pytest.raises(MiniMaxH3VAEContractError):
        validate_minimax_h3_vae_latent_stats(
            _StatsConfig(24, [0.0] * 24, [0.0] * 24), "video_vae", 24
        )


def test_video_vae_config_defaults() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.config import (
        MiniMaxH3VideoVAEArchConfig,
        MiniMaxH3VideoVAEConfig,
    )

    arch = MiniMaxH3VideoVAEArchConfig()
    assert arch.latent_channels == 24
    assert arch.temporal_compression_ratio == 4
    assert arch.spatial_compression_ratio == 16
    cfg = MiniMaxH3VideoVAEConfig()
    # Single-GPU port forces parallel decode off.
    assert cfg.use_parallel_decode is False


def test_audio_vae_config_defaults() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio.config import (
        MiniMaxH3AudioVAEArchConfig,
    )

    arch = MiniMaxH3AudioVAEArchConfig()
    assert arch.latent_channels == 32


def test_video_and_audio_vae_symbols_exported() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders import (
        minimax_h3_audio,
        minimax_h3_video,
    )

    assert hasattr(minimax_h3_video, "MiniMaxH3VideoVAE")
    assert hasattr(minimax_h3_audio, "MiniMaxH3AudioVAE")


def test_qwen3vl_encoder_module_imports_without_weights() -> None:
    # The module must import without constructing the HF backbone (which pulls
    # heavy deps). Construction is deferred to __init__/from_pretrained.
    import worldfoundry.base_models.diffusion_model.models.encoders.minimax_h3_qwen3vl as enc

    assert hasattr(enc, "MiniMaxH3Qwen3VLEncoder")


def test_qwen_checkpoint_stream_does_not_read_unused_layers(monkeypatch, tmp_path) -> None:
    import safetensors

    from worldfoundry.base_models.diffusion_model.models.encoders import minimax_h3_qwen3vl as enc

    (tmp_path / "model.safetensors").touch()
    read_names = []

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return [
                "lm_head.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.50.self_attn.q_proj.weight",
            ]

        def get_tensor(self, name):
            read_names.append(name)
            return torch.ones(1)

    monkeypatch.setattr(safetensors, "safe_open", lambda *_args, **_kwargs: FakeSafeOpen())
    pairs = list(enc._iter_checkpoint_weights(str(tmp_path)))

    assert [name for name, _ in pairs] == [
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ]
    assert read_names == ["model.language_model.layers.0.self_attn.q_proj.weight"]
