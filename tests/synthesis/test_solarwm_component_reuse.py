# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM loads its codecs through the canonical checkpoint graphs."""

from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import reference_22
from worldfoundry.base_models.diffusion_model.models.encoders.wan import model as wan_text
from worldfoundry.base_models.diffusion_model.models.encoders.wan import reference as wan_t5
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime import components

pytestmark = pytest.mark.unit


def test_solarwm_wan_text_loads_canonical_weights_and_masks_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = wan_t5.umt5_xxl

    def tiny_encoder(**kwargs: Any) -> torch.nn.Module:
        return factory(
            **kwargs,
            vocab_size=31,
            dim=16,
            dim_attn=16,
            dim_ffn=32,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            num_buckets=8,
            dropout=0.0,
        )

    class Tokenizer:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"name": "tokenizer", "seq_len": 512, "clean": "whitespace"}

        def __call__(self, prompts: list[str], **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            assert prompts == ["a", "b"]
            assert kwargs == {"return_mask": True, "add_special_tokens": True}
            ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
            return ids, ids != 0

    reference = tiny_encoder(encoder_only=True).eval()
    weights = tmp_path / "umt5.pth"
    torch.save(reference.state_dict(), weights)
    monkeypatch.setattr(wan_t5, "umt5_xxl", tiny_encoder)
    monkeypatch.setattr(wan_text, "HuggingfaceTokenizer", Tokenizer)
    encoder = components.WanTextEncoder(weights, "tokenizer")
    assert isinstance(encoder.module, wan_t5.T5Encoder)
    assert encoder.module.state_dict().keys() == reference.state_dict().keys()
    result = encoder(["a", "b"])["prompt_embeds"]
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    expected = reference(ids, ids != 0)
    expected[0, 2:] = 0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_solarwm_wan_vae_loads_canonical_weights_and_streams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    factory = reference_22._video_vae
    small = {"dim": 4, "dec_dim": 4, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1}
    reference = factory(z_dim=48, temperal_downsample=[False, True, True], **small).eval()
    weights = tmp_path / "wan.pth"
    torch.save(reference.state_dict(), weights)

    def tiny_vae(**kwargs: Any) -> torch.nn.Module:
        assert kwargs == {
            "pretrained_path": str(weights),
            "z_dim": 48,
            "temperal_downsample": [False, True, True],
        }
        return factory(**kwargs, **small)

    monkeypatch.setattr(reference_22, "_video_vae", tiny_vae)
    codec = components.Wan5BVAE(weights)
    assert isinstance(codec.module, reference_22.WanVAE_)
    assert codec.module.state_dict().keys() == reference.state_dict().keys()
    assert codec.mean == reference_22.WAN_2P2_VAE_MEAN
    assert codec.std == reference_22.WAN_2P2_VAE_STD
    pixels = torch.randn(1, 3, 5, 32, 32)
    with torch.no_grad():
        latents = codec.encode(pixels)
        expected = reference.encode(pixels, codec._scale(pixels)).permute(0, 2, 1, 3, 4)
        torch.testing.assert_close(latents, expected, rtol=0, atol=0)
        decoded = codec.decode(latents)
        codec.module.clear_cache()
        streamed = torch.cat([codec.decode(chunk, use_cache=True) for chunk in latents.split(1, 1)], dim=1)
    torch.testing.assert_close(streamed, decoded, rtol=0, atol=0)


def test_solarwm_diffusion_decoder_loads_canonical_weights(tmp_path: Path) -> None:
    import json

    from safetensors.torch import save_file

    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
        DiffusionVideoDecoder,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.single_gpu_model_builder import (
        SingleGPUModelBuilder,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.video_decoder import (
        build_diffvae_mode_op,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.video_vae import (
        model_configurator,
    )

    metadata = {
        "config": {
            "vae": {
                "_class_name": "CausalDiffusionVAE",
                "latent_channels": 16,
                "decoder": {
                    "in_channels": 16,
                    "head_dim": 16,
                    "stage_channels": [16] * 5,
                    "stage_depths": [1] * 5,
                    "stage_kernels": [[1, 1, 1]] * 5,
                    "stage5_kernel": [1, 1, 1],
                    "upsamples": [[[1, 1, 1], 1]] * 4,
                    "t_emb_dim": 16,
                },
            }
        }
    }
    reference = model_configurator.VideoDecoderConfigurator.from_metadata(metadata).eval()
    assert type(reference) is DiffusionVideoDecoder
    with torch.no_grad():
        reference.per_channel_statistics.get_buffer("std-of-means").fill_(1)
        reference.per_channel_statistics.get_buffer("mean-of-means").zero_()
    state = {f"decoder.{key}": value for key, value in reference.state_dict().items()}
    for key in list(state):
        if ".qkv.to_q." not in key:
            continue
        state[key.replace(".qkv.to_q.", ".qkv.")] = torch.cat(
            [state.pop(key.replace(".to_q.", f".to_{axis}.")) for axis in ("q", "k", "v")]
        )
    weights = tmp_path / "video.safetensors"
    save_file(state, weights, metadata={"config": json.dumps(metadata["config"])})
    ops = model_configurator.video_decoder_sd_ops_for_checkpoint(str(weights))
    loaded = SingleGPUModelBuilder(
        model_path=str(weights),
        model_class_configurator=model_configurator.VideoDecoderConfigurator,
        model_sd_ops=ops,
    ).build(device=torch.device("cpu"), dtype=torch.float32)
    assert type(loaded) is DiffusionVideoDecoder
    assert loaded.state_dict().keys() == reference.state_dict().keys()
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    from dataclasses import replace

    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.apply import (
        apply_diffvae_config,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import (
        DiffVAEMode,
        NAttentionKind,
    )

    op = build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER)
    assert op.matcher(loaded)
    assert op.mutator(loaded) is loaded
    assert op.mutator(reference) is reference
    assert loaded.deferred_stage4_upsample and reference.deferred_stage4_upsample
    # Loading parity uses the same CPU neighborhood attention on both sides.
    cpu_config = replace(DiffVAEMode.CHUNKED_EAGER.resolve(), attention=NAttentionKind.EAGER_SDPA, natten_backend=None)
    apply_diffvae_config(loaded, cpu_config)
    apply_diffvae_config(reference, cpu_config)
    latent = torch.randn(1, 16, 2, 2, 3)
    with torch.no_grad():
        actual = tuple(loaded.decode_video(latent, generator=torch.Generator().manual_seed(3)))
        expected = tuple(reference.decode_video(latent, generator=torch.Generator().manual_seed(3)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_solarwm_disposal_retains_nonpersistent_loader_buffers() -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.disposable import dispose_model

    module = torch.nn.Linear(4, 4)
    module.register_buffer("checkpoint_scale", torch.ones(4))
    scratch = torch.arange(4)
    module.register_buffer("scratch", scratch, persistent=False)
    dispose_model(module)
    assert module.weight.device.type == "meta"
    assert module.checkpoint_scale.device.type == "meta"
    assert module.scratch is scratch
