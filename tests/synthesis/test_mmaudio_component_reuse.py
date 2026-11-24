# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading through MMAudio's canonical audio components."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Literal

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.edm2_utils import (
    MPConv1D,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.vae import (
    Decoder1D,
    Encoder1D,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.bigvgan_v2.alias_free_activation.torch.act import (
    Activation1d,
)
from worldfoundry.base_models.perception_core.general_perception.dino.models.vision_transformer import Mlp
from worldfoundry.synthesis.visual_generation.mmaudio.checkpoints import (
    BIGVGAN_MODEL_ID,
    BIGVGAN_REQUIRED_FILES,
    DFN5B_MODEL_ID,
    DFN5B_REQUIRED_FILES,
    resolve_bigvgan_v2_checkpoint,
    resolve_dfn5b_checkpoint,
)
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.autoencoder import autoencoder
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.autoencoder.vae import VAE
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.bigvgan.bigvgan import BigVGAN
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.bigvgan.models import BigVGANVocoder
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.bigvgan_v2.env import AttrDict
from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.ext.synchformer import vit_helper

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("mode", "data_dim"), [("16k", 80), ("44k", 128)])
def test_mmaudio_loads_canonical_vae_and_vocoder_components(
    mode: Literal["16k", "44k"], data_dim: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(7)
    reference = VAE(data_dim=data_dim, embed_dim=2, hidden_dim=4).eval()
    vae_path = tmp_path / "vae.pth"
    torch.save(reference.state_dict(), vae_path)
    reference.remove_weight_norm()

    def build_vae(name: str) -> VAE:
        assert name == mode
        return VAE(data_dim=data_dim, embed_dim=2, hidden_dim=4)

    monkeypatch.setattr(autoencoder, "get_my_vae", build_vae)
    config = AttrDict(
        num_mels=data_dim,
        upsample_initial_channel=8,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        activation="snakebeta",
        snake_logscale=True,
    )
    vocoder = BigVGANVocoder(config) if mode == "16k" else BigVGANv2(config)
    vocoder.eval()
    (tmp_path / "config.json").write_text(json.dumps(config))
    vocoder_path = tmp_path / "bigvgan_generator.pt"
    torch.save({"generator": vocoder.state_dict()}, vocoder_path)
    vocoder.remove_weight_norm()
    if mode == "16k":
        monkeypatch.setattr(autoencoder, "BigVGAN", partial(BigVGAN, config_path=tmp_path / "config.json"))
    else:
        monkeypatch.setattr(autoencoder, "resolve_bigvgan_v2_checkpoint", lambda: tmp_path)

    loaded = autoencoder.AutoEncoderModule(vae_ckpt_path=vae_path, vocoder_ckpt_path=str(vocoder_path), mode=mode)
    assert isinstance(loaded.vae.encoder, Encoder1D)
    assert isinstance(loaded.vae.decoder, Decoder1D)
    assert all(module.weight_norm_removed for module in loaded.vae.modules() if isinstance(module, MPConv1D))
    assert any(isinstance(module, Activation1d) for module in loaded.vocoder.modules())
    assert loaded.vae.state_dict().keys() == reference.state_dict().keys()
    x = torch.randn(1, data_dim, 8)
    with torch.inference_mode():
        expected_posterior = reference.encode(x)
        posterior = loaded.encode(x)
        torch.testing.assert_close(posterior.mean, expected_posterior.mean, rtol=0, atol=0)
        torch.testing.assert_close(posterior.logvar, expected_posterior.logvar, rtol=0, atol=0)
        spec = loaded.decode(posterior.mode())
        torch.testing.assert_close(spec, reference.decode(expected_posterior.mode()), rtol=0, atol=0)
        torch.testing.assert_close(loaded.vocode(spec), vocoder(spec), rtol=0, atol=0)


def test_mmaudio_synchformer_uses_dino_mlp() -> None:
    assert vit_helper.Mlp is Mlp
    block = vit_helper.DividedSpaceTimeBlock(dim=8, num_heads=2, qkv_bias=True)
    assert isinstance(block.mlp, Mlp)


@pytest.mark.parametrize("rate", ["16k", "44k"])
def test_mmaudio_replay_hashes_the_resolved_support_weights(
    rate: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    repositories = [(DFN5B_MODEL_ID, DFN5B_REQUIRED_FILES, resolve_dfn5b_checkpoint)]
    if rate == "44k":
        repositories.append((BIGVGAN_MODEL_ID, BIGVGAN_REQUIRED_FILES, resolve_bigvgan_v2_checkpoint))
    required = set()
    for model_id, files, resolve in repositories:
        directory = tmp_path / model_id.replace("/", "--")
        directory.mkdir()
        for name in files:
            (directory / name).touch()
        required.update(resolve() / name for name in files)

    matrix = Path(__file__).resolve().parents[1] / "manual" / "geometry_regression_cases.json"
    case = json.loads(matrix.read_text(encoding="utf-8"))[f"mmaudio-small-{rate}-video-audio-short"]
    staged = {Path(path.replace("${WORLDFOUNDRY_CKPT_DIR}", str(tmp_path))) for path in case["assets"].values()}
    assert required <= staged
    assert set(case["asset_sha256"]) == set(case["assets"])
