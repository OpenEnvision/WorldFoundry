# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""JoyAI's checkpoint loaders construct canonical LTX audio and upsampling graphs."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import AudioEncoder
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import LatentUpsampler

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("runtime", ["echo_wm_runtime", "echo_longvideo_runtime"])
def test_joyai_audio_encoder_loads_canonical_graph(runtime: str, tmp_path: Path) -> None:
    package = f"worldfoundry.synthesis.visual_generation.joyai_echo.{runtime}.ltx_core"
    configs = importlib.import_module(f"{package}.model.audio_vae.model_configurator")
    builder = importlib.import_module(f"{package}.loader.single_gpu_model_builder").SingleGPUModelBuilder
    config = {
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "ch": 8,
                        "ch_mult": [1, 2],
                        "num_res_blocks": 1,
                        "attn_resolutions": [],
                        "in_channels": 2,
                        "resolution": 16,
                        "z_channels": 2,
                        "mel_bins": 8,
                    }
                }
            }
        }
    }
    reference = configs.AudioEncoderConfigurator.from_config(config).eval()
    assert type(reference) is AudioEncoder
    with torch.no_grad():
        reference.per_channel_statistics.get_buffer("std-of-means").fill_(1)
        reference.per_channel_statistics.get_buffer("mean-of-means").zero_()
    checkpoint = {}
    for key, value in reference.state_dict().items():
        if key.startswith("per_channel_statistics."):
            checkpoint[f"audio_vae.{key}"] = value
        else:
            checkpoint[f"audio_vae.encoder.{key}"] = value
    path = tmp_path / "audio.safetensors"
    save_file(checkpoint, path, metadata={"config": json.dumps(config)})
    loaded = builder(
        model_path=str(path),
        model_class_configurator=configs.AudioEncoderConfigurator,
        model_sd_ops=configs.AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device("cpu"), dtype=torch.float32)
    assert type(loaded) is AudioEncoder
    assert loaded.state_dict().keys() == reference.state_dict().keys()
    inputs = torch.randn(1, 2, 16, 8)
    with torch.no_grad():
        torch.testing.assert_close(loaded(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("runtime", ["echo_wm_runtime", "echo_longvideo_runtime"])
def test_joyai_upsampler_uses_canonical_graph(runtime: str) -> None:
    ledger = importlib.import_module(
        f"worldfoundry.synthesis.visual_generation.joyai_echo.{runtime}.ltx_pipelines.utils.model_ledger"
    )
    module = ledger.LatentUpsamplerConfigurator.from_config(
        {"in_channels": 4, "mid_channels": 32, "num_blocks_per_stage": 1}
    )
    assert type(module) is LatentUpsampler
    with torch.no_grad():
        assert module(torch.randn(1, 4, 2, 3, 3)).shape == (1, 4, 2, 6, 6)


def test_joyai_prope_preserves_meta_construction_and_feature_dtype() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.prope import (
        EchoPropeAttention,
        _apply_tiled_projmat,
    )

    with torch.device("meta"):
        attention = EchoPropeAttention(16, 2, 2, 8, 8)
    assert attention.coeffs_x_0 is None
    viewmats = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    viewmats[:, 1, 0, 3] = 0.5
    attention._precompute_and_cache_apply_fns(viewmats, None)
    features = torch.ones(1, 2, 8, 16)
    assert torch.isfinite(attention._apply_to_q(features)).all()
    features = torch.ones(1, 1, 2, 8, dtype=torch.bfloat16)
    projected = _apply_tiled_projmat(features, viewmats)
    expected = features.clone()
    expected[:, :, 1, 0::4] = 1.5
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)


@pytest.mark.parametrize("runtime", ["echo_wm_runtime", "echo_longvideo_runtime"])
def test_joyai_quantization_imports_before_the_loader(runtime: str) -> None:
    package = f"worldfoundry.synthesis.visual_generation.joyai_echo.{runtime}.ltx_core"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

package = sys.argv[1]
for name in ("quantization", "quantization.fp8_cast", "quantization.fp8_scaled_mm", "quantization.policy"):
    importlib.import_module(f"{package}.{name}")
loader = importlib.import_module(f"{package}.loader")
assert callable(loader.apply_loras)
assert loader.SingleGPUModelBuilder is not None
""",
            package,
        ],
        check=True,
        timeout=60,
    )
