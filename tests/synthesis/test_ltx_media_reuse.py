# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared LTX media graphs, sample windows and family precision/lifecycle policies."""

import importlib
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.variants.joyai_echo import (
    EchoVocoderWithBWE,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import (
    MelSTFT,
    Vocoder,
    VocoderWithBWE,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.image import resize_and_center_crop
from worldfoundry.core.media.codecs.audio_window import decode_audio_from_file

pytestmark = pytest.mark.unit

_ROOTS = (
    "worldfoundry.synthesis.visual_generation.joyai_echo.echo_wm_runtime.ltx_core",
    "worldfoundry.synthesis.visual_generation.joyai_echo.echo_longvideo_runtime.ltx_core",
    "worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core",
)


def _vocoder_config(rate: int = 2) -> dict[str, Any]:
    return {
        "resblock_kernel_sizes": [3],
        "upsample_rates": [rate],
        "upsample_kernel_sizes": [2 * rate],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_initial_channel": 16,
        "resblock": "AMP1",
        "activation": "snakebeta",
    }


def _initialize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for key, value in module.state_dict().items():
            value.uniform_(-0.05, 0.05)
            if key.endswith("mel_basis"):
                value.abs_().add_(0.01)


@pytest.mark.parametrize("root", _ROOTS, ids=["echo-wm", "echo-longvideo", "solarwm"])
@pytest.mark.parametrize("with_bwe", [False, True])
def test_ltx_vocoder_checkpoint_loads_shared_graph(tmp_path: Path, root: str, with_bwe: bool) -> None:
    from safetensors.torch import save_file

    cfg: dict[str, Any] = _vocoder_config() | {"resblock": "1", "stereo": True}
    if with_bwe:
        cfg = {
            "vocoder": _vocoder_config() | {"stereo": True},
            "bwe": _vocoder_config(8)
            | {
                "stereo": True,
                "input_sampling_rate": 24000,
                "output_sampling_rate": 48000,
                "n_fft": 8,
                "hop_length": 4,
                "num_mels": 64,
            },
        }
    config = {"vocoder": cfg}
    factory_module = importlib.import_module(root + ".model.audio_vae.model_configurator")
    factory = factory_module.VocoderConfigurator
    solar = ".solarwm." in root
    reference = (factory.from_metadata({"config": config}) if solar else factory.from_config(config)).eval()
    assert isinstance(reference, VocoderWithBWE if with_bwe else Vocoder)
    if with_bwe and not solar:
        assert type(reference) is EchoVocoderWithBWE
    _initialize(reference)
    dtype = torch.float32 if solar else torch.bfloat16
    reference = reference.to(dtype)
    state = {"vocoder." + key: value for key, value in reference.state_dict().items()}
    path = tmp_path / "vocoder.safetensors"
    save_file(state, path, metadata={"config": json.dumps(config)})
    builder = importlib.import_module(root + ".loader.single_gpu_model_builder").SingleGPUModelBuilder
    loaded = builder(
        model_path=str(path),
        model_class_configurator=factory,
        model_sd_ops=factory_module.VOCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device("cpu"), dtype=dtype)
    torch.testing.assert_close(loaded.state_dict(), reference.state_dict(), rtol=0, atol=0)
    mel = torch.randn(1, 2, 5, 64, dtype=dtype)
    with torch.no_grad():
        actual = loaded(mel)
        torch.testing.assert_close(actual, reference(mel), rtol=0, atol=0)
    assert actual.shape == (1, 2, 20 if with_bwe else 10)
    if solar:
        scratch = torch.ones(2)
        loaded.register_buffer("scratch", scratch, persistent=False)
        loaded.dispose()
        assert all(value.device.type == "meta" for value in loaded.state_dict().values())
        assert loaded.scratch is scratch
        if with_bwe:
            assert loaded.resampler.filter.device.type == "cpu"


def test_echo_bandwidth_extension_retains_ambient_autocast() -> None:
    import copy

    echo = EchoVocoderWithBWE(
        Vocoder(**_vocoder_config()),
        Vocoder(**_vocoder_config(8), apply_final_activation=False),
        MelSTFT(8, 4, 8, 64),
        24000,
        48000,
        4,
    ).eval()
    _initialize(echo)
    native = copy.deepcopy(echo)
    native.__class__ = VocoderWithBWE
    mel = torch.randn(1, 2, 5, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ambient = echo(mel)
        with pytest.warns(UserWarning, match="target dtype"):
            forced = native(mel)
    assert ambient.dtype == torch.bfloat16
    assert forced.dtype == torch.float32
    assert (ambient.float() - forced).abs().max() > 0


@pytest.mark.parametrize("start,duration", [(0.0, None), (0.013, 0.039), (0.14, 0.04), (0.0, 0.0)])
def test_audio_window_decodes_exact_pcm_samples(tmp_path: Path, start: float, duration: float | None) -> None:
    pytest.importorskip("av")
    samples = (np.arange(2560, dtype=np.int16) - 1280).reshape(1280, 2)
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())
    decoded = decode_audio_from_file(str(path), torch.device("cpu"), start, duration)
    assert decoded is not None
    waveform, sample_rate = decoded
    assert sample_rate == 8000
    expected = torch.from_numpy(samples.T.astype(np.float32) / 32768.0).unsqueeze(0)
    expected = expected[..., round(start * 8000) :]
    if duration is not None:
        expected = expected[..., : round(duration * 8000)]
    torch.testing.assert_close(waveform, expected, rtol=0, atol=0)


def test_audio_window_returns_none_for_video_only(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "video.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    assert decode_audio_from_file(str(path), torch.device("cpu")) is None


@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_ltx_center_crop_uses_ceil_geometry(frames: int, dtype: torch.dtype) -> None:
    values = torch.arange(frames * 9 * 15 * 3).remainder(256).reshape(frames, 9, 15, 3).to(dtype)
    image = values[0] if frames == 1 else values
    expected = torch.nn.functional.interpolate(
        values.permute(0, 3, 1, 2), size=(8, 14), mode="bilinear", align_corners=False
    )[..., 1:13]
    actual = resize_and_center_crop(image, height=8, width=12)
    torch.testing.assert_close(actual, expected.permute(1, 0, 2, 3).unsqueeze(0), rtol=0, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("role", ["encoder", "decoder", "upsampler"])
def test_solar_media_disposal_wraps_shared_graphs(role: str) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio import audio_vae
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.causality_axis import CausalityAxis
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import NormType
    from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import LatentUpsampler

    settings: dict[str, Any] = {
        "ch": 8,
        "ch_mult": (1, 2),
        "num_res_blocks": 1,
        "attn_resolutions": {8},
        "resolution": 16,
        "z_channels": 2,
        "norm_type": NormType.PIXEL,
        "causality_axis": CausalityAxis.HEIGHT,
        "mel_bins": 8,
    }
    if role == "upsampler":
        native_type = LatentUpsampler
        family_type = importlib.import_module(_ROOTS[2] + ".model.upsampler.model").LatentUpsampler
        settings = {"in_channels": 4, "mid_channels": 32, "num_blocks_per_stage": 1, "temporal_upsample": True}
        inputs = torch.randn(1, 4, 3, 4, 4)
    else:
        native_type = audio_vae.AudioEncoder if role == "encoder" else audio_vae.AudioDecoder
        family_type = getattr(importlib.import_module(_ROOTS[2] + ".model.audio_vae.audio_vae"), native_type.__name__)
        settings["in_channels" if role == "encoder" else "out_ch"] = 2
        inputs = torch.randn(1, 2, 16, 8) if role == "encoder" else torch.randn(1, 2, 8, 4)
    reference = native_type(**settings).eval()
    _initialize(reference)
    for key, value in reference.named_buffers():
        if "std" in key:
            value.abs_().add_(0.5)
    family = family_type(**settings).eval()
    assert isinstance(family, native_type)
    family.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(family(inputs), reference(inputs), rtol=0, atol=0)
    scratch = torch.arange(3)
    family.register_buffer("scratch", scratch, persistent=False)
    family.dispose()
    assert family.scratch is scratch
    assert all(value.device.type == "meta" for value in family.state_dict().values())
