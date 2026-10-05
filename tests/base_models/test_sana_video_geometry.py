from __future__ import annotations

from types import SimpleNamespace

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
    DiffusersLTX2TensorVideoCodec,
    LTXTensorVideoCodec,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import (
    _module_execution_target,
)
from worldfoundry.base_models.diffusion_model.models.denoisers import sana as sana_denoisers
from worldfoundry.base_models.diffusion_model.models.initializers.sana import (
    SanaNoiseInitializer,
    SanaWorldInitializer,
)
from worldfoundry.core.geometry.trajectory import rollout_wasd_camera_actions
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.pipelines.sana.pipeline_sana import SanaVideo2b480pPipeline


def test_sana_video_480p_config_matches_official_checkpoint_graph() -> None:
    config = sana_denoisers.sana_video_config(resolution="480p")

    assert config == {
        "input_size": 60,
        "patch_size": (1, 2, 2),
        "in_channels": 16,
        "hidden_size": 2240,
        "depth": 20,
        "num_heads": 20,
        "mlp_ratio": 3.0,
        "class_dropout_prob": 0.1,
        "pred_sigma": False,
        "caption_channels": 2304,
        "model_max_length": 300,
        "qk_norm": True,
        "y_norm": True,
        "y_norm_scale_factor": 0.01,
        "attn_type": "LiteLAReLURope",
        "ffn_type": "GLUMBConvTemp",
        "mlp_acts": ("silu", "silu", None),
        "use_pe": True,
        "pos_embed_type": "wan_rope",
        "linear_head_dim": 112,
        "cross_norm": True,
        "t_kernel_size": 3,
    }


def test_sana_video_720p_config_matches_official_checkpoint_graph() -> None:
    config = sana_denoisers.sana_video_config(resolution="720p")

    assert config["input_size"] == 22
    assert config["patch_size"] == (1, 1, 1)
    assert config["in_channels"] == 128
    assert config["hidden_size"] == 2240
    assert config["depth"] == 20
    assert config["qk_norm"] is True
    assert config["cross_norm"] is True
    assert config["ffn_type"] == "GLUMBConvTemp"
    assert config["t_kernel_size"] == 3


def test_sana_video_builder_selects_video_graph(monkeypatch) -> None:
    captured = {}

    def fake_build(context, *, module_class, config, output_scale=1.0):
        del context, output_scale
        captured["module_class"] = module_class
        captured["config"] = config
        return "denoiser"

    monkeypatch.setattr(sana_denoisers, "_build", fake_build)
    context = SimpleNamespace(component_options={"resolution": "480p"})

    result = sana_denoisers.build_sana_video_denoiser(context)

    assert result == "denoiser"
    assert captured["module_class"].__name__ == "SanaMSVideo"
    assert captured["config"] == sana_denoisers.sana_video_config(resolution="480p")


def test_sana_video_pipeline_prefers_catalog_variant_over_family_id() -> None:
    model_id = SanaVideo2b480pPipeline._requested_model_id(
        {
            "model_id": "sana",
            "variant_id": "sana-video-2b-480p",
            "profile_id": "sana",
        }
    )

    assert model_id == "sana-video-2b-480p"


@pytest.mark.parametrize(
    "model_id",
    (
        "sana-video-2b-480p",
        "sana-video-2b-720p",
        "longsana-video-2b-480p",
    ),
)
def test_sana_video_runtime_profiles_emit_mp4(model_id: str) -> None:
    profile = load_runtime_profile(model_id)

    assert profile.artifact_kind == "generated_video"
    assert profile.artifact_filename.endswith(".mp4")
    assert profile.integration_status == "integrated"


def test_sana_720p_geometry_pads_latent_height_without_reducing_resolution() -> None:
    initializer = SanaNoiseInitializer(
        channels=128,
        spatial_compression=32,
        temporal_compression=8,
        allow_spatial_padding=True,
    )

    shape = initializer.latent_shape(
        DiffusionRequest(prompt="test", height=720, width=1280, num_frames=81)
    )

    assert shape == (1, 128, 11, 23, 40)


def test_sana_world_camera_actions_fall_back_to_interactions_when_default_is_none(
    monkeypatch,
) -> None:
    captured = {}

    def rollout(actions, *, num_frames):
        captured["actions"] = actions
        return torch.eye(4).repeat(num_frames, 1, 1)

    monkeypatch.setattr(
        "worldfoundry.core.geometry.trajectory.rollout_sana_wm_camera_actions",
        rollout,
    )
    request = DiffusionRequest(
        prompt="test",
        height=704,
        width=1280,
        num_frames=9,
        inputs={"camera_actions": None, "interactions": ["forward"]},
    )

    conditions = SanaWorldInitializer._camera_conditions(request)

    assert captured["actions"] == ["forward"]
    assert conditions["camera_conditions"].shape == (1, 2, 20)
    assert conditions["chunk_plucker"].shape == (1, 48, 2, 22, 40)


def test_sana_world_named_camera_action_maps_to_wasd_key() -> None:
    poses = rollout_wasd_camera_actions("forward", num_frames=3)

    assert poses.shape == (3, 4, 4)
    assert poses[-1, 2, 3] > poses[0, 2, 3]


def test_ltx_tensor_codec_crops_padded_decode_to_requested_pixels() -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    class FakeDecoderBody(torch.nn.Module):
        def forward(self, latents: torch.Tensor, *, generator=None):
            del latents, generator
            return torch.zeros(1, 3, 9, 64, 64)

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decoder = FakeDecoderBody()

    codec = LTXTensorVideoCodec(FakeEncoder(), FakeDecoder(), tiling=None)

    video = codec.decode(
        torch.zeros(1, 128, 2, 2, 2),
        DiffusionRequest(prompt="test", height=33, width=63, num_frames=9),
    )

    assert video.shape == (1, 3, 9, 33, 63)


def test_diffusers_ltx2_codec_normalizes_and_denormalizes_official_latents() -> None:
    class Posterior:
        def mode(self):
            return torch.full((1, 128, 1, 1, 1), 10.0)

    class FakeVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.latents_mean = torch.full((128,), 2.0)
            self.latents_std = torch.full((128,), 4.0)
            self.config = SimpleNamespace(scaling_factor=0.5)
            self.decoded_latents = None

        def encode(self, pixels, return_dict=False):
            del pixels, return_dict
            return (Posterior(),)

        def decode(self, latents, return_dict=False):
            del return_dict
            self.decoded_latents = latents
            return (torch.zeros(1, 3, 9, 64, 64),)

    vae = FakeVAE()
    codec = DiffusersLTX2TensorVideoCodec(vae)
    latents = codec.encode(torch.zeros(1, 3, 1, 32, 32))
    video = codec.decode(
        latents,
        DiffusionRequest(prompt="test", height=33, width=63, num_frames=9),
    )

    assert torch.equal(latents, torch.ones_like(latents))
    assert torch.equal(vae.decoded_latents, torch.full_like(latents, 10.0))
    assert video.shape == (1, 3, 9, 33, 63)


def test_diffusers_ltx2_codec_stages_vae_around_each_call() -> None:
    class Posterior:
        def mode(self):
            return torch.zeros(1, 128, 1, 1, 1)

    class FakeVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.latents_mean = torch.zeros(128)
            self.latents_std = torch.ones(128)
            self.config = SimpleNamespace(scaling_factor=1.0)
            self.moves: list[torch.device] = []

        def to(self, *args, **kwargs):
            raw_device = kwargs.get("device", args[0] if args else None)
            if raw_device is not None:
                self.moves.append(torch.device(raw_device))
            return super().to(*args, **kwargs)

        def encode(self, pixels, return_dict=False):
            del pixels, return_dict
            return (Posterior(),)

        def decode(self, latents, return_dict=False):
            del latents, return_dict
            return (torch.zeros(1, 3, 1, 32, 32),)

    vae = FakeVAE()
    codec = DiffusersLTX2TensorVideoCodec(
        vae,
        compute_device="cpu",
        compute_dtype=torch.float32,
        offload_after_call=True,
    )
    latents = codec.encode(torch.zeros(1, 3, 1, 32, 32))
    codec.decode(latents)

    assert vae.moves == [
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cpu"),
    ]


def test_ltx_tensor_codec_uses_explicit_compute_target_for_wrapped_modules() -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    class FakeDecoderBody(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: torch.Tensor | None = None

        def forward(self, latents: torch.Tensor, *, generator=None):
            del generator
            self.received = latents
            return torch.zeros(1, 3, 9, 32, 32, dtype=latents.dtype, device=latents.device)

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.decoder = FakeDecoderBody()

    decoder = FakeDecoder()
    codec = LTXTensorVideoCodec(
        FakeEncoder(),
        decoder,
        tiling=None,
        compute_device="cpu",
        compute_dtype=torch.bfloat16,
    )

    codec.decode(torch.zeros(1, 128, 2, 1, 1, dtype=torch.float32))

    assert decoder.decoder.received is not None
    assert decoder.decoder.received.dtype is torch.bfloat16


def test_ltx_tiled_encoder_prefers_vram_wrapper_execution_target() -> None:
    class FakeManagedLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.computation_device = torch.device("cuda:3")
            self.computation_dtype = torch.bfloat16

    module = torch.nn.Sequential(FakeManagedLayer())

    device, dtype = _module_execution_target(module)

    assert device == torch.device("cuda:3")
    assert dtype is torch.bfloat16
