# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free public pipeline, operator, and catalog tests for pi-H3."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.processor import VAEProcessor
from worldfoundry.base_models.diffusion_model.models.denoisers.pi_h3 import pi_h3_arch_config
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import MiniMaxH3DiTBlock
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3 import (
    PiH3DiTModel,
    PiH3PrunedAdalnProj,
    SolH3Block,
)
from worldfoundry.base_models.diffusion_model.schedulers.pi_h3 import pi_h3_flow_euler_sigmas
from worldfoundry.operators.pi_h3_operator import PiH3Operator, normalize_pi_h3_variant
from worldfoundry.pipelines.pi_h3 import pipeline_pi_h3 as pi_h3_module
from worldfoundry.pipelines.pi_h3.pipeline_pi_h3 import PiH3Pipeline
from worldfoundry.runtime.inference_specs.pi_h3 import PI_H3_INFERENCE_SPEC

pytestmark = pytest.mark.unit


def _tiny_pi_h3_dit(*, with_dwd: bool, seed: int = 12) -> PiH3DiTModel:
    cfg = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 64,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 16,
        "freq_dim": 8,
        "time_embed_hidden_dim": 32,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "adaln_rank": 8,
        "time_table_size": 17,
    }
    dwd_cfg = (
        {
            "pixel_patch_size": [4, 32, 32],
            "first_frame_pixel_patch_size": [1, 32, 32],
            "first_frame_period": 5,
            "pixel_channels": 3,
            "linear_head": False,
            "pixel_refiner": {
                "type": "SolH3DWDHead",
                "num_layers": 2,
                "total_vae_layers": 24,
                "num_heads": 2,
                "head_dim": 16,
                "ffn_mult": 2,
                "num_register_tokens": 4,
                "window_size": 4,
                "block_size": 5,
                "t_shift": 2,
                "time_norm": 5,
                "spatial_patch": 16,
                "patch_t": 4,
                "frames_to_trim": 3,
            },
        }
        if with_dwd
        else None
    )
    torch.manual_seed(seed)
    model = PiH3DiTModel(
        pi_h3_arch_config(cfg),
        adaln_rank=8,
        time_table_size=17,
        dwd_head_config=dwd_cfg,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            assert isinstance(block, MiniMaxH3DiTBlock)
            assert isinstance(block.adaln_proj, PiH3PrunedAdalnProj)
            block.adaln_proj.linear.weight.zero_()
            block.adaln_proj.linear.bias.zero_()
        model.final_layer.adaln_proj.linear.weight.zero_()
        model.final_layer.adaln_proj.linear.bias.zero_()
        if model.dwd_pixel_refiner is not None:
            model.dwd_pixel_refiner.register_tokens.normal_(0.0, 0.02)
            for blk in model.dwd_pixel_refiner.blocks:
                assert isinstance(blk, SolH3Block)
                blk.scale1.fill_(0.1)
                blk.scale2.fill_(0.1)
        for param in model.parameters():
            param.mul_(0.05)
    return model


class _StubTextEncoder:
    def __init__(self, text_dim: int = 16) -> None:
        self.text_dim = text_dim

    def encode_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(int(input_ids.numel()), self.text_dim, dtype=torch.float32) * 0.1

    def encode_presentation(
        self,
        prompt: str,
        *,
        tokenizer: object,
        processor: object,
        images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt, tokenizer, processor
        n = 4 + len(images)
        return torch.ones(n, self.text_dim, dtype=torch.float32) * 0.1, torch.ones(n, dtype=torch.long)


class _StubTokenizer:
    def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt" and add_special_tokens is False
        n = max(2, len(prompt.split()))
        return {"input_ids": torch.arange(1, n + 1, dtype=torch.long).unsqueeze(0)}


class _StubAudioVAE:
    def __init__(self) -> None:
        self.calls = 0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # [2, 32, audio_t] -> stereo [2, 1, samples]
        return torch.zeros(int(latent.shape[0]), 1, int(latent.shape[-1]) * 80, dtype=torch.float32)


class _StubVideoVAE:
    def __init__(self, *, forbid_decode: bool = False) -> None:
        self.forbid_decode = forbid_decode
        self.decode_calls = 0
        self.encode_calls = 0
        self.sglang_config = SimpleNamespace(
            arch_config=SimpleNamespace(latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
        )
        self.processor = VAEProcessor(
            vae_ratio=16,
            vae_ratio_t=4,
            clip_length=17,
            frame_overlap=0,
            token_overlap=0,
            tokens_chunk_size=4,
            isolated_last_frame=False,
            latent_patch_size=1,
            crop_mode="center",
            use_3d_conv=True,
        )

    def encode_images(self, images: list[Image.Image], *, deterministic: bool = False) -> list[torch.Tensor]:
        del deterministic
        self.encode_calls += len(images)
        return [torch.zeros(24, 1, img.height // 16, img.width // 16, dtype=torch.float32) for img in images]

    def decode_base(self, latent: torch.Tensor) -> torch.Tensor:
        if self.forbid_decode:
            raise AssertionError("video VAE decode must be skipped in DWD mode")
        self.decode_calls += 1
        # latent: [1, 24, T, H_lat, W_lat] -> [1, 3, 5, H_lat*16, W_lat*16]
        t_lat, h_lat, w_lat = int(latent.shape[2]), int(latent.shape[3]), int(latent.shape[4])
        frames = 5 if t_lat <= 2 else ((t_lat - 2) // 5) * 17 + 5
        return torch.zeros(1, 3, frames, h_lat * 16, w_lat * 16, dtype=torch.float32)


def test_pi_h3_operator_validation_and_variant_normalization() -> None:
    assert normalize_pi_h3_variant("pi_h3_2step_dwd_adapter") == "2step-dwd"
    assert normalize_pi_h3_variant("4step_dwd") == "4step-dwd"
    assert normalize_pi_h3_variant("2step") == "2step"
    assert normalize_pi_h3_variant("4step-vae") == "4step"
    with pytest.raises(ValueError, match="Unsupported pi-H3 variant"):
        normalize_pi_h3_variant("8step")

    op = PiH3Operator(variant="2step-dwd")
    op.get_interaction("A red fox in snow")
    req = op.process_interaction(num_frames=124, seed=7)
    assert req["task"] == "t2va"
    assert req["variant"] == "2step-dwd"
    assert req["num_inference_steps"] == 2
    assert req["dwd"] is True

    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    assert op.validate_request(prompt="cat", images=img)["task"] == "i2va"
    assert op.validate_request(prompt="cat", last_image=img)["task"] == "l2va"
    assert op.validate_request(prompt="cat", images=[img, img])["task"] == "fl2va"
    assert len(op.process_perception(images=img)["images"]) == 1

    with pytest.raises(ValueError, match="prompt"):
        op.validate_request(prompt="   ")
    with pytest.raises(ValueError, match="requires 2 steps"):
        op.validate_request(prompt="fox", num_inference_steps=4)
    with pytest.raises(ValueError, match="supports \\(2, 4\\)"):
        op.validate_request(prompt="fox", num_inference_steps=6)
    with pytest.raises(ValueError, match="multiples of 32"):
        op.validate_request(prompt="fox", height=50, width=64)
    with pytest.raises(ValueError, match="24 fps"):
        op.validate_request(prompt="fox", fps=30)
    with pytest.raises(ValueError, match="requires 2 keyframe"):
        op.validate_request(prompt="fox", task="fl2va", images=[img])


def test_pi_h3_pipeline_dwd_skips_video_vae_and_outputs_unit_rgb_with_audio() -> None:
    dit = _tiny_pi_h3_dit(with_dwd=True)
    forward_flags: list[bool] = []
    original_forward = dit.forward

    def spy_forward(*args, **kwargs):
        forward_flags.append(bool(kwargs.get("return_features", False)))
        return original_forward(*args, **kwargs)

    dit.forward = spy_forward  # type: ignore[method-assign]
    video_vae = _StubVideoVAE(forbid_decode=True)
    audio_vae = _StubAudioVAE()
    pipe = PiH3Pipeline(
        transformer=dit,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_encoder=_StubTextEncoder(),
        tokenizer=_StubTokenizer(),
        device="cpu",
        variant="2step-dwd",
    )

    result = pipe(
        prompt="A red fox trotting through snow",
        height=32,
        width=32,
        num_frames=5,
        seed=42,
        return_dict=True,
    )

    assert forward_flags == [False, True]
    assert video_vae.decode_calls == 0
    assert audio_vae.calls == 1
    assert result["variant"] == "2step-dwd"
    assert result["dwd"] is True
    assert result["inference_steps"] == 2
    assert result["video"].shape == (1, 3, 5, 32, 32)
    assert float(result["video"].min()) >= 0.0
    assert float(result["video"].max()) <= 1.0
    assert torch.isfinite(result["video"]).all()
    assert result["audio"].shape[0] == 2
    assert result["sigmas_video"] == pi_h3_flow_euler_sigmas(2, shift=12.0)
    assert result["sigmas_audio"] == pi_h3_flow_euler_sigmas(2, shift=3.0)


def test_pi_h3_pipeline_non_dwd_uses_video_vae_decode() -> None:
    dit = _tiny_pi_h3_dit(with_dwd=False)
    forward_flags: list[bool] = []
    original_forward = dit.forward

    def spy_forward(*args, **kwargs):
        forward_flags.append(bool(kwargs.get("return_features", False)))
        return original_forward(*args, **kwargs)

    dit.forward = spy_forward  # type: ignore[method-assign]
    video_vae = _StubVideoVAE(forbid_decode=False)
    audio_vae = _StubAudioVAE()
    pipe = PiH3Pipeline(
        transformer=dit,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_encoder=_StubTextEncoder(),
        tokenizer=_StubTokenizer(),
        device="cpu",
        variant="4step",
    )

    result = pipe(
        prompt="Ocean waves crashing at sunset",
        height=32,
        width=32,
        num_frames=5,
        seed=7,
        return_dict=True,
    )

    assert forward_flags == [False, False, False, False]
    assert video_vae.decode_calls == 1
    assert audio_vae.calls == 1
    assert result["variant"] == "4step"
    assert result["dwd"] is False
    assert result["video"].shape == (1, 3, 5, 32, 32)
    assert float(result["video"].min()) >= 0.0 and float(result["video"].max()) <= 1.0


def test_pi_h3_pipeline_fl2va_keyframe_encoding_and_output_writing(monkeypatch, tmp_path: Path) -> None:
    dit = _tiny_pi_h3_dit(with_dwd=True)
    video_vae = _StubVideoVAE(forbid_decode=True)
    audio_vae = _StubAudioVAE()
    pipe = PiH3Pipeline(
        transformer=dit,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_encoder=_StubTextEncoder(),
        tokenizer=_StubTokenizer(),
        processor=object(),
        device="cpu",
        variant="2step-dwd",
    )
    written: dict[str, object] = {}

    def fake_write(*, video, audio, output_path, fps):
        written["shape"] = tuple(video.shape)
        written["has_audio"] = audio is not None
        written["fps"] = fps
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return str(output_path)

    monkeypatch.setattr(PiH3Pipeline, "_write_mp4_with_audio", staticmethod(fake_write))
    first = Image.new("RGB", (64, 64), color=(10, 20, 30))
    last = Image.new("RGB", (64, 64), color=(200, 150, 100))
    out_file = tmp_path / "clip.mp4"

    artifact = pipe(
        prompt="Camera glides from morning to dusk",
        images=[first, last],
        height=32,
        width=32,
        num_frames=5,
        seed=1,
        output_path=out_file,
    )

    assert artifact == str(out_file)
    assert out_file.is_file()
    assert video_vae.encode_calls == 2
    assert video_vae.decode_calls == 0
    assert written == {"shape": (1, 3, 5, 32, 32), "has_audio": True, "fps": 24}

    with pytest.raises(ValueError, match="loaded with variant"):
        pipe(prompt="x", variant="4step-dwd", height=32, width=32, num_frames=5)
    with pytest.raises(ValueError, match="not a video input"):
        pipe(prompt="x", video="clip.mp4")


def test_pi_h3_from_pretrained_resolves_variant_and_optional_dwd_vae(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_load_transformer(base_dir, adapter_dir, *, device, dtype, lora_scale):
        captured["base_dir"] = Path(base_dir)
        captured["adapter_dir"] = Path(adapter_dir)
        captured["device"] = device
        captured["dtype"] = dtype
        captured["lora_scale"] = lora_scale
        return SimpleNamespace(has_dwd_head=True)

    monkeypatch.setattr(pi_h3_module, "load_pi_h3_transformer", fake_load_transformer)
    monkeypatch.setattr(
        PiH3Pipeline,
        "_load_pruned_audio_vae",
        classmethod(lambda _cls, _root, *, device: ("audio_vae", device)),
    )
    monkeypatch.setattr(
        PiH3Pipeline,
        "_load_text_encoder",
        classmethod(lambda _cls, _root, *, device: ("text_encoder", device)),
    )
    monkeypatch.setattr(PiH3Pipeline, "_load_tokenizer", classmethod(lambda _cls, _root: "tokenizer"))

    base_root = tmp_path / "MiniMax-H3-Pruned"
    (base_root / "transformer").mkdir(parents=True)
    adapter_root = tmp_path / "pi-H3"
    (adapter_root / "pi_h3_2step_dwd_adapter").mkdir(parents=True)

    # DWD mode succeeds even when no video VAE directory is staged (T2VA-only footprint).
    pipe = PiH3Pipeline.from_pretrained(
        base_model_path=base_root,
        adapter_root=adapter_root,
        variant="2step-dwd",
        device="cpu",
        dtype="float32",
    )
    assert pipe.variant == "2step-dwd"
    assert pipe.use_dwd is True
    assert pipe.video_vae is None
    assert captured["adapter_dir"] == (adapter_root / "pi_h3_2step_dwd_adapter").resolve()
    assert captured["dtype"] == torch.float32

    # Non-DWD mode strictly requires the video VAE directory.
    with pytest.raises(FileNotFoundError, match="video VAE"):
        PiH3Pipeline.from_pretrained(
            base_model_path=base_root,
            adapter_root=adapter_root,
            variant="2step",
            device="cpu",
        )


def test_pi_h3_catalog_binding_profile_and_inference_spec_resolve() -> None:
    from worldfoundry.zoo.bindings import resolve_pipeline_binding
    from worldfoundry.zoo.catalog import load_model_zoo_registry

    registry = load_model_zoo_registry()
    for alias in (
        "pi-h3",
        "pi_h3",
        "pih3",
        "pi-h3-dwd",
        "pi-h3-2step-dwd",
        "pi-h3-4step-dwd",
        "pi-h3-2step",
        "pi-h3-4step",
        "minimax-h3-pi",
    ):
        entry = registry.get(alias)
        assert entry.model_id == "pi-h3"
        assert entry.pipeline_binding == "pi-h3"
        assert entry.runtime_profile == "runtime-profile:pi-h3"
        binding = resolve_pipeline_binding(alias)
        assert binding.pipeline_target == "worldfoundry.pipelines.pi_h3.pipeline_pi_h3:PiH3Pipeline"

    assert PI_H3_INFERENCE_SPEC.model_family_id == "pi-h3"
    assert PI_H3_INFERENCE_SPEC.default_variant_id == "pi-h3-4step-dwd"
    assert [v.variant_id for v in PI_H3_INFERENCE_SPEC.variants] == [
        "pi-h3-4step-dwd",
        "pi-h3-2step-dwd",
        "pi-h3-4step",
        "pi-h3-2step",
    ]
    assert [t.task_id for t in PI_H3_INFERENCE_SPEC.tasks] == ["t2va", "i2va", "l2va", "fl2va"]
