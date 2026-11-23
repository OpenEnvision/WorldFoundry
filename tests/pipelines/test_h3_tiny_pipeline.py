# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free pipeline, operator, invocation, and catalog/binding tests for H3-Tiny."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from worldfoundry.base_models.diffusion_model.models.denoisers.h3_tiny import h3_tiny_arch_config
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_tiny import H3TinyDiTModel
from worldfoundry.operators.h3_tiny_operator import H3TinyOperator
from worldfoundry.pipelines.h3_tiny import H3_TINY_VARIANTS, H3TinyPipeline
from worldfoundry.pipelines.h3_tiny import pipeline_h3_tiny as h3_tiny_module
from worldfoundry.runtime.inference_specs.h3_tiny import H3_TINY_INFERENCE_SPEC
from worldfoundry.zoo.bindings import resolve_pipeline_binding
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

pytestmark = pytest.mark.unit


def _tiny_h3_tiny_model(*, endpoint_mode: str = "meanflow") -> H3TinyDiTModel:
    cfg = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "refiner_ffn_dim": 48,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 16,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "adaln_rank": 4,
    }
    arch, variant_kwargs = h3_tiny_arch_config(cfg)
    torch.manual_seed(11)
    return H3TinyDiTModel(arch, endpoint_mode=endpoint_mode, **variant_kwargs).float().eval()


class _StubVideoVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sglang_config = SimpleNamespace(
            arch_config=SimpleNamespace(
                latents_mean=[0.0] * 24,
                latents_std=[1.0] * 24,
            )
        )
        self.processor = SimpleNamespace(revert_tensor=lambda tensor: tensor.clamp(0.0, 1.0))

    def encode_images(self, images: list[Any], deterministic: bool = False) -> list[torch.Tensor]:
        del deterministic
        out = []
        for image in images:
            w, h = image.size
            out.append(torch.zeros((1, 24, 1, h // 16, w // 16), dtype=torch.float32))
        return out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        _, _, latent_t, latent_h, latent_w = latent.shape
        return torch.full((1, 3, latent_t, latent_h * 16, latent_w * 16), 0.5, dtype=torch.float32)


class _StubAudioVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sglang_config = SimpleNamespace(arch_config=SimpleNamespace(latents_mean=None, latents_std=None))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        channels, _, steps = latent.shape
        return torch.zeros((channels, 1, steps * 16), dtype=torch.float32)


class _StubTextEncoder:
    def encode_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((int(input_ids.numel()), 16), dtype=torch.float32)

    def encode_presentation(
        self,
        prompt: str,
        *,
        tokenizer: Any,
        processor: Any,
        images: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt, tokenizer, processor
        n = 4 + len(images)
        return torch.zeros((n, 16), dtype=torch.float32), torch.ones((n,), dtype=torch.long)


def _stub_tokenizer(
    prompt: str, return_tensors: str = "pt", add_special_tokens: bool = False
) -> dict[str, torch.Tensor]:
    del prompt, return_tensors, add_special_tokens
    return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}


def test_h3_tiny_operator_validates_all_tasks_and_constraints(tmp_path: Path) -> None:
    op = H3TinyOperator()
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img1)
    Image.new("RGB", (64, 64), (40, 50, 60)).save(img2)

    assert op.validate_request(prompt="hello", task="t2va", endpoint_mode="none")["num_inference_steps"] == 50
    assert op.validate_request(prompt="hello", references=[str(img1)], endpoint_mode="meanflow")["task"] == "ref2va"
    assert (
        op.validate_request(prompt="hello", references=[{"kind": "image", "path": str(img1)}])["num_inference_steps"]
        == 6
    )
    assert op.validate_request(prompt="hello", image=str(img1))["keyframe_indices"] == (0,)
    assert op.validate_request(prompt="hello", last_image=str(img2))["keyframe_indices"] == (-1,)
    assert op.validate_request(prompt="hello", images=[str(img1), str(img2)])["task"] == "fl2va"
    assert op.validate_request(prompt="hello", task="l2va", images=[str(img2)])["keyframe_indices"] == (-1,)
    assert op.validate_request(prompt="hello", workflow="fl2va", image=str(img1))["task"] == "i2va"

    with pytest.raises(ValueError, match="non-empty"):
        op.validate_request(prompt="   ")
    with pytest.raises(ValueError, match="at least 4"):
        op.validate_request(prompt="hello", references=[str(img1)], num_inference_steps=3, endpoint_mode="meanflow")
    with pytest.raises(ValueError, match="not both"):
        op.validate_request(prompt="hello", images=[str(img1)], references=[str(img2)])
    with pytest.raises(ValueError, match="image references"):
        op.validate_request(prompt="hello", references=[{"kind": "video", "path": "clip.mp4"}])
    with pytest.raises(ValueError, match="multiple of 32"):
        op.validate_request(prompt="hello", task="t2va", height=60, width=64, endpoint_mode="none")
    with pytest.raises(ValueError, match="together"):
        op.validate_request(prompt="hello", task="t2va", height=64, endpoint_mode="none")
    with pytest.raises(ValueError, match="seed"):
        op.validate_request(prompt="hello", task="t2va", seed=-1, endpoint_mode="none")


def test_from_pretrained_loads_h3_tiny_transformer_and_skips_base_dit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel_transformer = SimpleNamespace(endpoint_mode="meanflow")
    captured_loader: dict[str, Any] = {}
    captured_base: dict[str, Any] = {}

    def fake_load_h3_tiny(weights_path: Path, *, endpoint_mode: str | None, device: str) -> Any:
        captured_loader.update(weights_path=weights_path, endpoint_mode=endpoint_mode, device=device)
        return sentinel_transformer

    def fake_base_from_pretrained(cls, *, model_path, required_components, device, model_id, component_devices=None):
        del cls, device, model_id
        captured_base.update(
            model_path=model_path,
            required_components=dict(required_components),
            component_devices=dict(component_devices or {}),
        )
        return SimpleNamespace(
            transformer=required_components["transformer"],
            video_vae=None,
            audio_vae=None,
            text_encoder=None,
            tokenizer=None,
            device="cpu",
            component_devices={"transformer": "cpu"},
        )

    monkeypatch.setattr(h3_tiny_module, "load_h3_tiny_transformer", fake_load_h3_tiny)
    monkeypatch.setattr(
        h3_tiny_module.NativeMiniMaxH3Pipeline,
        "from_pretrained",
        classmethod(fake_base_from_pretrained),
    )
    monkeypatch.setattr(H3TinyPipeline, "_load_processor", staticmethod(lambda _root: object()))

    pipe = H3TinyPipeline.from_pretrained(
        weights_path=tmp_path / "weights",
        base_model_path=tmp_path / "base",
        variant="17b-fewstep-ref2va",
        device="cpu",
    )

    assert pipe.MODEL_ID == "h3-tiny"
    assert pipe.variant == "17b-fewstep-ref2va"
    assert pipe.workflow == "ref2va"
    assert pipe.endpoint_mode == "meanflow"
    assert captured_loader["weights_path"] == (tmp_path / "weights").resolve()
    assert captured_loader["device"] == "cpu"
    assert captured_base["required_components"]["transformer"] is sentinel_transformer
    assert captured_base["required_components"]["partition"] == "Ref2VA"

    # FL2VA variant routes to FL2VA partition
    pipe_fl = H3TinyPipeline.from_pretrained(
        weights_path=tmp_path / "weights_fl",
        base_model_path=tmp_path / "base",
        variant="17b-pruned-fl2va",
        device="cpu",
    )
    assert pipe_fl.workflow == "fl2va"
    assert captured_base["required_components"]["partition"] == "FL2VA"


def test_h3_tiny_pipeline_end_to_end_ref2va_and_fl2va_and_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _tiny_h3_tiny_model(endpoint_mode="meanflow")
    seen_endpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_forward = model.forward

    def spy_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        seen_endpoints.append((kwargs["timestep"].detach().clone(), kwargs["endpoint_timestep"].detach().clone()))
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", spy_forward)

    written: dict[str, Any] = {}

    def fake_write_mp4(*, video: torch.Tensor, audio: torch.Tensor | None, output_path: Path, fps: int) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        written.update(
            video_shape=tuple(video.shape), audio_shape=None if audio is None else tuple(audio.shape), fps=fps
        )
        return str(output_path)

    monkeypatch.setattr(H3TinyPipeline, "_write_mp4_with_audio", staticmethod(fake_write_mp4))

    ref_img = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (120, 80, 40)).save(ref_img)

    pipe_ref = H3TinyPipeline(
        transformer=model,
        video_vae=_StubVideoVAE(),
        audio_vae=_StubAudioVAE(),
        text_encoder=_StubTextEncoder(),
        tokenizer=_stub_tokenizer,
        processor=object(),
        device="cpu",
        variant="17b-fewstep-ref2va",
    )
    out_file = tmp_path / "out" / "clip.mp4"
    result = pipe_ref(
        prompt="<Picture 1> Test scene",
        references=[str(ref_img)],
        num_frames=5,
        height=32,
        width=32,
        num_inference_steps=4,
        seed=7,
        output_path=out_file,
        return_dict=True,
    )
    assert len(seen_endpoints) == 4
    # Every step passes r >= t, and step 0 has distinct endpoints for video vs audio
    for ts, rs in seen_endpoints:
        assert bool((rs >= ts).all())
    assert len(seen_endpoints[0][0]) >= 3
    assert result["model_id"] == "h3-tiny"
    assert result["variant"] == "17b-fewstep-ref2va"
    assert result["endpoint_mode"] == "meanflow"
    assert result["inference_steps"] == 4
    assert result["artifact_path"] == str(out_file)
    assert out_file.is_file()
    assert written["fps"] == 24
    assert written["video_shape"] == (1, 3, 2, 32, 32)
    assert written["audio_shape"] is not None

    # Mismatched workflow rejected
    with pytest.raises(ValueError, match="FL2VA"):
        pipe_ref(prompt="no refs", task="t2va", num_inference_steps=4)

    # FL2VA pipeline runs t2va & fl2va
    model_fl = _tiny_h3_tiny_model(endpoint_mode="none")
    pipe_fl = H3TinyPipeline(
        transformer=model_fl,
        video_vae=_StubVideoVAE(),
        audio_vae=_StubAudioVAE(),
        text_encoder=_StubTextEncoder(),
        tokenizer=_stub_tokenizer,
        processor=object(),
        device="cpu",
        variant="17b-pruned-fl2va",
    )
    fl_res = pipe_fl(
        prompt="keyframe clip",
        images=[str(ref_img), str(ref_img)],
        num_frames=5,
        height=32,
        width=32,
        num_inference_steps=2,
        seed=5,
        return_dict=True,
    )
    assert fl_res["task"] == "fl2va"
    assert fl_res["endpoint_mode"] == "none"
    with pytest.raises(ValueError, match="Ref2VA"):
        pipe_fl(prompt="refs on fl2va", references=[str(ref_img)], num_inference_steps=2)


def test_catalog_binding_profile_environment_and_inference_spec_resolve() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog = yaml.safe_load((repo_root / "worldfoundry/data/models/catalog/video/h3-tiny.yaml").read_text())
    env_doc = yaml.safe_load(
        (repo_root / "worldfoundry/data/models/runtime/environments/video/h3-tiny.yaml").read_text()
    )

    assert catalog["id"] == "h3-tiny"
    assert catalog["pipeline_binding"] == "h3-tiny"
    assert catalog["runtime_profile"] == "runtime-profile:h3-tiny"
    assert catalog["source"]["revision"] == "a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c"
    assert len(catalog["checkpoints"]) == 6
    assert env_doc["model_id"] == "h3-tiny"

    for identifier in ("h3-tiny", "minimax-h3-tiny", "h3-tiny-17b-fewstep", "h3-tiny-14b-okd"):
        binding = resolve_pipeline_binding(identifier)
        assert binding.binding_id == "h3-tiny"
        assert binding.pipeline_target == "worldfoundry.pipelines.h3_tiny.pipeline_h3_tiny:H3TinyPipeline"

    profile = load_runtime_profile("h3-tiny", check_conda_env_exists=False)
    assert profile.model_id == "h3-tiny"
    assert profile.artifact_filename == "h3-tiny.mp4"

    assert H3_TINY_INFERENCE_SPEC.model_family_id == "h3-tiny"
    assert H3_TINY_INFERENCE_SPEC.default_variant_id == "17b-fewstep-ref2va"
    assert set(v.variant_id for v in H3_TINY_INFERENCE_SPEC.variants) == set(H3_TINY_VARIANTS)
    assert [t.task_id for t in H3_TINY_INFERENCE_SPEC.tasks] == ["ref2va", "t2va", "i2va", "l2va", "fl2va"]
