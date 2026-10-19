# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Image, mask, timestep and request ownership through canonical Wan batching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from tests.base_models.diffusion_model.test_request_batching import Conditioner, Decoder, requests
from worldfoundry.base_models.diffusion_model.contracts import Conditioning, DiffusionRequest, LatentInitialization
from worldfoundry.base_models.diffusion_model.extensions.frozen_mask import FrozenLatentMaskExtension
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import Wan22DualExpertDenoiser, WanDenoiser
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanImageToVideoLatentInitializer,
    WanTextImageToVideoLatentInitializer,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import AttentionModule, WanModel
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import install_static_cross_kv_cache
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.runners.base import (
    NativeDiffusionRunner,
    RunnerComponents,
    Wan22DualExpertGuidanceRunner,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowUniPCScheduler

pytestmark = pytest.mark.unit


class Codec:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.fail_at: int | None = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pixels = images
        self.calls.append(tuple(pixels.shape))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("image encoding failed")
        sampled = pixels[:, :, ::4].mean(1, keepdim=True).repeat(1, 4, 1, 1, 1)
        return F.avg_pool3d(sampled, (1, 2, 2))


class ImageConditioner(Conditioner):
    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        condition = super().encode(request, device=device, dtype=dtype)
        image: Any = request.inputs["images"]
        pixel = float(image.getpixel((0, 0))[0]) / 255
        return replace(
            condition,
            shared={"clip_feature": torch.full((request.batch_size, 257, 1280), pixel, device=device, dtype=dtype)},
        )


def image_pipeline(
    kind: str,
    *,
    device: str = "cpu",
    graph: bool = False,
    compute_dtype: torch.dtype = torch.float32,
) -> NativeDiffusionPipeline:
    torch.manual_seed(712)
    masked = kind.startswith("masked")
    model = (
        WanModel(
            dim=48,
            in_dim=4 if masked else 12,
            ffn_dim=64,
            out_dim=4,
            text_dim=16,
            freq_dim=16,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=4,
            num_layers=2,
            has_image_input=kind == "clip",
            per_token_timestep=masked,
        )
        .eval()
        .to(device)
    )
    install_static_cross_kv_cache(model)
    initializer = (WanTextImageToVideoLatentInitializer if masked else WanImageToVideoLatentInitializer)(
        channels=4, spatial_compression=2
    )
    return NativeDiffusionPipeline(
        NativeDiffusionRunner(
            model_id="small-wan-image-batching",
            device=device,
            dtype=compute_dtype,
            components=RunnerComponents(
                denoiser=WanDenoiser(
                    model,
                    compute_dtype=compute_dtype,
                    enable_cuda_graph=graph,
                    channel_condition_key="condition_latents" if kind == "channel" else None,
                ),
                conditioner=ImageConditioner() if kind == "clip" else Conditioner(),
                latent_initializer=initializer,
                latent_encoder=Codec(),
                scheduler=WanFlowUniPCScheduler(),
                decoder=Decoder(),
            ),
            extensions=(FrozenLatentMaskExtension(),) if masked else (),
        )
    )


def image_requests(*, text_only: bool = False) -> tuple[DiffusionRequest, ...]:
    images = (Image.new("RGB", (8, 8), (32, 64, 96)), Image.new("RGB", (8, 8), (192, 160, 128)))
    return tuple(
        replace(case, inputs={} if text_only else {"images": image}) for case, image in zip(requests(), images)
    )


@pytest.mark.parametrize("kind", ["clip", "channel", "masked", "masked-text"])
def test_image_batches_match_singletons_and_reverse_order(kind: str) -> None:
    pipe = image_pipeline(kind)
    cases = image_requests(text_only=kind == "masked-text")
    expected = tuple(pipe(case) for case in cases)
    rng = torch.get_rng_state().clone()
    outputs = pipe.run_batch(cases)
    assert torch.equal(rng, torch.get_rng_state())
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output.sample, reference.sample, rtol=1e-5, atol=1e-6)
        assert output.metadata["seed"] == reference.metadata["seed"]
        if kind == "masked":
            torch.testing.assert_close(output.latents[:, :, :1], reference.latents[:, :, :1], rtol=0, atol=0)
    reversed_outputs = pipe.run_batch(tuple(reversed(cases)))
    for output, reference in zip(reversed_outputs, reversed(expected)):
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)
    if kind != "masked-text":
        changed = pipe.run_batch(
            tuple(replace(case, inputs={"images": Image.new("RGB", (8, 8), "white")}) for case in cases)
        )
        assert not torch.equal(changed[0].latents, outputs[0].latents)
    repeated = pipe.run_batch(cases)
    for output, reference in zip(repeated, outputs):
        torch.testing.assert_close(output.sample, reference.sample, rtol=0, atol=0)
    assert pipe.components.denoiser._optimization_request_windows == {}
    assert pipe.components.denoiser.model._worldfoundry_static_cross_kv.report()["entries"] == 0


def test_per_token_time_projections_keep_original_request_shapes() -> None:
    pipe = image_pipeline("masked")
    seen: list[tuple[int, ...]] = []
    model = pipe.components.denoiser.model
    handle = model.time_embedding.register_forward_pre_hook(lambda _module, args: seen.append(tuple(args[0].shape)))
    try:
        pipe.run_batch(image_requests())
    finally:
        handle.remove()
    assert seen == [(8, 16), (16, 16)] * 8


def test_batch_attention_kernels_keep_original_request_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan import model as wan

    pipe = image_pipeline("masked")
    attend = wan.flash_attention
    rows: list[int] = []

    def record(**kwargs: Any) -> torch.Tensor:
        rows.append(kwargs["q"].shape[0])
        return attend(**kwargs)

    monkeypatch.setattr(wan, "flash_attention", record)
    pipe.run_batch(image_requests())
    assert rows == [1, 2] * 32


@pytest.mark.parametrize("kind", ["channel", "masked"])
def test_image_encoding_failure_cleans_batch_and_retains_request_artifacts(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = image_pipeline(kind)
    cases = image_requests()
    codec = pipe.components.latent_encoder
    codec.fail_at = 2
    with pytest.raises(RuntimeError, match="image encoding failed"):
        pipe.run_batch(cases)
    assert pipe.components.denoiser._optimization_request_windows == {}
    codec.fail_at = None
    initializer = pipe.components.latent_initializer
    initialize: Callable[..., LatentInitialization] = initializer.initialize_with_encoder

    def with_artifacts(request: DiffusionRequest, **kwargs: Any) -> LatentInitialization:
        value = initialize(request, **kwargs)
        return replace(value, artifacts={"seed": request.sampling.seed})

    monkeypatch.setattr(initializer, "initialize_with_encoder", with_artifacts)
    expected = tuple(pipe(case) for case in cases)
    outputs = pipe.run_batch(cases)
    for output, reference in zip(outputs, expected):
        assert output.artifacts == reference.artifacts
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("branch", ["self_attn", "cross_attn"])
def test_custom_attention_is_rejected_before_image_encoding(branch: str) -> None:
    pipe = image_pipeline("masked")
    attention = getattr(pipe.components.denoiser.model.blocks[0], branch)
    attention.set_processor(object())
    with pytest.raises(ValueError, match="canonical attention processors"):
        pipe.run_batch(image_requests())
    assert pipe.components.conditioner.calls == 0 and not pipe.components.latent_encoder.calls


@pytest.mark.parametrize("kind", ["channel", "masked"])
def test_mixed_image_presence_is_rejected_before_encoding(kind: str) -> None:
    pipe = image_pipeline(kind)
    first, second = image_requests()
    with pytest.raises(ValueError, match="matching image-conditioning presence"):
        pipe.run_batch((first, replace(second, inputs={})))
    assert pipe.components.conditioner.calls == 0 and not pipe.components.latent_encoder.calls


def test_image_batch_keeps_both_experts_and_guidance_routes() -> None:
    pipe = image_pipeline("channel")
    high, low = pipe.components.denoiser, image_pipeline("channel").components.denoiser
    denoiser = Wan22DualExpertDenoiser(high, low, boundary_ratio=0.8)
    pipe.runner = Wan22DualExpertGuidanceRunner(
        model_id=pipe.model_id,
        components=replace(pipe.components, denoiser=denoiser),
        boundary_ratio=0.8,
        low_noise_guidance_scale=2,
        high_noise_guidance_scale=4,
    )
    cases = image_requests()
    expected = tuple(pipe(case) for case in cases)
    outputs = pipe.run_batch(cases)
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)
    runtime = denoiser.runtime_optimization_report()["runtime"]
    assert isinstance(runtime, dict)
    assert all(value > 0 for value in runtime["dual_expert"]["route_calls"].values())


@pytest.mark.gpu
@pytest.mark.parametrize("sizes", [(1, 1), (1, 2)])
def test_cuda_attention_preserves_request_shaped_split_kv_reductions(sizes: tuple[int, ...]) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this contract"
    attention = AttentionModule(24).eval().cuda()
    attention.set_attention_backend("torch")
    generator = torch.Generator(device="cuda").manual_seed(12)
    q = torch.randn(sum(sizes), 320, 3072, generator=generator, device="cuda")
    k = torch.randn(q.shape, generator=generator, device="cuda")
    v = torch.randn(q.shape, generator=generator, device="cuda").bfloat16()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = torch.cat(
            [attention(query, key, value) for query, key, value in zip(q.split(sizes), k.split(sizes), v.split(sizes))]
        )
        actual = attention(q, k, v, request_batch_sizes=sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["clip", "channel", "masked", "masked-text"])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_cuda_image_batch_warmup_replays_changed_images(kind: str, compute_dtype: torch.dtype) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this contract"
    pipe = image_pipeline(kind, device="cuda", graph=True, compute_dtype=compute_dtype)
    cases = image_requests(text_only=kind == "masked-text")
    expected = tuple(pipe(case) for case in cases)
    pipe.warmup([cases, tuple(reversed(cases))])
    outputs = pipe.run_batch(cases)
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)
    graph = pipe.components.denoiser.graph_report()
    assert graph["replay"] == 8
    assert graph["capture"] == graph["capture_failed"] == graph["eager"] == 0
    assert graph["capture_on_miss"] is False
    retained = outputs[0].sample.clone()
    pipe.run_batch(tuple(reversed(cases)))
    torch.testing.assert_close(outputs[0].sample, retained, rtol=0, atol=0)
