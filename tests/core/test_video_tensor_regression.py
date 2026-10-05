"""Numerical video/world contracts using real small operators and analytic flows.

Requires CPU Torch and einops; no weights, tokenizers, diffusers or CUDA. These
protect shared math, not the quality or every variant of a full trained model.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserOutput,
    DiffusionRequest,
    ModalityState,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx import component as ltx_codec
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoDecoder
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import CausalConv3d
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    install_static_cross_kv_cache,
    reset_static_cross_kv,
)
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.wan import (
    WanFlowMatchEulerScheduler,
    add_flow_noise,
    flow_prediction_to_x0,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def tiny_ltx_decoder():
    """Real noisy VAE with all three upsampling blocks and small channel widths."""
    generator = torch.Generator().manual_seed(123)
    decoder = VideoDecoder(
        base_channels=4, patch_size=4, timestep_conditioning=True,
        decoder_blocks=[("res_x", {"num_layers": 1, "inject_noise": True}),
                        ("compress_all", {"multiplier": 2}),
                        ("compress_all", {"multiplier": 2}),
                        ("compress_all", {"multiplier": 2})],
    ).eval()
    # These tables/statistics normally come from checkpoints, not initialization.
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.uniform_(-0.05, 0.05, generator=generator)
        decoder.per_channel_statistics.get_buffer("std-of-means").fill_(1)
        decoder.per_channel_statistics.get_buffer("mean-of-means").zero_()
    return decoder


@pytest.mark.parametrize("kind", ["video", "audio_video", "tensor"])
@pytest.mark.parametrize("tiled", [False, True])
def test_ltx_stochastic_vae_decode_is_request_seeded_and_matches_controlled_legacy_math(kind, tiled, monkeypatch):
    decoder = tiny_ltx_decoder()
    generator = torch.Generator().manual_seed(51)
    latent = torch.randn(1, 128, 2, 3, 4, generator=generator)
    original = latent.clone()
    tiling = ltx_codec.TilingConfig.default() if tiled else None

    def request(seed):
        return DiffusionRequest(prompt="A", height=96, width=128, num_frames=9,
                                sampling=SamplingConfig(seed=seed))

    def state(value):
        return ModalityState(latent=value, denoise_mask=torch.ones_like(value),
                             positions=torch.zeros(1), clean_latent=torch.zeros_like(value))

    states = {"video": state(VideoLatentPatchifier(1).patchify(latent))}
    wrapper = torch.nn.Module()
    wrapper.decoder = decoder
    if kind == "tensor":
        codec = ltx_codec.LTXTensorVideoCodec(None, wrapper, tiling=tiling)

        def decode(seed):
            return codec.decode(latent, request(seed))

    elif kind == "audio_video":
        _, shape = ltx_codec.LTXMediaDecoder._shapes(request(43))
        audio = torch.randn(shape.to_torch_shape(), generator=generator)
        states["audio"] = state(AudioPatchifier(1).patchify(audio))
        waveform = torch.randn(1, 128, generator=generator)
        monkeypatch.setattr(ltx_codec, "decode_audio", lambda *args: SimpleNamespace(
            waveform=waveform, sampling_rate=16000
        ))
        codec = ltx_codec.LTXMediaDecoder(wrapper, SimpleNamespace(decoder=None),
                                        SimpleNamespace(vocoder=None), compute_dtype=torch.float32, tiling=tiling)

        def decode(seed):
            output = codec.decode_modalities(states, request(seed))
            torch.testing.assert_close(output["audio"], waveform, atol=0, rtol=0)
            assert output["audio_sampling_rate"] == 16000
            return output["video"]

    else:
        codec = ltx_codec.LTXVideoMediaDecoder(wrapper, tiling=tiling)

        def decode(seed):
            return codec.decode_modalities(states, request(seed))["video"]

    with torch.no_grad():
        global_before = torch.random.get_rng_state()
        first = decode(43)
        torch.testing.assert_close(torch.random.get_rng_state(), global_before, atol=0, rtol=0)
        other = decode(44)
        assert not torch.equal(first, other)
        # Global random draws and another request cannot alter repeat decoding.
        torch.randn(17)
        global_before = torch.random.get_rng_state()
        repeated = decode(43)
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), global_before, atol=0, rtol=0)
        # Compare to the unchanged VAE using the same global RNG state the old
        # adapter would have needed; only random-number ownership changed.
        with torch.random.fork_rng():
            torch.manual_seed(43)
            if kind == "tensor":
                legacy = (torch.cat(list(decoder.tiled_decode(latent, tiling)), dim=2) if tiled
                          else decoder(latent)).clamp(-1, 1)
            else:
                legacy = torch.cat(list(decoder.decode_video(latent, tiling)), dim=0)
        torch.testing.assert_close(first, legacy, atol=0, rtol=0)
        torch.testing.assert_close(latent, original, atol=0, rtol=0)
        assert torch.isfinite(first).all()
        assert first.shape == ((1, 3, 9, 96, 128) if kind == "tensor" else (9, 96, 128, 3))


def tiny_wan():
    torch.manual_seed(13)
    model = WanModel(
        dim=48, in_dim=2, ffn_dim=64, out_dim=2, text_dim=16, freq_dim=16,
        patch_size=(1, 2, 2), num_heads=4, num_layers=2, eps=1e-6,
        has_image_input=False, require_vae_embedding=False, require_clip_embedding=False,
    ).eval()
    model.set_attention_compatibility_mode(True)
    return model


@pytest.mark.parametrize("shape", [(1, 2, 3, 4, 6), (2, 2, 1, 6, 4)])
@pytest.mark.parametrize("strategy", ["packed", "split", "auto"])
def test_real_wan_fused_transformer_matches_dense_across_requests(shape, strategy):
    reference = tiny_wan()
    optimized = copy.deepcopy(reference)
    assert fuse_qkv_projections(optimized, strategy=strategy, split_threshold=8) == 2
    cache = install_static_cross_kv_cache(optimized)
    # A transposed spatial view catches accidental layout assumptions in fusion.
    latents = torch.randn(shape).transpose(-1, -2)
    context_a = torch.randn(shape[0], 5, 16)
    context_b = torch.randn(shape[0], 7, 16)
    timestep = torch.full((shape[0],), 500.0)
    original = latents.clone()
    with torch.no_grad():
        for context in (context_a, context_b, context_a):
            reset_static_cross_kv(cache)
            for time in (timestep, timestep * 0.3):
                expected = reference(latents, time, context)
                actual = optimized(latents, time, context)
                assert actual.shape == latents.shape
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        # Reuse the same context storage with changed contents in a new request.
        context_a.add_(0.25)
        reset_static_cross_kv(cache)
        torch.testing.assert_close(
            optimized(latents, timestep, context_a), reference(latents, timestep, context_a),
            atol=2e-6, rtol=2e-5,
        )
    torch.testing.assert_close(latents, original, atol=0, rtol=0)
    assert cache.report()["lifetime_hits"] > 0


@pytest.mark.parametrize("chunk_sizes", [(1, 1, 3, 2), (2, 3, 2), (4, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wan_vae_causal_chunks_match_independent_full_convolution(chunk_sizes, dtype):
    torch.manual_seed(19)
    conv = CausalConv3d(2, 3, kernel_size=3, padding=1).to(dtype)
    signal = torch.randn(1, 2, sum(chunk_sizes), 4, 5, dtype=dtype)
    expected = F.conv3d(F.pad(signal, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
    chunks, offset = [], 0
    for size in chunk_sizes:
        # Exactly the previous two input frames are needed by this kernel.
        history = signal[:, :, max(0, offset - 2):offset] if offset else None
        chunks.append(conv(signal[:, :, offset:offset + size], cache_x=history))
        offset += size
    torch.testing.assert_close(torch.cat(chunks, dim=2), expected)
    changed_future = signal.clone()
    changed_future[:, :, 4:] += 100
    torch.testing.assert_close(conv(changed_future)[:, :, :4], expected[:, :, :4], atol=0, rtol=0)
    # A new clip must start with fresh history, even when the operator is reused.
    new_clip = torch.randn_like(signal)
    independent = F.conv3d(F.pad(new_clip, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
    torch.testing.assert_close(conv(new_clip), independent)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flow_noise_and_x0_recover_known_clean_video_per_sample(batch, dtype):
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn(batch, 2, 3, 2, 4, generator=generator, dtype=dtype)
    noise = torch.randn(clean.shape, generator=generator, dtype=dtype)
    sigma = torch.linspace(0, 1, batch, dtype=dtype) if batch > 1 else torch.tensor(0.4, dtype=dtype)
    expected = torch.stack([(1 - weight) * x + weight * n for x, n, weight in
                            zip(clean, noise, sigma.reshape(-1).expand(batch))])
    mixed = add_flow_noise(clean, noise, sigma)
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(flow_prediction_to_x0(noise - clean, mixed, sigma), clean)


@pytest.mark.parametrize("steps", [1, 4, 7])
@pytest.mark.parametrize("shift", [1.0, 5.0])
def test_unipc_constant_flow_recovers_analytic_endpoint_and_resets_history(steps, shift):
    generator = torch.Generator().manual_seed(31)
    initial = torch.randn(1, 2, 3, 2, 2, generator=generator)
    velocity = torch.full_like(initial, 0.125)
    scheduler = FlowUniPCMultistepScheduler(shift=1, solver_order=2)
    # The native training grid starts at 999/1000, rather than sigma=1.
    sigma_start = shift * 0.999 / (1 + (shift - 1) * 0.999)
    results = []
    for _ in range(2):
        scheduler.set_timesteps(steps, shift=shift, device="cpu")
        sample = initial.clone()
        for timestep in scheduler.timesteps:
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        torch.testing.assert_close(sample, initial - sigma_start * velocity, atol=2e-6, rtol=2e-6)
        assert scheduler.step_index == steps
        results.append(sample)
    torch.testing.assert_close(*results, atol=0, rtol=0)


class AnalyticComponents:
    def __init__(self):
        self.finalized = []
        self.calls = []

    def encode(self, request, *, device, dtype):
        value = 2.0 if request.prompts == ("A",) else 4.0
        return Conditioning(
            positive={"velocity": value}, negative={"velocity": 0.25},
            shared={"fail_denoise": request.inputs.get("fail_denoise", False)},
        )

    def initialize(self, request, *, generator, device, dtype):
        return torch.randn(1, 2, request.num_frames, 2, 2, generator=generator, device=device, dtype=dtype)

    def __call__(self, model_input):
        self.calls.append((model_input.request_id, model_input.branch))
        if model_input.conditioning.get("fail_denoise"):
            raise RuntimeError("injected denoising failure")
        return DenoiserOutput(sample=torch.full_like(model_input.latents, model_input.conditioning["velocity"]))

    def decode(self, latents, request):
        if request.inputs.get("fail_decode"):
            raise RuntimeError("injected decoding failure")
        return latents.clone()

    def end_request(self, request_id, *, error=None):
        self.finalized.append((request_id, error))


def analytic_runner(mode="standard"):
    parts = AnalyticComponents()
    runner = NativeDiffusionRunner(
        model_id="analytic-video", device="cpu", dtype=torch.float32, guidance_mode=mode,
        components=RunnerComponents(parts, parts, parts, WanFlowMatchEulerScheduler(shift=5), parts),
    )
    return runner, parts


def request(prompt="A", *, seed=37, scale=3.0, inputs=None):
    return DiffusionRequest(
        prompt=prompt, negative_prompt="negative", height=16, width=16, num_frames=3,
        sampling=SamplingConfig(seed=seed, num_inference_steps=4, guidance_scale=scale),
        inputs=inputs or {},
    )


@pytest.mark.parametrize("mode", ["standard", "positive"])
@pytest.mark.parametrize("scale", [0.0, 1.0, 3.0])
def test_native_cfg_sampler_matches_closed_form_not_another_runner(mode, scale):
    runner, parts = analytic_runner(mode)
    output = runner.run(request(scale=scale))
    initial = torch.randn(output.latents.shape, generator=torch.Generator().manual_seed(37))
    guided = 0.25 + scale * (2.0 - 0.25) if mode == "standard" else 2.0 + scale * (2.0 - 0.25)
    torch.testing.assert_close(output.latents, initial - guided, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.sample, output.latents, atol=0, rtol=0)
    assert len(parts.finalized) == 1 and parts.finalized[0][1] is None


@pytest.mark.parametrize("failure", ["fail_denoise", "fail_decode"])
def test_native_sampler_resident_a_b_a_and_failure_do_not_leak_state_or_global_rng(failure):
    runner, parts = analytic_runner()
    torch.manual_seed(101)
    global_rng = torch.random.get_rng_state().clone()
    first = runner.run(request()).latents.clone()
    other = runner.run(request("B")).latents
    assert not torch.equal(first, other)
    with pytest.raises(RuntimeError, match="injected") as error:
        runner.run(request(inputs={failure: True}))
    repeated = runner.run(request()).latents
    fresh, _ = analytic_runner()
    torch.testing.assert_close(repeated, first, atol=0, rtol=0)
    torch.testing.assert_close(repeated, fresh.run(request()).latents, atol=0, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, atol=0, rtol=0)
    assert len(parts.finalized) == 4
    assert parts.finalized[2][1] is error.value
    assert len({name for name, _ in parts.finalized}) == 4
    assert all(err is None for index, (_, err) in enumerate(parts.finalized) if index != 2)


def native_adapter():
    from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline

    adapter = object.__new__(NativeVisualDiffusionPipeline)
    adapter.model_id = "video-boundary-test"
    adapter.process = lambda **kwargs: kwargs
    requests = []

    def infer(value):
        requests.append(value)
        return SimpleNamespace(sample=torch.zeros(1, 3, 1, 2, 2), latents=torch.zeros(1), metadata={})

    adapter.native_pipeline = infer
    return adapter, requests


@pytest.mark.parametrize("field", ["height", "width", "num_frames", "num_inference_steps", "fps"])
@pytest.mark.parametrize("value", [0, -1])
def test_public_native_video_adapter_rejects_explicit_invalid_values_without_inference(field, value):
    adapter, requests = native_adapter()
    with pytest.raises(ValueError, match="must be positive"):
        adapter(prompt="A", return_dict=True, **{field: value})
    assert requests == []


def test_public_native_video_adapter_applies_defaults_only_for_none():
    adapter, requests = native_adapter()
    adapter(prompt="A", height=None, width=None, num_frames=None, num_inference_steps=None, fps=None, return_dict=True)
    value = requests[0]
    assert (value.height, value.width, value.num_frames, value.sampling.num_inference_steps, value.inputs["fps"]) == (
        adapter.DEFAULT_HEIGHT, adapter.DEFAULT_WIDTH, adapter.DEFAULT_NUM_FRAMES,
        adapter.DEFAULT_NUM_INFERENCE_STEPS, adapter.DEFAULT_FPS,
    )
