"""Numerical and stream-scheduling regressions for MG2's optimization boundary."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import causal_action_21 as wan
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
    CausalWanSelfAttention,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
    CausalInferenceSession,
)


class _CudaSchedulingTensor(torch.Tensor):
    """CPU arithmetic with CUDA scheduling enabled, without a CUDA dependency."""

    @property
    def is_cuda(self):
        return True


class _AnalyticalGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(local_attn_size=2)

    def get_scheduler(self):
        return SimpleNamespace()

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, kv_cache, **kwargs):
        kv_cache[0]["calls"] += 1
        return None, noisy_image_or_video * 0.5 + conditional_dict["cond_concat"] * 0.25


class _AnalyticalDecoder(torch.nn.Module):
    def forward(self, latent, *cache):
        count = int(cache[0] or 0) + 1
        return latent[:, :, :3].float() * 2, [count]


def _small_session(device="cpu"):
    args = SimpleNamespace(denoising_step_list=[1], warp_denoising_step=False, num_frame_per_block=1, context_noise=0)
    pipeline = CausalInferencePipeline(args, generator=_AnalyticalGenerator(), vae_decoder=_AnalyticalDecoder())
    condition = {
        "cond_concat": torch.ones(1, 16, 3, 2, 2, device=device),
        "visual_context": torch.zeros(1, 1, 2, device=device),
        "keyboard_cond": torch.zeros(1, 9, 7, device=device),
    }
    session = CausalInferenceSession(
        conditional_dict=condition,
        mode="templerun",
        batch_size=1,
        dtype=torch.float32,
        device=condition["cond_concat"].device,
        kv_cache=[{"calls": 0}],
        mouse_kv_cache=[],
        keyboard_kv_cache=[],
        crossattn_cache=[],
        vae_cache=[None],
    )
    return pipeline, session


@pytest.mark.parametrize("profile", [False, True])
def test_timing_uses_only_opt_in_events_on_the_callers_stream(monkeypatch, profile):
    events = []
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.recorded_on = None
            self.waits = 0
            events.append(self)

        def record(self, selected_stream):
            self.recorded_on = selected_stream

        def synchronize(self):
            self.waits += 1

        def elapsed_time(self, other):
            assert events[-1].waits == 1
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda **kwargs: pytest.fail("MG2 must not synchronize the entire device")
    )
    pipeline, session = _small_session()
    noise = torch.ones(1, 16, 1, 2, 2).as_subclass(_CudaSchedulingTensor)
    block = pipeline.generate_next_block(session, noise, profile=profile)
    torch.testing.assert_close(block.latent.as_subclass(torch.Tensor), torch.full((1, 16, 1, 2, 2), 0.75))
    torch.testing.assert_close(block.video.as_subclass(torch.Tensor), torch.full((1, 1, 3, 2, 2), 1.5))
    assert session.current_start_frame == 1
    assert session.kv_cache[0]["calls"] == 2  # denoising plus clean-cache commit
    assert session.vae_cache == [1]
    assert len(events) == (3 if profile else 0)
    if profile:
        assert [event.waits for event in events] == [0, 0, 1]
        assert all(event.recorded_on is stream for event in events)
        assert block.model_ms == block.decode_ms == 2.0
    else:
        assert block.model_ms is block.decode_ms is session.last_model_ms is session.last_decode_ms is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
@pytest.mark.gpu
def test_nondefault_stream_returns_correct_results_without_device_sync(monkeypatch):
    pipeline, session = _small_session("cuda")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda **kwargs: pytest.fail("unexpected device barrier"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        noise = torch.full((1, 16, 1, 2, 2), 1.0, device="cuda")
        block = pipeline.generate_next_block(session, noise, profile=False)
        completed = torch.cuda.Event()
        completed.record(stream)
    completed.synchronize()
    torch.testing.assert_close(block.video.cpu(), torch.full((1, 1, 3, 2, 2), 1.5))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"runtime_options": []}, "runtime_options"),
        ({"fuse_qkv": "true"}, "fuse_qkv"),
        ({"qkv_strategy": "adaptive"}, "qkv_strategy"),
        ({"qkv_split_threshold": 1.5}, "qkv_split_threshold"),
        ({"compile": "true"}, "compile"),
        ({"compile": True, "compile_dynamic": "false"}, "compile_dynamic"),
        ({"cuda_graph": True}, "cuda_graph"),
        ({"sequence_parallel": True}, "sequence_parallel"),
        ({"offload": "block"}, "offload"),
        ({"attention_backend": "sage"}, "attention_backend"),
        ({"vae_channels_last_3d": "true"}, "vae_channels_last_3d"),
        ({"overlap_vae_decode": "true"}, "overlap_vae_decode"),
        ({"runtime_options": {"fuse_qqkv": True}}, "fuse_qqkv"),
    ],
)
def test_invalid_or_unsupported_options_fail_before_loading(monkeypatch, options, match):
    monkeypatch.setattr(
        runtime.OmegaConf,
        "load",
        lambda path: pytest.fail("options must be validated before config/checkpoint loading"),
    )
    with pytest.raises((ValueError, TypeError), match=match):
        runtime.MatrixGame2Runtime.from_pretrained("missing", device="cpu", weight_dtype=torch.float32, **options)


def _install_tiny_loader(monkeypatch, decoder=None):
    class Generator(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = CausalWanSelfAttention(32, 2)

    class Pipeline(torch.nn.Module):
        def __init__(self, args, generator, vae_decoder):
            super().__init__()
            self.generator = generator
            self.vae_decoder = vae_decoder

    import worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline as pipeline_module
    import worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.wan_wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "WanDiffusionWrapper", Generator)
    monkeypatch.setattr(pipeline_module, "CausalInferencePipeline", Pipeline)
    prefix = "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime"
    decoder_module = ModuleType(prefix + ".utils.vae_runtime.vae_block3")
    decoder_module.VAEDecoderWrapper = torch.nn.Identity if decoder is None else lambda: copy.deepcopy(decoder)
    monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
    vae_module = ModuleType(prefix + ".extension_modules.wanx_vae.wanx_vae")
    vae_module.get_wanx_vae_wrapper = lambda *args: torch.nn.Identity()
    monkeypatch.setitem(sys.modules, vae_module.__name__, vae_module)
    monkeypatch.setattr(
        runtime.OmegaConf, "load", lambda path: OmegaConf.create({"model_kwargs": {"model_config": "tiny.pt"}})
    )
    monkeypatch.setattr(runtime, "_resolve_model_root", lambda *args: "/unused")
    monkeypatch.setattr(
        runtime.MatrixGame2Runtime, "_resolve_checkpoint_path", lambda **kwargs: "/unused/model.safetensors"
    )
    monkeypatch.setattr(
        runtime.torch, "load", lambda *args, **kwargs: {} if decoder is None else copy.deepcopy(decoder.state_dict())
    )
    monkeypatch.setattr(runtime, "_enable_torch_compile", lambda: False)
    torch.manual_seed(91)
    restored = Generator()
    monkeypatch.setattr(runtime, "load_file", lambda path: copy.deepcopy(restored.state_dict()))
    return restored


@pytest.mark.parametrize("conv2d,conv3d", [(True, False), (False, True), (True, True)])
def test_loader_uses_shared_vae_layout_transform_with_numeric_parity(monkeypatch, conv2d, conv3d):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = torch.nn.Conv2d(4, 6, 3, padding=1)
            self.decoder = torch.nn.Conv3d(6, 8, 3, padding=1)

        def forward(self, value):
            return self.decoder(self.conv2(value).unsqueeze(2))

    torch.manual_seed(95)
    restored = Decoder().half().eval()
    _install_tiny_loader(monkeypatch, decoder=restored)
    loaded = runtime.MatrixGame2Runtime.from_pretrained(
        "unused",
        device="cpu",
        weight_dtype=torch.float32,
        vae_channels_last=conv2d,
        vae_channels_last_3d=conv3d,
    )
    decoder = loaded.pipeline.vae_decoder
    value = torch.randn(1, 4, 4, 4).half()
    with torch.no_grad():
        torch.testing.assert_close(decoder(value), restored(value))
    assert decoder.conv2.weight.is_contiguous(memory_format=torch.channels_last) is conv2d
    assert decoder.decoder.weight.is_contiguous(memory_format=torch.channels_last_3d) is conv3d
    report = loaded.pipeline._worldfoundry_vae_convolution_layout
    assert report.conv2d_converted == int(conv2d)
    assert report.conv3d_converted == int(conv3d)
    assert loaded._worldfoundry_applied_optimizations.effective["vae_convolution_layout"]["conv3d_converted"] == int(
        conv3d
    )


@pytest.mark.parametrize("fuse", [False, True])
def test_loader_applies_fusion_after_restoring_checkpoint_with_direct_precedence(monkeypatch, fuse):
    restored = _install_tiny_loader(monkeypatch)
    loaded = runtime.MatrixGame2Runtime.from_pretrained(
        "unused",
        device="cpu",
        weight_dtype=torch.float32,
        runtime_options={"fuse_qkv": not fuse, "qkv_strategy": "packed"},
        fuse_qkv=fuse,
        qkv_strategy="split",
        qkv_split_threshold=8,
    )
    model = loaded.pipeline.generator.model
    if fuse:
        torch.testing.assert_close(
            model.qkv.weight, torch.cat([restored.model.q.weight, restored.model.k.weight, restored.model.v.weight])
        )
        assert model.forward.__func__ is CausalWanSelfAttention.forward
    else:
        assert not hasattr(model, "qkv")
        torch.testing.assert_close(model.q.weight, restored.model.q.weight)
    receipt = loaded._worldfoundry_applied_optimizations
    assert receipt.requested["fuse_qkv"] is fuse
    if fuse:
        assert receipt.effective["fuse_qkv_blocks"] == 1
        assert receipt.effective["qkv_strategy"] == "split"


def test_loader_quantization_keeps_a_truthful_dense_fallback_receipt(monkeypatch):
    restored = _install_tiny_loader(monkeypatch)
    loaded = runtime.MatrixGame2Runtime.from_pretrained(
        "unused",
        device="cpu",
        weight_dtype=torch.float32,
        fuse_qkv=True,
        quantization={"mode": "fp8", "min_features": 1, "keep_dense_fallback": True},
    )
    hidden = torch.randn(1, 5, 32)
    model = loaded.pipeline.generator.model
    with torch.no_grad():
        torch.testing.assert_close(
            model.qkv(hidden),
            torch.cat([restored.model.q(hidden), restored.model.k(hidden), restored.model.v(hidden)], dim=-1),
        )
    receipt = loaded._worldfoundry_applied_optimizations
    assert receipt.requested["quantization"] == "fp8"
    assert receipt.effective["quantization"] == "fp8-wrapper-installed (runtime-pending)"
    assert receipt.effective["quantization_dense_fallback_retained"] is True


def test_bfloat16_generator_placement_preserves_float16_vae_checkpoint_values(monkeypatch):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = torch.nn.Conv2d(4, 6, 3, padding=1)

    torch.manual_seed(97)
    restored = Decoder().half().eval()
    _install_tiny_loader(monkeypatch, decoder=restored)
    loaded = runtime.MatrixGame2Runtime.from_pretrained("unused", device="cpu", weight_dtype=torch.bfloat16)
    assert loaded.pipeline.generator.model.q.weight.dtype is torch.bfloat16
    assert loaded.pipeline.vae_decoder.conv2.weight.dtype is torch.float16
    torch.testing.assert_close(loaded.pipeline.vae_decoder.conv2.weight, restored.conv2.weight, atol=0, rtol=0)
    torch.testing.assert_close(loaded.pipeline.vae_decoder.conv2.bias, restored.conv2.bias, atol=0, rtol=0)


@pytest.mark.parametrize("requested", [False, True])
def test_loader_decode_overlap_is_explicit_and_auditable(monkeypatch, requested):
    _install_tiny_loader(monkeypatch)
    loaded = runtime.MatrixGame2Runtime.from_pretrained(
        "unused",
        device="cpu",
        weight_dtype=torch.float32,
        overlap_vae_decode=requested,
    )
    assert loaded.pipeline.overlap_vae_decode is requested
    receipt = loaded._worldfoundry_applied_optimizations
    assert receipt.requested["overlap_vae_decode"] is requested
    if requested:
        assert receipt.effective["overlap_vae_decode"] == "synchronous (non-CUDA fallback)"
        assert any("overlap_vae_decode" in reason for reason in receipt.fallbacks)


def test_loader_compiles_bound_model_forward_and_preserves_checkpoint_type(monkeypatch):
    restored = _install_tiny_loader(monkeypatch)

    def cpu_flex(*, query, key, value, block_mask):
        positions = torch.arange(query.shape[2])
        allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] < 8)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)

    monkeypatch.setattr(wan, "flex_attention", cpu_flex)
    loaded = runtime.MatrixGame2Runtime.from_pretrained(
        "unused",
        device="cpu",
        weight_dtype=torch.float32,
        fuse_qkv=True,
        compile=True,
        compile_backend="eager",
    )
    model = loaded.pipeline.generator.model
    assert type(model) is CausalWanSelfAttention
    assert model._worldfoundry_compile_runtime["wrapper_installed"] is True
    assert loaded._worldfoundry_applied_optimizations.effective["compile"] == "compile-wrapper-installed (lazy)"
    hidden = torch.randn(1, 8, 32)
    kwargs = dict(
        seq_lens=torch.tensor([8]), grid_sizes=torch.tensor([1, 1, 8]), freqs=wan.rope_params(8, 16), block_mask=None
    )
    with torch.no_grad():
        expected = restored.model(hidden, **kwargs)
        for _ in range(2):
            torch.testing.assert_close(model(hidden, **kwargs), expected, atol=2e-6, rtol=2e-5)
    assert model._worldfoundry_compile_runtime["calls"] == 2
    assert model._worldfoundry_compile_runtime["failures"] == 0
