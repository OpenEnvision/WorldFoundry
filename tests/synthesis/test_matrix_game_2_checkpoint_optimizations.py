"""Opt-in numerical contracts using local MG2 checkpoint operators.

Set WORLDFOUNDRY_MG2_CHECKPOINT_ROOT to an existing Matrix-Game-2.0 directory.
This loads production block/FFN/streaming VAE weights and preserves their real
dimensions. Seeded operator inputs and controls are not a generation-quality
evaluation or a throughput benchmark; no checkpoints are downloaded.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import causal_action_21 as wan
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections, qkv_fusion_report
from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts
from worldfoundry.core.acceleration.quantization.fused_ffn import FusedFP8GELUFeedForward
from worldfoundry.core.acceleration.quantization.linear import Float8Linear, quantization_runtime_report
from worldfoundry.core.model_loading.optimize import apply_quantization_policy
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
    VAEDecoderWrapper,
)

pytestmark = pytest.mark.gpu


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


@pytest.fixture(scope="module")
def checkpoint_evidence():
    configured = os.getenv("WORLDFOUNDRY_MG2_CHECKPOINT_ROOT")
    if not configured:
        pytest.skip("opt-in contract: set WORLDFOUNDRY_MG2_CHECKPOINT_ROOT to existing local weights")
    assert torch.cuda.is_available(), "real checkpoint operator contracts require CUDA"
    assert torch.cuda.get_device_capability()[0] >= 9, "real FP8 checkpoint contracts require SM90+"
    root = Path(configured).expanduser().resolve()
    paths = {
        "dit": root / "base_distilled_model/base_distill.safetensors",
        "config": root / "base_distilled_model/config.json",
        "vae": root / "Wan2.1_VAE.pth",
    }
    provenance = {}
    for name, path in paths.items():
        assert path.is_file(), f"required local checkpoint is absent: {path}"
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        provenance[name] = {"path": str(path), "bytes": after.st_size, "sha256": digest}
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    evidence = {
        "schema_version": 1,
        "scope": "checkpoint_loaded_production_operators",
        "full_generation_quality_certified": False,
        "performance_certified": False,
        "synthetic_seeded_inputs": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "checkpoints": provenance,
        "source_sha256": {
            str(Path(inspect.getfile(operator)).resolve()): _sha256_file(Path(inspect.getfile(operator)))
            for operator in (wan.CausalWanAttentionBlock, wan.ActionModule, VAEDecoderWrapper,
                             fuse_qkv_projections, apply_quantization_policy, convert_convolution_weight_layouts,
                             FusedFP8GELUFeedForward, Float8Linear, quantize_rowwise_fp8_triton)
        },
        "test_source_sha256": _sha256_file(Path(__file__)),
        "contracts": {},
    }
    yield paths, json.loads(paths["config"].read_text()), evidence
    output = os.getenv("WORLDFOUNDRY_MG2_CHECKPOINT_REPORT")
    if output:
        report = Path(output).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def _load_block_state(path):
    prefix = "model.blocks.0."
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        state = {key.removeprefix(prefix): checkpoint.get_tensor(key) for key in checkpoint.keys() if key.startswith(prefix)}
    assert state and len(state) == 46, "checkpoint block layout changed; audit the loaded operator contract"
    return state


def _block(config, state):
    block = wan.CausalWanAttentionBlock(
        "i2v_cross_attn", config["dim"], config["ffn_dim"], config["num_heads"],
        local_attn_size=6, sink_size=1, qk_norm=True, cross_attn_norm=True,
        action_config=config["action_config"], block_idx=0, eps=config["eps"],
    )
    block.load_state_dict(state, strict=True)
    return block.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)


class _NoDeviceScalarRead(torch.Tensor):
    def item(self):
        raise AssertionError("managed cache index unexpectedly read a CUDA scalar")


def _cache(batch, capacity, heads, width, managed=False):
    cache = {
        "k": torch.zeros(batch, capacity, heads, width, device="cuda", dtype=torch.bfloat16),
        "v": torch.zeros(batch, capacity, heads, width, device="cuda", dtype=torch.bfloat16),
        "global_end_index": torch.tensor(0, device="cuda"),
        "local_end_index": torch.tensor(0, device="cuda"),
    }
    if managed:
        cache.update(_host_global_end_index=0, _host_local_end_index=0)
        for name in ("global_end_index", "local_end_index"):
            cache[name] = cache[name].as_subclass(_NoDeviceScalarRead)
    return cache


def _error_metrics(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item(),
    }


@torch.inference_mode()
def test_real_checkpoint_action_block_qkv_and_managed_caches_preserve_denoising_and_rollover(checkpoint_evidence):
    paths, config, evidence = checkpoint_evidence
    state = _load_block_state(paths["dit"])
    dense = _block(config, state)
    optimized = copy.deepcopy(dense)
    assert fuse_qkv_projections(optimized, strategy="packed") == 1
    assert config["dim"] == 1536 and config["ffn_dim"] == 8960 and config["num_heads"] == 12
    caches = []
    for managed in (False, True):
        caches.append({
            "self": _cache(1, 6 * 880, 12, 128, managed),
            "mouse": _cache(880, 6, 16, 64, managed),
            "keyboard": _cache(1, 6, 16, 64, managed),
            "cross": {"is_init": False},
        })
    generator = torch.Generator(device="cuda").manual_seed(730)
    context = torch.randn(1, 257, config["dim"], device="cuda", dtype=torch.bfloat16, generator=generator)
    head_dim = config["dim"] // config["num_heads"]
    freqs = torch.cat([
        wan.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        wan.rope_params(1024, 2 * (head_dim // 6)),
        wan.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1).to("cuda")
    grid = torch.tensor([3, 22, 40])
    metrics, input_hashes = [], []
    starts = [0, 0, 3, 3, 6, 6, 9, 9]
    for step, start in enumerate(starts):
        hidden = torch.randn(1, 3 * 880, config["dim"], device="cuda", dtype=torch.bfloat16, generator=generator)
        modulation = torch.randn(1, 3, 6, config["dim"], device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
        control_frames = 1 + 4 * (start + 3 - 1)
        keyboard = torch.zeros(1, control_frames, 4, device="cuda", dtype=torch.bfloat16)
        keyboard[:, :, step % 4] = 1
        mouse = torch.linspace(-0.1, 0.1, control_frames, device="cuda", dtype=torch.bfloat16).view(1, -1, 1).repeat(1, 1, 2)
        input_hashes.append({name: _tensor_sha256(value) for name, value in (
            ("hidden", hidden), ("modulation", modulation), ("keyboard", keyboard), ("mouse", mouse),
        )})
        outputs = []
        for block, cache in zip((dense, optimized), caches):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs.append(block(
                    hidden, modulation, torch.tensor([3 * 880]), grid, freqs, context,
                    None, None, None, num_frame_per_block=3, use_rope_keyboard=True,
                    mouse_cond=mouse, keyboard_cond=keyboard, kv_cache=cache["self"],
                    kv_cache_mouse=cache["mouse"], kv_cache_keyboard=cache["keyboard"],
                    crossattn_cache=cache["cross"], current_start=start * 880,
                ))
        assert all(torch.isfinite(output).all() for output in outputs)
        measured = _error_metrics(outputs[1], outputs[0])
        assert measured["relative_l2"] < 0.003 and measured["cosine"] > 0.9999
        metrics.append(measured)
        for name in ("self", "mouse", "keyboard"):
            for kind in ("k", "v"):
                measured_cache = _error_metrics(caches[1][name][kind], caches[0][name][kind])
                assert measured_cache["relative_l2"] < 0.004
            multiplier = 880 if name == "self" else 1
            assert caches[1][name]["_host_global_end_index"] == (start + 3) * multiplier
            assert caches[1][name]["_host_local_end_index"] == min(start + 3, 6) * multiplier
    report = qkv_fusion_report(optimized)
    assert report["eager_packed_projection_calls"] == len(starts)
    evidence["contracts"]["action_block"] = {
        "passed": True, "block_index": 0, "geometry": [1, 3, 22, 40, 1536],
        "dtype": "bfloat16", "latent_frame_starts": starts, "seed": 730,
        "input_sha256": input_hashes, "context_sha256": _tensor_sha256(context),
        "error_metrics": metrics, "qkv_receipt": report,
        "checkpoint_tensor_sha256": {name: _tensor_sha256(value) for name, value in state.items()},
    }


@torch.inference_mode()
def test_real_checkpoint_ffn_fp8_fusion_preserves_quantized_math_and_obeys_dense_error_budget(checkpoint_evidence):
    paths, config, evidence = checkpoint_evidence
    with safe_open(str(paths["dit"]), framework="pt", device="cpu") as checkpoint:
        state = {name: checkpoint.get_tensor("model.blocks.0.ffn." + name) for name in ("0.weight", "0.bias", "2.weight", "2.bias")}
    dense = torch.nn.ModuleDict({"ffn": torch.nn.Sequential(
        torch.nn.Linear(config["dim"], config["ffn_dim"]), torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(config["ffn_dim"], config["dim"]),
    )})
    dense["ffn"].load_state_dict(state, strict=True)
    dense = dense.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    baseline, fused = copy.deepcopy(dense), copy.deepcopy(dense)
    for model, fuse in ((baseline, False), (fused, True)):
        receipt = apply_quantization_policy(model, QuantizationPolicy(
            mode=QuantizationMode.FP8, options={"min_features": 16, "fuse_fp8_ffn": fuse},
        ))
        assert receipt.extra["fused_fp8_ffn_blocks"] == int(fuse)
    generator = torch.Generator(device="cuda").manual_seed(731)
    measured, inputs = [], []
    for amplitude in (0.1, 1.0, 3.0):
        value = torch.randn(1, 880, config["dim"], device="cuda", dtype=torch.bfloat16, generator=generator) * amplitude
        expected, quantized, actual = dense["ffn"](value), baseline["ffn"](value), fused["ffn"](value)
        assert torch.isfinite(actual).all()
        fusion_error, quantization_error = _error_metrics(actual, quantized), _error_metrics(actual, expected)
        assert fusion_error["relative_l2"] < 0.006 and fusion_error["cosine"] > 0.9999
        assert quantization_error["relative_l2"] < 0.04 and quantization_error["cosine"] > 0.999
        measured.append({"amplitude": amplitude, "fusion_error": fusion_error, "dense_error": quantization_error})
        inputs.append(_tensor_sha256(value))
    receipt = quantization_runtime_report(fused)
    assert receipt["low_precision_kernel_calls"] >= 6 and receipt["dense_fallback_calls"] == 0
    assert fused["ffn"].fused_calls == 3
    evidence["contracts"]["fp8_ffn"] = {
        "passed": True, "shape": [1, 880, 1536], "dtype": "bfloat16", "seed": 731,
        "input_sha256": inputs, "error_metrics": measured, "quantization_receipt": receipt,
        "checkpoint_tensor_sha256": {name: _tensor_sha256(value) for name, value in state.items()},
        "acceptance": {"fusion_relative_l2": 0.006, "fusion_cosine_minimum": 0.9999,
                       "dense_relative_l2": 0.04, "dense_cosine_minimum": 0.999},
    }


@torch.inference_mode()
def test_real_checkpoint_streaming_vae_layout_preserves_pixels_and_recurrent_caches(checkpoint_evidence):
    paths, _, evidence = checkpoint_evidence
    restored = torch.load(paths["vae"], map_location="cpu", weights_only=True)
    state = {name: value for name, value in restored.items() if "decoder." in name or "conv2" in name}
    dense = VAEDecoderWrapper()
    dense.load_state_dict(state, strict=True)
    dense = dense.to(device="cuda", dtype=torch.float16).eval().requires_grad_(False)
    optimized = copy.deepcopy(dense)
    receipt = convert_convolution_weight_layouts(optimized, conv2d=True, conv3d=True)
    assert receipt.conv2d_converted > 0 and receipt.conv3d_converted > 0
    caches = [[None] * dense.decoder.decoder_conv_num for _ in range(2)]
    generator = torch.Generator(device="cuda").manual_seed(732)
    metrics, inputs = [], []
    for _ in range(3):
        latent = torch.randn(1, 1, 16, 4, 5, device="cuda", dtype=torch.float16, generator=generator) * 0.2
        inputs.append(_tensor_sha256(latent))
        expected, caches[0] = dense(latent, *caches[0])
        actual, caches[1] = optimized(latent, *caches[1])
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.004)
        measured = _error_metrics(actual, expected)
        assert measured["relative_l2"] < 0.001
        assert measured["max_abs"] <= 0.002
        reference_bytes = ((expected + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        actual_bytes = ((actual + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        byte_delta = (actual_bytes.to(torch.int16) - reference_bytes.to(torch.int16)).abs()
        assert byte_delta.max().item() <= 1
        measured["maximum_rgb_byte_delta"] = byte_delta.max().item()
        measured["output_shape"] = list(actual.shape)
        measured["cache_error_metrics"] = []
        metrics.append(measured)
        for actual_cache, expected_cache in zip(caches[1], caches[0]):
            if torch.is_tensor(expected_cache):
                # FP16 convolution layouts can differ by several ULPs near
                # zero. Bound aggregate and peak feature error, then require
                # the continued decode to keep every RGB byte within one.
                cache_error = _error_metrics(actual_cache, expected_cache)
                assert torch.isfinite(actual_cache).all()
                assert cache_error["relative_l2"] < 0.003
                assert cache_error["cosine"] > 0.999995
                assert cache_error["max_abs"] < 0.01 * max(1.0, expected_cache.abs().max().item())
                measured["cache_error_metrics"].append(cache_error)
            else:
                assert actual_cache == expected_cache
    evidence["contracts"]["streaming_vae_layout"] = {
        "passed": True, "latent_shape": [1, 1, 16, 4, 5], "dtype": "float16", "seed": 732,
        "input_sha256": inputs, "error_metrics": metrics,
        "layout_receipt": {name: getattr(receipt, name) for name in (
            "conv2d_total", "conv2d_converted", "conv3d_total", "conv3d_converted",
        )},
        "acceptance": {"output_relative_l2": 0.001, "output_max_abs": 0.002, "rgb_byte_delta": 1,
                       "cache_relative_l2": 0.003, "cache_cosine_minimum": 0.999995, "cache_relative_peak": 0.01},
    }
