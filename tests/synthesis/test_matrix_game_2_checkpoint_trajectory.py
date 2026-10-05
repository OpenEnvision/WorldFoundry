"""Opt-in full-checkpoint MG2 autoregressive numerical trajectory contracts.

WORLDFOUNDRY_MG2_CHECKPOINT_ROOT must point to existing local universal weights;
no download occurs. All 30 production DiT layers and the streaming VAE run at
352x640 with the shipped three denoising timesteps plus clean refresh. Four
blocks exercise continuation and rollover of the shipped six-frame window.
Image latents and visual embeddings are seeded test inputs, so this certifies
numerical behavior on that trajectory, not public-prompt generation quality,
long-horizon stability, FP8, CUDA Graphs, or acceleration throughput.
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
from omegaconf import OmegaConf
from safetensors.torch import load_file

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import causal_action_21 as wan
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections, qkv_fusion_report
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.scheduler import (
    FlowMatchScheduler,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
    VAEDecoderWrapper,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.wan_wrapper import (
    WanDiffusionWrapper,
)

pytestmark = pytest.mark.gpu

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "worldfoundry/data/models/runtime/configs/matrix_game_2"
_BUDGET = {
    "latent_relative_l2_max": 0.0,
    "latent_cosine_min": 1.0,
    "cache_relative_l2_max": 0.0,
    "pixel_rmse_max": 0.0,
    "pixel_mae_max": 0.0,
    "pixel_max_abs_max": 0.0,
    "pixel_byte_mean_abs_max": 0.0,
    "pixel_byte_max_abs_max": 0,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    maximum = delta.abs().max().item()
    # FP32 normalization/reduction can report cosine below one even for
    # identical large vectors. Preserve exact equality and use FP64 otherwise.
    cosine = 1.0 if maximum == 0 else torch.nn.functional.cosine_similarity(
        actual.double().flatten(), expected.double().flatten(), dim=0,
    ).item()
    return {
        "max_abs": maximum,
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected).clamp_min(1e-8)).item(),
        "cosine": cosine,
    }


class _NoDeviceScalarRead(torch.Tensor):
    def item(self):
        raise AssertionError("managed causal cache read a CUDA index scalar during model inference")


def _cache_indices(cache):
    # Test-only audits may synchronize; production scalar reads on the managed
    # path remain prohibited by the tensor subclass above.
    return {
        name: int(cache[name].as_subclass(torch.Tensor).item())
        for name in ("global_end_index", "local_end_index")
    }


class _TracedGenerator(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.trace = []

    @property
    def model(self):
        return self.core.model

    def get_scheduler(self):
        return self.core.get_scheduler()

    def forward(self, **kwargs):
        flow, prediction = self.core(**kwargs)
        conditions = kwargs["conditional_dict"]
        self.trace.append({
            "current_start_tokens": kwargs["current_start"],
            "timestep": kwargs["timestep"].detach().cpu().clone(),
            "input": kwargs["noisy_image_or_video"].detach().cpu().clone(),
            "flow": flow.detach().cpu().clone(),
            "x0": prediction.detach().cpu().clone(),
            "condition_sha256": {name: _tensor_sha256(value) for name, value in conditions.items()},
            "cache_indices": {
                name: [_cache_indices(cache) for cache in kwargs[key]]
                for name, key in (("self", "kv_cache"), ("mouse", "kv_cache_mouse"), ("keyboard", "kv_cache_keyboard"))
            },
        })
        return flow, prediction


def _restore_generator(config, checkpoint):
    # Avoid allocating randomly initialized full-model CPU weights before
    # replacing every parameter with the original checkpoint's tensors.
    with torch.device("meta"):
        generator = WanDiffusionWrapper(model_config=config, timestep_shift=5.0, is_causal=True)
    width = config["dim"] // config["num_heads"]
    generator.model.freqs = torch.cat([
        wan.rope_params(1024, width - 4 * (width // 6)),
        wan.rope_params(1024, 2 * (width // 6)),
        wan.rope_params(1024, 2 * (width // 6)),
    ], dim=1)
    generator.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    generator.scheduler.set_timesteps(1000, training=True)
    generator.post_init()
    loaded = load_file(str(checkpoint), device="cpu")
    generator.load_state_dict(loaded, strict=True, assign=True)
    assert len(generator.model.blocks) == 30
    assert generator.model.dim == 1536 and generator.model.ffn_dim == 8960
    assert sum(block.action_model is not None for block in generator.model.blocks) == 15
    return generator.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def full_checkpoint_models():
    configured = os.getenv("WORLDFOUNDRY_MG2_CHECKPOINT_ROOT")
    if not configured:
        pytest.skip("set WORLDFOUNDRY_MG2_CHECKPOINT_ROOT for the opt-in full trajectory contract")
    assert torch.cuda.is_available(), "full checkpoint trajectory requires CUDA"
    assert torch.cuda.get_device_capability()[0] >= 9, "this exact same-backend trajectory gate requires SM90+"
    root = Path(configured).expanduser().resolve()
    paths = {
        "dit": root / "base_distilled_model/base_distill.safetensors",
        "checkpoint_config": root / "base_distilled_model/config.json",
        "vae": root / "Wan2.1_VAE.pth",
        "runtime_config": _CONFIG_ROOT / "inference_yaml/inference_universal.yaml",
        "model_config": _CONFIG_ROOT / "distilled_model/universal/config.yaml",
    }
    provenance = {}
    for name, path in paths.items():
        assert path.is_file(), f"required local checkpoint/config absent: {path}"
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        provenance[name] = {"path": str(path), "bytes": after.st_size, "sha256": digest}
    runtime_config = OmegaConf.load(paths["runtime_config"])
    model_config = OmegaConf.to_container(OmegaConf.load(paths["model_config"]), resolve=True)
    checkpoint_config = json.loads(paths["checkpoint_config"].read_text())
    for name in ("dim", "ffn_dim", "num_heads", "num_layers", "in_dim", "out_dim", "action_config"):
        assert model_config[name] == checkpoint_config[name]
    assert runtime_config.denoising_step_list == [1000, 666, 333]
    assert runtime_config.context_noise == 0 and runtime_config.num_frame_per_block == 3
    assert runtime_config.warp_denoising_step is True
    assert model_config["local_attn_size"] == 6 and model_config["sink_size"] == 0
    dense = _restore_generator(model_config, paths["dit"])
    optimized = copy.deepcopy(dense)
    assert fuse_qkv_projections(optimized.model, strategy="split") == 30
    assert {parameter.data_ptr() for parameter in dense.parameters()}.isdisjoint(
        parameter.data_ptr() for parameter in optimized.parameters()
    ), "baseline and optimized DiTs must own distinct restored parameter storage"
    restored = torch.load(paths["vae"], map_location="cpu", weights_only=True)
    decoder = VAEDecoderWrapper()
    decoder.load_state_dict({key: value for key, value in restored.items() if "decoder." in key or "conv2" in key}, strict=True)
    decoder = decoder.to(device="cuda", dtype=torch.float16).eval().requires_grad_(False)
    del restored
    evidence = {
        "schema_version": 1,
        "scope": "full_checkpoint_short_autoregressive_trajectory",
        "full_generation_quality_certified": False,
        "long_horizon_stability_certified": False,
        "performance_certified": False,
        "fp8_certified": False,
        "seeded_image_latents_and_visual_embeddings": True,
        "checkpoint_and_config_provenance": provenance,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(), "visible_device": torch.cuda.current_device(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model_parameter_count": sum(parameter.numel() for parameter in dense.parameters()),
        "baseline_and_optimized_parameter_storage_distinct": True,
        "decoder_weights_shared_read_only_with_independent_recurrent_caches": True,
        "source_sha256": {
            str(Path(inspect.getfile(operator)).resolve()): _sha256_file(Path(inspect.getfile(operator)))
            for operator in (wan.CausalWanModel, wan.ActionModule, VAEDecoderWrapper, WanDiffusionWrapper,
                             CausalInferencePipeline, FlowMatchScheduler, fuse_qkv_projections)
        },
        "test_source_sha256": _sha256_file(Path(__file__)),
        "acceptance": _BUDGET,
        "contracts": {},
    }
    try:
        yield dense, optimized, decoder, runtime_config, evidence
    finally:
        output = os.getenv("WORLDFOUNDRY_MG2_TRAJECTORY_REPORT")
        if output:
            report = Path(output).expanduser()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def _conditions():
    sample = torch.Generator(device="cuda").manual_seed(1730)
    image = torch.randn(1, 16, 1, 44, 80, dtype=torch.bfloat16, device="cuda", generator=sample) * 0.2
    conditions = {
        "cond_concat": torch.zeros(1, 20, 12, 44, 80, dtype=torch.bfloat16, device="cuda"),
        "visual_context": torch.randn(1, 257, 1280, dtype=torch.bfloat16, device="cuda", generator=sample) * 0.1,
        "keyboard_cond": torch.zeros(1, 45, 4, dtype=torch.bfloat16, device="cuda"),
        "mouse_cond": torch.zeros(1, 45, 2, dtype=torch.bfloat16, device="cuda"),
    }
    conditions["cond_concat"][:, :4, :1] = 1
    conditions["cond_concat"][:, 4:, :1] = image
    for block in range(4):
        start, end = (0 if block == 0 else 1 + (block * 3 - 1) * 4), 1 + ((block + 1) * 3 - 1) * 4
        conditions["keyboard_cond"][:, start:end, block] = 1
        conditions["mouse_cond"][:, start:end, 0] = (block - 1.5) * 0.025
        conditions["mouse_cond"][:, start:end, 1] = (1.5 - block) * 0.05
    noise = torch.randn(1, 16, 12, 44, 80, dtype=torch.bfloat16, device="cuda", generator=sample)
    return conditions, noise


def _audit_cache_equality(baseline, optimized, exact, failures):
    receipts = []
    for family in ("kv_cache", "mouse_kv_cache", "keyboard_kv_cache"):
        for layer, (legacy, managed) in enumerate(zip(getattr(baseline, family), getattr(optimized, family))):
            assert _cache_indices(legacy) == _cache_indices(managed)
            for name, value in _cache_indices(managed).items():
                assert managed["_host_" + name] == value
            for name in ("k", "v"):
                measured = _metrics(managed[name], legacy[name])
                receipts.append({"family": family, "layer": layer, "kind": name, **measured})
                if measured["relative_l2"] > _BUDGET["cache_relative_l2_max"] or (exact and measured["max_abs"] != 0):
                    failures.append(f"{family}.{layer}.{name}: {measured}")
    return receipts


@pytest.mark.parametrize("strategy", ["split", "packed"])
@torch.inference_mode()
def test_full_checkpoint_rollout_preserves_every_denoise_clean_refresh_and_recurrent_pixel(strategy, full_checkpoint_models):
    dense, optimized, decoder, runtime_config, evidence = full_checkpoint_models
    state = optimized.model._worldfoundry_qkv_fusion
    state.strategy = strategy
    state.reset_request_window()
    baseline_trace, optimized_trace = _TracedGenerator(dense), _TracedGenerator(optimized)
    baseline = CausalInferencePipeline(copy.deepcopy(runtime_config), generator=baseline_trace, vae_decoder=decoder)
    accelerated = CausalInferencePipeline(copy.deepcopy(runtime_config), generator=optimized_trace, vae_decoder=decoder)
    accelerated.overlap_vae_decode = True
    assert baseline.num_transformer_blocks == accelerated.num_transformer_blocks == 30
    assert baseline.frame_seq_length == accelerated.frame_seq_length == 880
    conditions, noise = _conditions()
    sessions = [pipe.start_session(copy.deepcopy(conditions), reference_tensor=noise) for pipe in (baseline, accelerated)]
    for family in ("kv_cache", "mouse_kv_cache", "keyboard_kv_cache"):
        for cache in getattr(sessions[0], family):
            cache.pop("_host_global_end_index")
            cache.pop("_host_local_end_index")
        for cache in getattr(sessions[1], family):
            for name in ("global_end_index", "local_end_index"):
                cache[name] = cache[name].as_subclass(_NoDeviceScalarRead)
    noise_receipts = [[], []]
    originals = [pipe.scheduler.add_noise for pipe in (baseline, accelerated)]
    for owner, pipe in enumerate((baseline, accelerated)):
        def audit_noise(clean, sampled_noise, timestep, *, owner=owner):
            noise_receipts[owner].append({"noise_sha256": _tensor_sha256(sampled_noise), "timestep": timestep.cpu().tolist()})
            return originals[owner](clean, sampled_noise, timestep)
        pipe.scheduler.add_noise = audit_noise
    # This is a same-device, same-backend exact-transform gate. A future
    # provider/rounding difference must trigger review, not widen this budget.
    exact = True
    failures = []
    contract = {
        "passed": False, "qkv_strategy": strategy, "bitwise_required": exact,
        "geometry": [1, 16, 12, 44, 80], "decoded_resolution": [352, 640],
        "model_layers": 30, "action_layers": 15, "local_attention_window": 6,
        "latent_frame_starts": [0, 3, 6, 9], "initial_input_seed": 1730,
        "schedule": baseline.denoising_step_list.tolist(), "context_noise": 0,
        "quantization": "none", "compile": False, "vae_weight_dtype": "float16",
        "ambient_autocast": "cuda_bfloat16",
        "vae_convolution_layout_changed": False, "overlap_vae_decode": True,
        "condition_sha256": {name: _tensor_sha256(value) for name, value in conditions.items()},
        "initial_noise_sha256": _tensor_sha256(noise), "blocks": [], "failures": failures,
    }
    evidence["contracts"][strategy] = contract
    torch.cuda.reset_peak_memory_stats()
    try:
        for block in range(4):
            outputs = []
            # The production loop samples fresh inter-step noise. Restore the
            # same global RNG chain for both trajectories without replacing the
            # sampler or injecting optimized outputs into the legacy rollout.
            for pipe, session in zip((baseline, accelerated), sessions):
                torch.cuda.manual_seed(1800 + block)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs.append(pipe.generate_next_block(session, noise[:, :, block * 3:(block + 1) * 3]))
            assert noise_receipts[0] == noise_receipts[1]
            stage_receipts = []
            for stage, (legacy, current) in enumerate(zip(baseline_trace.trace[-4:], optimized_trace.trace[-4:])):
                assert legacy["current_start_tokens"] == current["current_start_tokens"] == block * 3 * 880
                assert legacy["condition_sha256"] == current["condition_sha256"]
                assert legacy["cache_indices"] == current["cache_indices"]
                for family, indices in current["cache_indices"].items():
                    multiplier = 880 if family == "self" else 1
                    for layer, counters in enumerate(indices):
                        active = family == "self" or layer < 15
                        expected = {
                            "global_end_index": (block + 1) * 3 * multiplier if active else 0,
                            "local_end_index": min((block + 1) * 3, 6) * multiplier if active else 0,
                        }
                        assert counters == expected
                torch.testing.assert_close(legacy["timestep"], current["timestep"], rtol=0, atol=0)
                measured = {kind: _metrics(current[kind], legacy[kind]) for kind in ("input", "flow", "x0")}
                for kind, metrics in measured.items():
                    if metrics["relative_l2"] > _BUDGET["latent_relative_l2_max"] or metrics["cosine"] < _BUDGET["latent_cosine_min"]:
                        failures.append(f"block{block}.stage{stage}.{kind}: {metrics}")
                    if exact and metrics["max_abs"] != 0:
                        failures.append(f"{strategy} trajectory differs at block{block}.stage{stage}.{kind}: {metrics}")
                stage_receipts.append({
                    "stage": "clean_refresh" if stage == 3 else f"denoise_{stage}",
                    "timestep": current["timestep"].tolist(), "metrics": measured,
                    "baseline_output_sha256": {kind: _tensor_sha256(legacy[kind]) for kind in ("flow", "x0")},
                    "optimized_output_sha256": {kind: _tensor_sha256(current[kind]) for kind in ("flow", "x0")},
                    "cache_indices": current["cache_indices"],
                })
            latent_error = _metrics(outputs[1].latent, outputs[0].latent)
            if latent_error["max_abs"] != 0:
                failures.append(f"block{block}.final_latent: {latent_error}")
            pixels = _metrics(outputs[1].video, outputs[0].video)
            bytes_ = [((output.video + 1) * 127.5).round().clamp(0, 255).to(torch.uint8) for output in outputs]
            byte_delta = (bytes_[1].to(torch.int16) - bytes_[0].to(torch.int16)).abs()
            pixel_bytes = {"mean_abs": byte_delta.float().mean().item(), "max_abs": byte_delta.max().item()}
            for metric, budget in (("rmse", "pixel_rmse_max"), ("mae", "pixel_mae_max"), ("max_abs", "pixel_max_abs_max")):
                if pixels[metric] > _BUDGET[budget]:
                    failures.append(f"block{block}.pixel.{metric}: {pixels[metric]}")
            if pixel_bytes["mean_abs"] > _BUDGET["pixel_byte_mean_abs_max"] or (exact and pixels["max_abs"] != 0):
                failures.append(f"block{block}.pixels: {pixels}; bytes: {pixel_bytes}")
            if pixel_bytes["max_abs"] > _BUDGET["pixel_byte_max_abs_max"]:
                failures.append(f"block{block}.RGB8: {pixel_bytes}")
            cache_receipts = _audit_cache_equality(*sessions, exact, failures)
            recurrent = []
            for slot, (legacy, current) in enumerate(zip(sessions[0].vae_cache, sessions[1].vae_cache)):
                assert (legacy is None) == (current is None)
                if legacy is not None:
                    measured = _metrics(current, legacy)
                    recurrent.append({"slot": slot, **measured})
                    if measured["relative_l2"] > _BUDGET["cache_relative_l2_max"] or (exact and measured["max_abs"] != 0):
                        failures.append(f"block{block}.vae_cache{slot}: {measured}")
            contract["blocks"].append({
                "latent_start": block * 3, "denoise_and_refresh": stage_receipts,
                "latent_error": latent_error, "pixel_error": pixels, "pixel_byte_error": pixel_bytes,
                "pixel_shape": list(outputs[1].video.shape), "all_layer_cache_error": cache_receipts,
                "recurrent_vae_cache_error": recurrent,
                "baseline_latent_sha256": _tensor_sha256(outputs[0].latent),
                "optimized_latent_sha256": _tensor_sha256(outputs[1].latent),
                "baseline_video_sha256": _tensor_sha256(outputs[0].video),
                "optimized_video_sha256": _tensor_sha256(outputs[1].video),
            })
            print(f"MG2 trajectory {strategy}: completed block{block}, latent relative_l2={latent_error['relative_l2']:.6g}, "
                  f"pixel rmse={pixels['rmse']:.6g}, byte mean={pixel_bytes['mean_abs']:.6g}", flush=True)
        assert len(baseline_trace.trace) == len(optimized_trace.trace) == 16
        receipt = qkv_fusion_report(optimized.model)
        assert receipt["fused_blocks"] == 30 and receipt[f"eager_{strategy}_projection_calls"] == 30 * 16
        assert receipt["eager_projection_calls"] == 480
        assert accelerated._decode_overlap_runtime == {"calls": 4, "execution": "decode-stream-overlap-enqueued"}
        assert accelerated._decode_stream is not None
        assert all(session.current_start_frame == 12 for session in sessions)
        contract.update(qkv_receipt=receipt, overlap_receipt=dict(accelerated._decode_overlap_runtime),
                        inter_step_noise=noise_receipts[0], peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        assert not failures, "\n".join(failures)
        contract["passed"] = True
    finally:
        for pipe, original in zip((baseline, accelerated), originals):
            pipe.scheduler.add_noise = original
            pipe.close()
