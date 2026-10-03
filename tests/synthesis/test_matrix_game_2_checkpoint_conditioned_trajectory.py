"""Opt-in exact long MG2 rollout with checkpoint VAE and CLIP conditioning.

Six cases exercise three continuous noise streams, two projection strategies,
12 native blocks, and the real public resident configure/generate APIs. The
independent baseline casts original image-conditioning weights directly to
BF16 and encodes the entire 141-frame condition before rollout. Every tensor
comparison requires bitwise equality; this is not a throughput benchmark.
Every accepted trajectory also matches frozen historical output and state
receipts, so a shared regression in the two current paths cannot self-certify.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21 import WanVAE
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.action_clip import CLIPModel
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import qkv_fusion_report
from worldfoundry.operators.matrix_game_2_operator import MatrixGame2Operator
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.extension_modules.wanx_vae.wanx_vae import (
    WanxVAEWrapper,
    get_wanx_vae_wrapper,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
    MatrixGame2RealtimeSession,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.worldfoundry_runtime import (
    MatrixGame2Runtime,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis

pytestmark = pytest.mark.gpu

_HELPER_PATH = Path(__file__).with_name("test_matrix_game_2_checkpoint_trajectory.py")
_SPEC = importlib.util.spec_from_file_location("mg2_frozen_trajectory_helpers", _HELPER_PATH)
_short = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_short)
_metrics = _short._metrics
_sha256_file = _short._sha256_file
_SEEDS = (7, 42, 1730)
_BLOCKS = 12
_REFERENCE_PATH = Path(__file__).with_name("mg2_historical_reference.py")
_REFERENCE_SPEC = importlib.util.spec_from_file_location("mg2_historical_reference", _REFERENCE_PATH)
_historical = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(_historical)


def _tensor_sha256(value):
    # PyTorch requires a dimension when reinterpreting a scalar's element size.
    return _short._tensor_sha256(value.reshape(1) if value.ndim == 0 else value)


def _image():
    rows, cols = np.indices((407, 733))
    pixels = np.stack(((cols * 3 + rows) % 256, (rows * 7 + cols) % 256,
                       (cols ^ rows) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _public_pipeline(core, vae):
    runtime = MatrixGame2Runtime(core, vae, weight_dtype=torch.bfloat16, device="cuda", mode="universal")
    return MatrixGame2Pipeline(
        synthesis_model=MatrixGame2Synthesis(runtime=runtime),
        operators=MatrixGame2Operator(mode="universal"),
        device="cuda", weight_dtype=torch.bfloat16,
    )


def _cache_digest(caches):
    return [{key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_sha256(value)}
             if isinstance(value, torch.Tensor) else value
             for key, value in cache.items() if key in ("k", "v", "is_init")}
            for cache in caches]


class _TracedGenerator(_short._TracedGenerator):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        self.trace[-1]["full_cache_sha256"] = {
            family: _cache_digest(kwargs[key])
            for family, key in (("self", "kv_cache"), ("mouse", "kv_cache_mouse"),
                                ("keyboard", "kv_cache_keyboard"), ("cross", "crossattn_cache"))
        }
        return result


def _compare_condition_weights(canonical, resident):
    result = {}
    for name in ("vae", "clip"):
        left, right = getattr(canonical, name), getattr(resident, name)
        left_parameters, right_parameters = dict(left.named_parameters()), dict(right.named_parameters())
        assert left_parameters.keys() == right_parameters.keys()
        assert {parameter.data_ptr() for parameter in left_parameters.values()}.isdisjoint(
            parameter.data_ptr() for parameter in right_parameters.values()
        ), "independent conditioning models must own distinct parameter storage"
        for key in left_parameters:
            assert left_parameters[key].dtype == right_parameters[key].dtype == torch.bfloat16
            assert torch.equal(left_parameters[key].contiguous().reshape(-1).view(torch.uint8),
                               right_parameters[key].contiguous().reshape(-1).view(torch.uint8)), f"conditioning weight differs: {name}.{key}"
        left_buffers, right_buffers = dict(left.named_buffers()), dict(right.named_buffers())
        assert left_buffers.keys() == right_buffers.keys()
        for key in left_buffers:
            assert left_buffers[key].shape == right_buffers[key].shape
            assert left_buffers[key].dtype == right_buffers[key].dtype
            assert _tensor_sha256(left_buffers[key]) == _tensor_sha256(right_buffers[key])
        result[name] = {"parameter_tensors": len(left_parameters),
                        "parameters": sum(parameter.numel() for parameter in left_parameters.values()),
                        "bitwise_equal": True, "storage_distinct": True}
    normalization = {"mean": canonical.vae.mean, "std": canonical.vae.std,
                     "scale_mean": canonical.vae.scale[0], "scale_inverse_std": canonical.vae.scale[1]}
    resident_normalization = {"mean": resident.vae.mean, "std": resident.vae.std,
                              "scale_mean": resident.vae.scale[0], "scale_inverse_std": resident.vae.scale[1]}
    for name, value in normalization.items():
        _assert_exact_metrics(value, resident_normalization[name], f"conditioning normalization {name}")
    result["vae_normalization_sha256"] = {name: _tensor_sha256(value) for name, value in normalization.items()}
    return result


@pytest.fixture(scope="module")
def conditioned_checkpoint_models():
    # The frozen fixture restores all original weights onto meta-created DiTs.
    # Its report environment remains unset so the old evidence is untouched.
    assert not os.getenv("WORLDFOUNDRY_MG2_TRAJECTORY_REPORT"), "use the distinct conditioned report environment"
    historical = _historical.MG2HistoricalReference()
    source_revision = source_tree = None
    if os.getenv("WORLDFOUNDRY_MG2_CHECKPOINT_ROOT"):
        snapshot = subprocess.run(
            ["git", "rev-parse", "HEAD", "HEAD^{tree}"],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        assert len(snapshot) == 2 and all(
            len(value) == 40 and all(character in "0123456789abcdef" for character in value)
            for value in snapshot
        ), "strict conditioned trajectory requires a committed source revision and tree"
        source_revision, source_tree = snapshot
    frozen = _short.full_checkpoint_models.__wrapped__()
    dense, optimized, decoder, runtime_config, short_evidence = next(frozen)
    root = Path(os.environ["WORLDFOUNDRY_MG2_CHECKPOINT_ROOT"]).expanduser().resolve()
    evidence = copy.deepcopy(short_evidence)
    evidence.update(
        scope="full_checkpoint_real_conditioned_long_autoregressive_trajectory",
        seeded_image_latents_and_visual_embeddings=False,
        real_wan_vae_and_clip_conditioning=True,
        public_resident_configure_and_generate=True,
        independent_baseline_raw_checkpoint_direct_bfloat16=True,
        long_horizon_stability_certified=False,
        contracts={},
        acceptance=copy.deepcopy(_short._BUDGET),
        frozen_helper_sha256=_sha256_file(_HELPER_PATH),
        test_source_sha256=_sha256_file(Path(__file__)),
        torch_version=str(torch.__version__),
        source_revision=source_revision,
        source_tree=source_tree,
        historical_reference={**historical.receipt, "conditioning_passed": False, "contracts_passed": []},
    )
    try:
        assert evidence["frozen_helper_sha256"] == "84b77737b8dc055d399a8afea7a3be1db111b2cea6d0cf2584085b907005282c"
        clip_path = root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        for path in (clip_path, *(root / "xlm-roberta-large").iterdir()):
            if path.is_file():
                evidence["checkpoint_and_config_provenance"][str(path.relative_to(root))] = {
                    "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path),
                }
        historical.assert_backend_and_weights(evidence)
        print("MG2 conditioned: loading independent raw BF16 WanVAE and CLIP", flush=True)
        canonical = WanxVAEWrapper(
            WanVAE(pretrained_path=str(root / "Wan2.1_VAE.pth")).to(device="cuda", dtype=torch.bfloat16),
            CLIPModel(checkpoint_path=str(clip_path), tokenizer_path=str(root / "xlm-roberta-large")).to(
                device="cuda", dtype=torch.bfloat16),
        )
        resident = get_wanx_vae_wrapper(str(root), torch.bfloat16).to("cuda", torch.bfloat16)
        evidence["conditioning_weights"] = _compare_condition_weights(canonical, resident)
        for operator in (WanVAE, CLIPModel, MatrixGame2Operator, MatrixGame2Pipeline, MatrixGame2Runtime,
                         MatrixGame2RealtimeSession, get_wanx_vae_wrapper):
            path = Path(inspect.getfile(operator)).resolve()
            evidence["source_sha256"][str(path)] = _sha256_file(path)
        condition_path = Path(inspect.getfile(MatrixGame2RealtimeSession)).with_name("conditioning.py")
        assert condition_path.is_file()
        evidence["source_sha256"][str(condition_path)] = _sha256_file(condition_path)
        evidence["source_sha256"][str(_REFERENCE_PATH.resolve())] = _sha256_file(_REFERENCE_PATH)

        image = _image()
        evidence["input_image"] = {"width": image.width, "height": image.height,
                                    "rgb_sha256": _tensor_sha256(torch.from_numpy(np.array(image)))}
        baseline_core = CausalInferencePipeline(copy.deepcopy(runtime_config), generator=dense, vae_decoder=decoder)
        accelerated_core = CausalInferencePipeline(copy.deepcopy(runtime_config), generator=optimized, vae_decoder=decoder)
        baseline_public = _public_pipeline(baseline_core, canonical)
        accelerated_public = _public_pipeline(accelerated_core, resident)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            print("MG2 conditioned: canonical 36-latent image/CLIP condition", flush=True)
            full = baseline_public.process(image, 36, interaction_signal=["forward"], seed=1730)
            print("MG2 conditioned: public resident condition prefetch and exact-tail prerequisite", flush=True)
            configured = accelerated_public.configure_realtime(image, seed=1730)
            adapter = accelerated_public._ensure_realtime_session()
            assert adapter.condition_prefetch_blocks == 11
            assert adapter._condition_concat.shape == (1, 20, 33, 44, 80)
            prefix = _metrics(adapter._condition_concat, full["cond_concat"][:, :, :33])
            tail = _metrics(adapter._condition_block(33), full["cond_concat"][:, :, 33:36])
            visual = _metrics(adapter._visual_context, full["visual_context"])
            old_tail = _metrics(full["cond_concat"][:, 4:, 12:15], full["cond_concat"][:, 4:, 15:18])
            prerequisite = {
                "bitwise_required": True,
                "prefetch_blocks": 11, "prefetch_latent_frames": 33, "prefetch_rgb_frames": 129,
                "canonical_latent_frames": 36, "canonical_rgb_frames": 141,
                "prefix_error": prefix, "future_reused_tail_error": tail, "visual_context_error": visual,
                "old_15_latent_prefetch_tail_error": old_tail,
                "canonical_condition_sha256": {key: _tensor_sha256(full[key]) for key in ("cond_concat", "visual_context")},
                "resident_prefetch_sha256": _tensor_sha256(adapter._condition_concat),
                "configured": configured,
                "passed": prefix["max_abs"] == tail["max_abs"] == visual["max_abs"] == 0,
            }
            evidence["conditioning_prerequisite"] = prerequisite
            assert prerequisite["passed"], f"image-conditioning prerequisite failed before DiT rollout: {prerequisite}"
            _assert_exact_metrics(adapter._condition_concat, full["cond_concat"][:, :, :33], "condition prefix prerequisite")
            _assert_exact_metrics(adapter._condition_block(33), full["cond_concat"][:, :, 33:36], "condition tail prerequisite")
            _assert_exact_metrics(adapter._visual_context, full["visual_context"], "visual context prerequisite")
            assert old_tail["max_abs"] > 0, "regression image must expose the earlier unsafe tail reuse"
            historical.assert_conditioning(evidence)
            evidence["historical_reference"]["conditioning_passed"] = True
        baseline_public.reset_realtime()
        accelerated_public.reset_realtime()
        baseline_core.close()
        accelerated_core.close()
        print("MG2 conditioned: 33-latent prefix, future tail, and real CLIP are bitwise exact", flush=True)
        yield dense, optimized, decoder, runtime_config, canonical, resident, image, full, evidence, historical
    finally:
        output = os.getenv("WORLDFOUNDRY_MG2_CONDITIONED_TRAJECTORY_REPORT")
        if output:
            destination = Path(output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        frozen.close()


def _controls(block):
    controls = (
        {"interactions": []},
        {"interactions": ["forward_left", "camera_l"]},
        {"interactions": ["forward_right", "camera_r"]},
        {"interactions": ["backward", "camera_down"]},
        {"interactions": [], "realtime_segments": [
            {"duration": 0.25, "keys": ["w", "a", "j"]},
            {"duration": 0.5, "keys": []},
            {"duration": 0.25, "keys": ["s", "d", "l"]}]},
        {"interactions": ["left", "camera_up"]},
        {"interactions": ["right", "camera_down"]},
        {"interactions": []},
        {"interactions": ["forward", "camera_right"]},
        {"interactions": [], "realtime_segments": [
            {"duration": 0.5, "keys": ["i", "j"]},
            {"duration": 0.5, "keys": ["k", "l"]}]},
        {"interactions": ["back_left", "camera_left"]},
        {"interactions": ["forward_right", "camera_r"]},
    )
    return copy.deepcopy(controls[block])


def _assert_exact_metrics(current, legacy, label):
    assert current.shape == legacy.shape, f"{label}: tensor shape changed"
    assert current.dtype == legacy.dtype, f"{label}: tensor dtype changed"
    assert _tensor_sha256(current) == _tensor_sha256(legacy), f"{label}: tensor bytes changed"
    measured = _metrics(current, legacy)
    assert measured["max_abs"] == measured["relative_l2"] == 0, f"{label}: {measured}"
    assert measured["cosine"] == 1
    return measured


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("strategy", ("split", "packed"))
@torch.inference_mode()
def test_real_conditioned_public_rollout_preserves_12_blocks_exactly(strategy, seed, conditioned_checkpoint_models):
    dense, optimized, decoder, runtime_config, canonical, resident, image, full, evidence, historical = conditioned_checkpoint_models
    fusion = optimized.model._worldfoundry_qkv_fusion
    fusion.strategy = strategy
    fusion.reset_request_window()
    traces = [_TracedGenerator(core) for core in (dense, optimized)]
    cores = [CausalInferencePipeline(copy.deepcopy(runtime_config), generator=trace, vae_decoder=decoder) for trace in traces]
    cores[1].overlap_vae_decode = True
    publics = [_public_pipeline(core, vae) for core, vae in zip(cores, (canonical, resident))]
    key = f"{strategy}_seed{seed}"
    contract = {"passed": False, "qkv_strategy": strategy, "seed": seed, "bitwise_required": True,
                "geometry": [1, 16, 36, 44, 80], "decoded_resolution": [352, 640],
                "rgb_frames": 141, "blocks_count": 12, "rollovers": 10,
                "latent_frame_starts": list(range(0, 36, 3)), "model_layers": 30, "action_layers": 15,
                "local_attention_window": 6, "schedule": cores[0].denoising_step_list.tolist(),
                "context_noise": 0, "quantization": "none", "compile": False,
                "conditioning_weight_dtype": "bfloat16", "vae_decode_weight_dtype": "float16",
                "ambient_autocast": "cuda_bfloat16", "overlap_vae_decode": True,
                "continuous_rng_without_per_block_reseed": True, "historical_reference_passed": False, "blocks": []}
    evidence["contracts"][key] = contract
    historical.assert_contract_identity(key, contract)
    torch.cuda.reset_peak_memory_stats()
    noise_receipts = [[], []]
    originals = [core.scheduler.add_noise for core in cores]
    expected_initial_noise = torch.Generator(device="cuda").manual_seed(seed)
    expected_global_noise = [torch.Generator(device="cuda") for _ in cores]
    try:
        for public in publics:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                receipt = public.configure_realtime(image, seed=seed)
            assert receipt["status"] == "configured"
            adapter = public._ensure_realtime_session()
            _assert_exact_metrics(adapter._condition_concat, full["cond_concat"][:, :, :33], "public configured prefix")
            _assert_exact_metrics(adapter._visual_context, full["visual_context"], "public configured visual context")
            assert adapter._noise_generator.initial_seed() == seed
        # Baseline supplies the independently encoded complete finite condition;
        # the optimized public adapter must exercise its actual prefetched tail.
        publics[0]._ensure_realtime_session()._condition_concat = full["cond_concat"].clone()
        sessions = [core.session for core in cores]
        for family in ("kv_cache", "mouse_kv_cache", "keyboard_kv_cache"):
            for cache in getattr(sessions[0], family):
                cache.pop("_host_global_end_index")
                cache.pop("_host_local_end_index")
            for cache in getattr(sessions[1], family):
                for name in ("global_end_index", "local_end_index"):
                    cache[name] = cache[name].as_subclass(_short._NoDeviceScalarRead)
        for family in ("kv_cache", "mouse_kv_cache", "keyboard_kv_cache", "crossattn_cache"):
            left, right = getattr(sessions[0], family), getattr(sessions[1], family)
            for legacy, managed in zip(left, right):
                assert legacy is not managed
                for kind in ("k", "v"):
                    if kind in legacy:
                        assert legacy[kind].data_ptr() != managed[kind].data_ptr()
        for owner, core in enumerate(cores):
            def audit_noise(clean, sampled_noise, timestep, *, owner=owner):
                expected = torch.empty_like(sampled_noise).normal_(generator=expected_global_noise[owner])
                assert torch.equal(sampled_noise, expected), "inter-step noise differs from the independent continuous RNG chain"
                noise_receipts[owner].append({"noise_sha256": _tensor_sha256(sampled_noise), "timestep": timestep.cpu().tolist()})
                return originals[owner](clean, sampled_noise, timestep)
            core.scheduler.add_noise = audit_noise
        for block in range(_BLOCKS):
            controls = _controls(block)
            rng_before = torch.cuda.get_rng_state().clone()
            for generator in expected_global_noise:
                generator.set_state(rng_before)
            expected_input = torch.randn((1, 16, 3, 44, 80), device="cuda", dtype=torch.bfloat16,
                                         generator=expected_initial_noise)
            outputs = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs.append(publics[0].stream_realtime(**controls))
                rng_after = torch.cuda.get_rng_state().clone()
                torch.cuda.set_rng_state(rng_before)
                outputs.append(publics[1].stream_realtime(**controls))
            assert torch.equal(torch.cuda.get_rng_state(), rng_after), "optimized trajectory changed the continuous CUDA RNG chain"
            assert not torch.equal(rng_before, rng_after), "inter-step sampling did not advance the global RNG"
            assert all(torch.equal(generator.get_state(), rng_after) for generator in expected_global_noise), (
                "public rollout consumed unexpected global randomness"
            )
            torch.cuda.set_rng_state(rng_after)
            assert noise_receipts[0] == noise_receipts[1]
            adapters = [public._ensure_realtime_session() for public in publics]
            assert torch.equal(adapters[0]._noise_generator.get_state(), adapters[1]._noise_generator.get_state())
            for adapter, trace in zip(adapters, traces):
                assert torch.equal(adapter._noise_generator.get_state(), expected_initial_noise.get_state()), (
                    "resident initial noise stream diverged from independent unreseeded sampling"
                )
                _assert_exact_metrics(trace.trace[0]["input"], expected_input.cpu(), "independent initial noise")
            stages = []
            for stage, (legacy, current) in enumerate(zip(traces[0].trace, traces[1].trace)):
                assert legacy["current_start_tokens"] == current["current_start_tokens"] == block * 3 * 880
                assert legacy["condition_sha256"] == current["condition_sha256"]
                assert legacy["cache_indices"] == current["cache_indices"]
                assert legacy["full_cache_sha256"] == current["full_cache_sha256"], "some complete attention K/V cache changed"
                for family, indices in current["cache_indices"].items():
                    multiplier = 880 if family == "self" else 1
                    for layer, counters in enumerate(indices):
                        active = family == "self" or layer < 15
                        assert counters == {"global_end_index": (block + 1) * 3 * multiplier if active else 0,
                                            "local_end_index": min((block + 1) * 3, 6) * multiplier if active else 0}
                torch.testing.assert_close(current["timestep"], legacy["timestep"], rtol=0, atol=0)
                stages.append({"stage": "clean_refresh" if stage == 3 else f"denoise_{stage}",
                               "timestep": current["timestep"].tolist(),
                               "metrics": {kind: _assert_exact_metrics(current[kind], legacy[kind], f"block{block}.stage{stage}.{kind}")
                                           for kind in ("input", "flow", "x0")},
                               "condition_sha256": current["condition_sha256"],
                               "cache_indices": current["cache_indices"],
                               "full_cache_sha256": current["full_cache_sha256"],
                               "baseline_output_sha256": {kind: _tensor_sha256(legacy[kind]) for kind in ("flow", "x0")},
                               "optimized_output_sha256": {kind: _tensor_sha256(current[kind]) for kind in ("flow", "x0")}})
            assert len(stages) == 4
            native = [core.last_block for core in cores]
            latent = _assert_exact_metrics(native[1].latent, native[0].latent, f"block{block}.latent")
            pixels = _assert_exact_metrics(native[1].video, native[0].video, f"block{block}.pixels")
            assert np.array_equal(outputs[0]["video"], outputs[1]["video"]), f"public RGB8 differs at block{block}"
            assert outputs[1]["video"].shape == (9 if block == 0 else 12, 352, 640, 3)
            recurrent = []
            assert len(sessions[0].vae_cache) == len(sessions[1].vae_cache) == 32
            for slot, (legacy, current) in enumerate(zip(sessions[0].vae_cache, sessions[1].vae_cache)):
                assert (legacy is None) == (current is None)
                if legacy is not None:
                    measured = _assert_exact_metrics(current, legacy, f"block{block}.decoder_cache{slot}")
                    recurrent.append({"slot": slot, **measured, "sha256": _tensor_sha256(current)})
            for family in ("kv_cache", "mouse_kv_cache", "keyboard_kv_cache"):
                for cache in getattr(sessions[1], family):
                    for name, value in _short._cache_indices(cache).items():
                        assert cache["_host_" + name] == value
            contract["blocks"].append({"latent_start": block * 3, "controls": controls, "denoise_and_refresh": stages,
                                        "latent_error": latent, "pixel_error": pixels,
                                        "pixel_byte_error": {"mean_abs": 0.0, "max_abs": 0},
                                        "public_rgb8_shape": list(outputs[1]["video"].shape),
                                        "public_rgb8_sha256": _tensor_sha256(torch.from_numpy(outputs[1]["video"])),
                                        "baseline_latent_sha256": _tensor_sha256(native[0].latent),
                                        "optimized_latent_sha256": _tensor_sha256(native[1].latent),
                                        "baseline_video_sha256": _tensor_sha256(native[0].video),
                                        "optimized_video_sha256": _tensor_sha256(native[1].video),
                                        "recurrent_vae_cache_error": recurrent,
                                        "independent_initial_noise_sha256": _tensor_sha256(expected_input),
                                        "continuous_global_rng_sha256": _tensor_sha256(rng_after)})
            historical.assert_block(key, block, contract["blocks"][-1])
            for trace in traces:
                trace.trace.clear()
            print(f"MG2 conditioned {key}: block {block + 1}/12, current paths and historical receipts bitwise exact", flush=True)
        receipt = qkv_fusion_report(optimized.model)
        assert receipt["fused_blocks"] == 30
        assert receipt[f"eager_{strategy}_projection_calls"] == receipt["eager_projection_calls"] == 1440
        assert cores[1]._decode_overlap_runtime == {"calls": 12, "execution": "decode-stream-overlap-enqueued"}
        assert cores[1]._decode_stream is not None
        assert all(session.current_start_frame == 36 for session in sessions)
        completion = {"qkv_receipt": receipt, "overlap_receipt": dict(cores[1]._decode_overlap_runtime),
                      "inter_step_noise": noise_receipts[0], "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "passed": True}
        historical.assert_complete_contract(key, {**contract, **completion})
        contract.update(**completion, historical_reference_passed=True)
        evidence["historical_reference"]["contracts_passed"].append(key)
    finally:
        for public, core, original in zip(publics, cores, originals):
            core.scheduler.add_noise = original
            public.reset_realtime()
            core.close()
