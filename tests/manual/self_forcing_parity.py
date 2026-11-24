#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare native Self-Forcing or Causal-Forcing with a recorded official reference run.

The reference directory records one request run through the official
Self-Forcing or Causal-Forcing ``inference.py`` and contains ``prompt.txt``, ``initial_noise.pt``,
``renoise_draws.pt``, ``prompt_embeds.pt``, ``step_trace.pt`` (every generator
call), ``latents.pt`` and ``video.pt``; an image-to-video record adds
``i2v_image_input.pt`` and ``initial_latent.pt``. See ``meta.json`` in that
directory.

``--model-id`` selects the native recipe (``self-forcing`` or one of the
``causal-forcing-*`` recipes), loaded through ``NativeSelfForcingPipeline`` or
``NativeCausalForcingPipeline`` and compared stage by stage: checkpoint
conversion, prompt embeddings, for image-to-video the codec input and image
latent, the denoiser on the recorded inputs of all generator calls, the
``autoregressive-window`` rollout with the recorded initial noise and re-noise
draws, and the codec decode of the reference latents as one clip and as streamed
chunks. Noise injection replaces the rollout's noise source on the runner
instance only. Each comparison reports bitwise equality, the maximum and mean
absolute difference, and the maximum absolute difference relative to the
largest reference magnitude. Run in the unified GPU environment on the GPU that
recorded the reference; this is accepted-reference component and rollout
evidence for one request, not a regression-matrix replay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
import torch.version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    EncodedLatentInitializer,
    LatentInitialization,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (  # noqa: E402
    SelfForcingWanDenoiser,
    convert_self_forcing_generator_state_dict,
)
from worldfoundry.base_models.diffusion_model.runners.autoregressive import (  # noqa: E402
    AutoregressiveWindowRunner,
)
from worldfoundry.pipelines.forcing.pipeline_native_forcing import (  # noqa: E402
    NativeCausalForcingPipeline,
    NativeSelfForcingPipeline,
)

COMMIT_KINDS = ("cache_commit", "initial_latent_commit")


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return elementwise agreement of ``actual`` with the reference ``expected``.

    A native tensor in another dtype is first cast to the reference dtype, as
    the denoiser does with its float32 rollout latents; ``cast_lossless``
    records whether that cast was exact.
    """

    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    actual = actual.to(device=expected.device)
    cast = {}
    if actual.dtype != expected.dtype:
        converted = actual.to(dtype=expected.dtype)
        cast = {
            "native_dtype": str(actual.dtype).removeprefix("torch."),
            "cast_lossless": bool(torch.equal(converted.to(dtype=actual.dtype), actual)),
        }
        actual = converted
    actual64 = actual.double()
    expected64 = expected.double()
    difference = (actual64 - expected64).abs()
    max_abs = float(difference.max())
    scale = float(expected64.abs().max())
    return {
        "shape": list(expected.shape),
        "dtype": str(expected.dtype).removeprefix("torch."),
        **cast,
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(difference.mean()),
        "max_rel_diff": max_abs / scale if scale else max_abs,
    }


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-tensor comparisons into one stage verdict."""

    return {
        "bitwise_equal": all(entry["bitwise_equal"] for entry in entries),
        "max_abs_diff": max(entry["max_abs_diff"] for entry in entries),
        "max_rel_diff": max(entry["max_rel_diff"] for entry in entries),
    }


def to_bcthw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert the official ``[B, T, C, H, W]`` latent layout to native ``[B, C, T, H, W]``."""

    return tensor.permute(0, 2, 1, 3, 4)


class FlowCapture:
    """Record the flow prediction of every network call on one model instance."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.flows: list[torch.Tensor] = []

    @contextmanager
    def active(self) -> Generator[None, None, None]:
        forward = self.model.forward

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            flow = forward(*args, **kwargs)
            self.flows.append(flow.detach().clone())
            return flow

        self.model.forward = capture
        try:
            yield
        finally:
            del self.model.forward


class CallRecorder(DiffusionExtension):
    """Copy each denoiser call's input, timestep and x0 prediction, optionally timing it."""

    extension_id = "self-forcing-parity-recorder"

    def __init__(self, *, timing: bool = False) -> None:
        self.timing = timing
        self.calls: list[dict[str, Any]] = []
        self._start = 0.0

    def before_denoiser(self, context: Any, model_input: DenoiserInput) -> DenoiserInput:
        del context
        if self.timing:
            torch.cuda.synchronize()
            self._start = time.perf_counter()
            self.calls.append({"branch": model_input.branch, "start": self._start})
            return model_input
        self.calls.append(
            {
                "branch": model_input.branch,
                "timestep": float(model_input.timestep),
                "current_start": cast(int, model_input.conditioning["current_start"]),
                "noisy_input": model_input.latents.detach().clone(),
            }
        )
        return model_input

    def after_denoiser(self, context: Any, model_output: DenoiserOutput) -> DenoiserOutput:
        del context
        if self.timing:
            torch.cuda.synchronize()
            end = time.perf_counter()
            self.calls[-1].update(end=end, ms=(end - self._start) * 1000.0)
            return model_output
        self.calls[-1]["denoised_pred"] = model_output.sample.detach().clone()
        return model_output


@contextmanager
def instrumented_runner(
    runner: AutoregressiveWindowRunner,
    *,
    initial_noise: torch.Tensor,
    renoise_draws: list[torch.Tensor],
    recorder: CallRecorder,
) -> Generator[list[torch.Tensor], None, None]:
    """Inject reference noise and a recorder into one runner instance, restoring it afterwards.

    An initializer bound to the codec still encodes the request image; only its
    noise is replaced, so the first-frame latent stays the native encoding.
    """

    initializer = runner.components.latent_initializer
    pending = list(renoise_draws)
    encoded = runner.components.latent_encoder is not None
    initialize_name = "initialize_with_encoder" if encoded else "initialize"
    native_initialize = getattr(initializer, initialize_name)

    def initialize(request: DiffusionRequest, **kwargs: Any) -> torch.Tensor | LatentInitialization:
        if not encoded:
            return initial_noise.clone()
        native = native_initialize(request, **kwargs)
        if isinstance(native, LatentInitialization):
            return LatentInitialization(initial_noise.clone(), native.conditioning, native.artifacts)
        return initial_noise.clone()

    def noise_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        del generator
        draw = to_bcthw(pending.pop(0).unflatten(0, (value.shape[0], -1)))
        if draw.shape != value.shape:
            raise ValueError(f"re-noise draw {tuple(draw.shape)} does not match {tuple(value.shape)}")
        return draw.to(device=value.device, dtype=value.dtype)

    extensions = runner.extensions
    setattr(initializer, initialize_name, initialize)
    setattr(runner, "_noise_like", noise_like)
    runner.extensions = (*extensions, recorder)
    try:
        yield pending
    finally:
        runner.extensions = extensions
        delattr(runner, "_noise_like")
        delattr(initializer, initialize_name)


def compare_checkpoint(path: Path, denoiser: SelfForcingWanDenoiser) -> dict[str, Any]:
    """Count key and shape mismatches and compare every loaded tensor with the cast source tensor."""

    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    role = "generator_ema" if "generator_ema" in raw else "generator"
    source = cast(dict[str, torch.Tensor], convert_self_forcing_generator_state_dict(raw))
    loaded = denoiser.model.state_dict()
    shared = sorted(set(source) & set(loaded))
    shape_mismatches = [name for name in shared if tuple(source[name].shape) != tuple(loaded[name].shape)]
    value_mismatches = [
        name
        for name in shared
        if name not in shape_mismatches
        and not torch.equal(loaded[name].cpu(), source[name].to(dtype=loaded[name].dtype))
    ]
    return {
        "state_key": role,
        "source_dtypes": sorted({str(value.dtype).removeprefix("torch.") for value in source.values()}),
        "loaded_dtype": str(next(iter(loaded.values())).dtype).removeprefix("torch."),
        "tensors": len(loaded),
        "missing_keys": len(set(loaded) - set(source)),
        "unexpected_keys": len(set(source) - set(loaded)),
        "shape_mismatches": len(shape_mismatches),
        "value_mismatches_after_cast": len(value_mismatches),
    }


def compare_trace(
    calls: list[dict[str, Any]],
    flows: list[torch.Tensor],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare native generator calls with the recorded official calls, in call order."""

    if len(calls) != len(trace) or len(flows) != len(trace):
        raise ValueError(f"expected {len(trace)} calls, got {len(calls)} denoiser and {len(flows)} network calls")
    per_call = []
    first_divergence = None
    for call, flow, expected in zip(calls, flows, trace, strict=True):
        commit = call["branch"] == "positive-cache-commit"
        kind = (
            expected["kind"] if commit and expected["kind"] in COMMIT_KINDS else "cache_commit" if commit else "denoise"
        )
        timestep = float(expected["timestep"].flatten()[0])
        if (
            kind != expected["kind"]
            or call["timestep"] != timestep
            or call["current_start"] != expected["current_start"]
        ):
            raise ValueError(f"call {expected['call_index']} schedule differs: {call} vs {expected}")
        entry: dict[str, Any] = {
            "call_index": expected["call_index"],
            "block_index": expected["block_index"],
            "kind": kind,
            "step_index": expected["step_index"],
            "timestep": timestep,
            "noisy_input": compare(call["noisy_input"], to_bcthw(expected["noisy_input"])),
            "flow_pred": compare(flow, to_bcthw(expected["flow_pred"])),
            "denoised_pred": compare(call["denoised_pred"], to_bcthw(expected["denoised_pred"])),
        }
        if first_divergence is None:
            for boundary in ("noisy_input", "flow_pred", "denoised_pred"):
                if not entry[boundary]["bitwise_equal"]:
                    first_divergence = {
                        "call_index": entry["call_index"],
                        "block_index": entry["block_index"],
                        "kind": kind,
                        "step_index": entry["step_index"],
                        "boundary": boundary,
                    }
                    break
        per_call.append(entry)
    return {
        **summarize([entry[name] for entry in per_call for name in ("noisy_input", "flow_pred", "denoised_pred")]),
        "calls": len(per_call),
        "first_divergence": first_divergence,
        "per_call": per_call,
    }


def run_teacher_forced(
    denoiser: SelfForcingWanDenoiser,
    context: torch.Tensor,
    trace: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Feed every recorded official generator input, in order, through the native denoiser and one cache."""

    frame_tokens = int(trace[0]["noisy_input"].shape[-2] * trace[0]["noisy_input"].shape[-1] // 4)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=denoiser.cache_frames,
        frame_sequence_length=frame_tokens,
        device=device,
        dtype=denoiser.compute_dtype,
    )
    capture = FlowCapture(denoiser.model)
    calls = []
    with torch.no_grad(), capture.active():
        for entry in trace:
            noisy = to_bcthw(entry["noisy_input"]).to(device)
            timestep = entry["timestep"].flatten()[0].to(device=device, dtype=torch.float32)
            branch = "positive-cache-commit" if entry["kind"] in COMMIT_KINDS else "positive"
            output = denoiser(
                DenoiserInput(
                    latents=noisy,
                    timestep=timestep,
                    next_timestep=torch.zeros((), device=device),
                    conditioning={"context": context, "kv_cache": cache, "current_start": entry["current_start"]},
                    step_index=entry["step_index"] or 0,
                    total_steps=4,
                    branch=branch,
                )
            )
            calls.append(
                {
                    "branch": branch,
                    "timestep": float(timestep),
                    "current_start": entry["current_start"],
                    "noisy_input": noisy,
                    "denoised_pred": output.sample,
                }
            )
    return compare_trace(calls, capture.flows, trace)


def run_rollout(
    runner: AutoregressiveWindowRunner,
    request: DiffusionRequest,
    *,
    initial_noise: torch.Tensor,
    renoise_draws: list[torch.Tensor],
    recorder: CallRecorder,
    capture: FlowCapture | None,
) -> torch.Tensor:
    """Run the native window rollout on the reference noise and return its latents."""

    decoder = runner.components.decoder
    setattr(decoder, "decode", lambda latents, request: latents)
    try:
        with instrumented_runner(
            runner,
            initial_noise=initial_noise,
            renoise_draws=renoise_draws,
            recorder=recorder,
        ) as pending:
            if capture is None:
                output = runner.run(request)
            else:
                with capture.active():
                    output = runner.run(request)
            if pending:
                raise ValueError(f"{len(pending)} re-noise draws were not consumed")
    finally:
        delattr(decoder, "decode")
    return output.latents


def block_timings(recorder: CallRecorder, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group timed calls by the reference blocks: its denoising steps, then one cache commit."""

    blocks: dict[int, list[dict[str, Any]]] = {}
    for call, expected in zip(recorder.calls, trace, strict=True):
        if expected["kind"] != "initial_latent_commit":
            blocks.setdefault(int(expected["block_index"]), []).append(call)
    return [
        {
            "block_index": index,
            "denoise_step_ms": [call["ms"] for call in block[:-1]],
            "denoise_total_ms": sum(call["ms"] for call in block[:-1]),
            "cache_commit_ms": block[-1]["ms"],
            "block_wall_ms": (block[-1]["end"] - block[0]["start"]) * 1000.0,
        }
        for index, block in sorted(blocks.items())
    ]


def reference_blocks(trace: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Latent frame range ``[start, end)`` of every denoised reference block, in order."""

    frame_tokens = int(trace[0]["noisy_input"].shape[-2] * trace[0]["noisy_input"].shape[-1] // 4)
    ranges: dict[int, tuple[int, int]] = {}
    for entry in trace:
        if entry["kind"] == "cache_commit":
            start = int(entry["current_start"]) // frame_tokens
            ranges[int(entry["block_index"])] = (start, start + int(entry["noisy_input"].shape[1]))
    return [ranges[index] for index in sorted(ranges)]


def to_reference_video(pixels: torch.Tensor) -> torch.Tensor:
    """Map native ``[-1, 1]`` BCTHW pixels to the official ``[0, 1]`` BTCHW float32 output.

    The official pipeline returns ``(clamp(decode.float(), -1, 1) * 0.5 + 0.5).clamp(0, 1)``.
    """

    return (pixels.float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4).cpu()


def compare_decode(
    codec: Any,
    latents: torch.Tensor,
    expected_video: torch.Tensor,
    request: DiffusionRequest,
    chunk_ends: list[int],
) -> dict[str, Any]:
    """Decode the reference latents as one clip and as the realtime session's streamed chunks."""

    timings = []
    with torch.no_grad():
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            pixels = codec.decode(latents, request)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
        whole = {
            "decoder_output_dtype": str(pixels.dtype).removeprefix("torch."),
            "cold_ms": timings[0],
            "warm_ms": timings[1],
            **compare(to_reference_video(pixels), expected_video),
        }
        del pixels
        state = codec.new_stream_state()
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunks = [
            codec.decode_chunk(latents[:, :, start:end], state=state, is_first_chunk=start == 0)
            for start, end in zip([0, *chunk_ends[:-1]], chunk_ends, strict=True)
        ]
        torch.cuda.synchronize()
        streamed = {
            "chunks": len(chunks),
            "ms": (time.perf_counter() - start) * 1000.0,
            **compare(to_reference_video(torch.cat(chunks, dim=2)), expected_video),
        }
    return {"whole_clip": whole, "block_chunks": streamed}


def compare_image_latent(
    runner: AutoregressiveWindowRunner,
    request: DiffusionRequest,
    reference: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Compare the codec input and first-frame latent the native initializer builds from the image."""

    encoder = runner.components.latent_encoder
    initializer = runner.components.latent_initializer
    if encoder is None or not isinstance(initializer, EncodedLatentInitializer):
        raise TypeError("an image-to-video comparison needs a recipe that binds the latent encoder")
    pixels: list[torch.Tensor] = []
    encode = encoder.encode

    def record(images: torch.Tensor) -> torch.Tensor:
        pixels.append(images.detach().clone())
        return encode(images)

    setattr(encoder, "encode", record)
    try:
        initialized = initializer.initialize_with_encoder(
            request,
            latent_encoder=encoder,
            generator=torch.Generator(device=device).manual_seed(0),
            device=device,
            dtype=runner.dtype,
        )
    finally:
        delattr(encoder, "encode")
    if not isinstance(initialized, LatentInitialization):
        raise TypeError("the image request did not produce a first-frame latent")
    first = cast(torch.Tensor, initialized.conditioning["first_frame_latent"])

    def load(name: str) -> torch.Tensor:
        return torch.load(reference / name, map_location="cpu", weights_only=True).to(device)

    return {
        "codec_input": compare(pixels[0], load("i2v_image_input.pt")),
        "latent": compare(first, to_bcthw(load("initial_latent.pt"))),
    }


def main() -> None:
    """Load the native pipeline once and compare every stage with the reference record."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, required=True, help="Recorded official run directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Generator checkpoint, e.g. self_forcing_dmd.pt")
    parser.add_argument("--wan-root", type=Path, required=True, help="Wan2.1-T2V-1.3B official layout directory")
    parser.add_argument(
        "--model-id",
        default="self-forcing",
        help="self-forcing, causal-forcing-chunkwise, causal-forcing-framewise, causal-forcing-framewise-1step "
        "or causal-forcing-framewise-2step",
    )
    parser.add_argument("--image", type=Path, help="Source image of an image-to-video record")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/self-forcing-parity.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Forcing reference parity requires the CUDA device that recorded the reference")
    reference = args.reference_dir
    meta = json.loads((reference / "meta.json").read_text(encoding="utf-8"))

    def load(name: str) -> Any:
        return torch.load(reference / name, map_location="cpu", weights_only=True)

    load_start = time.perf_counter()
    pipeline_class = NativeSelfForcingPipeline if args.model_id == "self-forcing" else NativeCausalForcingPipeline
    pipeline = pipeline_class.from_pretrained(
        model_id=args.model_id,
        device=str(device),
        torch_dtype="bf16",
        offload_mode="none",
        checkpoint_overrides={
            "dit": str(args.checkpoint),
            "text-encoder": str(args.wan_root),
            "tokenizer": str(args.wan_root),
            "vae": str(args.wan_root),
        },
    )
    load_seconds = time.perf_counter() - load_start
    runner = pipeline.native_pipeline.runner
    if not isinstance(runner, AutoregressiveWindowRunner):
        raise TypeError(f"expected the autoregressive-window runner, got {type(runner).__name__}")
    denoiser = runner.components.denoiser
    if not isinstance(denoiser, SelfForcingWanDenoiser):
        raise TypeError(f"expected SelfForcingWanDenoiser, got {type(denoiser).__name__}")

    prompt = (reference / "prompt.txt").read_text(encoding="utf-8")
    image = None if args.image is None else str(args.image)
    request, _ = pipeline._prepare_call(prompt=prompt, images=image, seed=int(meta["request"].get("seed", 42)))
    report: dict[str, Any] = {
        "evidence_level": "accepted-reference components and rollout (one request)",
        "model_id": pipeline.model_id,
        "reference_dir": str(reference),
        "reference_variant": meta.get("variant"),
        "reference_commit": meta.get("commit"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "reference_gpu": meta["versions"].get("gpu_name"),
        "dtype": "bfloat16",
        "backend_flags": {
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "load_seconds": round(load_seconds, 2),
        "request": {
            "prompt_matches_reference": request.prompt == prompt,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_inference_steps": request.sampling.num_inference_steps,
            "guidance_scale": request.sampling.guidance_scale,
            "scheduler_options": dict(request.sampling.scheduler_options),
        },
    }

    report["checkpoint"] = compare_checkpoint(args.checkpoint, denoiser)

    conditioning = runner.components.conditioner.encode(request, device=device, dtype=runner.dtype)
    context = conditioning.positive["context"]
    report["text"] = compare(context, load("prompt_embeds.pt").to(device))

    if image is not None:
        report["image"] = compare_image_latent(runner, request, reference, device)

    trace = load("step_trace.pt")
    report["denoiser_teacher_forced"] = run_teacher_forced(denoiser, context, trace, device)

    # The native initializer draws float32 noise; the bf16 reference noise is exact in float32.
    initial_noise = to_bcthw(load("initial_noise.pt")).to(device=device, dtype=torch.float32).contiguous()
    if image is not None:
        # The official I2V noise covers frames 1..; frame 0 is the image latent, so its noise is never read.
        initial_noise = torch.cat((torch.zeros_like(initial_noise[:, :, :1]), initial_noise), dim=2)
    # Device-resident draws keep host-to-device copies out of the block timings.
    renoise_draws = [draw["tensor"].to(device) for draw in load("renoise_draws.pt")]
    expected_latents = to_bcthw(load("latents.pt")).to(device)
    recorder = CallRecorder()
    capture = FlowCapture(denoiser.model)
    latents = run_rollout(
        runner,
        request,
        initial_noise=initial_noise,
        renoise_draws=renoise_draws,
        recorder=recorder,
        capture=capture,
    )
    block_ranges = reference_blocks(trace)
    blocks = [
        {
            "block_index": index,
            "frames": [start, end],
            **compare(latents[:, :, start:end], expected_latents[:, :, start:end]),
        }
        for index, (start, end) in enumerate(block_ranges)
    ]
    report["rollout"] = {
        "final_latents": compare(latents, expected_latents),
        "first_divergent_block": next((block["block_index"] for block in blocks if not block["bitwise_equal"]), None),
        "blocks": blocks,
        "calls": compare_trace(recorder.calls, capture.flows, trace),
    }

    repeat = run_rollout(
        runner,
        request,
        initial_noise=initial_noise,
        renoise_draws=renoise_draws,
        recorder=CallRecorder(),
        capture=None,
    )
    report["rollout"]["repeat_bitwise_equal"] = bool(torch.equal(repeat, latents))

    timed = CallRecorder(timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    run_rollout(
        runner,
        request,
        initial_noise=initial_noise,
        renoise_draws=renoise_draws,
        recorder=timed,
        capture=None,
    )
    torch.cuda.synchronize()
    rollout_ms = (time.perf_counter() - start) * 1000.0
    native_blocks = block_timings(timed, trace)
    reference_timings: Mapping[str, Any] = meta["timings"]
    report["timing"] = {
        "method": "Third native rollout in the process, eager; torch.cuda.synchronize() before and after "
        "every denoiser call. block_wall_ms spans the first step's start to the cache commit's end.",
        "native": {
            "blocks": native_blocks,
            "denoise_total_ms": sum(block["denoise_total_ms"] for block in native_blocks),
            "cache_commit_total_ms": sum(block["cache_commit_ms"] for block in native_blocks),
            "block_wall_total_ms": sum(block["block_wall_ms"] for block in native_blocks),
            "rollout_without_decode_ms": rollout_ms,
        },
        "reference": {
            "block_wall_ms": [block["block_wall_ms"] for block in reference_timings["blocks"]],
            "denoise_total_ms": reference_timings["denoise_total_ms"],
            "cache_commit_total_ms": reference_timings["cache_commit_total_ms"],
            "block_wall_total_ms": reference_timings["block_wall_total_ms"],
            "decode_ms": reference_timings["decode_ms"],
        },
    }

    # Realtime chunks: one per block; an image-seeded first chunk also carries frame 0.
    report["decode"] = compare_decode(
        runner.components.decoder,
        to_bcthw(load("latents.pt")).to(device=device, dtype=latents.dtype),
        load("video.pt"),
        request,
        [end for _, end in block_ranges],
    )
    report["timing"]["native"]["decode_ms"] = report["decode"]["whole_clip"]["warm_ms"]

    report["summary"] = {
        "checkpoint_strict": all(
            report["checkpoint"][name] == 0
            for name in ("missing_keys", "unexpected_keys", "shape_mismatches", "value_mismatches_after_cast")
        ),
        "text_bitwise_equal": report["text"]["bitwise_equal"],
        "image_latent_bitwise_equal": report["image"]["latent"]["bitwise_equal"] if image is not None else None,
        "denoiser_teacher_forced_bitwise_equal": report["denoiser_teacher_forced"]["bitwise_equal"],
        "rollout_final_latents_bitwise_equal": report["rollout"]["final_latents"]["bitwise_equal"],
        "rollout_first_divergence": report["rollout"]["calls"]["first_divergence"],
        "decode_bitwise_equal": report["decode"]["whole_clip"]["bitwise_equal"],
        "decode_chunk_bitwise_equal": report["decode"]["block_chunks"]["bitwise_equal"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
