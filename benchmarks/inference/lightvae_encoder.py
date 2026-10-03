"""Matched LightVAE21 encoder quality and latency on real generated Wan RGB.

The native FP32 untiled teacher first verifies saved RGB provenance. Both
encoders consume that same RGB, and the unchanged teacher decodes both results.
Only cases passing quality and execution gates can qualify encoder-only timing.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.harness import _paired_bootstrap_ci
from benchmarks.inference.plugin_diagnostics import BUDGET, compare_generation, cuda_device_admission, file_manifest
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    Wan21LightVAECodec,
    WanVideoDecoder,
    load_wan_video_codec,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.light_21 import Wan21LightVAE
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 1 or not args.quality_only and args.rounds < 3:
        parser.error("performance diagnostics need at least three paired rounds")
    if len({path.resolve() for path in args.reference}) != len(args.reference):
        parser.error("duplicate reference files are not independent cases")
    return args


def _load_reference(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"sample", "latents"}:
        raise ValueError("saved generation must contain only sample and latents tensors")
    pixels, latents = payload["sample"], payload["latents"]
    if not all(isinstance(value, torch.Tensor) and value.is_floating_point() for value in (pixels, latents)):
        raise ValueError("saved generation requires floating-point tensors")
    if pixels.dtype != torch.float32 or pixels.ndim != 5 or pixels.shape[1] != 3 or min(pixels.shape) <= 0:
        raise ValueError("saved RGB must be nonempty FP32 BCTHW with three channels")
    if latents.ndim != 5 or latents.shape[1] != 16 or min(latents.shape) <= 0:
        raise ValueError("LightVAE21 requires normalized 16-channel BCFHW latents")
    batch, _, frames, height, width = pixels.shape
    if (frames - 1) % 4 or height % 8 or width % 8:
        raise ValueError("Wan21 RGB requires 4n+1 frames and spatial dimensions divisible by eight")
    if tuple(latents.shape) != (batch, 16, 1 + (frames - 1) // 4, height // 8, width // 8):
        raise ValueError("saved RGB and normalized latents have inconsistent Wan21 geometry")
    if not all(bool(torch.isfinite(value).all()) for value in (pixels, latents)):
        raise ValueError("saved generation must be finite")
    if bool((pixels.abs() > 1).any()):
        raise ValueError("saved RGB must be normalized to [-1, 1]")
    return payload


def _source_paths(repo):
    # Loading traverses converters, policy, IO, VRAM and runtime helpers. Hash
    # the full local Python source trees rather than an incomplete short list.
    return sorted({path for name in ("worldfoundry", "benchmarks") for path in (repo / name).rglob("*.py")})


def _validate_codec(codec, *, student):
    expected_codec = Wan21LightVAECodec if student else WanVideoDecoder
    expected_vae = Wan21LightVAE if student else WanVideoVAE
    if type(codec) is not expected_codec or type(codec.vae) is not expected_vae:
        raise TypeError("encoder diagnostic requires the matched native Wan21 teacher/student architectures")
    if codec.dtype != torch.float32 or any(module.training for module in codec.vae.modules()):
        raise ValueError("encoder diagnostic requires FP32 eval codecs")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in codec.vae.state_dict().values()):
        raise ValueError("all codec floating-point weights must be FP32")
    if codec.tiled or codec.temporal_chunk_size or codec.parallel_degree != 1:
        raise ValueError("encoder diagnostic requires untiled resident codecs without extra chunking or parallelism")
    if codec.offload_effective != "resident" or codec.vae.decode_autocast_dtype is not None:
        raise ValueError("encoder diagnostic requires resident FP32 decoding without autocast")


def _encode_pilot(codec, pixels):
    calls = 0

    def observed(module, inputs, output):
        nonlocal calls
        calls += 1

    before = codec.runtime_optimization_report()
    handle = codec.vae.model.encoder.register_forward_hook(observed)
    try:
        with torch.inference_mode():
            latents = codec.encode(pixels)
    finally:
        handle.remove()
    after = codec.runtime_optimization_report()
    return latents, {
        "encoder_module_calls": calls,
        "expected_encoder_module_calls": pixels.shape[0] * (1 + (pixels.shape[2] - 1) // 4),
        "latent_dtype": str(latents.dtype),
        "lightvae_encode_call_delta": int(after["runtime"].get("lightvae_encode_calls", 0))
        - int(before["runtime"].get("lightvae_encode_calls", 0)),
        "report": after,
    }


def _execution_gate(teacher_receipt, student_receipt, teacher_decode_before, teacher_decode_after):
    teacher_report, student_report = teacher_receipt["report"], student_receipt["report"]
    checks = {
        "teacher_encoder_executed": teacher_receipt["encoder_module_calls"]
        == teacher_receipt["expected_encoder_module_calls"]
        > 0,
        "student_encoder_executed": student_receipt["encoder_module_calls"]
        == student_receipt["expected_encoder_module_calls"]
        > 0,
        "student_variant": student_report["effective"].get("vae_variant") == "lightvae-wan21",
        "student_encode_count": student_receipt["lightvae_encode_call_delta"] == 1,
        "encoded_latents_fp32": teacher_receipt["latent_dtype"] == student_receipt["latent_dtype"] == "torch.float32",
        "teacher_is_original": not teacher_report["effective"].get("vae_variant")
        and teacher_report["quality_tier"] == "exact",
        "student_decoder_unused": int(student_report["runtime"]["lifetime"].get("lightvae_decode_calls", 0)) == 0,
        "common_teacher_decoder_executed": int(teacher_decode_after["runtime"]["lifetime"]["dense_decode_calls"])
        - int(teacher_decode_before["runtime"]["lifetime"]["dense_decode_calls"])
        == 2,
        "no_fallbacks": not any(
            report.get("fallbacks") for report in (teacher_report, student_report, teacher_decode_after)
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _evaluate_case(teacher, student, payload, row):
    original_latents = payload["latents"].to(device=teacher.device, dtype=torch.float32)
    pixels = payload["sample"].to(device=teacher.device, dtype=torch.float32)
    with torch.inference_mode():
        provenance_rgb = teacher.decode(original_latents)
    torch.testing.assert_close(provenance_rgb.float().cpu(), payload["sample"], rtol=1e-4, atol=1e-5)
    row["saved_rgb_provenance"] = {
        "passed": True,
        "codec_dtype": "fp32",
        "codec_tiled": False,
        "rtol": 1e-4,
        "atol": 1e-5,
        "teacher_execution": teacher.runtime_optimization_report(),
    }
    del provenance_rgb, original_latents
    teacher_latents, teacher_receipt = _encode_pilot(teacher, pixels)
    student_latents, student_receipt = _encode_pilot(student, pixels)
    decode_before = teacher.runtime_optimization_report()
    with torch.inference_mode():
        teacher_rgb = teacher.decode(teacher_latents)
        student_rgb = teacher.decode(student_latents)
    decode_after = teacher.runtime_optimization_report()
    reference = SimpleNamespace(sample=teacher_rgb, latents=teacher_latents)
    candidate = SimpleNamespace(sample=student_rgb, latents=student_latents)
    row.update(
        video_shape=list(pixels.shape),
        latent_shape=list(teacher_latents.shape),
        comparison_codec_dtype="fp32",
        comparison_codec_tiled=False,
        quality=compare_generation(reference, candidate),
        teacher_encoder_execution=teacher_receipt,
        student_encoder_execution=student_receipt,
        common_teacher_decoder_execution=decode_after,
        execution_gate=_execution_gate(teacher_receipt, student_receipt, decode_before, decode_after),
    )
    return pixels, reference, candidate


def _measure_encode(codec, pixels, pilot_latents, *, student):
    before = codec.runtime_optimization_report()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        latents = codec.encode(pixels)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    after = codec.runtime_optimization_report()
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("encoder timing must be positive and finite")
    delta = int(after["runtime"].get("lightvae_encode_calls", 0)) - int(
        before["runtime"].get("lightvae_encode_calls", 0)
    )
    checks = {
        "output_matches_pilot": latents.dtype == pilot_latents.dtype and torch.equal(latents, pilot_latents),
        "student_variant_and_counter": not student
        or after["effective"].get("vae_variant") == "lightvae-wan21"
        and delta == 1,
        "no_fallbacks": not after.get("fallbacks"),
    }
    return elapsed, {"passed": all(checks.values()), "checks": checks, "lightvae_encode_call_delta": delta}


def _time_case(teacher, student, pixels, reference, candidate, row, *, rounds, quality_only):
    if not row["quality"]["passed"] or not row["execution_gate"]["passed"]:
        row["status"] = "rejected_quality" if not row["quality"]["passed"] else "rejected_execution"
        return
    row["device_admission"] = cuda_device_admission()
    if quality_only or not row["device_admission"]["timing_qualified"]:
        row["status"] = "quality_passed_timing_unqualified"
        return
    if rounds < 3:
        raise ValueError("performance diagnostics need at least three paired rounds")
    teacher_times, student_times, devices, receipts = [], [], [], []
    for round_index in range(rounds):
        for use_student in (False, True) if round_index % 2 == 0 else (True, False):
            devices.append(cuda_device_admission())
            elapsed, receipt = _measure_encode(
                student if use_student else teacher,
                pixels,
                candidate.latents if use_student else reference.latents,
                student=use_student,
            )
            (student_times if use_student else teacher_times).append(elapsed)
            receipts.append({"round": round_index, "encoder": "student" if use_student else "teacher", **receipt})
            devices.append(cuda_device_admission())
    row.update(
        teacher_wall_s=teacher_times,
        student_wall_s=student_times,
        timing_device_samples=devices,
        timing_execution_receipts=receipts,
    )
    if not all(item["timing_qualified"] for item in devices):
        row["status"] = "rejected_shared_device_timing"
    elif not all(item["passed"] for item in receipts):
        row["status"] = "rejected_timing_execution"
    else:
        row.update(
            status="qualified_for_test_case",
            speedup_median=statistics.median(teacher_times) / statistics.median(student_times),
            speedup_ci95=list(_paired_bootstrap_ci(teacher_times, student_times)),
        )


def main(argv=None):
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    paths = [path.resolve() for path in (args.teacher, args.student, *args.reference)]
    source_paths = _source_paths(repo)
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "matched Wan21 encoder diagnostic on real saved RGB; common native teacher decode; not end-to-end speed",
        "budget": BUDGET,
        "quality_aggregation": "worst batch latent relative L2 and video PSNR; minimum SSIM over every frame",
        "timing_scope": "paired resident FP32 untiled encoder-only wall time; decode, loading and pilot hooks excluded",
        "timing_receipts": "pilot encoder module hooks; every timed output must match its pilot and student count advance",
        "parameters": vars(args)
        | {key: str(getattr(args, key)) for key in ("teacher", "student", "out")}
        | {"reference": [str(path) for path in args.reference]},
        "cases": [],
        "status": "running",
    }

    def save():
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "results.json")

    save()
    try:
        record["files"] = file_manifest(paths, Path("/"))
        record["source_files"] = file_manifest(source_paths, repo)
        save()
        torch.cuda.init()
        record["runtime"] = capture_runtime_fingerprint(device_index=0).to_dict()
        teacher = load_wan_video_codec(args.teacher, device="cuda", dtype=torch.float32)
        student = load_wan_video_codec(args.student, variant="lightvae-wan21", device="cuda", dtype=torch.float32)
        _validate_codec(teacher, student=False)
        _validate_codec(student, student=True)
        record["teacher_parameters"] = sum(value.numel() for value in teacher.vae.parameters())
        record["student_parameters"] = sum(value.numel() for value in student.vae.parameters())
        for index, path in enumerate(args.reference):
            row = {"reference": str(path.resolve()), "status": "running"}
            record["cases"].append(row)
            pixels, reference, candidate = _evaluate_case(teacher, student, _load_reference(path), row)
            for name, output in (("teacher", reference), ("student", candidate)):
                torch.save(
                    {"sample": output.sample.cpu(), "latents": output.latents.cpu()},
                    args.out / f"{name}_encoded_{index}.pt",
                )
            print(json.dumps({"reference": str(path), "quality": row["quality"]}, allow_nan=False), flush=True)
            _time_case(
                teacher, student, pixels, reference, candidate, row, rounds=args.rounds, quality_only=args.quality_only
            )
            del pixels, reference, candidate
            save()
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            if "source_files" in record:
                record["source_files_after"] = file_manifest(_source_paths(repo), repo)
                record["source_unchanged"] = record["source_files_after"] == record["source_files"]
            if "files" in record:
                record["files_after"] = file_manifest(paths, Path("/"))
                record["inputs_unchanged"] = record["files_after"] == record["files"]
            if not record.get("source_unchanged", True) or not record.get("inputs_unchanged", True):
                raise RuntimeError("source or inputs changed during diagnostics")
        except BaseException as error:
            already_failed = record["status"] == "failed"
            record.update(status="failed", manifest_verification_error=repr(error))
            if not already_failed:
                raise
        finally:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
