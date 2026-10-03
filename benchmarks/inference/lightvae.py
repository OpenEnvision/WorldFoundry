"""Matched LightVAE quality and decode latency on previously generated Wan16 latents.

The teacher is decoded again and checked against the saved RGB video before
accepting a student comparison. This diagnostic measures the decoder only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.harness import _paired_bootstrap_ci
from benchmarks.inference.plugin_diagnostics import BUDGET, compare_generation, cuda_device_admission, file_manifest
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import load_wan_video_codec
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def _execution_source_paths(repo):
    """Cover codec computation and the shared attention/loading implementation."""
    paths = [
        Path(__file__).resolve(),
        repo / "benchmarks/harness.py",
        repo / "benchmarks/inference/plugin_diagnostics.py",
        repo / "worldfoundry/runtime/performance.py",
    ]
    paths += sorted((repo / "worldfoundry/core").rglob("*.py"))
    paths += sorted((repo / "worldfoundry/base_models/diffusion_model").rglob("*.py"))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--saved-codec-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--saved-codec-tiled", action="store_true")
    parser.add_argument("--saved-tile-size", type=int, nargs=2, default=(34, 34))
    parser.add_argument("--saved-tile-stride", type=int, nargs=2, default=(18, 16))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or not args.quality_only and args.rounds < 3:
        parser.error("performance diagnostics need at least three rounds")
    if len({path.resolve() for path in args.reference}) != len(args.reference):
        parser.error("duplicate reference files are not independent cases")
    if any(
        stride <= 0 or size < stride for size, stride in zip(args.saved_tile_size, args.saved_tile_stride, strict=True)
    ):
        parser.error("saved codec tiles need positive strides no larger than tile sizes")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.init()
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "matched Wan2.1 teacher/student decode of identical real generated normalized latents; not end-to-end speed",
        "budget": BUDGET,
        "quality_aggregation": "minimum SSIM over all frames; worst batch video PSNR",
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "parameters": vars(args)
        | {
            "teacher": str(args.teacher),
            "student": str(args.student),
            "out": str(args.out),
            "reference": [str(path) for path in args.reference],
        },
        "cases": [],
        "status": "running",
    }

    def save():
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "results.json")

    save()
    try:
        paths = [path.resolve() for path in (args.teacher, args.student, *args.reference)]
        record["files"] = file_manifest(paths, Path("/"))
        repo = Path(__file__).resolve().parents[2]
        source_paths = _execution_source_paths(repo)
        record["source_files"] = file_manifest(source_paths, repo)
        teacher = load_wan_video_codec(args.teacher, device="cuda", dtype=torch.float32)
        student = load_wan_video_codec(args.student, variant="lightvae-wan21", device="cuda", dtype=torch.float32)
        saved_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.saved_codec_dtype]
        saved_codec = (
            teacher
            if saved_dtype is torch.float32 and not args.saved_codec_tiled
            else load_wan_video_codec(args.teacher, device="cuda", dtype=saved_dtype)
        )
        if args.saved_codec_tiled:
            saved_codec.tiled = True
            saved_codec.tile_size = tuple(args.saved_tile_size)
            saved_codec.tile_stride = tuple(args.saved_tile_stride)

        def decode(codec, latents):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                video = codec.decode(latents)
            torch.cuda.synchronize()
            return video, time.perf_counter() - start

        for index, reference in enumerate(args.reference):
            payload = torch.load(reference, map_location="cpu", weights_only=True)
            if set(payload) != {"sample", "latents"}:
                raise ValueError("saved generation must contain only sample and latents tensors")
            latents = payload["latents"].to("cuda")
            if latents.ndim != 5 or latents.shape[1] != 16:
                raise ValueError("LightVAE21 requires normalized 16-channel BCFHW latents")
            baseline, _ = decode(teacher, latents)
            saved_baseline = baseline if saved_codec is teacher else decode(saved_codec, latents)[0]
            # Native generation converts final RGB to FP32 for storage; this
            # cast preserves the actual low-precision decoder values.
            torch.testing.assert_close(saved_baseline.float().cpu(), payload["sample"].float(), rtol=1e-4, atol=1e-5)
            precision_control = compare_generation(
                SimpleNamespace(sample=payload["sample"], latents=payload["latents"]),
                SimpleNamespace(sample=baseline.cpu(), latents=payload["latents"]),
            )
            del saved_baseline
            candidate, _ = decode(student, latents)
            quality = compare_generation(
                SimpleNamespace(sample=baseline, latents=latents), SimpleNamespace(sample=candidate, latents=latents)
            )
            row = {
                "reference": reference.name,
                "video_shape": list(baseline.shape),
                "latent_shape": list(latents.shape),
                "saved_codec_matches_saved_rgb": True,
                "saved_codec_dtype": args.saved_codec_dtype,
                "saved_codec_tiled": args.saved_codec_tiled,
                "comparison_codec_dtype": "fp32",
                "fp32_teacher_vs_saved_rgb": precision_control,
                "quality": quality,
                "teacher_execution": teacher.runtime_optimization_report(),
                "student_execution": student.runtime_optimization_report(),
                "teacher_parameters": sum(value.numel() for value in teacher.vae.parameters()),
                "student_parameters": sum(value.numel() for value in student.vae.parameters()),
            }
            record["cases"].append(row)
            execution = row["student_execution"]
            row["execution_gate"] = {
                "passed": execution.get("effective", {}).get("vae_variant") == "lightvae-wan21"
                and int(execution.get("runtime", {}).get("lightvae_decode_calls", 0)) > 0,
            }
            torch.save({"sample": candidate.cpu(), "latents": payload["latents"]}, args.out / f"student_{index}.pt")
            print(json.dumps({"reference": reference.name, "quality": quality}), flush=True)
            del candidate, baseline, payload
            if not quality["passed"] or not row["execution_gate"]["passed"]:
                row["status"] = "rejected_quality" if not quality["passed"] else "rejected_execution"
                save()
                continue
            admission = cuda_device_admission()
            row["device_admission"] = admission
            if args.quality_only or not admission["timing_qualified"]:
                row["status"] = "quality_passed_timing_unqualified"
                save()
                continue
            teacher_times, student_times, devices = [], [], []
            for round_index in range(args.rounds):
                for use_student in (False, True) if round_index % 2 == 0 else (True, False):
                    devices.append(cuda_device_admission())
                    video, elapsed = decode(student if use_student else teacher, latents)
                    (student_times if use_student else teacher_times).append(elapsed)
                    del video
                    devices.append(cuda_device_admission())
            row.update(teacher_wall_s=teacher_times, student_wall_s=student_times, timing_device_samples=devices)
            if not all(item["timing_qualified"] for item in devices):
                row["status"] = "rejected_shared_device_timing"
            else:
                row.update(
                    status="qualified_for_test_case",
                    speedup_median=statistics.median(teacher_times) / statistics.median(student_times),
                    speedup_ci95=list(_paired_bootstrap_ci(teacher_times, student_times)),
                )
            save()
        record["source_unchanged"] = file_manifest(source_paths, repo) == record["source_files"]
        record["inputs_unchanged"] = file_manifest(paths, Path("/")) == record["files"]
        if not record["source_unchanged"] or not record["inputs_unchanged"]:
            raise RuntimeError("source or inputs changed during diagnostics")
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
