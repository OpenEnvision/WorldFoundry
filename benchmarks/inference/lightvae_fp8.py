"""Calibrated FP8 encoder versus the same FP32 LightVAE student checkpoint.

This isolates quantization drift. It does not certify student/teacher parity.
Only quality, execution and device-admitted cases may report encoder timings.
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
from benchmarks.inference.calibrate_plugins import execution_sources
from benchmarks.inference.lightvae_encoder import _load_reference, _validate_codec
from benchmarks.inference.plugin_diagnostics import BUDGET, compare_generation, cuda_device_admission, file_manifest
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import load_wan_video_codec
from worldfoundry.base_models.diffusion_model.optimizations.lightvae_fp8 import install_lightvae_fp8
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def qualify_encoder(receipt, modules):
    layers = receipt.get("layers", {})
    checks = {
        "enabled": receipt.get("enabled") is True,
        "all_selected_convolutions_executed": bool(modules)
        and set(layers) == set(modules)
        and all(value.get("kernel_calls", 0) > 0 for value in layers.values()),
        "no_clipping": receipt.get("clipped_input_operands", -1) == 0,
        "no_dense_fallback": receipt.get("dense_fallback_calls", -1) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 3 and not args.quality_only:
        parser.error("at least three paired timing rounds required")
    if len(set(path.resolve() for path in args.reference)) != len(args.reference):
        parser.error("duplicate references do not constitute independent cases")
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    inputs = [path.resolve() for path in (args.checkpoint, args.artifact, *args.reference)]
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "FP8 encoder vs identical FP32 LightVAE student; common unchanged student decoder; not teacher parity",
        "timing_scope": "resident encoder-only wall time; install, calibration and common decoding excluded",
        "quality_aggregation": "worst batch latent L2/PSNR; minimum SSIM across all frames and batch members",
        "budget": BUDGET,
        "cases": [],
        "status": "running",
    }

    def save():
        temp = args.out / "results.tmp"
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temp.replace(args.out / "results.json")

    def encode(codec, pixels, *, enabled):
        session = install_lightvae_fp8(codec.vae, args.artifact) if enabled else None
        try:
            if session:
                session.bind_runtime(codec)
            modules = session.report()["installed"][0]["modules"] if session else []
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                latents = codec.encode(pixels)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            receipt = codec.runtime_optimization_report()["fp8_encoder"]
            gate = qualify_encoder(receipt, modules) if enabled else {"passed": not receipt["enabled"]}
            return latents, elapsed, {"execution_gate": gate, "runtime": receipt}
        finally:
            if session:
                session.uninstall()

    save()
    try:
        record["files"] = file_manifest(inputs, Path("/"))
        record["source_files"] = file_manifest(execution_sources(repo), repo)
        torch.cuda.init()
        record["runtime"] = capture_runtime_fingerprint(device_index=0).to_dict()
        codec = load_wan_video_codec(args.checkpoint, variant="lightvae-wan21")
        _validate_codec(codec, student=True)
        for path in args.reference:
            payload = _load_reference(path)
            pixels = payload["sample"].cuda()
            reference_latents, _, baseline_receipt = encode(codec, pixels, enabled=False)
            candidate_latents, _, candidate_receipt = encode(codec, pixels, enabled=True)
            with torch.inference_mode():
                reference = SimpleNamespace(latents=reference_latents, sample=codec.decode(reference_latents))
                candidate = SimpleNamespace(latents=candidate_latents, sample=codec.decode(candidate_latents))
            row = {
                "reference": str(path.resolve()),
                "video_shape": list(pixels.shape),
                "quality": compare_generation(reference, candidate),
                "baseline_execution": baseline_receipt,
                "candidate_execution": candidate_receipt,
            }
            record["cases"].append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            if not row["quality"]["passed"] or not candidate_receipt["execution_gate"]["passed"]:
                row["status"] = "rejected_quality" if not row["quality"]["passed"] else "rejected_execution"
            else:
                row["device_admission"] = cuda_device_admission()
                if args.quality_only or not row["device_admission"]["timing_qualified"]:
                    row["status"] = "quality_passed_timing_unqualified"
                else:
                    times, devices, receipts = {False: [], True: []}, [], []
                    for round_index in range(args.rounds):
                        for enabled in (False, True) if round_index % 2 == 0 else (True, False):
                            devices.append(cuda_device_admission())
                            output, elapsed, receipt = encode(codec, pixels, enabled=enabled)
                            devices.append(cuda_device_admission())
                            expected = candidate_latents if enabled else reference_latents
                            receipt.update(
                                round=round_index,
                                candidate_enabled=enabled,
                                output_matches_pilot=output.dtype == expected.dtype and torch.equal(output, expected),
                            )
                            receipts.append(receipt)
                            times[enabled].append(elapsed)
                    row.update(
                        reference_wall_s=times[False],
                        candidate_wall_s=times[True],
                        timing_device_samples=devices,
                        timing_execution_receipts=receipts,
                    )
                    if not all(value["timing_qualified"] for value in devices):
                        row["status"] = "rejected_shared_device_timing"
                    elif not all(
                        value["execution_gate"]["passed"] and value["output_matches_pilot"] for value in receipts
                    ):
                        row["status"] = "rejected_timing_execution"
                    else:
                        row.update(
                            status="qualified_for_test_case",
                            speedup_median=statistics.median(times[False]) / statistics.median(times[True]),
                            speedup_ci95=list(_paired_bootstrap_ci(times[False], times[True])),
                        )
            save()
        record["selection"] = (
            "validated_for_cases"
            if all(row["status"] == "qualified_for_test_case" and row["speedup_ci95"][0] > 1 for row in record["cases"])
            else "no_validated_speedup"
        )
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            record["source_unchanged"] = record.get("source_files") == file_manifest(execution_sources(repo), repo)
            record["inputs_unchanged"] = record.get("files") == file_manifest(inputs, Path("/"))
            if not record["source_unchanged"] or not record["inputs_unchanged"]:
                raise RuntimeError("diagnostic sources or inputs changed")
        except BaseException as error:
            failed = record["status"] == "failed"
            record.update(status="failed", verification_error=repr(error))
            if not failed:
                raise
        finally:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
