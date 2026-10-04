"""Real native SANA/Wan plugin diagnostics with fixed quality and execution gates.

Candidate JSON maps names to acceleration options. Loading/JIT/install time is
excluded from paired wall time. Shared-device runs can verify correctness, but
never qualify timing or select a purported performance winner.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.harness import _paired_bootstrap_ci
from benchmarks.inference.plugin_diagnostics import (
    BUDGET,
    attention_scope_receipts,
    compare_generation,
    cuda_device_admission,
    file_metadata,
    qualify_execution,
)
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.core.acceleration.quantization.linear import (
    quantization_runtime_report,
    reset_quantization_runtime_window,
)
from worldfoundry.core.attention.backends.dispatch import (
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.model_loading.policy import RuntimePolicy
from worldfoundry.runtime.performance import capture_runtime_fingerprint

_MODELS = ("sana-video-2b-480p", "wan2.1-t2v-1.3b")


def _qualify_timed_execution(options, evidence, pilot_evidence):
    """Validate fresh timing receipts against the untimed scope proof.

    Per-module attention hooks belong to the pilot. Timed requests retain
    their own installed scope manifest and provider counters; their successful
    provider call counts must match the pilot without errors or fallbacks.
    """
    gate = (
        qualify_execution(options, evidence | {"attention_scopes": pilot_evidence.get("attention_scopes", {})})
        if options
        else {"passed": True, "plugins": {}}
    )

    def successes(report):
        return {
            backend: int(counters.get("successes", 0))
            for backend, counters in report.items()
            if int(counters.get("successes", 0)) > 0
        }

    providers = evidence.get("attention", {})
    checks = {
        "attention_configuration_matches_pilot": evidence.get("attention_configuration", {})
        == pilot_evidence.get("attention_configuration", {}),
        "attention_call_counts_match_pilot": successes(providers) == successes(pilot_evidence.get("attention", {})),
        "attention_no_errors_or_fallbacks": not any(
            int(counters.get(key, 0))
            for counters in providers.values()
            for key in ("errors", "fallbacks", "quarantined_skips")
        ),
    }
    return {
        "passed": gate["passed"] and all(checks.values()),
        "plugins": gate["plugins"],
        "timing_checks": checks,
        "attention_scope_proof": "untimed pilot hooks; fresh timed configuration and provider counts",
    }


def _assets(model_id, root):
    if model_id == "sana-video-2b-480p":
        model = root / "Efficient-Large-Model--SANA-Video_2B_480p"
        gemma = root / "Efficient-Large-Model--gemma-2-2b-it"
        shards = sorted(gemma.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"Gemma safetensors shards missing: {gemma}")
        overrides = {
            "dit": str(model / "checkpoints/SANA_Video_2B_480p.pth"),
            "text-encoder": CheckpointSpec(source=str(gemma), files=tuple(path.name for path in shards)),
            "tokenizer": str(gemma),
            "codec": str(model / "vae/Wan2.1_VAE.pth"),
        }
        weights = [Path(overrides["dit"]), Path(overrides["codec"]), *shards]
    else:
        model = root / "Wan-AI--Wan2.1-T2V-1.3B"
        overrides = {
            "dit": str(model / "diffusion_pytorch_model.safetensors"),
            "text-encoder": str(model / "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": str(model),
            "vae": str(model / "Wan2.1_VAE.pth"),
        }
        weights = [Path(overrides[key]) for key in ("dit", "text-encoder", "vae")]
    return overrides, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=_MODELS, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument(
        "--prompt",
        default="A small red fox walks through a snowy forest, soft morning sunlight, cinematic tracking shot.",
    )
    args = parser.parse_args()
    if (
        min(args.height, args.width, args.frames, args.steps, args.rounds) <= 0
        or not args.quality_only
        and args.rounds < 3
    ):
        parser.error("positive geometry/steps required; performance diagnostics need at least three rounds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds do not establish independent cases")
    candidates = json.loads(args.candidates.read_text())
    if (
        not isinstance(candidates, dict)
        or not candidates
        or not all(
            isinstance(name, str) and isinstance(options, dict) and options for name, options in candidates.items()
        )
    ):
        parser.error("candidate JSON must map nonempty names to nonempty plugin configurations")
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.assets.resolve()
    overrides, weights = _assets(args.model, assets)
    policy = RuntimePolicy(device="cuda", dtype=torch.bfloat16, options={"dit_weight_dtype": "bf16"})
    torch.cuda.init()
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real native generation diagnostics; fixed cases, not model-wide quality certification",
        "budget": BUDGET,
        "quality_aggregation": "worst batch latent relative L2 and video PSNR; minimum SSIM over all frames and batch members",
        "parameters": vars(args) | {key: str(getattr(args, key)) for key in ("assets", "candidates", "out")},
        "candidates": candidates,
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "device_admission": cuda_device_admission(),
        "timing_scope": "paired wall time including text encoding, denoise and VAE; loading/JIT/plugin installation excluded",
        "timing_receipts": "request-local kernel and provider receipts; per-module attention hooks run only in untimed pilots",
        "cases": [],
        "status": "running",
    }

    def save():
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "results.json")

    save()
    try:
        record["weights"] = file_metadata(weights, assets)
        calibration_paths = sorted(
            {
                Path(config["artifact"]).resolve()
                for options in candidates.values()
                for config in options.values()
                if isinstance(config, dict) and "artifact" in config
            }
        )
        record["calibration_files"] = file_metadata(calibration_paths, Path("/"))
        metadata = sorted(
            path
            for path in Path(overrides["tokenizer"]).rglob("*")
            if path.is_file() and path.suffix in {".json", ".model", ".txt"}
        )
        record["asset_metadata"] = file_metadata(metadata, assets)
        save()
        pipeline = NativeDiffusionPipeline.from_pretrained(args.model, policy=policy, checkpoint_overrides=overrides)
        denoiser = pipeline.components.denoiser

        def generate(request, options, *, pilot=False):
            session = install_diffusion_accelerations(denoiser.model, options, policy) if options else None
            kernels = {}
            try:
                installed = session.report()["installed"] if session is not None else []
                reset_quantization_runtime_window(denoiser.model)
                reset_attention_provider_runtime()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with (
                    torch.inference_mode(),
                    kernel_dispatch_receipt_scope(kernels),
                    attention_scope_receipts(denoiser.model, installed) if pilot else nullcontext({}) as scopes,
                ):
                    output = pipeline(request)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                evidence = {
                    "denoiser": denoiser.runtime_optimization_report(),
                    "kernels": kernels,
                    "quantization": quantization_runtime_report(denoiser.model),
                    "attention": attention_provider_runtime_report(),
                    "attention_scopes": scopes,
                    "attention_configuration": next(
                        (item["scopes"] for item in installed if item["name"] == "attention_policy"), {}
                    ),
                    "codec": pipeline.components.decoder.runtime_optimization_report(),
                }
                if session is not None:
                    for plugin in session.report()["installed"]:
                        if plugin["name"] == "wan_cross_kv_fusion":
                            counters = plugin.get("runtime", {})
                            evidence["cross_kv_fusion"] = counters | {
                                "projection_calls": int(counters.get("text_packed_calls", 0))
                                + int(counters.get("image_packed_calls", 0))
                            }
                return output, elapsed, evidence
            finally:
                if session is not None:
                    session.uninstall()

        for seed in args.seeds:
            request = DiffusionRequest(
                prompt=args.prompt,
                negative_prompt="",
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                sampling=SamplingConfig(
                    num_inference_steps=args.steps,
                    guidance_scale=6.0,
                    seed=seed,
                    scheduler_options={"shift": args.shift},
                ),
            )
            reference, _, reference_evidence = generate(request, {}, pilot=True)
            record.setdefault(
                "output_geometry",
                {
                    "video": list(reference.sample.shape),
                    "latents": list(reference.latents.shape),
                    "video_dtype": str(reference.sample.dtype),
                    "latent_dtype": str(reference.latents.dtype),
                },
            )
            torch.save(
                {"sample": reference.sample.cpu(), "latents": reference.latents.cpu()},
                args.out / f"reference_{seed}.pt",
            )
            for name, options in candidates.items():
                candidate, _, evidence = generate(request, options, pilot=True)
                quality = compare_generation(reference, candidate)
                gate = qualify_execution(options, evidence)
                # Candidate names never become paths; arbitrary JSON labels cannot escape the output directory.
                index = len(record["cases"])
                torch.save(
                    {"sample": candidate.sample.cpu(), "latents": candidate.latents.cpu()},
                    args.out / f"candidate_{index}.pt",
                )
                row = {
                    "seed": seed,
                    "candidate": name,
                    "quality": quality,
                    "execution_gate": gate,
                    "evidence": evidence,
                }
                record["cases"].append(row)
                pilot_tensors = {"sample": candidate.sample.cpu(), "latents": candidate.latents.cpu()}
                del candidate
                print(
                    json.dumps({"seed": seed, "candidate": name, "quality": quality, "execution_gate": gate}),
                    flush=True,
                )
                if not quality["finite"] or not gate["passed"]:
                    row["status"] = "rejected_nonfinite" if not quality["finite"] else "rejected_execution"
                    save()
                    continue
                admission = cuda_device_admission()
                row["device_admission"] = admission
                if args.quality_only or not admission["timing_qualified"]:
                    row["status"] = "not_timed"
                    save()
                    continue
                reference_times, candidate_times, devices, timing_receipts = [], [], [], []
                for round_index in range(args.rounds):
                    for enabled in (False, True) if round_index % 2 == 0 else (True, False):
                        devices.append(cuda_device_admission())
                        timed_options = options if enabled else {}
                        output, elapsed, timed_evidence = generate(request, timed_options)
                        (candidate_times if enabled else reference_times).append(elapsed)
                        expected = (
                            pilot_tensors if enabled else {"sample": reference.sample, "latents": reference.latents}
                        )
                        output_matches = all(
                            getattr(output, field).dtype == value.dtype
                            and torch.equal(getattr(output, field).cpu(), value.cpu())
                            for field, value in expected.items()
                        )
                        del output
                        devices.append(cuda_device_admission())
                        timing_receipts.append(
                            {
                                "round": round_index,
                                "candidate_enabled": enabled,
                                "output_matches_pilot": output_matches,
                                "evidence": timed_evidence,
                                "execution_gate": _qualify_timed_execution(
                                    timed_options, timed_evidence, evidence if enabled else reference_evidence
                                ),
                            }
                        )
                row.update(
                    reference_wall_s=reference_times,
                    candidate_wall_s=candidate_times,
                    timing_device_samples=devices,
                    timing_execution_receipts=timing_receipts,
                )
                if not all(
                    item["execution_gate"]["passed"] and item["output_matches_pilot"] for item in timing_receipts
                ):
                    row["status"] = "rejected_timing_execution"
                elif not all(item["timing_qualified"] for item in devices):
                    row["status"] = "rejected_shared_device_timing"
                else:
                    row.update(
                        status="qualified_for_test_case" if quality["passed"] else "measured_quality_tradeoff",
                        speedup_median=statistics.median(reference_times) / statistics.median(candidate_times),
                        speedup_ci95=list(_paired_bootstrap_ci(reference_times, candidate_times)),
                    )
                save()
            del reference
        record["calibration_unchanged"] = record["calibration_files"] == file_metadata(calibration_paths, Path("/"))
        record["weights_unchanged"] = record["weights"] == file_metadata(weights, assets)
        record["asset_metadata_unchanged"] = record["asset_metadata"] == file_metadata(metadata, assets)
        if not all(record[key] for key in ("calibration_unchanged", "weights_unchanged", "asset_metadata_unchanged")):
            raise RuntimeError("diagnostic weights, calibration or asset metadata changed")
        eligible = {}
        for name in candidates:
            rows = [row for row in record["cases"] if row["candidate"] == name]
            if len(rows) == len(args.seeds) and all(row["status"] == "qualified_for_test_case" for row in rows):
                eligible[name] = min(row["speedup_ci95"][0] for row in rows)
        record["selection"] = (
            max(eligible, key=eligible.get) if eligible and max(eligible.values()) > 1 else "no_validated_speedup"
        )
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
