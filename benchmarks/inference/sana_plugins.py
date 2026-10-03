"""Quality-constrained plugin comparison through real SANA video generation.

Loads local DiT, Gemma and Wan VAE weights. Fixed prompt/seed/geometry/scheduler
and quality limits are shared by every candidate. Selection covers only the
tested cases; it is not model-wide or publication certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from skimage.metrics import structural_similarity

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.model_loading.policy import RuntimePolicy
from worldfoundry.runtime.performance import capture_runtime_fingerprint

_BUDGET = {"latent_relative_l2_max": 0.03, "video_psnr_min_db": 33.0, "video_ssim_min": 0.98}
_CACHE = {"threshold": 0.1, "warmup_steps": 5, "max_skip_steps": 2, "subsample_stride": 8}
_CANDIDATES = {
    "fusion": {"sana_block_fusion": True},
    "repacked_fusion": {"sana_block_fusion": {"repack_tokens": True}},
    "full_norm_fusion": {"sana_block_fusion": {"fuse_layer_norm": True, "repack_tokens": True}},
    "easycache": {"easycache": _CACHE},
    "combined": {"sana_block_fusion": True, "easycache": _CACHE},
    "conservative": {"sana_block_fusion": True, "easycache": _CACHE | {"threshold": 0.025}},
    "balanced": {"sana_block_fusion": True, "easycache": _CACHE | {"threshold": 0.05}},
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution(receipt, report, options):
    counts = {}
    for item in receipt.get("dispatches", []):
        key = f"{item['op']}:{item['backend']}"
        counts[key] = counts.get(key, 0) + 1
    accelerated = {item["op"] for item in receipt.get("dispatches", []) if item["accelerated"]}
    fusion = options.get("sana_block_fusion")
    if fusion:
        full_norm = isinstance(fusion, dict) and fusion.get("fuse_layer_norm", False)
        passed = {"layer_norm_scale_shift" if full_norm else "scale_shift"} <= accelerated
    else:
        passed = report["feature_cache"]["skipped_block_calls"] > 0
    return {
        "passed": passed,
        "kernel_calls": counts,
        "skipped_block_calls": report["feature_cache"]["skipped_block_calls"],
    }


def _paired_ci(reference, candidate):
    ratios = [a / b for a, b in zip(reference, candidate, strict=True)]
    rng = random.Random(20261003)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {"ratios": ratios, "bootstrap_median_ci95": [samples[125], samples[4874]]}


def _quality(reference, candidate):
    if reference.sample.shape != candidate.sample.shape or reference.latents.shape != candidate.latents.shape:
        raise ValueError("candidate changed generation geometry")
    ref_latents, got_latents = reference.latents.float().cpu(), candidate.latents.float().cpu()
    left, right = reference.sample.float().cpu(), candidate.sample.float().cpu()
    finite = all(bool(torch.isfinite(value).all()) for value in (left, right, ref_latents, got_latents))
    if not finite:
        return {"finite": False, "passed": False, "reason": "nonfinite-output"}
    relative = ((ref_latents - got_latents).norm() / ref_latents.norm().clamp_min(1e-12)).item()
    mse = (left - right).square().mean().item()
    psnr = math.inf if mse == 0 else 10 * math.log10(4.0 / mse)
    # Native Wan VAE decoder returns BCTHW RGB in [-1,1]; compare all frames.
    frames_left, frames_right = left[0].permute(1, 2, 3, 0).numpy(), right[0].permute(1, 2, 3, 0).numpy()
    ssim = float(
        statistics.mean(
            structural_similarity(a, b, channel_axis=2, data_range=2.0)
            for a, b in zip(frames_left, frames_right, strict=True)
        )
    )
    return {
        "finite": finite,
        "latent_relative_l2": relative,
        "video_psnr_db": psnr if math.isfinite(psnr) else "infinity",
        "video_ssim": ssim,
        "video_bitwise_equal": torch.equal(left, right),
        "passed": finite
        and relative <= _BUDGET["latent_relative_l2_max"]
        and psnr >= _BUDGET["video_psnr_min_db"]
        and ssim >= _BUDGET["video_ssim_min"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--prompt",
        default="A small red fox walks through a snowy forest, soft morning sunlight, cinematic tracking shot.",
    )
    args = parser.parse_args()
    if min(args.height, args.width, args.frames, args.steps, args.rounds) <= 0:
        parser.error("geometry, steps and rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    policy = RuntimePolicy(device="cuda", dtype=torch.bfloat16)
    assets = args.assets.resolve()
    gemma = assets / "Efficient-Large-Model--gemma-2-2b-it"
    overrides = {
        "dit": str(assets / "Efficient-Large-Model--SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"),
        "text-encoder": CheckpointSpec(
            source=str(gemma),
            files=tuple(path.name for path in sorted(gemma.glob("*.safetensors"))),
        ),
        "tokenizer": str(gemma),
        "codec": str(assets / "Efficient-Large-Model--SANA-Video_2B_480p/vae/Wan2.1_VAE.pth"),
    }
    record = {
        "scope": "native end-to-end SANA-Video 2B; explicit local assets; fixed test cases",
        "publication_certified": False,
        "budget": _BUDGET,
        "candidates": _CANDIDATES,
        "parameters": vars(args) | {"assets": str(assets), "out": str(args.out)},
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "timing_scope": "paired wall time including conditioning, all denoising steps and VAE decode; excludes loading/JIT pilots",
        "cases": [],
        "status": "running",
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    print("Loading local SANA, Gemma and Wan VAE", flush=True)
    try:
        weights = [Path(overrides["dit"]), Path(overrides["codec"]), *sorted(gemma.glob("*.safetensors"))]
        record["weights"] = [
            {"file": path.relative_to(assets).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in weights
        ]
        save()
        pipeline = NativeDiffusionPipeline.from_pretrained(
            "sana-video-2b-480p", policy=policy, checkpoint_overrides=overrides
        )
        denoiser = pipeline.components.denoiser

        def generate(request, options, *, collect_receipts=False):
            session = install_diffusion_accelerations(denoiser.model, options, policy) if options else None
            receipt = {}
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                scope = kernel_dispatch_receipt_scope(receipt) if collect_receipts else nullcontext()
                with torch.inference_mode(), scope:
                    result = pipeline(request)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                report = denoiser.runtime_optimization_report()
                return result, elapsed, report, receipt
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
                    num_inference_steps=args.steps, guidance_scale=6.0, seed=seed, scheduler_options={"shift": 7.0}
                ),
            )
            reference, _, _, _ = generate(request, {})  # same-shape warmup
            record.setdefault(
                "output_geometry", {"video": list(reference.sample.shape), "latents": list(reference.latents.shape)}
            )
            torch.save(
                {"sample": reference.sample.cpu(), "latents": reference.latents.cpu()},
                args.out / f"reference_{seed}.pt",
            )
            print(f"Reference seed {seed} complete", flush=True)
            for name, options in _CANDIDATES.items():
                candidate, _, report, receipt = generate(request, options, collect_receipts=True)
                quality = _quality(reference, candidate)
                torch.save(
                    {"sample": candidate.sample.cpu(), "latents": candidate.latents.cpu()},
                    args.out / f"{name}_{seed}.pt",
                )
                row = {
                    "seed": seed,
                    "candidate": name,
                    "quality": quality,
                    "execution": report,
                    "execution_gate": _execution(receipt, report, options),
                    "reference_wall_s": [],
                    "candidate_wall_s": [],
                }
                record["cases"].append(row)
                save()
                print(json.dumps({"seed": seed, "candidate": name, "quality": quality}), flush=True)
                del candidate
                if not quality["passed"]:
                    row["status"] = "rejected_quality"
                    save()
                    continue
                if not row["execution_gate"]["passed"]:
                    row["status"] = "rejected_execution"
                    save()
                    continue
                for index in range(args.rounds):
                    order = (False, True) if index % 2 == 0 else (True, False)
                    for enabled in order:
                        output, elapsed, _, _ = generate(request, options if enabled else {})
                        row["candidate_wall_s" if enabled else "reference_wall_s"].append(elapsed)
                        del output
                    save()
                row["speedup_median"] = statistics.median(row["reference_wall_s"]) / statistics.median(
                    row["candidate_wall_s"]
                )
                row["paired_speedup"] = _paired_ci(row["reference_wall_s"], row["candidate_wall_s"])
                row["status"] = "qualified_for_test_case"
                save()
                print(json.dumps({"seed": seed, "candidate": name, "speedup": row["speedup_median"]}), flush=True)
            del reference
        qualified = {name: [row for row in record["cases"] if row["candidate"] == name] for name in _CANDIDATES}
        eligible = {
            name: min(row["paired_speedup"]["bootstrap_median_ci95"][0] for row in rows)
            for name, rows in qualified.items()
            if len(rows) == len(args.seeds) and all(row["status"] == "qualified_for_test_case" for row in rows)
        }
        record["selection"] = max(eligible, key=eligible.get) if eligible and max(eligible.values()) > 1 else "baseline"
        record["selection_scope"] = (
            "worst lower CI bound across these seeds, subject to fixed quality and actual execution gates"
        )
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
