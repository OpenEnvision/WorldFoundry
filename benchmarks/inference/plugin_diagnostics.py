"""Quality, execution and device admission for native acceleration diagnostics."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from contextlib import contextmanager

import torch
from skimage.metrics import structural_similarity

BUDGET = {"latent_relative_l2_max": 0.03, "video_psnr_min_db": 33.0, "video_ssim_min": 0.98}


@contextmanager
def attention_scope_receipts(model, installed):
    """Observe successful eager calls per configured scope during pilots only."""
    scopes = next((item.get("scopes", {}) for item in installed if item["name"] == "attention_policy"), {})
    receipts, handles = {}, []
    try:
        for scope, configuration in scopes.items():
            class_name = "SelfAttention" if scope == "self" else "CrossAttention"
            targets = [
                child.attn
                for child in model.modules()
                if type(child).__name__ == class_name and type(child).__module__.endswith(".wan.model")
            ]
            row = receipts[scope] = dict(configuration) | {
                "observed_modules": len(targets),
                "executed_modules": 0,
                "successful_calls": 0,
                "backend_mismatches": 0,
            }
            seen = set()

            def observed(module, inputs, output, *, row=row, seen=seen):
                seen.add(id(module))
                row["executed_modules"] = len(seen)
                row["successful_calls"] += 1
                row["backend_mismatches"] += int(module.attention_backend != row["resolved"])

            for target in targets:
                handles.append(target.register_forward_hook(observed))
        yield receipts
    finally:
        for handle in handles:
            handle.remove()


def compare_generation(reference, candidate):
    """Compare every batch member and frame; a bad member cannot be averaged away."""
    if reference.sample.shape != candidate.sample.shape or reference.latents.shape != candidate.latents.shape:
        raise ValueError("candidate changed output geometry")
    left, right = reference.sample.float().cpu(), candidate.sample.float().cpu()
    ref_latents, got_latents = reference.latents.float().cpu(), candidate.latents.float().cpu()
    if left.ndim != 5 or left.shape[1] != 3 or ref_latents.ndim != 5 or left.shape[0] != ref_latents.shape[0]:
        raise ValueError("native RGB BCTHW video and BCFHW latents are required")
    if min(left.shape) <= 0 or min(ref_latents.shape) <= 0 or min(left.shape[-2:]) < 7:
        raise ValueError("nonempty outputs and at least 7x7 RGB frames are required")
    if not all(bool(torch.isfinite(value).all()) for value in (left, right, ref_latents, got_latents)):
        return {"finite": False, "passed": False, "reason": "nonfinite-output"}
    relative = (
        ((ref_latents - got_latents).flatten(1).norm(dim=1) / ref_latents.flatten(1).norm(dim=1).clamp_min(1e-12))
        .max()
        .item()
    )
    mse = (left - right).square().flatten(1).mean(dim=1).max().item()
    psnr = math.inf if mse == 0 else 10 * math.log10(4.0 / mse)
    ssim = min(
        float(structural_similarity(a, b, channel_axis=2, data_range=2.0))
        for batch_left, batch_right in zip(left, right, strict=True)
        for a, b in zip(batch_left.permute(1, 2, 3, 0).numpy(), batch_right.permute(1, 2, 3, 0).numpy(), strict=True)
    )
    return {
        "finite": True,
        "latent_relative_l2": relative,
        "video_psnr_db": psnr if math.isfinite(psnr) else "infinity",
        "video_ssim": ssim,
        "latent_bitwise_equal": reference.latents.dtype == candidate.latents.dtype
        and torch.equal(ref_latents, got_latents),
        "video_bitwise_equal": reference.sample.dtype == candidate.sample.dtype and torch.equal(left, right),
        "passed": relative <= BUDGET["latent_relative_l2_max"]
        and psnr >= BUDGET["video_psnr_min_db"]
        and ssim >= BUDGET["video_ssim_min"],
    }


def qualify_execution(options, evidence):
    """Require evidence for every requested plugin, including combined candidates."""
    checks = {}
    dispatches = evidence.get("kernels", {}).get("dispatches", ())
    accelerated = {item["op"] for item in dispatches if item.get("accelerated")}
    for name, config in options.items():
        if config is False or config is None:
            continue
        values = {} if config is True else config
        if name == "sana_block_fusion":
            op = "layer_norm_scale_shift" if values.get("fuse_layer_norm", False) else "scale_shift"
            checks[name] = op in accelerated
        elif name == "selective_fp8":
            counters = evidence.get("quantization") or {}
            checks[name] = (
                int(counters.get("low_precision_kernel_calls", 0)) > 0
                and int(counters.get("dense_compute_calls", -1)) == 0
                and int(counters.get("dense_fallback_calls", -1)) == 0
                and not counters.get("fallback_reasons")
            )
        elif name == "easycache":
            denoiser = evidence.get("denoiser", {})
            runtime = denoiser.get("runtime") or {}
            cache = runtime.get("feature_cache", denoiser.get("feature_cache", {})) or {}
            counter = "skipped_block_calls" if values.get("threshold", 0) > 0 else "dense_block_calls"
            checks[name] = int(cache.get(counter, 0)) > 0
        elif name == "attention_policy":
            from worldfoundry.core.attention.backends.probe import normalize_attention_backend

            checks[name] = bool(values)
            for scope_name, scope in values.items():
                requested = normalize_attention_backend(scope if isinstance(scope, str) else scope["backend"])
                receipt = evidence.get("attention_scopes", {}).get(scope_name, {})
                backend = receipt.get("resolved")
                counters = evidence.get("attention", {}).get(backend, {})
                count = int(receipt.get("configured_modules", 0))
                checks[name] &= (
                    receipt.get("requested") == requested
                    and count > 0
                    and receipt.get("observed_modules") == receipt.get("executed_modules") == count
                    and int(receipt.get("successful_calls", 0)) >= count
                    and not receipt.get("backend_mismatches")
                    and int(counters.get("successes", 0)) > 0
                    and not any(int(counters.get(key, 0)) for key in ("errors", "fallbacks", "quarantined_skips"))
                )
        elif name == "wan_cross_kv_fusion":
            counters = evidence.get("cross_kv_fusion", {})
            checks[name] = int(counters.get("projection_calls", 0)) > 0 and (
                not values.get("include_image", False) or int(counters.get("image_packed_calls", 0)) > 0
            )
        else:
            checks[name] = False
    return {"passed": bool(checks) and all(checks.values()), "plugins": checks}


def file_manifest(paths, root):
    rows = []
    for path in paths:
        before = path.stat()
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024**2):
                hasher.update(chunk)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"file changed during hashing: {path}")
        rows.append({"file": path.relative_to(root).as_posix(), "bytes": before.st_size, "sha256": hasher.hexdigest()})
    return rows


def cuda_device_admission():
    """Use our live context to identify the physical GPU despite CUDA_VISIBLE_DEVICES."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        rows = [tuple(field.strip() for field in line.split(",")) for line in output.splitlines() if line.strip()]
        if any(len(row) != 3 for row in rows):
            return {"timing_qualified": False, "reason": "GPU process telemetry returned an unexpected row"}
        own = {row[0] for row in rows if int(row[1]) == os.getpid()}
        if len(own) != 1:
            return {"timing_qualified": False, "reason": "physical GPU context could not be identified"}
        uuid = own.pop()
        foreign = [
            {"pid": int(row[1]), "memory_mib": row[2]} for row in rows if row[0] == uuid and int(row[1]) != os.getpid()
        ]
        return {"gpu_uuid": uuid, "foreign_contexts": foreign, "timing_qualified": not foreign}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"timing_qualified": False, "reason": str(error)}
