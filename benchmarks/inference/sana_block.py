"""Paired SANA-Video block benchmark with real weights and execution receipts.

This measures block forwards with synthetic activations, not video generation.
Quality thresholds are fixed before timing. Failed quality or kernel fallback
raises instead of turning an unqualified run into a speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from benchmarks.harness import bench_paired
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import sana_video_config
from worldfoundry.base_models.diffusion_model.models.networks.sana.block_ops import (
    SanaBlockFusionPolicy,
    gated_residual,
    modulated_norm,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import WanRotaryPosEmbed
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale_video import SanaVideoMSBlock
from worldfoundry.core.kernels import clear_kernel_dispatch_cache
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def _grid(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("grid must be positive F,H,W")
    return parts


def _quality(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    left, right = reference.float(), candidate.float()
    difference = left - right
    relative_l2 = (difference.norm() / left.norm().clamp_min(1e-12)).item()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {
        "finite": finite,
        "bitwise_equal": torch.equal(reference, candidate),
        "max_abs": difference.abs().max().item(),
        "relative_l2": relative_l2,
        "cosine": torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item(),
        "passed": finite and relative_l2 <= 0.005,
    }


def _load_block(checkpoint: Path, index: int) -> tuple[nn.Module, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload["state_dict"]
    prefix = f"blocks.{index}."
    weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not weights:
        raise ValueError(f"checkpoint has no {prefix} weights")
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(memoryview(value.contiguous().view(torch.uint8).numpy()))
    config = sana_video_config(resolution="480p")
    block_keys = (
        "hidden_size",
        "num_heads",
        "mlp_ratio",
        "qk_norm",
        "cross_norm",
        "attn_type",
        "ffn_type",
        "mlp_acts",
        "linear_head_dim",
        "t_kernel_size",
    )
    with torch.device("meta"):
        block = SanaVideoMSBlock(**{key: config[key] for key in block_keys})
    block.load_state_dict(weights, strict=True, assign=True)
    return block.to(device="cuda", dtype=torch.bfloat16).eval(), digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="official SANA_Video_2B_480p.pth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=_grid, action="append")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--candidate", choices=("auto", "triton"), default="auto")
    parser.add_argument("--fuse-layer-norm", action="store_true", help="opt into a different reduction order")
    parser.add_argument("--strided-input", action="store_true", help="use the native patch-embedding token strides")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.rounds) <= 0:
        parser.error("warmup, iters and rounds must be positive")
    torch.cuda.set_device(0)
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "scope": "real checkpoint block; synthetic inputs; no semantic or end-to-end certification",
        "publication_certified": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "block": args.block,
        "dtype": "bfloat16",
        "candidate": args.candidate,
        "thresholds": {"finite": True, "relative_l2_max": 0.005},
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "parameters": vars(args) | {"checkpoint": str(args.checkpoint), "out": str(args.out)},
        "cases": [],
        "status": "running",
    }
    output = args.out / "results.json"

    def save() -> None:
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    policies = {
        "torch": None,
        args.candidate: SanaBlockFusionPolicy(backend=args.candidate, fuse_layer_norm=args.fuse_layer_norm),
    }
    try:
        block, digest = _load_block(args.checkpoint, args.block)
        record["block_weights_sha256"] = digest
        clear_kernel_dispatch_cache()
        rotary = WanRotaryPosEmbed(112, (1, 2, 2), 1024)
        with torch.inference_mode():
            for grid in args.grid or [(3, 8, 8), (21, 30, 52)]:
                tokens = math.prod(grid)
                frequencies = rotary(grid, torch.device("cuda"))
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    value = torch.randn(1, tokens, 2240, device="cuda", dtype=torch.bfloat16)
                    if args.strided_input:
                        value = value.transpose(1, 2).contiguous().transpose(1, 2)
                    context = torch.randn(1, 300, 2240, device="cuda", dtype=torch.bfloat16) * 0.1
                    timestep = torch.randn(1, 6 * 2240, device="cuda", dtype=torch.bfloat16) * 0.1
                    scale, shift, gate = (
                        torch.randn(1, 1, 2240, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                    )

                    def forward(backend: str) -> torch.Tensor:
                        block._worldfoundry_block_fusion = policies[backend]
                        return block(value, context, timestep, THW=grid, rotary_emb=frequencies)

                    reference = forward("torch")
                    receipt = {}
                    with kernel_dispatch_receipt_scope(receipt):
                        candidate = forward(args.candidate)
                    quality = _quality(reference, candidate)
                    row = {"grid": grid, "seed": seed, "tokens": tokens, "quality": quality, "execution": receipt}
                    record["cases"].append(row)
                    save()
                    if not quality["passed"]:
                        raise RuntimeError(f"SANA block quality failed: {quality}")
                    dispatches = receipt.get("dispatches", [])
                    if any(item["failures"] for item in dispatches):
                        raise RuntimeError(f"SANA kernel fallback: {dispatches}")
                    norm_op = "layer_norm_scale_shift" if args.fuse_layer_norm else "scale_shift"
                    required_ops = {norm_op} if args.strided_input else {norm_op, "residual_gate_add"}
                    if tokens * 2240 >= 12 * 1024**2 and not required_ops.issubset(
                        {item["op"] for item in dispatches if item["accelerated"]}
                    ):
                        raise RuntimeError("large-grid run did not execute both fusion kernels")
                    del reference, candidate
                    row["block_timing"] = bench_paired(
                        f"sana_block_{tokens}_seed{seed}",
                        lambda: forward("torch"),
                        lambda: forward(args.candidate),
                        device=torch.device("cuda", 0),
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                    ).to_dict()

                    def glue(backend: str) -> torch.Tensor:
                        policy = policies[backend]
                        normalized = modulated_norm(value, block.norm1, shift, scale, policy=policy)
                        return gated_residual(value, normalized, gate, block.drop_path, policy=policy)

                    row["glue_timing"] = bench_paired(
                        f"sana_glue_{tokens}_seed{seed}",
                        lambda: glue("torch"),
                        lambda: glue(args.candidate),
                        device=torch.device("cuda", 0),
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                    ).to_dict()
                    save()
                    print(
                        json.dumps(
                            {
                                "grid": grid,
                                "seed": seed,
                                "quality": quality,
                                "block_speedup": row["block_timing"]["speedup_median"],
                                "glue_speedup": row["glue_timing"]["speedup_median"],
                            }
                        ),
                        flush=True,
                    )
        record["status"] = "completed_diagnostic"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
