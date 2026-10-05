"""Paired Sol-Attn operator diagnostics on random and correlated video tokens.

Uses actual optional CUDA kernels, records the selected upstream backend and
execution counters, and times only cases passing a predeclared numerical gate.
Synthetic tokens are not a model/video quality certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.harness import bench_paired
from worldfoundry.core.attention.backends.dispatch import (
    attention_forward,
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def main():
    from sol_attn import get_sol_attn_backend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192, 32768])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    args = parser.parse_args()
    if any(count <= 0 or count % 64 for count in args.tokens) or min(args.heads, args.rounds) <= 0:
        parser.error("tokens must be positive multiples of 64; heads and rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "scope": "synthetic attention operator; no model or video certification",
        "publication_certified": False,
        "thresholds": {"relative_l2_max": 0.01},
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "upstream_backend": get_sol_attn_backend(torch.device("cuda", 0)),
        "parameters": vars(args) | {"out": str(args.out)},
        "cases": [],
        "status": "running",
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    try:
        with torch.inference_mode():
            for tokens in args.tokens:
                for kind in ("random", "block_correlated"):
                    torch.manual_seed(42)
                    if kind == "random":
                        values = [
                            torch.randn(1, tokens, args.heads, 128, device="cuda", dtype=torch.bfloat16)
                            for _ in range(3)
                        ]
                    else:
                        centers = torch.randn(1, tokens // 64, args.heads, 128, device="cuda", dtype=torch.bfloat16)
                        values = [
                            centers.repeat_interleave(64, dim=1)
                            + torch.randn(1, tokens, args.heads, 128, device="cuda", dtype=torch.bfloat16) * 0.05
                            for _ in range(3)
                        ]

                    def forward(backend):
                        return attention_forward(
                            *values,
                            q_pattern="b s n d",
                            k_pattern="b s n d",
                            v_pattern="b s n d",
                            out_pattern="b s n d",
                            backend=backend,
                            backend_options={"tau": args.tau} if backend == "sol_attn" else None,
                        )

                    reset_attention_provider_runtime()
                    reference, candidate = forward("torch"), forward("sol_attn")
                    relative = (
                        (reference.float() - candidate.float()).norm() / reference.float().norm().clamp_min(1e-12)
                    ).item()
                    quality = {
                        "relative_l2": relative,
                        "finite": bool(torch.isfinite(candidate).all()),
                        "passed": relative <= 0.01 and bool(torch.isfinite(candidate).all()),
                    }
                    row = {
                        "tokens": tokens,
                        "kind": kind,
                        "quality": quality,
                        "execution": attention_provider_runtime_report(),
                    }
                    record["cases"].append(row)
                    if quality["passed"]:
                        row["timing"] = bench_paired(
                            f"sol_{kind}_{tokens}",
                            lambda: forward("torch"),
                            lambda: forward("sol_attn"),
                            device=torch.device("cuda", 0),
                            warmup=5,
                            iters=3,
                            rounds=args.rounds,
                        ).to_dict()
                    else:
                        row["status"] = "rejected_quality"
                    save()
                    print(
                        json.dumps(
                            {
                                "tokens": tokens,
                                "kind": kind,
                                "quality": quality,
                                "speedup": row.get("timing", {}).get("speedup_median"),
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
