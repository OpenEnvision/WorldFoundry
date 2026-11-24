# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Sol-Engine-style automated 5-pillar tuner and quality-gated integrator.

Implements a reproducible three-stage optimization loop modeled after NVIDIA's
Sol Video Inference Engine:

1. **Baseline**: Measure L0 eager BF16 reference latency and save reference
   latents/pixels.
2. **Skill Sweep**: Evaluate each optimization pillar in isolation (Kernel
   Fusion + CUDA Graph, FP8 W8A8 Quantization, Sparse Attention via Sol-Attn,
   Cross-Step Caching via TeaCache, Mid-Stack Token Pruning) against the L0
   reference under an explicit :class:`QualityBudget`.
3. **Integrator + Validator**: Greedily compose winning compatible pillars
   through :func:`compose_sol_stack`, re-measuring latency and quality at each
   composition step and rolling back any pillar that violates the budget.

Outputs a certified JSON recipe (reloadable via
:meth:`SolEngineStack.from_recipe_dict`) and a Markdown ablation report.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from benchmarks.inference.plugin_diagnostics import compare_generation
from benchmarks.videogen.report import format_markdown_leaderboard
from benchmarks.videogen.runner import (
    _MODEL_DEFAULTS,
    DEFAULT_NEGATIVE,
    DEFAULT_PROMPT,
    _instrument_pipeline_stages,
    _sync,
)
from worldfoundry.core.acceleration.sol_engine import (
    CachePillar,
    ExecutionPillar,
    KernelFusionPillar,
    ModelHardwareProfile,
    QualityBudget,
    QuantizationPillar,
    SparseAttentionPillar,
    TokenPruningPillar,
    compose_sol_stack,
)
from worldfoundry.core.attention.backends.probe import probe_attention_backends


def _detect_hardware_profile(model: str, device: torch.device) -> ModelHardwareProfile:
    meta = _MODEL_DEFAULTS[model]
    sm_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    num_layers = 30 if model == "wan2.1-t2v-1.3b" else 40
    caps = probe_attention_backends(device)
    usable = tuple(sorted(name for name, cap in caps.items() if cap.usable))
    return ModelHardwareProfile(
        model_id=model,
        num_heads=int(meta["num_heads"]),
        head_dim=128,
        num_layers=num_layers,
        world_size=1,
        sm_major=sm_major,
        supports_cfg=bool(meta["supports_cfg"]),
        is_causal_ar=False,
        available_attention_backends=usable,
    )


def _measure_options(
    *,
    model: str,
    label: str,
    opts: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    ref_ns: SimpleNamespace | None,
) -> tuple[dict[str, Any], SimpleNamespace | None]:
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import (
        Wan2p1T2V14BPipeline,
        Wan2p1T2VPipeline,
    )

    ckpt_dir = args.checkpoint_dir or _MODEL_DEFAULTS[model]["checkpoint_dir"]
    pipeline_cls = Wan2p1T2VPipeline if model == "wan2.1-t2v-1.3b" else Wan2p1T2V14BPipeline
    req_kwargs: dict[str, Any] = {
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg_scale,
        "seed": args.seed,
        "return_dict": True,
    }

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_load0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(model_path=ckpt_dir, device=str(device), **dict(opts))
    _sync(device)
    load_s = round(time.perf_counter() - t_load0, 2)
    timings, orig_den = _instrument_pipeline_stages(pipe.native_pipeline, device)

    t_cold0 = time.perf_counter()
    warm = pipe(**req_kwargs)
    _sync(device)
    cold_s = round(time.perf_counter() - t_cold0, 3)
    del warm

    runs: list[dict[str, Any]] = []
    last_ns: SimpleNamespace | None = None
    for _ in range(args.repeats):
        for bucket in timings.values():
            bucket.clear()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        res = pipe(**req_kwargs)
        _sync(device)
        e2e_s = time.perf_counter() - t0
        runs.append(
            {
                "e2e_s": round(e2e_s, 3),
                "text_s": round(sum(timings["text"]), 3),
                "dit_s": round(sum(timings["dit"]), 3),
                "vae_s": round(sum(timings["vae"]), 3),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
                if device.type == "cuda"
                else 0.0,
            }
        )
        last_ns = SimpleNamespace(sample=res["sample"].detach().cpu(), latents=res["latents"].detach().cpu())
        del res

    receipt = orig_den.runtime_optimization_report() if hasattr(orig_den, "runtime_optimization_report") else {}
    if ref_ns is None:
        quality: dict[str, Any] = {"reference": True}
    elif last_ns is not None:
        quality = compare_generation(ref_ns, last_ns)
    else:
        quality = {}

    record = {
        "label": label,
        "model": model,
        "world_size": 1,
        "topology_label": "1GPU",
        "load_s": load_s,
        "cold_first_run_s": cold_s,
        "median_e2e_s": round(statistics.median(r["e2e_s"] for r in runs), 3),
        "median_text_s": round(statistics.median(r["text_s"] for r in runs), 3),
        "median_dit_s": round(statistics.median(r["dit_s"] for r in runs), 3),
        "median_vae_s": round(statistics.median(r["vae_s"] for r in runs), 3),
        "peak_alloc_gib": max(r["peak_alloc_gib"] for r in runs),
        "quality": quality,
        "options": dict(opts),
        "fallbacks": receipt.get("fallbacks") if isinstance(receipt, Mapping) else [],
        "runs": runs,
    }
    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, last_ns


def run_sol_integrator(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the 3-stage Sol-Engine skill sweep + quality-gated integration."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hw = _detect_hardware_profile(args.model, device)
    budget = QualityBudget.for_tier(args.budget)
    if args.min_psnr is not None:
        budget = replace(budget, min_psnr_db=float(args.min_psnr))
    if args.min_ssim is not None:
        budget = replace(budget, min_ssim=float(args.min_ssim))

    records: list[dict[str, Any]] = []
    # Stage 0: L0 Baseline
    baseline_opts = {
        "torch_dtype": "bfloat16",
        "offload_mode": "resident",
        "attention_backend": "torch",
        "dit_weight_dtype": "bfloat16",
    }
    base_rec, ref_ns = _measure_options(
        model=args.model, label="L0_baseline_bf16", opts=baseline_opts, device=device, args=args, ref_ns=None
    )
    records.append(base_rec)
    base_e2e = float(base_rec["median_e2e_s"])
    print(f"[Integrator] L0 baseline: {base_e2e:.2f}s (dit={base_rec['median_dit_s']}s)", flush=True)

    # Current best accepted stack starts at lossless fusion + graph
    best_stack = compose_sol_stack(
        hw,
        budget=budget,
        cache=CachePillar(algorithm="none"),
        sparse_attention=SparseAttentionPillar(backend="torch"),
        token_pruning=TokenPruningPillar(enabled=False),
        quantization=QuantizationPillar(mode="none"),
        kernel_fusion=KernelFusionPillar(fuse_qkv=True, fused_rope=False, static_cross_kv=True, inplace_residual=True),
        execution=ExecutionPillar(cuda_graph="whole"),
    )
    s1_rec, _ = _measure_options(
        model=args.model,
        label="S1_exact_fusion_graph",
        opts=best_stack.to_pipeline_kwargs(),
        device=device,
        args=args,
        ref_ns=ref_ns,
    )
    records.append(s1_rec)
    s1_e2e = float(s1_rec["median_e2e_s"])
    s1_accepted = budget.accepts(s1_rec["quality"]) and s1_e2e <= base_e2e * 1.01
    if s1_accepted:
        best_e2e = s1_e2e
    else:
        best_stack = compose_sol_stack(
            hw,
            budget=budget,
            cache=CachePillar(algorithm="none"),
            sparse_attention=SparseAttentionPillar(backend="torch"),
            token_pruning=TokenPruningPillar(enabled=False),
            quantization=QuantizationPillar(mode="none"),
            kernel_fusion=KernelFusionPillar(
                fuse_qkv=False,
                fused_rope=False,
                static_cross_kv=False,
                inplace_residual=False,
            ),
            execution=ExecutionPillar(cuda_graph="off"),
        )
        best_e2e = base_e2e
    print(
        f"[Integrator] S1 lossless fusion+graph: {s1_e2e:.2f}s ({base_e2e / s1_e2e:.2f}x) accepted={s1_accepted}",
        flush=True,
    )

    # Stage 2: Greedy quality-gated integration across the remaining Sol-Engine pillars
    pillar_trials: list[tuple[str, dict[str, Any]]] = []
    if hw.sm_major >= 9 and budget.tier != "lossless":
        pillar_trials.append(
            (
                "P4_quant_fp8_exact_accum",
                {
                    "quantization": QuantizationPillar(
                        mode="fp8",
                        scaling="rowwise",
                        use_fast_accum=False,
                        fuse_fp8_ffn=True,
                        keep_dense_fallback=False,
                        exclude_edge_blocks=hw.num_heads <= 16,
                    ),
                },
            )
        )
    if "sol_attn" in hw.available_attention_backends and budget.tier != "lossless":
        dense_steps = max(4, int(args.steps) // 4)
        for tau in (0.5, 1.0):
            pillar_trials.append(
                (
                    f"P2_sol_attn_1x8x8_tau{tau:g}",
                    {
                        "sparse_attention": SparseAttentionPillar(
                            backend="sol_attn",
                            tau=tau,
                            thresh_type="diag",
                            compress_kv=True,
                            dense_steps=dense_steps,
                            dense_blocks=(0, 1, -1),
                            tile_3d=(1, 8, 8),
                        ),
                    },
                )
            )
    if budget.tier != "lossless":
        for thresh in (0.10, 0.15):
            pillar_trials.append(
                (
                    f"P1_teacache_{thresh:g}",
                    {"cache": CachePillar(algorithm="teacache", threshold=thresh)},
                )
            )
        pillar_trials.append(
            (
                "P3_token_pruning_0.8",
                {"token_pruning": TokenPruningPillar(enabled=True, keep_ratio=0.8)},
            )
        )

    integration_log: list[dict[str, Any]] = []
    for trial_name, patch in pillar_trials:
        candidate_stack = compose_sol_stack(
            hw,
            budget=budget,
            cache=patch.get("cache", best_stack.cache),
            sparse_attention=patch.get("sparse_attention", best_stack.sparse_attention),
            token_pruning=patch.get("token_pruning", best_stack.token_pruning),
            quantization=patch.get("quantization", best_stack.quantization),
            kernel_fusion=patch.get("kernel_fusion", best_stack.kernel_fusion),
            execution=patch.get("execution", best_stack.execution),
        )
        try:
            rec, _ = _measure_options(
                model=args.model,
                label=trial_name,
                opts=candidate_stack.to_pipeline_kwargs(),
                device=device,
                args=args,
                ref_ns=ref_ns,
            )
        except Exception as exc:
            integration_log.append({"trial": trial_name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append(rec)
        cand_e2e = float(rec["median_e2e_s"])
        passed = budget.accepts(rec["quality"])
        improved = cand_e2e < best_e2e * 0.99
        decision = "accepted" if (passed and improved) else ("rejected_quality" if not passed else "rejected_no_gain")
        if passed and improved:
            best_stack = candidate_stack
            best_e2e = cand_e2e
        integration_log.append(
            {
                "trial": trial_name,
                "decision": decision,
                "e2e_s": cand_e2e,
                "speedup_vs_l0": round(base_e2e / cand_e2e, 2) if cand_e2e > 0 else 0.0,
                "quality": rec["quality"],
                "active_pillars": list(candidate_stack.active_pillars()),
            }
        )
        print(
            f"[Integrator] {trial_name:28s}: {cand_e2e:6.2f}s ({base_e2e / cand_e2e:.2f}x) -> {decision} "
            f"psnr={rec['quality'].get('video_psnr_db')} ssim={rec['quality'].get('video_ssim')}",
            flush=True,
        )

    markdown = format_markdown_leaderboard(records, baseline_e2e_s=base_e2e)
    report = {
        "model": args.model,
        "budget": budget.tier,
        "baseline_e2e_s": base_e2e,
        "best_e2e_s": best_e2e,
        "best_speedup": round(base_e2e / best_e2e, 2) if best_e2e > 0 else 1.0,
        "certified_recipe": best_stack.to_recipe_dict(),
        "integration_log": integration_log,
        "records": records,
        "markdown": markdown,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.model}_{args.budget}_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / f"{args.model}_{args.budget}_recipe.json").write_text(
        json.dumps(best_stack.to_recipe_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / f"{args.model}_{args.budget}_leaderboard.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown, flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sol-Engine 5-pillar automated integrator for WorldFoundry.")
    parser.add_argument("--model", choices=sorted(_MODEL_DEFAULTS), required=True)
    parser.add_argument("--budget", choices=("lossless", "high_fidelity", "balanced", "aggressive"), default="balanced")
    parser.add_argument("--min-psnr", type=float, default=None)
    parser.add_argument("--min-ssim", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_sol_integrator(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
