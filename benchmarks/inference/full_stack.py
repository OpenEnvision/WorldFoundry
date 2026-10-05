"""Full-stack composed-optimization A/B over a denoise loop.

Run:
  python -m benchmarks.inference.full_stack [--profile service|ar] [--out ...]

Composes merged-QKV and FP8 with either static cross-attention K/V caching or
steady-state CUDA Graph. Python-managed static caches cannot be combined with
graph replay. Every changing denoise step must pass an explicit FP8 error
budget before timing. This synthetic operator benchmark does not certify
checkpoint video quality.

The optimizations are applied via the same functions the loader uses
(fuse_qkv_projections / install_static_cross_kv_cache / apply_quantization_policy
/ InferenceCUDAGraphRunner), so the benchmark exercises the real transform path.
"""

from __future__ import annotations

import argparse
import copy

import torch

from benchmarks.harness import bench_paired, summarize_markdown, write_result
from benchmarks.inference.correctness import FP8_BUDGET, compare_step, require_graph_execution, validate_steps
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention, SelfAttention
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    install_static_cross_kv_cache,
    reset_static_cross_kv,
)
from worldfoundry.core.acceleration.quantization.linear import quantization_runtime_report
from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner
from worldfoundry.core.model_loading.optimize import AppliedOptimizations, apply_quantization_policy
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy


class _DiTBlock(torch.nn.Module):
    """Wan-style block: self-attn (freqs) + cross-attn (context) + SwiGLU FFN."""

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = SelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, heads)
        self.up = torch.nn.Linear(dim, ffn)
        self.gate = torch.nn.Linear(dim, ffn)
        self.down = torch.nn.Linear(ffn, dim)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x, freqs)
        x = x + self.cross_attn(x, ctx)
        return x + self.down(self.act(self.gate(x)) * self.up(x))


class _DiT(torch.nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DiTBlock(dim, heads, ffn) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, freqs, ctx)
        return x


_PROFILES = {
    "service": {"dim": 3072, "heads": 24, "ffn": 8192, "n_blocks": 6, "seq": 2048, "ctx": 512, "steps": 20},
    "ar": {"dim": 1536, "heads": 12, "ffn": 4096, "n_blocks": 12, "seq": 512, "ctx": 256, "steps": 30},
}


def _freqs(seq: int, head_dim: int, device: torch.device) -> torch.Tensor:
    ang = torch.randn(seq, 1, head_dim // 2, device=device)
    return torch.polar(torch.ones_like(ang), ang)


def run(
    profile: str = "service",
    dtype: torch.dtype = torch.bfloat16,
    *,
    cuda_graph: bool = False,
    static_cross_kv: bool = True,
):
    if cuda_graph and static_cross_kv:
        raise ValueError("static_cross_kv and cuda_graph have incompatible Python cache lifecycles")
    cfg = _PROFILES[profile]
    validate_steps(cfg["steps"])
    assert torch.cuda.is_available(), "full-stack benchmark needs a CUDA device"
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    dim, heads, ffn, n_blocks = cfg["dim"], cfg["heads"], cfg["ffn"], cfg["n_blocks"]
    seq, ctx_tokens, steps = cfg["seq"], cfg["ctx"], cfg["steps"]
    head_dim = dim // heads

    base = _DiT(dim, heads, ffn, n_blocks).to(device=device, dtype=dtype).eval()
    freqs = _freqs(seq, head_dim, device)
    ctx = torch.randn(1, ctx_tokens, dim, device=device, dtype=dtype)
    latents = [torch.randn(1, seq, dim, device=device, dtype=dtype) for _ in range(4)]

    # Compose the full stack via the same transforms the loader applies.
    opt = copy.deepcopy(base)
    applied = AppliedOptimizations()
    applied.record_fusion(requested=True, fused_blocks=fuse_qkv_projections(opt))
    cross_cache = install_static_cross_kv_cache(opt) if static_cross_kv else None
    applied.record_cross_kv(
        requested=static_cross_kv, wrapped_blocks=0 if cross_cache is None else cross_cache.wrapped_blocks
    )
    applied.record_quantization(
        apply_quantization_policy(opt, QuantizationPolicy(mode=QuantizationMode.FP8, options={"min_features": 512}))
    )
    graph = InferenceCUDAGraphRunner(opt.forward) if cuda_graph else None
    optimized_forward = graph if graph is not None else opt

    def baseline() -> torch.Tensor:
        with torch.no_grad():
            out = None
            for step in range(steps):
                out = base(latents[step % len(latents)], freqs, ctx)
            return out

    def full_stack() -> torch.Tensor:
        with torch.no_grad():
            if cross_cache is not None:
                reset_static_cross_kv(cross_cache)
            out = None
            for step in range(steps):
                out = optimized_forward(latents[step % len(latents)], freqs, ctx)
            return out

    with torch.no_grad():
        correctness = []
        for step in range(steps):
            latent = latents[step % len(latents)]
            correctness.append(
                compare_step(
                    base(latent, freqs, ctx), optimized_forward(latent, freqs, ctx), step=step, budget=FP8_BUDGET
                )
            )
    graph_report = graph.report() if graph is not None else None
    if graph_report is not None:
        require_graph_execution(graph_report, steps=steps)
    quantization_report = quantization_runtime_report(opt)
    if quantization_report is None or int(quantization_report.get("low_precision_kernel_calls", 0)) < 1:
        raise AssertionError(f"FP8 kernel did not execute: {quantization_report}")
    cache_report = cross_cache.report() if cross_cache is not None else None
    if cache_report is not None and steps > 1 and int(cache_report.get("hits", 0)) < n_blocks * (steps - 1):
        raise AssertionError(f"static cross K/V was not reused: {cache_report}")
    applied.requested["cuda_graph"] = cuda_graph
    applied.effective["cuda_graph"] = graph_report is not None
    applied.effective["quantization"] = quantization_report["effective"]
    result = bench_paired(
        f"full_stack_{profile}",
        baseline,
        full_stack,
        label_a="bf16_dense",
        label_b="qkv+"
        + str(quantization_report["effective"])
        + ("+xkv" if static_cross_kv else "")
        + ("+graph" if cuda_graph else ""),
        device=device,
        warmup=3,
        iters=1,
        rounds=6,
        workload={
            **cfg,
            "correctness": correctness,
            "numerical_budget": vars(FP8_BUDGET),
            "graph_report": graph_report,
            "quantization_report": quantization_report,
            "cross_kv_report": cache_report,
            "checkpoint_quality_certified": False,
        },
    )
    print(
        f"full_stack_{profile} speedup={result.speedup_median:.2f}x "
        f"CI[{result.speedup_ci_low:.2f},{result.speedup_ci_high:.2f}] correctness=passed"
    )
    return [result], applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-stack composed optimization A/B")
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--profile", choices=sorted(_PROFILES), default="service")
    parser.add_argument("--cuda-graph", action="store_true", help="use graphs instead of static cross K/V")
    args = parser.parse_args()
    results, applied = run(profile=args.profile, cuda_graph=args.cuda_graph, static_cross_kv=not args.cuda_graph)
    suite = f"full_stack_{args.profile}"
    print()
    print(summarize_markdown(results, suite=suite), end="")
    json_path, md_path = write_result(
        results, suite=suite, out_dir=args.out, optimization=applied.to_optimization_snapshot()
    )
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
