"""End-to-end multi-step denoise-loop A/B: CUDA graph on vs off.

The per-operator microbenchmarks show launch-bound kernels get 2-3x from a CUDA
graph, but the honest question for inference is: what does that buy across a
REAL multi-step denoise loop, where the same inner DiT call is replayed once per
diffusion step? This benchmark drives the real ``Cosmos2Denoiser`` preconditioning
around a stub DiT (a launch-bound stack of many small layers, standing in for the
28-36 transformer blocks whose shapes are fixed across steps), then runs an
N-step loop with ``enable_cuda_graph`` off (eager, the reference-framework
baseline for a loop with no graph capture) vs on (our capture/replay path).

The stub is intentionally launch-bound (many tiny layers) because that is where
graphs help and where a real DiT's per-step CPU-launch overhead lives; the point
is to measure the *loop-level* speedup our graph wiring delivers, not to model a
specific checkpoint's arithmetic. Shapes are fixed across steps (only the latent
values and the timestep change), which is exactly the regime a CUDA graph
requires and the denoise loop provides.

Run:
  python -m benchmarks.inference.denoise_loop [--out benchmarks/results] [--steps 30]
"""

from __future__ import annotations

import argparse
import copy

import torch

from benchmarks.harness import PairedABResult, bench_paired, summarize_markdown, write_result
from benchmarks.inference.correctness import compare_step, require_graph_execution, validate_steps
from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos2 import Cosmos2Denoiser
from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner


class _LaunchBoundDiT(torch.nn.Module):
    """Stub DiT: a deep stack of small ops, matching the Cosmos2 inner signature.

    Fixed shapes across calls; only the input values change per step. This mimics
    the launch-bound nature of a real transformer stack's per-step dispatch cost
    without needing a checkpoint.
    """

    def __init__(self, channels: int, layers: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Conv3d(channels, channels, 1) for _ in range(layers)]).to(
            device, dtype
        )

    def forward(
        self,
        network_input: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        fps: float = 16.0,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = network_input
        for block in self.blocks:
            y = torch.nn.functional.silu(block(y))
        return y + timestep[:, None, :, None, None] + context.mean() * float(fps) * 1e-4


class _GraphModel(torch.nn.Module):
    """Keep the module/parameter contract while graphing only pure DiT math."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.graph = InferenceCUDAGraphRunner(model.forward)

    def forward(self, *args, **kwargs):
        return self.graph(*args, **kwargs)


# (channels, layers, T, H, W, tag). More layers = more launch-bound.
_CASES = [
    (16, 28, 4, 16, 16, "cosmos2_2b_28L"),
    (16, 36, 4, 16, 16, "cosmos2_14b_36L"),
]


def _make_input(
    channels: int, T: int, H: int, W: int, device: torch.device, seed: int, dtype: torch.dtype
) -> DenoiserInput:
    gen = torch.Generator(device=device).manual_seed(seed)
    B = 1
    shape = (B, channels, T, H, W)
    lat = torch.randn(*shape, device=device, dtype=dtype, generator=gen)
    cl = torch.randn(*shape, device=device, dtype=dtype, generator=gen)
    cm = torch.randint(0, 2, shape, device=device, generator=gen).to(dtype)
    ci = torch.zeros(*shape, device=device, dtype=dtype)
    ctx = torch.randn(B, 12, 1024, device=device, dtype=dtype, generator=gen)
    ts = torch.rand(B, device=device, dtype=dtype, generator=gen) + 0.1
    return DenoiserInput(
        latents=lat,
        timestep=ts,
        next_timestep=ts,
        conditioning={
            "context": ctx,
            "condition_latents": cl,
            "condition_mask": cm,
            "condition_indicator": ci,
            "fps": 16.0,
        },
        step_index=0,
        total_steps=1,
    )


def run(steps: int = 30, dtype: torch.dtype = torch.bfloat16) -> list[PairedABResult]:
    validate_steps(steps)
    assert torch.cuda.is_available(), "denoise-loop benchmark needs a CUDA device"
    device = torch.device("cuda", 0)
    results: list[PairedABResult] = []
    for channels, layers, T, H, W, tag in _CASES:
        model = _LaunchBoundDiT(channels, layers, device, dtype).eval()
        graph_model = _GraphModel(copy.deepcopy(model))
        eager = Cosmos2Denoiser(model)
        graphed = Cosmos2Denoiser(graph_model)
        # A fixed pool of per-step inputs (values change each step, shapes fixed).
        step_inputs = [_make_input(channels, T, H, W, device, seed=s, dtype=dtype) for s in range(steps)]

        def loop_eager() -> torch.Tensor:
            out = None
            for di in step_inputs:
                out = eager(di).sample
            return out

        def loop_graph() -> torch.Tensor:
            out = None
            for di in step_inputs:
                out = graphed(di).sample
            return out

        # Check every changing step before timing; the last step cannot hide a
        # stale input or a bad intermediate result. Verify actual capture/replay.
        with torch.inference_mode():
            correctness = []
            for step, di in enumerate(step_inputs):
                ref, got = eager(di).sample, graphed(di).sample
                if ref.shape != di.latents.shape:
                    raise AssertionError("denoiser changed the latent shape")
                correctness.append(compare_step(ref, got, step=step))
            graph_report = graph_model.graph.report()
            require_graph_execution(graph_report, steps=steps)

        with torch.inference_mode():
            result = bench_paired(
                tag,
                loop_eager,
                loop_graph,
                label_a="eager_loop",
                label_b="cuda_graph_loop",
                device=device,
                warmup=3,
                iters=3,
                rounds=6,
                workload={
                    "channels": channels,
                    "layers": layers,
                    "steps": steps,
                    "grid": f"{T}x{H}x{W}",
                    "correctness": correctness,
                    "graph_report": graph_report,
                },
            )
        results.append(result)
        print(
            f"{tag:18s} steps={steps} layers={layers} speedup={result.speedup_median:.2f}x "
            f"CI[{result.speedup_ci_low:.2f},{result.speedup_ci_high:.2f}] correctness=passed"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end denoise-loop CUDA-graph A/B")
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    results = run(steps=args.steps)
    print()
    print(summarize_markdown(results, suite="denoise_loop"), end="")
    json_path, md_path = write_result(results, suite="denoise_loop", out_dir=args.out)
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
