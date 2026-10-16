# Compile, Warmup, Cache, and Offload Lifecycles

Use this reference when changing one of these execution paths. Run commands
from the repository root. The
[native diffusion architecture](../../../docs/fumadocs/content/docs/maintainers/architecture/native-diffusion.mdx)
owns components and runners; the
[validation guide](../../../docs/fumadocs/content/docs/reference/validation.mdx)
owns acceptance. Apply the relevant sections to the selected workload.

## Follow the Real Compute Route

Trace the public pipeline or resident worker through the recipe, assembler,
runner, component adapter, compute callable, and leaf kernel. Resolve final
runtime policy and confirm the target checkpoint, input media, and adapters.
Establish a runnable eager baseline at the requested task and shape before
changing acceleration. A smaller or substitute workload is a smoke test, not
acceptance for a failing or unavailable target workload.

Evaluate only existing modes affected by the change. For each selected eager,
compile, graph, or offload mode, identify the actual compute entrypoint, module
owner, weight/storage lifetime, and cleanup boundary. Parser acceptance and a
neighboring model's implementation do not establish execution support.

## Keep Compilation and Capture Owner-Local

Reuse [compile caching](../../../worldfoundry/core/execution/compile_cache.py)
and [inference graphs](../../../worldfoundry/core/execution/graphs/inference_graph.py).
Keep stable tensor compute inside the compiled/captured region and checkpoint
loading, offload transfers, Python request setup, and cleanup outside it.

- Reuse wrappers only for their actual owner and compile policy/options.
  Offload staging objects may differ from logical layers; inspect which one
  the callable captures before changing cache identity.
- Check shape, dtype, device, layout/stride, conditioning/CFG branch, and stage.
  For packed or distributed paths, include post-padding local token lengths
  and per-rank tensor shapes. Equal output resolution is not an equal graph.
- Keep capture storage stable and inputs current. Model replacement or changed
  graph-visible storage must follow the existing invalidation path. Returned
  outputs must not alias buffers overwritten by the next replay.
- Check Dynamo guards and leaf-kernel dispatch. A compiled callable can select
  a different attention or third-party kernel; label that as an optimized-stack
  comparison rather than a compile-only gain.
- Use request-window execution receipts where provided. Previous warmup capture
  or lifetime trace counts cannot certify the measured request's replay.
  Record eager fallbacks and signature misses as such.

## Warm the Same Path Without Carrying Request State

When implementing warmup and compile together, validate eager warmup first.
Reuse the production component methods and shared dispatch rather than a second
sampling or compilation implementation. Cover the target signatures and the
branches/stages actually used by the workload, including decoder execution if
pipeline warmup promises it. Fully consume a decoder iterator; constructing it
alone does not execute its kernels. Distributed warmup must preserve the actual
rank participation and collective order.

Separate reusable weights, compiled code, and captured storage from mutable
request RNG, latents, masks, scheduler history, conditioning, and cache cursors.
Use the existing per-run context and reset/cleanup APIs after warmup, including
failure paths. The first real request must use its own seed and inputs. Preserve
offload prefetch, staging-buffer, residency, and stream-ordering contracts.

Compare a fresh no-warmup request with a post-warmup request using identical
inputs and seed, then exercise changed inputs/seeds and A/B/A reuse. Check
exception recovery and cancellation only where the route exposes them. Reuse
existing lifecycle regressions instead of adding model-specific copies of
shared ownership tests.

## Distinguish Startup from Steady-State Benefit

Keep loading, compilation/capture, warmup, first request, steady compute,
offload transfer, decode, and artifact/transport time separate. Record warmup
plus the first completed request as well as the steady-state result. Preserve
per-run measurements and peak memory; mean latency alone can hide cold-start
or graph-pool costs. Report allocator-reserved and live allocated memory
separately when diagnosing an OOM.

A slow first step after warmup is not proof of recompilation. Inspect compile
and kernel traces, signature coverage, first allocation, H2D residency,
synchronization, and input-dependent work. Request-time model movement and
state reset can remain necessary even after successful warmup.

For Studio streaming, measure first delivered chunk/frame, cadence, final
artifact completion, and queue/transport cost at the consumer boundary. A
model callback alone does not establish public streaming or cancellation.
Use a bounded next-request recovery check for changed worker/cache ownership;
run longer reliability or concurrency validation when making those claims.

The diagnosis-first, native-path, and lifecycle procedures draw on
[SGLang diffusion](https://github.com/sgl-project/sglang),
[LightX2V](https://github.com/ModelTC/LightX2V), and
[vLLM-Omni](https://github.com/vllm-project/vllm-omni), adapted to WorldFoundry's
execution and validation contracts.
