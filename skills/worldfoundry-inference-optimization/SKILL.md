---
name: worldfoundry-inference-optimization
description: >
  Use this skill when diagnosing slow WorldFoundry inference or changing
  attention backends, torch.compile, CUDA graphs, warmup, KV or feature caches,
  VAE decoding, precision, offload, transfers, or Studio streaming latency.
  Guides matched-stack profiling and optimizations that preserve numerical
  outputs and request isolation.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and prepared CUDA environment.
  Measurements need local checkpoints and representative inputs.
---

# Optimize WorldFoundry Inference

Run commands from the WorldFoundry repository root. Follow
[contribution rules](../../CONTRIBUTING.md). Read
[native diffusion ownership](../../docs/fumadocs/content/docs/maintainers/architecture/native-diffusion.mdx)
when changing diffusion components, and the relevant
[validation](../../docs/fumadocs/content/docs/reference/validation.mdx)
sections when selecting numerical and performance acceptance gates.

## Measure the Executed Path

Trace the public pipeline or worker to its actual compute caller. Establish a
baseline with fixed weights, input, prompt or control schedule, seed, shape,
frame count, sampler, and precision. Record the command, revision, GPU count
and model, driver, CUDA, PyTorch, and selected runtime policy. For Studio,
record worker startup, the single request, and the repeated benchmark command.
Confirm the resolved runtime and actual kernel/compile/graph dispatch through
existing logs and request-scoped receipts. A configured fast path or lifetime
warmup counter does not prove that the measured request used it; label fallback
measurements by their executed path.

Separate loading, cold compilation, warmup, cache fill, steady-state compute,
decode, transfer, encoding, and queue wait. Use CUDA events or explicit
synchronization for GPU attribution; CPU launch time is not completed GPU time.
Use the existing performance harness for the affected integration. Report
measured latency or throughput, memory, warmup policy, and fallback paths.
Profiler-instrumented timings are attribution evidence, not headline timings.
Use at least three measured repeats for a local latency comparison, keeping
per-run values and spread. Collect a short operator/shape trace for compute
hotspots or a separate host-stack trace for launch and synchronization gaps.
Change one performance variable at a time and inspect existing core fast paths
before proposing a custom kernel.

## Change the Measured Bottleneck

- Shared backend selection and attention kernels belong in
  `worldfoundry/core/attention/`; shared execution, graph, cache, and offload
  behavior belongs in the relevant `worldfoundry/core/` domain.
- Diffusion-specific composition uses existing runtime policy, components,
  runners, and recipes. Keep generic infrastructure out of family packages.
- Compile regions should contain stable compute. CUDA graph capture requires
  stable shapes and storage, explicit warmup, and correct stream ordering.
- Cache changes must preserve fill, eviction, reset, cancellation, and reuse
  semantics. Exercise changed inputs and failures between repeated requests.
- Decoder changes compare identical latents. Frame order and queue latency
  remain part of the streaming contract; dropping output frames changes it.
- Precision or approximate-attention changes require the existing acceptance
  contract for the affected case. Do not substitute a new tolerance or quality
  reference to make a candidate pass.

Keep the change narrow and reuse existing configuration surfaces. Add a new
public option only when users need a distinct supported behavior; performance
work does not automatically require another flag or fallback implementation.

## Validate Execution Lifecycles

For compile, graphs, warmup, cache, or offload changes, load
[execution lifecycle](references/execution-lifecycle.md). Verify the real eager
route first, then warmup state isolation, then the accelerated route for the
same supported workload. Cover actual shapes, conditioning branches, object
lifetimes, and leaf kernels rather than inferring support from config names.
Keep compile, CUDA graphs, approximation, and deployment topology as separate
comparisons so their costs and quality effects remain attributable.

## Verify and Measure the Candidate

Add small numerical and state-isolation regressions before a full replay. Use
[inference validation](../worldfoundry-inference-validation/SKILL.md) for CUDA
and affected real-checkpoint gates.

Measure reference and candidate with matched stacks and settings. Use fresh
processes when studying compilation, backend probing, or persistent caches.
Report startup and steady-state costs separately, including warmup plus the
first completed request for one-shot workloads. Moving compilation into warmup
changes when its cost is paid, not necessarily total latency. A speedup is
accepted only with the required numerical, artifact, lifecycle, and quality
evidence.

## Report the Result

State the measured improvement and its hardware and workload scope. Explain
memory, startup, precision, or quality tradeoffs with their measured values.
Update the paired model, environment, or Studio pages that own the changed
behavior. Keep traces, generated media, and replay outputs outside Git.
