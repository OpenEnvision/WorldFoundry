---
name: worldfoundry-inference-validation
description: >
  Use this skill when selecting or running validation after WorldFoundry model
  or runtime changes, updating integration tests, or reviewing inference claims.
  Covers component parity, checkpoint conversion, numerical drift diagnosis,
  weight-free contracts, CUDA checks, accepted-reference checkpoint replays,
  and artifact comparisons.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and prepared Python environment.
  CUDA and local checkpoints are required for GPU and replay gates.
---

# Validate WorldFoundry Inference

Run commands from the WorldFoundry repository root. Follow
[contribution rules](../../CONTRIBUTING.md). Read the relevant change-type and
evidence sections in the
[validation guide](../../docs/fumadocs/content/docs/reference/validation.mdx)
before selecting gates; it owns resource and accepted-reference contracts.

## Select Evidence for the Change

Identify the changed public surface and its affected models and variants.
Before a check, identify the failure it would detect and the action that a
failure would require. Keep these evidence levels distinct:

- Catalog, binding, and mocked tests verify resolution and request wiring.
- Small real components verify numerical behavior and request isolation.
- CUDA contracts verify selected kernels, streams, caches, and precision paths.
- Accepted-reference checkpoint replays verify the declared weights, inputs,
  numerical outputs, and exported artifacts.
- Benchmark runs verify the configured evaluator on the executed cohort.

Imports, mocks, and `plan_only` are not checkpoint-backed generation evidence.
Retain tests exposing defects in supported integrations. Remove tests only
when their implementation or contract has actually been removed.

## Run the Applicable Fast Gates

Run from the repository root in the prepared GPU development environment:

```bash
make test-infer
make test-infer-tensors
```

Use `make test-infer-contracts` when the additional model/serializer dependencies
are prepared. Use the relevant CUDA targets, including
`make test-infer-cuda-contracts`, on supported hardware. Select only the gates
required by the changed surface, as described in the validation guide.

For stateful execution, cover changed inputs, A/B/A repeatability, failed
requests, cancellation, reset, and subsequent reuse. For public wrappers, verify
supported input validation, loading delegation, and observable outputs.
Required Make gates reject skipped, missing, duplicate, or stale test evidence.

## Compare Components and Locate Numerical Drift

For model ports and changed tensor math, compare deterministic component outputs
before the full sampling loop. Align weights, inputs, normalization, timestep,
dtype, and attention kernel; verify missing/unexpected checkpoint keys and
conversion transforms before investigating activations. Use FP32 and small real
components when supported, then validate the actual deployment precision and
shape. The owning acceptance contract determines tolerances.

When parity fails, load [numerical parity](references/numerical-parity.md).
Check the initial latents and conditioning, then identify the first divergent
step or block instead of explaining only the final image. Use existing replay
capture or instance-local extension hooks, narrowing instrumentation to the
failed boundary. Keep debug traces under `tmp/` and remove temporary hooks.

## Replay Affected Checkpoint Cases

Use a committed candidate and an accepted reference. The model-host profile
supplies private checkpoint and input paths; keep it outside Git.

```bash
make plan-inference INFERENCE_BASE=<comparison-commit>
make replay-inference INFERENCE_PROFILE=/private/path/profile.json
make verify-inference \
  INFERENCE_PROFILE=/private/path/profile.json \
  INFERENCE_REPORT=/private/path/completed-run/report.json
```

The plan and report must refer to the same candidate and affected cases. Follow
the validation guide for additional fixed-reference, GPU, and trusted CI
attestation requirements. Where completed reference and candidate directories
are required, run `make test-geometry` with `GEOMETRY_REFERENCE` and
`GEOMETRY_CANDIDATE` set to those directories.

## Present the Evidence

Record exact commands, selected cases, environment, pass/fail/skip counts, and
artifact paths in the PR. Label component-only versus public-loader coverage,
and separate functional, numerical, performance, and lifecycle evidence.
Distinguish a defined replay recipe from an executed, accepted replay. Missing
resources or failed gates remain blockers; do not promote readiness or replace
the accepted reference to hide them.
