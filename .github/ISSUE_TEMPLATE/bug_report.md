---
name: Bug report
about: Report a reproducible failure in a WorldFoundry pipeline, benchmark, runtime, or doc workflow
title: "[Bug]: "
labels: ["bug", "needs-triage"]
assignees: ""
---

## Summary

<!-- Required: describe the observed failure in one or two sentences. -->

## Affected Pipeline / Benchmark

<!-- Required: name the pipeline, benchmark, runtime profile, or command path involved. -->

- Pipeline:
- Benchmark:
- Runtime profile:
- Public entrypoint or script:

## Environment

<!-- Required: include enough detail for a maintainer to reproduce. Do not paste secrets. -->

- Python version:
- Install command or profile:
- Device / CUDA / GPU:
- OS:
- Relevant package overrides:

## Reproduction

<!-- Required: provide the smallest command or code path that triggers the bug. -->

```bash

```

## Expected Behavior

<!-- Required: what should have happened? -->

## Actual Behavior

<!-- Required: what happened instead? Include the error message or log excerpt. -->

```text

```

## Checkpoint / API Key Needs

<!-- Required: list required checkpoints, local caches, API providers, or env vars. Write "None" if not needed. Do not paste secrets. -->

- Checkpoints / weights:
- Required environment variables:
- API providers:
- Local cache assumptions:

## Sample Artifact Evidence

<!-- Required when relevant: attach generated outputs, screenshots, audit summaries, or paths under tmp/. -->

- Artifact path or link:
- What it shows:

## Validation Tried

<!-- Required: list commands already run and their results. -->

| Command | Result | Notes |
| --- | --- | --- |
| `conda run -n worldfoundry …` |  |  |

## Preflight / Scorecard Output

<!-- Required when the failure involves evaluation, model integration, or benchmark readiness. -->

- `wf prepare` or preflight report:
- Scorecard path:
- Run manifest path:
- Validation matrix or zoo report output:

## Additional Context

<!-- Optional: related PRs, commits, datasets, or benchmark assets. -->
