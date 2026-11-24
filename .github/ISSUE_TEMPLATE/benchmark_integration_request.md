---
name: Benchmark integration request
about: Propose adding or updating a benchmark, dataset, metric, or evaluation runner
title: "[Benchmark Integration]: "
labels: ["benchmark-integration", "needs-triage"]
assignees: ""
---

## Summary

<!-- Required: what benchmark should WorldFoundry support, and what capability does it measure? -->

## Benchmark Details

<!-- Required: provide canonical references and access details. -->

- Benchmark name:
- Paper / project / repository:
- Dataset or asset source:
- License:
- Expected task family:
- Proposed WorldFoundry benchmark runner:

## Data And Asset Requirements

<!-- Required: describe data size, storage expectations, download steps, and whether assets can live in data/benchmarks/. -->

- Dataset size:
- Suggested local path:
- Download or generation command:
- Redistribution constraints:
- Fixture subset for `data/test_cases/`:

## Pipeline Coverage

<!-- Required: name the pipelines or model families that should be evaluated. -->

- Target pipeline(s):
- Required input representation:
- Expected output artifact:
- Streaming or multi-turn requirements:

## Metrics And Validation Plan

<!-- Required: identify metrics and a minimum verification matrix. -->

| Area | Proposed command or evidence | Required? |
| --- | --- | --- |
| Metric correctness check |  | Yes / No |
| Fixture-based regression |  | Yes / No |
| Full or sampled benchmark run |  | Yes / No |
| Real inference validation |  | Yes / No |
| Docs build |  | Yes / No |

## Preflight / Scorecard Evidence

<!-- Required if available: provide command output paths. Use "Not available yet" for proposals. -->

- `wf prepare` or preflight report:
- Contract or validation scorecard:
- Expected `leaderboard_valid` value:
- Known blockers before official validation:

## Checkpoint / API Key Needs

<!-- Required: list any model checkpoints, API providers, or env vars needed to run the benchmark. Write "None" if not needed. Do not paste secrets. -->

- Checkpoints / weights:
- Required environment variables:
- API providers:
- Local cache assumptions:

## Sample Artifact Evidence

<!-- Required if available: attach upstream examples, local audit summaries, metric outputs, or generated media. -->

- Artifact path or link:
- What it demonstrates:
- Known limitations:

## Additional Context

<!-- Optional: related standards, baseline numbers, known incompatibilities, or prior integration attempts. -->
