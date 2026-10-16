---
name: worldfoundry-benchmark-integration
description: >
  Use this skill when adding a benchmark to WorldFoundry, porting an evaluator,
  importing official results, or fixing benchmark readiness and scoring.
  Guides catalog metadata, task materialization, dataset contracts, official
  runners, metric correctness, and normalized scorecards.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and prepared evaluator environment.
  Official runs need the declared datasets, credentials, or simulators.
---

# Integrate a WorldFoundry Benchmark

Run commands from the WorldFoundry repository root. Follow
[contribution rules](../../CONTRIBUTING.md). Read
[benchmarks and data](../../docs/fumadocs/content/docs/maintainers/architecture/benchmarks-data.mdx)
when changing catalog or materialization contracts, and the
[CLI contract](../../docs/fumadocs/content/docs/reference/cli.mdx)
when constructing evaluation commands.

## Define the Evaluation Protocol

Record the upstream evaluator and license, dataset and split, sample IDs,
expected model artifacts, metric definitions and direction, judge or simulator
requirements, and official result format. Choose an existing benchmark with
the same execution contract as the implementation reference.

## Connect Data and Execution

1. Add metadata under `worldfoundry/data/benchmarks/catalog/` and, when
   required, task YAML under `worldfoundry/data/benchmarks/tasks/external/`.
   Declare dataset references, access requirements, metrics, and blockers.
2. Use the existing catalog parsers and registries. Keep runtime execution in
   `worldfoundry/evaluation/tasks/execution/runners/`; use existing official
   runner and normalizer contracts instead of creating a parallel evaluator.
3. Prove task materialization before generation. Inspect stable sample IDs,
   resolved input paths, and required artifacts using `worldfoundry-eval task
   materialize --help` and the benchmark's documented arguments.
4. Add deterministic fixtures for parsing, materialization, metric correctness,
   and normalization. Verify failed and skipped samples remain visible. Test
   scoring with known values, not merely the presence of a metric key.
5. Preserve the official cohort, denominator, aggregation, and preprocessing.
   Missing outputs must follow the benchmark's published policy; successful
   samples alone do not establish full-suite validity.

## Exercise the Supported Mode

Discover the declared surface before preparing assets:

```bash
worldfoundry-eval zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
worldfoundry-eval zoo benchmark-run --help
```

Use the benchmark Hub page for data roots and evaluator-specific arguments.
The modes have different evidence meanings:

- `contract` checks wiring and output contracts.
- `normalizer` converts supplied results without executing the official scorer.
- `official-validation` imports official-shaped results as integration evidence.
- `official-run` executes the declared evaluator with prepared resources.

Inspect `scorecard.json`, including `score_valid`, `leaderboard_valid`,
`normalizer_only`, fidelity, and blockers. CLI success is not leaderboard
validity. Keep integration status proportional to executed evidence.

## Deliver the Integration

Include deterministic regressions in the relevant Make/CI selection, and
record exact official execution commands and scorecard artifacts when run.
Maintain paired pages under
`docs/fumadocs/content/docs/evaluation/benchmark-hub/`. Keep datasets,
credentials, model outputs, and evaluator caches outside Git. Run the catalog,
packaging, and docs checks required by CONTRIBUTING.md.
