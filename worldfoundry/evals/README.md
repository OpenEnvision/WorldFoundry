# `worldfoundry.evals`

The benchmark side of WorldFoundry: resolves models and benchmarks from the catalog, turns
inputs into `GenerationRequest` rows, obtains `GenerationResult` rows (by running a model or
importing existing results), computes metrics, and writes the run's evidence —
`run_manifest.json`, `results.jsonl`, `metrics/summary.json`, `scorecard.json`, `report.md`.

## Usage

```bash
wf score --benchmark vbench --artifacts /path/to/videos --output-dir tmp/score --plan-only
wf run --model <model-id> --benchmark <benchmark-id> --mode official-run --output-dir tmp/bench
wf evaluate --mode existing-results --results-path tmp/results.jsonl --metric artifact_count --output-dir tmp/eval
wf metric list
```

Guide: [Evaluation](https://openenvision.github.io/WorldFoundry/docs/evaluation) ·
[Benchmark Hub](https://openenvision.github.io/WorldFoundry/docs/evaluation/benchmark-hub).

## Layout

| Path | Purpose |
| --- | --- |
| `api/` | Public contracts: `GenerationRequest` / `GenerationResult`, tasks, metrics, models, registries |
| `public.py` | Programmatic entry point: `run_worldfoundry`, `run_benchmark`, request/result types |
| `models/pipelines/` | Pipeline invocation, lifecycle, and result conversion for evaluation runs |
| `models/validate.py` | Runtime registry check: catalog, bindings, profiles, environments, and runner targets |
| `models/runners/` | `resolve_model_zoo_runner()` and runner registry/plugins |
| `tasks/execution/orchestration/` | Run modes: `run.py` (`wf run` dispatch), `service.py` (surface intents), evaluate, model × benchmark, suites, plans, resume, cache, scoring |
| `tasks/execution/runners/<benchmark>/` | Per-benchmark official runners (VBench, EvalCrafter, WorldScore, …) |
| `tasks/metrics/{distribution,perceptual,scorer}/<metric>/` | Categorized executable metrics (FID, FVD, JEDi, CMMD, Clean-FID, LPIPS, SSIM, PSNR, DINO, LAION aesthetic, OpenS2V, …), `_shared/`, and `registry.py` |
| `tasks/embodied/` | Simulator-backed closed-loop evaluation and policy adapters |
| `tasks/datasets/`, `tasks/catalog/`, `tasks/contracts/` | Dataset manifests, benchmark catalog loading, task contracts |
| `reporting/` | Run manifests, scorecards, reports, run index and comparison |

## Rules

- Benchmarks score artifacts, never checkpoints: a runner reads `GenerationResult` rows and files, not pipeline objects.
- A run that is missing assets or artifacts ends with blockers in `scorecard.json`, not a partial number presented as a score.
- Adding a benchmark: catalog entry in `data/benchmarks/catalog/`, a runner under `tasks/execution/runners/`, and metrics under `tasks/metrics/`. See [Evaluation core](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/evaluation-core).
