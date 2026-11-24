# OpenS2V-Nexus

OpenS2V-Nexus subject-to-video scorers.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| In-tree scorer metrics | `nexus_score`, `natural_score`, `gme_score` | `worldfoundry/evals/tasks/metrics/scorer/opens2v/` |
| In-tree scorer metrics | `facesim_cur` | `worldfoundry/evals/tasks/metrics/scorer/facesim_cur/wrapper.py` |

## Usage

Reached via `ensure_opens2v_eval_path()` from `worldfoundry.evals.tasks.metrics.scorer.opens2v` and `worldfoundry.evals.tasks.metrics.scorer.facesim_cur`.
