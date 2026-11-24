# `worldfoundry.zoo`

Loaders for model metadata under [`worldfoundry/data/models`](../data/README.md): which models
exist, how each one is run, and which pipeline class serves it. Inference, Studio, the CLI, and
evaluation all read model metadata through this package.

## Layout

| Path | Purpose |
| --- | --- |
| `catalog/` | Catalog schema (`ModelZooEntry`), manifest loading, policy checks, and the model zoo registry |
| `runtime/` | Runtime profiles (`load_runtime_profile`), conda environment profiles, and checkpoint asset manifests |
| `bindings.py` | Pipeline bindings and `resolve_pipeline_route()` — catalog id → pipeline class |
| `aliases.py` | Pipeline alias groups for runtime routing |

The cross-registry check (`make runtime-registry-check`) lives in
[`evals/models/validate.py`](../evals/models/validate.py) because it also validates evaluation runner targets.
