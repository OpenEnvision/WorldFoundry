# `worldfoundry.data`

Declarative, packaged data: which models and benchmarks exist, how they bind to code, what
they need to run, and their default configs. Everything here is YAML/JSON metadata read by
the CLI, Studio, and evaluation; checkpoints and datasets are never stored here.

## Layout

| Path | Contents |
| --- | --- |
| `models/catalog/<category>/<model>.yaml` | Model metadata: ID, aliases, tasks, checkpoints, VRAM, integration status. Categories: `video`, `world_models`, `three_d_four_d`, `vla_va_wam`, `multimodal`, `hosted_api` |
| `models/bindings/pipelines/` | Model ID → pipeline class bindings |
| `models/runtime/profiles/<model>.yaml` | Execution requirements and generation defaults |
| `models/runtime/environments/` | Conda environment specs |
| `models/runtime/configs/<package>/` | Inference presets and example inputs, one directory per synthesis package (snake_case package name) |
| `models/runtime/assets/` | Asset manifests for model checkpoints and demo examples |
| `base_models/` | Shared base-model capabilities (imports, checkpoints, datasets), stacks, and benchmark data assets, loaded by `worldfoundry/base_models/capabilities.py` |
| `benchmarks/catalog/{video,embodied}/` | Benchmark metadata and protocols |
| `benchmarks/tasks/external/` | Task manifests used to materialize requests |
| `benchmarks/runtime_profiles/`, `benchmarks/requirements/` | Benchmark runtime profiles and pip requirements |
| `benchmarks/assets/`, `benchmarks/eval_configs/` | Prompt sets, reference files, evaluation presets |
| `benchmarks/local_assets.example.yaml` | Template for mapping local benchmark data |
| `test_cases/` | Small demo inputs for Studio and smoke tests |
| `io.py` | Loaders for the files above |

## Inspecting the catalog

```bash
wf zoo models
wf zoo model-show --model-id <model-id> --include-manifest --json
wf zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
```

The full directory map and ownership rules are in
[Benchmarks and data](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/benchmarks-data).
A catalog entry may describe a model before it is runnable; only `integrated` entries with a
resolvable pipeline run. See [Add a model](https://openenvision.github.io/WorldFoundry/docs/guides/add-model)
and [Add a benchmark](https://openenvision.github.io/WorldFoundry/docs/guides/add-benchmark).
