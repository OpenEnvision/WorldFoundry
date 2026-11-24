# `worldfoundry` package

Source of the `worldfoundry` Python package. For installation and end-user usage, see the
[repository README](../README.md).

## Layout

| Package | Responsibility |
| --- | --- |
| [`cli/`](cli/README.md) | `wf` command and the terminal UI |
| [`mcp/`](mcp/README.md) | MCP server exposing discovery, runs, and Studio jobs to agents |
| [`studio/`](studio/README.md) | Browser workspace, single-model Studio frontends, visualizers |
| [`runtime/`](runtime/README.md) | Conda profiles, CUDA tiers, devices, compile cache, asset staging |
| [`data/`](data/README.md) | Model and benchmark catalogs, runtime profiles, configs, fixtures |
| [`zoo/`](zoo/README.md) | Model metadata loaders: catalog registry, runtime profiles, environments, assets, pipeline bindings |
| [`pipelines/`](pipelines/README.md) | Public per-family pipeline classes (`from_pretrained`, `__call__`) |
| [`operators/`](operators/README.md) | Input validation and shaping for each pipeline |
| [`synthesis/`](synthesis/README.md) | In-tree model execution: local runtimes, action policies, hosted APIs |
| [`base_models/`](base_models/README.md) | Inference-only model code: native diffusion, perception, 3D/4D, MLLM |
| [`core/`](core/README.md) | Shared infrastructure: loading, attention, VRAM, distributed, kernels, IO |
| [`evals/`](evals/README.md) | Run orchestration, benchmark runners, metrics, reporting, scorecards |
| [`training/`](training/README.md) | Training and post-training recipes; not on the inference/evaluation path |

## How a run flows

```text
cli / mcp / studio            user intent → request
  └─ runtime + zoo + data     environment, catalog, readiness
      └─ pipelines            public entry point for a model
          ├─ operators        validate and shape inputs
          └─ synthesis        execute the model
              └─ base_models + core
  └─ evals                    score artifacts → run_manifest.json, scorecard.json
```

Models and benchmarks meet only through serializable contracts (`GenerationRequest`,
`GenerationResult`, `scorecard.json`): a benchmark never imports a pipeline, and a pipeline
never imports an evaluator.

The full narrative is in the
[Architecture docs](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture) and the
[codebase map](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/file-index).
