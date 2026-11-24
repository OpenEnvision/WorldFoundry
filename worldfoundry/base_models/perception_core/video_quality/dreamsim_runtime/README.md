# DreamSim runtime

Vendored DreamSim package; `../dreamsim.py` is the accessor (`load_model`, `cache_dir`).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WBench | Spatial / consistency metrics | `worldfoundry/evals/tasks/execution/runners/wbench/runtime/wbench/src/evaluate.py` |

## Usage

Stage the weights with `wf models assets --capability wbench_dreamsim --execute-downloads`.
