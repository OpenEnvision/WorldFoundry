# HPSv3 runtime

Vendored HPSv3 human-preference model; `../hpsv3.py` is the accessor (`load_inferencer`).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WBench | Quality metrics | capability `wbench_hpsv3` |
| HarnessEval-W | HPSv3 quality | `worldfoundry/evals/tasks/execution/runners/harnesseval_w/runtime/metrics/video_quality/hpsv3_quality.py` |

## Usage

Stage the weights with `wf models assets --capability wbench_hpsv3 --execute-downloads`.
