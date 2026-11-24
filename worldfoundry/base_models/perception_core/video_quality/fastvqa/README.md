# FAST-VQA

FAST-VQA / FasterVQA video quality assessment.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| 4DWorldBench | Perceptual FAST-VQA score | `worldfoundry/evals/tasks/execution/runners/four_d_worldbench/runtime/four_d_worldbench/metric/perceptual_fastvqa.py` |

## Usage

Stage the weights with `wf models assets --capability fastvqa --execute-downloads`. Overrides: `WORLDFOUNDRY_FASTVQA_CKPT`, `WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT`.
