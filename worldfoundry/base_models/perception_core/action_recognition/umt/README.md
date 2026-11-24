# UMT action classifier

UMT-L/16 video classifier fine-tuned on Kinetics-400.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench | `human_action` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/human_action.py` |

## Usage

Weights: VBench asset `vbench_umt_checkpoint` (capability `vbench_metric_checkpoint_assets`). Stage the weights with `wf models assets --capability vbench_metric_checkpoint_assets --execute-downloads`.
