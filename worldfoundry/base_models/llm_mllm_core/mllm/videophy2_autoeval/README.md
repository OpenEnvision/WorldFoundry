# `mllm/videophy2_autoeval`

VideoPhy-2 AutoEval: the official mPLUG-Owl based judge for VideoPhy-2
(checkpoint `videophysics/videophy_2_auto`).

## Used by

| Benchmark | Metrics | Caller |
| --- | --- | --- |
| VideoPhy-2 | `semantic_adherence`, `physical_commonsense`, `joint_score`, `rule_followed_rate` | `worldfoundry/evals/tasks/execution/runners/videophy2/videophy2_runtime.py` |

## Usage

Stage the judge and dataset together with
`wf models assets --capability videophy2_autoeval_metric_stack`; the runner reads the checkpoint
from `WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_CKPT` or the staged HFD copy.
