# `mllm/videocon_physics`

VideoCon-Physics, the mPLUG-Owl based physical-commonsense judge released with VideoPhy.

## Used by

| Benchmark | Metrics | Caller |
| --- | --- | --- |
| VideoPhy | `semantic_adherence`, `physical_commonsense`, `joint_score` | `worldfoundry/evals/tasks/execution/runners/videophy/videophy_runtime.py` |

## Usage

The VideoPhy runner calls the entailment inference in this package (`runtime_root()` locates
it). Stage the checkpoint with `wf models assets --capability videocon_physics_model`, or the
whole VideoPhy set with `--capability videophy_metric_stack`.
