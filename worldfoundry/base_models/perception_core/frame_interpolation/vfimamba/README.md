# VFIMamba

VFIMamba frame interpolation.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WorldScore | `motion_smoothness` | `worldscore/.../motion_smoothness_metrics.py` |
| WorldArena | Motion smoothness | `worldfoundry/evals/tasks/execution/runners/worldarena/runtime/video_quality/WorldArena/motion_smoothness_metrics.py` |
| 4DWorldBench | `consistency_motion_smoothness` | `worldfoundry/evals/tasks/execution/runners/four_d_worldbench/runtime/four_d_worldbench/metric/` |

## Usage

`from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba import Model, InputPadder, checkpoint_path, init_model_config`. Stage the weights with `wf models assets --capability vfimamba --execute-downloads`.
