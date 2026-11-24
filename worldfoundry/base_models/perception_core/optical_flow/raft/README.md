# RAFT

RAFT optical flow.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench, HarnessEval-W, MiraBench, WBench, WorldArena | `dynamic_degree` | each runner's `dynamic_degree.py` |
| VBench-2.0 | Multi-view consistency | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/multi_view_consistency.py` |
| VBench++ long | Static filter, temporal flickering | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_long/` |
| WorldArena | Flow score | `WorldArena/flow_score.py` |
| WorldScore | Flow and motion-accuracy metrics (RAFT variant) | `worldscore/.../metric_impls/*_raft.py` |
| DEVIL | Dynamics | `worldfoundry/evals/tasks/execution/runners/devil_dynamics/` |

## Usage

`from worldfoundry.base_models.perception_core.optical_flow.raft import RAFT, InputPadder, checkpoint_path`. Stage the weights with `wf models assets --capability raft --execute-downloads`.
