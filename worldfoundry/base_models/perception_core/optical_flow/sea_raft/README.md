# SEA-RAFT

SEA-RAFT optical flow.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WorldScore | Default optical-flow, flow AEPE, and motion-accuracy metrics | `worldscore/.../metric_impls/flow_metrics.py` |
| WorldArena | Flow AEPE | `WorldArena/flow_aepe_metrics.py` |

## Usage

`from worldfoundry.base_models.perception_core.optical_flow.sea_raft import RAFT, load_ckpt, config_path, checkpoint_path`. Stage the weights with `wf models assets --capability sea_raft --execute-downloads`.
