# FlowFormer++

FlowFormer++ optical flow.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WorldScore | FlowFormer++ flow, AEPE, and motion-accuracy variants | `worldscore/.../metric_impls/*_flowformer.py` |

## Usage

`from worldfoundry.base_models.perception_core.optical_flow.flowformerplusplus import build_flowformer, get_cfg, checkpoint_path`. Stage the weights with `wf models assets --capability flowformerplusplus --execute-downloads`.
