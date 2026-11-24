# `perception_core/optical_flow`

Optical-flow estimators. Metrics use flow magnitude (how much things move) and flow error
against a reference.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`raft/`](raft/README.md) | RAFT | `dynamic_degree` in VBench, HarnessEval-W, MiraBench, WBench, WorldArena; VBench-2.0 → multi-view consistency; VBench++ long → static filter, temporal flickering; WorldArena → flow score; WorldScore → flow / motion-accuracy metrics (RAFT variant); DEVIL |
| [`sea_raft/`](sea_raft/README.md) | SEA-RAFT | WorldScore → default optical-flow, flow AEPE, and motion-accuracy metrics; WorldArena → flow AEPE |
| [`flowformerplusplus/`](flowformerplusplus/README.md) | FlowFormer++ | WorldScore → FlowFormer++ flow / AEPE / motion-accuracy variants |

## Usage

```python
from worldfoundry.base_models.perception_core.optical_flow.raft import RAFT, InputPadder, checkpoint_path
from worldfoundry.base_models.perception_core.optical_flow.sea_raft import RAFT as SEARAFT, load_ckpt
from worldfoundry.base_models.perception_core.optical_flow.flowformerplusplus import build_flowformer, get_cfg
```

Capabilities: `raft`, `sea_raft`, `flowformerplusplus` (all three: `motion_stack`). Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
