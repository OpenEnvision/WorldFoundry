# `perception_core/frame_interpolation`

Video frame interpolation models. Metrics interpolate the middle frame and compare it with the
real one; smooth motion reconstructs well.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`amt/`](amt/README.md) | AMT-S (and AMT-L/G, IFRNet configs) | `motion_smoothness` in VBench, VBench++ (I2V, trustworthiness), DEVIL, HarnessEval-W, MiraBench, and WBench; Open-Sora-Plan frame interpolation |
| [`vfimamba/`](vfimamba/README.md) | VFIMamba | `motion_smoothness` in WorldScore, WorldArena, and 4DWorldBench (`consistency_motion_smoothness`) |

## Usage

```python
from worldfoundry.base_models.perception_core.frame_interpolation.amt import checkpoint_path, config_path
from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba import Model, checkpoint_path, init_model_config
```

- AMT: configs in `worldfoundry/data/models/runtime/configs/amt/`; weights are VBench asset
  `vbench_amt_s_checkpoint`. `WORLDFOUNDRY_AMT_OUTPUT_OFFLOAD` moves intermediate frames to CPU.
- VFIMamba: capability `vfimamba`.

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
