# DINO v1

DINO v1 ViT-S/B backbones (facebookresearch/dino), with a `torch.hub` entry point in `hub/hubconf.py`.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench | `subject_consistency`, `background_consistency` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/subject_consistency.py` |
| VBench++ long | Subject consistency | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_long/` |

## Usage

VBench loads the official `dino_vitbase16_pretrain.pth` through the hub entry point; weights come from the local file, never the network:

```python
import torch
from worldfoundry.base_models.capabilities import vbench_dino_hub_config

model = torch.hub.load(**vbench_dino_hub_config())
```

Weights: VBench asset `vbench_dino_vitb16_checkpoint` (capability `vbench_metric_checkpoint_assets`). Direct use: `from worldfoundry.base_models.perception_core.general_perception.dino import dino_vitb16` (`pretrained=True, path=...`).
