# `perception_core/human_anatomy`

Body-part anomaly detection for generated humans.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`vit_detector/`](vit_detector/README.md) | YOLO-World person/face/hand crops + ViT anomaly classifiers | VBench-2.0 → `human_anatomy` (`worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/human_anatomy.py`) |

## Usage

```python
from worldfoundry.base_models.perception_core.human_anatomy.vit_detector import compute_abnormality
```

Weights are VBench-2.0 assets `vbench2_human_anomaly_checkpoint`, `vbench2_face_anomaly_checkpoint`,
`vbench2_hand_anomaly_checkpoint`, and `vbench2_yolo_world_checkpoint` (capability
`vbench_metric_checkpoint_assets`). Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
