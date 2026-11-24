# ViT anatomy detector

YOLO-World crops of people, faces, and hands scored by ViT anomaly classifiers.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench-2.0 | `human_anatomy` | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/human_anatomy.py` |

## Usage

`from worldfoundry.base_models.perception_core.human_anatomy.vit_detector import compute_abnormality`. Weights: VBench-2.0 assets `vbench2_human_anomaly_checkpoint`, `vbench2_face_anomaly_checkpoint`, `vbench2_hand_anomaly_checkpoint`, `vbench2_yolo_world_checkpoint` (capability `vbench_metric_checkpoint_assets`).
