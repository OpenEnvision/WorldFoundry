# ArcFace

ArcFace ResNet face-identity embeddings (`resnet_face18`).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench-2.0 | `human_identity` | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/human_identity.py` |

## Usage

`from worldfoundry.base_models.perception_core.face.arcface import resnet_face18`. Weights: VBench-2.0 asset `vbench2_arcface_checkpoint` (capability `vbench_metric_checkpoint_assets`).
