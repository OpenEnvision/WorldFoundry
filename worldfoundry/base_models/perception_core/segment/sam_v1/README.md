# SAM (v1)

Segment Anything ViT-H / ViT-B predictors and automatic mask generator.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| T2V-CompBench | Spatial relationships, motion binding | `worldfoundry/evals/tasks/execution/runners/t2v_compbench/runtime/t2v_compbench/grounded_sam_metrics/` |
| VMBench | Perceptible amplitude score | `worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official/` |
| WorldScore | Motion accuracy | `worldscore/.../motion_accuracy_metrics*.py` |
| Track-Anything | Mask initialisation | `../../tracking/track_anything/` |

## Usage

Capabilities `sam_v1` (ViT-H, `WORLDFOUNDRY_SAM_VIT_H_CKPT`) and `sam_vit_b` (`WORLDFOUNDRY_SAM_VIT_B_CKPT`). RepViT-SAM reuses its predictor.
