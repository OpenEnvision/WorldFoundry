# DINOv2

DINOv2 ViT backbones and hub constructors.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| EWMBench | Scene consistency | `worldfoundry/evals/tasks/execution/runners/ewmbench/runtime/ewmbench/EWMBench/scene_consistency.py` |
| DEVIL | Dynamic metrics | `worldfoundry/evals/tasks/execution/runners/devil_dynamics/runtime/official/tools/` |
| VBench++ long | Subject consistency (DINOv2 option) | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_long/subject_consistency.py` |
| Depth / 3D models | Encoder for Depth Anything v2/v3, Video-Depth-Anything, UniDepth, Pi3 | `worldfoundry/base_models/three_dimensions/` |

## Usage

Stage the weights with `wf models assets --capability dinov2_base --execute-downloads`. (HF `facebook/dinov2-base`).
