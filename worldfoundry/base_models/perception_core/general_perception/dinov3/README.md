# DINOv3

DINOv3 ViT backbones.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| MIND | DINO metric (DINOv3 ViT-B/16) | `worldfoundry/evals/tasks/execution/runners/mind/runtime/mind/src/utils/dino_utils.py` |
| Depth Anything v1 (DAP) | Encoder | `three_dimensions/depth/depth_anything/depth_anything_v1/` |

## Usage

Stage the weights with `wf models assets --capability dinov3_base --execute-downloads`. The weights are gated (request access from Meta first). MIND also reads `WORLDFOUNDRY_MIND_DINO_PATH` and `${WORLDFOUNDRY_CKPT_DIR}/mind/dinov3_vitb16`.
