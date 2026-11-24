# SAM 3

SAM 3 promptable concept segmentation (image and video).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| Apple-PI | Physical-law segmentation | `worldfoundry/evals/tasks/execution/runners/apple_pi/apple_pi_runtime.py` |
| MemoBench | Object-reappearance step (optional) | — |
| LiveWorld (generation) | Entity masks | `worldfoundry/synthesis/visual_generation/liveworld/` |

## Usage

`from worldfoundry.base_models.perception_core.segment.sam3 import build_sam3_image_model, build_sam3_predictor`. Stage the weights with `wf models assets --capability sam3 --execute-downloads`.
