# RepViT-SAM

RepViT-SAM (RepViT encoder, SAM decoder).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WonderWorld (generation) | Region proposals via `../scene_segmentation.py` | `worldfoundry/synthesis/visual_generation/wonderworld/` |

## Usage

No benchmark metric uses it. Stage the weights with `wf models assets --capability repvit_sam --execute-downloads`.
