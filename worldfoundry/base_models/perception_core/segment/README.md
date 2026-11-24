# `perception_core/segment`

Promptable segmentation (SAM family) and grounded segmentation.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`sam_v1/`](sam_v1/README.md) | SAM ViT-H / ViT-B | T2V-CompBench → spatial relationships, motion binding; VMBench → perceptible amplitude score; WorldScore → motion accuracy; Track-Anything |
| [`sam2/`](sam2/README.md) | SAM 2 / 2.1 image and video predictors, `SAM2MaskTracker` | VMBench → temporal coherence score; WorldScore → motion accuracy; WorldBench (SAM2 adapter); WBench (SAM2 tracking tool); Physical-AI-Bench (Grounded-SAM2) |
| [`sam3/`](sam3/README.md) | SAM 3 | Apple-PI; MemoBench (optional); LiveWorld pipeline |
| [`grounded_segment_anything/`](grounded_segment_anything/README.md) | Grounded-SAM / Grounded-SAM2 (Grounding DINO boxes → SAM masks, tracked over video) | Physical-AI-Bench → PAI-Bench-C `seg_m_iou`, `seg_recall` |
| [`repvit_sam/`](repvit_sam/README.md) | RepViT-SAM (lightweight SAM) | WonderWorld region proposals via `scene_segmentation.py` |
| `scene_segmentation.py` | ADE20K scene labels + RepViT-SAM region proposals | WonderWorld generation pipeline |
| `sam_core.py`, `mask_data.py` | Shared SAM model class and mask containers | Internal to the packages above |

## Usage

```python
from worldfoundry.base_models.perception_core.segment.sam2 import SAM2MaskTracker
from worldfoundry.base_models.perception_core.segment.grounded_segment_anything import GroundedSAM2VideoSegmenter

segmenter = GroundedSAM2VideoSegmenter(sam2_model_id="facebook/sam2-hiera-large")
records = segmenter.segment(frames, ["robot arm", "cup"])
```

| Capability | Weights |
| --- | --- |
| `sam_v1`, `sam_vit_b` | SAM ViT-H / ViT-B (`WORLDFOUNDRY_SAM_VIT_H_CKPT`, `WORLDFOUNDRY_SAM_VIT_B_CKPT`) |
| `sam2` | SAM 2.1 hiera-base-plus (`WORLDFOUNDRY_SAM2_CKPT`) |
| `sam2_hiera_large` | SAM 2 hiera-large, the official PAI-Bench-C tracker (`WORLDFOUNDRY_PHYSICAL_AI_BENCH_SAM2_CKPT`) |
| `sam3`, `repvit_sam` | SAM 3, RepViT-SAM |

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
