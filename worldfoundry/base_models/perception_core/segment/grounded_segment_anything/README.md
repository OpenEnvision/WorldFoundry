# Grounded-SAM / Grounded-SAM2

Grounding DINO boxes turned into SAM masks; the video variant tracks them with SAM 2.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| Physical-AI-Bench | PAI-Bench-C `seg_m_iou`, `seg_recall` | `worldfoundry/evals/tasks/execution/runners/physical_ai_bench/runtime/physical_ai_bench/conditional_generation.py` |

## Usage

```python
from worldfoundry.base_models.perception_core.segment.grounded_segment_anything import GroundedSAM2VideoSegmenter

segmenter = GroundedSAM2VideoSegmenter(sam2_model_id="facebook/sam2-hiera-large", box_threshold=0.3, text_threshold=0.25)
records = segmenter.segment(frames, ["robot arm", "cup"])
```

PAI-Bench-C uses the official configuration above (capabilities `grounding_dino` and `sam2_hiera_large`).
