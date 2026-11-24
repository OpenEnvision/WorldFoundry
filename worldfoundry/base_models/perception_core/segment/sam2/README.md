# SAM 2

SAM 2 / 2.1 image and video predictors, plus `SAM2MaskTracker` for mask propagation.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VMBench | Temporal coherence score | `worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official/temporal_coherence_score.py` |
| WorldScore | Motion accuracy | `worldscore/.../motion_accuracy_metrics*.py` |
| WorldBench | SAM 2 adapter | `worldfoundry/evals/tasks/execution/runners/worldbench/runtime/worldbench/sam2_adapter.py` |
| WBench | SAM 2 tracking tool | `worldfoundry/evals/tasks/execution/runners/wbench/runtime/wbench/tools/run_sam2_track.py` |
| Physical-AI-Bench | PAI-Bench-C segmentation (hiera-large) | `../grounded_segment_anything/` |

## Usage

```python
from worldfoundry.base_models.perception_core.segment.sam2 import SAM2MaskTracker

tracker = SAM2MaskTracker(model_id="facebook/sam2.1-hiera-base-plus")
```

Capabilities `sam2` (SAM 2.1 hiera-base-plus, `WORLDFOUNDRY_SAM2_CKPT`) and `sam2_hiera_large` (`WORLDFOUNDRY_PHYSICAL_AI_BENCH_SAM2_CKPT`). Any other `model_id` downloads its checkpoint from the Hugging Face hub.
