# CoTracker

CoTracker 2/3 point tracker (offline and online predictors).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench-2.0 | Camera motion, human identity, multi-view consistency | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/` |
| VBench++ I2V | Camera motion | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_i2v/camera_motion.py` |
| VMBench | Perceptible amplitude, temporal coherence | `worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official/` |
| ChronoMagic-Bench | CHScore | `worldfoundry/evals/tasks/execution/runners/chronomagic_bench/runtime/chronomagic_bench/CHScore/` |
| MiraBench | Motion metrics | `worldfoundry/evals/tasks/execution/runners/mirabench/` |

## Usage

```python
from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerPredictor

tracker = CoTrackerPredictor()          # WORLDFOUNDRY_COTRACKER2_CKPT or ${WORLDFOUNDRY_CKPT_DIR}
tracks, visibility = tracker(video, grid_size=10)   # video: (B, T, 3, H, W)
```
