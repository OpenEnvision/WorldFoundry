# `perception_core/tracking`

Point and object trackers for motion and identity metrics.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`cotracker/`](cotracker/README.md) | CoTracker 2/3 point tracker | VBench-2.0 → camera motion, human identity, multi-view consistency; VBench++ I2V → camera motion; VMBench → perceptible amplitude, temporal coherence; ChronoMagic-Bench → CHScore; MiraBench |
| [`dot/`](dot/README.md) | DOT dense point tracking (CoTracker initialisation + refinement) | T2V-CompBench → motion binding (`worldfoundry/evals/tasks/execution/runners/t2v_compbench/runtime/t2v_compbench/dot/`) |
| [`aot/`](aot/README.md) | DeAOT-L tracker checkpoint paths | Track-Anything |
| [`track_anything/`](track_anything/README.md) | Track-Anything (Grounding DINO + SAM + DeAOT) | ViPE preprocessing; `track-anything-prior` model; geometry priors for generation (not a metric) |

## Usage

```python
from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerPredictor, checkpoint_path

tracker = CoTrackerPredictor(checkpoint=str(checkpoint_path()))
tracks, visibility = tracker(video, grid_size=10)  # video: (B, T, 3, H, W)
```

- CoTracker weights: `WORLDFOUNDRY_COTRACKER2_CKPT` or `${WORLDFOUNDRY_CKPT_DIR}`.
- DeAOT-L: capability `aot_deaot_l` (`WORLDFOUNDRY_DEAOT_R50_CKPT`).

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
