# GRiT

GRiT dense captioner (DLA backbone): object boxes with region captions.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench | `object_class`, `multiple_objects`, `color`, `spatial_relationship` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/` |
| WorldArena | Video-quality object metrics | `worldfoundry/evals/tasks/execution/runners/worldarena/runtime/video_quality/WorldArena/third_party/grit_model.py` |

## Usage

Weights: VBench asset `vbench_grit_checkpoint` (capability `vbench_metric_checkpoint_assets`); override `WORLDFOUNDRY_GRIT_DLA_CKPT`. The default location is under `WORLDFOUNDRY_VBENCH_CACHE_DIR`.
