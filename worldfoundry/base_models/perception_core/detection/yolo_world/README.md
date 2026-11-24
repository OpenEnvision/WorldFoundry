# YOLO-World

YOLO-World open-vocabulary detector (vendored with its mmdet/mmyolo configs).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench-2.0 | `human_anatomy` (person, face, and hand crops) | `../../human_anatomy/vit_detector/` |

## Usage

`root_path()`, `config_path()`, and `ensure_import_path()` locate the runtime. Weights: VBench-2.0 asset `vbench2_yolo_world_checkpoint` (capability `vbench_metric_checkpoint_assets`).
