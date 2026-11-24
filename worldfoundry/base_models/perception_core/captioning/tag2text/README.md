# Tag2Text

Tag2Text image tagger and captioner (Swin backbone).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench | `scene` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/scene.py` |
| FETV | Reused vision components of the BLIP retrieval model | `../../video_text/fetv_blip/` |

## Usage

Weights: VBench asset `vbench_tag2text_checkpoint` (Hugging Face space `xinyu1205/recognize-anything`, capability `vbench_metric_checkpoint_assets`).
