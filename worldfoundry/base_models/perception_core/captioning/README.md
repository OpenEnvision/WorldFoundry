# `perception_core/captioning`

Dense captioning and tagging models used to check what objects, colours, and scenes appear in
generated videos.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`grit/`](grit/README.md) | GRiT dense captioner (object detection + region captions) | VBench → `object_class`, `multiple_objects`, `color`, `spatial_relationship`; WorldArena video-quality metrics |
| [`tag2text/`](tag2text/README.md) | Tag2Text image tagger/captioner | VBench → `scene`; FETV BLIP reuses its vision components |

## Usage

- GRiT weights: VBench asset `vbench_grit_checkpoint`, or `WORLDFOUNDRY_GRIT_DLA_CKPT`; the default
  location is under `WORLDFOUNDRY_VBENCH_CACHE_DIR`.
- Tag2Text weights: VBench asset `vbench_tag2text_checkpoint`.

Both are part of capability `vbench_metric_checkpoint_assets`. Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
