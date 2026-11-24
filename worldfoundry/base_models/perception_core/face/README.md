# `perception_core/face`

Face detection and face-identity embeddings for human-centric video metrics.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| `retinaface.py` | RetinaFace detector (checkpoint helpers) | VBench-2.0 → `human_identity`; VBench++ trustworthiness → `gender_bias`, `skin_bias` |
| [`arcface/`](arcface/README.md) | ArcFace ResNet face embeddings (`resnet_face18`) | VBench-2.0 → `human_identity` |

## Usage

- RetinaFace weights: VBench asset `vbench_retinaface_checkpoint`, override
  `WORLDFOUNDRY_VBENCH_RETINAFACE_CKPT`.
- ArcFace weights: VBench-2.0 asset `vbench2_arcface_checkpoint`.

Both belong to capability `vbench_metric_checkpoint_assets`. Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
