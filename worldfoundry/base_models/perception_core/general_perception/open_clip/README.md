# OpenCLIP

OpenCLIP / SigLIP models (model configs live in `worldfoundry/data/models/runtime/configs/open_clip/`).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| MemoBench | Visual quality | `worldfoundry/evals/tasks/execution/runners/memobench/runtime/memobench/evaluation/automated/metrics/visual_quality.py` |
| DreamSim, LVDM, LLaVA, OpenFlamingo | Vision towers and condition encoders | — |

## Usage

Capabilities `clip_vit_b32` (CLIP ViT-B/32) and `siglip_so400m`. Stage with `wf models assets --capability clip_vit_b32 --execute-downloads`.
