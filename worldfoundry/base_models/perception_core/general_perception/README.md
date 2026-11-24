# `perception_core/general_perception`

General-purpose visual and vision-language backbones. Benchmarks use them for frame embeddings
(consistency metrics) and text–image similarity; depth and 3D models use them as encoders.

| Path | Model | Used by |
| --- | --- | --- |
| [`dino/`](dino/README.md) | DINO v1 (ViT-S/B, `hub/hubconf.py` torch.hub entry point) | VBench → `subject_consistency`, `background_consistency` (via `vbench_dino_hub_config`); VBench++ long → subject consistency |
| `dino_embeddings.py` | DINO ViT-B/16 (HF `facebook/dino-vitb16`) feature loader | WorldScore → DINO consistency; WBench → subject consistency; WorldMemBench → feature metrics; DEVIL → quality distribution; MiraBench |
| [`dinov2/`](dinov2/README.md) | DINOv2 | EWMBench → scene consistency; DEVIL → dynamic metrics; VBench++ long (DINOv2 option); backbone for Depth Anything v2/v3, Video-Depth-Anything, UniDepth, Pi3 |
| [`dinov3/`](dinov3/README.md) | DINOv3 ViT-B/16 | MIND → DINO metric; Depth-Anything-v1 DAP backbone |
| `openai_clip.py`, `openai_clip_runtime/` | OpenAI CLIP (`clip.load`) | WorldOlympiad → interaction CLIP score; HarnessEval-W → aesthetic quality, background consistency; WBench quality metrics; ViCLIP tokenizer |
| [`open_clip/`](open_clip/README.md) | OpenCLIP / SigLIP | MemoBench → visual quality; DreamSim; LVDM condition encoders; LLaVA and OpenFlamingo vision towers |
| `qwen3_vl_entity.py` | Qwen3-VL entity extraction | LiveWorld generation pipeline (not a metric) |
| [`uniception/`](uniception/README.md) | UniCeption encoders/heads | CUT3R and Spatia 3D models (not a metric) |

## Usage

Registered weights:

| Capability | Model |
| --- | --- |
| `dino_vitb16` | DINO ViT-B/16 (HF) |
| `dinov2_base` | DINOv2-base |
| `dinov3_base` | DINOv3 ViT-B/16 (gated; also read from `WORLDFOUNDRY_MIND_DINO_PATH`) |
| `clip_vit_b32`, `siglip_so400m` | OpenCLIP ViT-B/32, SigLIP so400m |
| `openai_clip_vit_b32` | OpenAI CLIP ViT-B/32 (`${WORLDFOUNDRY_CKPT_DIR}/clip`) |

VBench's DINO uses the official `dino_vitbase16_pretrain.pth` through
`dino/hub/hubconf.py`:

```python
import torch
from worldfoundry.base_models.capabilities import vbench_dino_hub_config

model = torch.hub.load(**vbench_dino_hub_config())  # local checkpoint, no network access
```

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
