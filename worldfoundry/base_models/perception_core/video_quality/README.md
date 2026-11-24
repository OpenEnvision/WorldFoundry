# `perception_core/video_quality`

Learned video/image quality, preference, and perceptual-distance models.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`dover/`](dover/README.md) | DOVER | Physical-AI-Bench → PAI-Bench-C `dover_tech_score` |
| [`fastvqa/`](fastvqa/README.md) | FAST-VQA / FasterVQA | 4DWorldBench → perceptual FAST-VQA score |
| `lpips.py` | LPIPS (VGG) | Physical-AI-Bench → PAI-Bench-C `lpips_diversity` |
| `dreamsim.py`, `dreamsim_runtime/` | DreamSim perceptual similarity | WBench spatial/consistency metrics |
| `hpsv3.py`, `hpsv3_runtime/` | HPSv3 human-preference score | WBench quality; HarnessEval-W → HPSv3 quality |
| `transnetv2.py` | TransNetV2 shot-boundary detector | WBench → segment continuity |
| [`fetv_stylegan_v/`](fetv_stylegan_v/README.md) | StyleGAN-V FVD code | FETV → FVD |

## Usage

```python
from worldfoundry.base_models.perception_core.video_quality.dover import DOVERTechnicalScorer
from worldfoundry.base_models.perception_core.video_quality.lpips import LPIPSMetric

score = DOVERTechnicalScorer()(video_path)        # WORLDFOUNDRY_DOVER_CKPT or ${WORLDFOUNDRY_CKPT_DIR}/DOVER/...
distance = LPIPSMetric(net="vgg")(frames_a, frames_b)   # NHWC uint8 batches
```

| Capability | Model |
| --- | --- |
| `dover` | DOVER (official release) |
| `fastvqa` | FasterVQA (+ FAST-VQA-B/M), `WORLDFOUNDRY_FASTVQA_CKPT` |
| `lpips_vgg` | LPIPS; weights ship with the `lpips` package, the VGG backbone downloads on first use |
| `wbench_dreamsim`, `wbench_hpsv3`, `wbench_transnetv2` | WBench copies of DreamSim, HPSv3, TransNetV2 |
| `fetv_stylegan_v_fvd_source` | FETV FVD source |

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
