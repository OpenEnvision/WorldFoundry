# `perception_core/video_text`

Video–text alignment models: how well a generated video matches its prompt.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`viclip/`](viclip/README.md) | ViCLIP (InternVid) | VBench → `overall_consistency`, `temporal_style`; VBench++ trustworthiness → culture fairness; MiraBench → text–video consistency; DEVIL |
| [`vqa_score/`](vqa_score/README.md) | t2v_metrics VQAScore / ITMScore / CLIPScore model zoo (CLIP-FlanT5, BLIP-2, InternVideo2, UMT, LLaVA, Qwen2-VL, PerceptionLM backends) | GenAI-Bench → VQAScore (`clip-flant5-xxl`); CameraBench → BLIP-2 ITM; PhyGenBench → VQAScore; reached through `worldfoundry/evals/tasks/execution/runners/_scorers/` |
| `blip2_feature_extractor.py` | BLIP-2 feature extractor (LAVIS) | VBench++ trustworthiness → gender bias |
| [`fetv_blip/`](fetv_blip/README.md) | FETV BLIP retrieval model | FETV → BLIP-based text–video alignment |
| [`opens2v_nexus/`](opens2v_nexus/README.md) | OpenS2V-Nexus subject-to-video scorers | `tasks/metrics/scorer/opens2v` (`nexus_score`, `natural_score`, `gme_score`), `tasks/metrics/scorer/facesim_cur` (`facesim_cur`) |

## Usage

```python
from worldfoundry.evals.tasks.execution.runners._scorers.scoring import get_score_model

scorer = get_score_model("clip-flant5-xxl")       # VQAScore
scores = scorer(images=[video_path], texts=[prompt])  # one row per video, one column per prompt
```

| Capability | Weights |
| --- | --- |
| `t2v_metrics_source_assets` | t2v_metrics source tree |
| `t2v_metrics_clip_flant5_xxl` | CLIP-FlanT5-XXL VQAScore model |
| `t2v_metrics_blip2_itm`, `t2v_metrics_internvideo2_itm` | BLIP-2 and InternVideo2 ITM models |
| `fetv_blip_retrieval_source` | FETV BLIP source (`WORLDFOUNDRY_FETV_BLIP_TOKENIZER`) |

ViCLIP weights are the VBench asset `vbench_viclip_checkpoint`; `WORLDFOUNDRY_VICLIP_BPE_PATH`
overrides the tokenizer vocabulary. `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` sets the t2v_metrics
cache. Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
