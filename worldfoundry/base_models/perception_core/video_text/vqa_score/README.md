# VQAScore (t2v_metrics)

t2v_metrics model zoo for VQAScore, ITMScore, and CLIPScore (CLIP-FlanT5, BLIP-2, InternVideo2, UMT, LLaVA, Qwen2-VL, PerceptionLM backends).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| GenAI-Bench | VQAScore (`clip-flant5-xxl`) | `worldfoundry/evals/tasks/execution/runners/genai_bench/genai_bench_runtime.py` |
| CameraBench | BLIP-2 ITM | capability `t2v_metrics_blip2_itm` |
| PhyGenBench | VQAScore | capability `t2v_metrics_clip_flant5_xxl` |

## Usage

```python
from worldfoundry.evals.tasks.execution.runners._scorers.scoring import get_score_model

scorer = get_score_model("clip-flant5-xxl")
scores = scorer(images=[video_path], texts=[prompt])
```

Capabilities `t2v_metrics_source_assets`, `t2v_metrics_clip_flant5_xxl`, `t2v_metrics_blip2_itm`, `t2v_metrics_internvideo2_itm` (or `--capability t2v_metrics_metric_stack`). `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` sets the cache.
