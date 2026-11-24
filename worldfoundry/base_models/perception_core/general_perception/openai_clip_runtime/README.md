# OpenAI CLIP runtime

The OpenAI CLIP package (`import clip`); `../openai_clip.py` is the canonical accessor.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| WorldOlympiad | Interaction CLIP score (ViT-B/32) | `worldfoundry/evals/tasks/execution/runners/worldolympiad/base_model_resolver.py` |
| HarnessEval-W | Aesthetic quality, background consistency | `worldfoundry/evals/tasks/execution/runners/harnesseval_w/runtime/metrics/` |
| WBench | Quality metrics (CLIP ViT-B/32, ViT-L/14) | capability `wbench_quality_metric_assets` |

## Usage

```python
from worldfoundry.base_models.perception_core.general_perception import openai_clip

model, preprocess = openai_clip.load("ViT-B/32", device="cuda")
```

Stage the weights with `wf models assets --capability openai_clip_vit_b32 --execute-downloads`. `WORLDFOUNDRY_CLIP_STATE_CACHE` caches loaded state dicts.
