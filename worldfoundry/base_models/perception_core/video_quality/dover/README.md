# DOVER

DOVER video quality assessment (technical and aesthetic branches).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| Physical-AI-Bench | PAI-Bench-C `dover_tech_score` | `worldfoundry/evals/tasks/execution/runners/physical_ai_bench/runtime/physical_ai_bench/conditional_generation.py` |

## Usage

```python
from worldfoundry.base_models.perception_core.video_quality.dover import DOVERTechnicalScorer

score = DOVERTechnicalScorer()(video_path)
```

Stage the weights with `wf models assets --capability dover --execute-downloads`. Override: `WORLDFOUNDRY_DOVER_CKPT`.
