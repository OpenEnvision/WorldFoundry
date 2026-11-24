# `mllm/vila`

Inference-only VILA / NVILA runtime: model config, vision tower, generation, and video sampling.

## Used by

| Benchmark | Metrics | Caller |
| --- | --- | --- |
| WorldModelBench | `instruction_following`, `physical_adherence`, `common_sense` (and their average), scored by the VILA-based judge | `worldfoundry/evals/tasks/execution/runners/worldmodelbench/runtime/evaluation.py` |

## Usage

```python
from worldfoundry.base_models.llm_mllm_core.mllm.vila import generate_vila_response, load_vila_model

model = load_vila_model(judge_dir)
answer = generate_vila_response(model, prompt, videos=video_path)
```

Stage the judge with `wf models assets --capability worldmodelbench_vila_judge_model`
(or set `WORLDFOUNDRY_WORLDMODELBENCH_JUDGE`).
