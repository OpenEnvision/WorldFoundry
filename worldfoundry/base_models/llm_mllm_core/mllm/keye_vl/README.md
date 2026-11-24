# `mllm/keye_vl`

Path resolution for Kwai Keye-VL-1.5-8B, the multimodal judge of 4DWorldBench. The model itself
loads through `transformers` (remote code); this package only decides which local copy to use.

## Used by

| Benchmark | Metric (dimension) | Caller |
| --- | --- | --- |
| 4DWorldBench | Alignment, motion-QA, and caption-helper dimensions | `worldfoundry/evals/tasks/execution/runners/four_d_worldbench/runtime/four_d_worldbench/paths.py` |

## Usage

```python
from worldfoundry.base_models.llm_mllm_core.mllm.keye_vl import model_path

path = model_path()  # local directory when staged, otherwise "Kwai-Keye/Keye-VL-1_5-8B"
```

Lookup order: `WORLDFOUNDRY_KEYE_VL_MODEL`, `WORLDFOUNDRY_4DWORLDBENCH_KEYE_MODEL`,
`KEYE_MODEL_PATH`, then the HFD cache. Stage the weights with
`wf models assets --capability keye_vl`. 4DWorldBench also needs `keye-vl-utils` and
`qwen-vl-utils` installed.
