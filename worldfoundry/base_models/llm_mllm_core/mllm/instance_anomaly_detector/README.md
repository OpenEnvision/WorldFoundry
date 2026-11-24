# `mllm/instance_anomaly_detector`

Instance-level anomaly detector for generated videos: a ModelScope-Swift LoRA adapter on top of an
MLLM that answers whether an object instance stays intact across frames.

## Used by

| Benchmark | Metric (dimension) | Caller |
| --- | --- | --- |
| VBench-2.0 | Instance Preservation | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/instance_preservation.py` |

## Usage

```python
from worldfoundry.base_models.llm_mllm_core.mllm.instance_anomaly_detector import compute_anomaly

scores = compute_anomaly(prompt_dict_ls, device="cuda", submodules_dict={"model": adapter_dir})
```

`submodules_dict["model"]` is the Swift adapter directory; its saved `args.json` names the base
MLLM, which is loaded through [`modelscope_swift`](../modelscope_swift/). The VBench-2.0 runner
passes the adapter staged with its other metric weights (`wf models assets --capability
vbench_metric_checkpoint_assets`).
