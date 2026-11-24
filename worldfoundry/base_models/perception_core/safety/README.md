# `perception_core/safety`

Content-safety classifiers for generated videos.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`nudenet/`](nudenet/README.md) | NudeNet ONNX detector | VBench++ trustworthiness → `safety` (`worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_trustworthiness/safety.py`) |

## Usage

```python
from worldfoundry.base_models.perception_core.safety.nudenet import NudeDetector, model_path
```

The ONNX model path comes from `WORLDFOUNDRY_NUDENET_ONNX`, else the packaged default. VBench++
safety also uses the Stable Diffusion safety checker and Q16 prompts (VBench assets
`vbench_sd_safety_checker_model`, `vbench_q16_prompts`).
