# NudeNet

NudeNet ONNX nudity detector.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench++ trustworthiness | `safety` | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_trustworthiness/safety.py` |

## Usage

`from worldfoundry.base_models.perception_core.safety.nudenet import NudeDetector, model_path`. The ONNX file comes from `WORLDFOUNDRY_NUDENET_ONNX`, else the packaged default.
