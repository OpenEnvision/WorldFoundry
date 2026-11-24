# `worldfoundry.base_models.llm_mllm_core`

Inference-only (multimodal) language models. Their main use is as **judges and feature
extractors in benchmark evaluation** — physical-commonsense raters, video-QA scorers,
anomaly detectors. They also serve as language backbones for a few VLA policies and for
prompt expansion in diffusion pipelines.

## Layout

All models live under `mllm/`:

| Path | Model / role |
| --- | --- |
| [`mllm/qwen/`](mllm/qwen/README.md) | Qwen family: Qwen-VL judges, Qwen3 backbone, Qwen-Image, Cosmos-Reason1, and Being-H / QwenPi policy backbones |
| [`mllm/llava_next/`](mllm/llava_next/README.md), [`mllm/vila/`](mllm/vila/README.md), [`mllm/keye_vl/`](mllm/keye_vl/README.md), [`mllm/perceptionlm/`](mllm/perceptionlm/README.md) | General video/image MLLMs used as evaluators |
| [`mllm/internvideo2/`](mllm/internvideo2/README.md), [`mllm/umt/`](mllm/umt/README.md) | Video–language encoders |
| [`mllm/videophy2_autoeval/`](mllm/videophy2_autoeval/README.md), [`mllm/videocon_physics/`](mllm/videocon_physics/README.md) | Physical-commonsense judges (VideoPhy-2 AutoEval, VideoCon-Physics) |
| [`mllm/instance_anomaly_detector/`](mllm/instance_anomaly_detector/README.md) | Instance-level anomaly detection for generated videos |
| [`mllm/open_flamingo/`](mllm/open_flamingo/README.md) | OpenFlamingo backbone (used by RoboFlamingo) |
| [`mllm/modelscope_swift/`](mllm/modelscope_swift/README.md) | ModelScope Swift inference framework (used by the anomaly detector) |

Each model folder's README lists the benchmarks and metrics that use it and how to stage its
weights.

## Usage

These models are loaded on demand by benchmark runners under
`worldfoundry/evals/tasks/execution/runners/` (VideoScore, WorldModelBench, 4DWorldBench,
…), by VQAScore in `perception_core`, and by a few action policies under
`worldfoundry/synthesis/action_generation/`. Check what a benchmark needs before running it:

```bash
wf zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
python scripts/setup/prepare_benchmark_assets.py --benchmark-id <benchmark-id> --json
```

## Conventions

- Inference only; checkpoints resolve through the Hugging Face cache or `WORLDFOUNDRY_CKPT_DIR`.
- Keep imports lazy — these stacks pin large `transformers` / vision dependencies.
- Hosted-API judges (OpenAI, Gemini, DashScope) are not here; they live with the runners that call them and read credentials from environment variables.
- Retain upstream attribution and license notices in vendored code.
