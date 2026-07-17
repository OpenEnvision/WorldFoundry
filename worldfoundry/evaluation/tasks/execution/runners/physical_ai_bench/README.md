# Physical AI Bench (PAI-Bench)

This is an independent, inference-only, in-tree integration of SHI-Labs PAI-Bench-G and PAI-Bench-C. Runtime execution never clones or imports the upstream repository or its vendored `third_party` tree.

## Integration layout

- `generation`: reuses the checked-in VBench and VBench++ I2V runtimes. Binary physical VQA accepts a prediction manifest or a Qwen video judge.
- `conditional-generation`: implements the published CPU transfer metrics locally and reuses shared DOVER, LPIPS, Video-Depth-Anything, GroundingDINO, and SAM2 base models.

Prompts, evaluator settings, and provenance live under `worldfoundry/data/benchmarks/assets/physical-ai-bench`. There is deliberately no `SOURCE.json`.

## Commands

Generation quality only:

```bash
python -m worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.run_physical_ai_bench_official_runner \
  --track generation --run-official \
  --dataset-root /path/to/physical-ai-bench-generation \
  --generated-video-dir /path/to/videos \
  --metrics aesthetic_quality,background_consistency,imaging_quality,motion_smoothness,overall_consistency,subject_consistency \
  --output-dir outputs/pai-bench-g --json
```

Conditional transfer metrics with precomputed depth and segmentation sidecars:

```bash
python -m worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.run_physical_ai_bench_official_runner \
  --track conditional-generation --run-official \
  --dataset-root /path/to/physical-ai-bench-conditional-generation \
  --generated-video-dir /path/to/model-output \
  --pred-depth-dir /path/to/depth-npy \
  --pred-segmentation-dir /path/to/segmentation-json \
  --allow-trusted-pickle \
  --output-dir outputs/pai-bench-c --json
```

## Environment

The standard runtime profile is `worldfoundry-unified-cu128`. Result normalization and the CPU Canny/blur/depth/segmentation metrics do not require a separate environment. Model-backed metrics need the corresponding shared WorldFoundry checkpoints.

PAI-Bench-G's Qwen2.5-VL-72B binary-VQA judge usually needs a multi-GPU serving setup. When it does not fit the unified process, start a vLLM OpenAI-compatible server in its own serving environment and pass:

```bash
--judge-backend openai-compatible \
--judge-model Qwen/Qwen2.5-VL-72B-Instruct \
--judge-base-url http://localhost:8000/v1
```

That extra environment contains only the optional model server; benchmark code and aggregation remain in tree and run from the unified WorldFoundry environment.

Official dataset pickle files are loaded only when `--allow-trusted-pickle` is explicit. Prefer JSON prediction sidecars for generated segmentation.
