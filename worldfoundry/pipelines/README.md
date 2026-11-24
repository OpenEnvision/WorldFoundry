# `worldfoundry.pipelines`

Public entry points for models: one thin pipeline class per model family. A pipeline loads
the model (`from_pretrained`), turns user inputs into artifacts (`__call__` / `process`), and
optionally streams interactive output (`stream`). This is the API to use when calling a model
from Python.

## Usage

```python
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

pipe = Wan2p1T2VPipeline.from_pretrained(device="cuda")
result = pipe(
    "A paper boat drifting down a forest stream",
    num_frames=81,
    seed=42,
    output_path="tmp/wan21.mp4",
    return_dict=True,
)
print(result["artifact_path"])
```

Native diffusion pipelines share the call signature `prompt`, `images` / `image_path`,
`video` / `video_path`, `interactions`, `negative_prompt`, `height`, `width`, `num_frames`,
`num_inference_steps`, `guidance_scale`, `seed`, `output_path`, `return_dict`. Other families
add model-specific arguments (camera trajectories, reference images, control videos); see the
model page under [Models](https://openenvision.github.io/WorldFoundry/docs/guides/supported-models).

To find the class for a catalog model, run
`wf zoo model-show --model-id <id> --json` and read `pipeline_target`.

## Layout

| Path | Purpose |
| --- | --- |
| `pipeline_utils.py` | `PipelineABC`, the base class for all pipelines |
| `native_diffusion.py`, `native_diffusion_video.py` | Shared adapter over the native diffusion kernel in `base_models/diffusion_model` |
| `component_pipelines.py` | Pipelines assembled from reusable components |
| `api_runtime.py` | Shared helpers for hosted-API pipelines (Kling, Veo, Sora, …) |
| `<family>/pipeline_*.py` | One directory per model family (`wan/`, `cosmos/`, `matrix_game/`, `hunyuan_video/`, …) |

## Adding a pipeline

A runnable model needs a pipeline **and** an [operator](../operators/README.md). Keep the
pipeline thin: reuse canonical components from `base_models/diffusion_model` and `core`
instead of copying sampling loops or loaders, and bind it to its catalog ID under
`data/models/bindings/`. See [Model runtime](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/model-runtime).
