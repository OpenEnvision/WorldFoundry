# `worldfoundry.base_models.diffusion_model`

The single native diffusion kernel of WorldFoundry. It owns the sampling contracts and the
execution loop; every local diffusion family (Wan, Cosmos, HunyuanVideo, LTX, Sana,
Matrix-Game 3.5, Echo-Memory, Kandinsky 6, …) runs through it. Families contribute
component factories and a declarative recipe — never their own pipeline, runner, or
scheduler — and the package does not wrap DiffSynth, Diffusers, or upstream runtimes.

## Usage

Most users call a model through [`worldfoundry.pipelines`](../../pipelines/README.md) or
`wf run <model-id>`. The kernel can also be used directly:

```python
from worldfoundry.base_models.diffusion_model import (
    DiffusionRequest,
    NativeDiffusionPipeline,
    SamplingConfig,
)

pipe = NativeDiffusionPipeline.from_pretrained("wan2.1-t2v-1.3b")
output = pipe(
    DiffusionRequest(
        prompt="A paper boat drifting down a forest stream",
        height=480,
        width=832,
        num_frames=81,
        sampling=SamplingConfig(num_inference_steps=50, guidance_scale=6.0, seed=42),
    )
)
video = output.sample  # decoded tensor; output.latents keeps the final latents
```

`from_pretrained` also accepts `checkpoint_overrides` (point a component at a local
directory or file), `policy` (offload, precision, attention backend), and `extensions`
(per-instance research hooks).

## Call chain

1. `NativeDiffusionPipeline.from_pretrained(model_id)` looks the model up in the recipe registry.
2. `NativeDiffusionRegistry` lazily materializes a `NativeDiffusionRecipe`.
3. `NativeDiffusionAssembler` builds the components and hands them to an execution strategy.
4. A runner executes encode → initialize → schedule → denoise (+CFG) → decode.
5. The call returns an immutable `DiffusionOutput`.

## Layout

| Path | Purpose |
| --- | --- |
| `contracts.py` | `DiffusionRequest`, `SamplingConfig`, `DiffusionOutput`, and the `Denoiser` / encoder / scheduler / latent protocols |
| `components.py` | Component kinds, keys, build context, and inference/training purpose checks |
| `pipeline.py` | `NativeDiffusionPipeline` — public entry point |
| `assembly.py` | `NativeDiffusionAssembler` — recipe → executable runner |
| `registry.py` | Public registry exports |
| `recipes/` | One declarative recipe per model (`wan.py`, `cosmos2p5.py`, `hunyuan_video.py`, `ltx.py`, `matrix_game.py`, …), `spec.py`, and `registry.py` (`default_native_diffusion_registry`) |
| `models/networks/` | Family-specific DiT / UNet tensor math |
| `models/denoisers/` | State-dict conversion and the shared `Denoiser` adapter per family |
| `models/autoencoders/`, `models/encoders/` | VAEs and text / image / condition encoders |
| `models/initializers/`, `models/upsamplers/`, `models/representations/` | Latent initialization, upsamplers, intermediate representations |
| `loaders/` | Checkpoint description and materialization with full key/shape coverage |
| `runners/` | Sampling loops and execution strategies: standard, chunked, autoregressive, staged, segmented, prefix recompute, … |
| `schedulers/` | Flow-matching, UniPC, DPM, Karras, TrigFlow, and family-specific samplers |
| `optimizations/` | QKV fusion, fused RoPE, static cross-KV, sparse / approximate attention, FP8, sequence parallel, Wan caches (TeaCache, CFG skip) |
| `extensions/` | Instance-local research hooks (frozen context, depth guidance, …) |

[`models/README.md`](models/README.md) indexes the components by role and by model family.

## Adding a model

Add a recipe under `recipes/`, put new tensor math under `models/networks/` and its adapter
under `models/denoisers/`, and register the model ID in `recipes/registry.py`. Reuse existing
schedulers, runners, VAEs, and encoders, and take loading, offload, attention, and kernels
from [`worldfoundry/core`](../../core/README.md). Canonical components must never import a
family package. See [Native diffusion](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/native-diffusion).
