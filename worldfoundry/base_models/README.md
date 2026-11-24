# `worldfoundry.base_models`

Inference-only model implementations shared across pipelines. No training code, datasets,
losses, or experiment scripts live here.

## Layout

| Path | Purpose |
| --- | --- |
| [`diffusion_model/`](diffusion_model/README.md) | The single native diffusion kernel used by every local diffusion family |
| [`perception_core/`](perception_core/README.md) | Perception models: detection, segmentation, tracking, optical flow, captioning, face, video quality, safety, … |
| [`three_dimensions/`](three_dimensions/README.md) | 3D/4D models: `depth/`, `point_clouds/`, `general_3d/`, `slam/`, `three_d_four_d/` (VGGT, Depth Anything, pixelSplat, …) |
| [`llm_mllm_core/`](llm_mllm_core/README.md) | Multimodal LLM components |
| `capabilities.py` | Capability registry and readiness checks; declarations live in `worldfoundry/data/base_models/` |

### `diffusion_model/`

| Path | Purpose |
| --- | --- |
| `contracts.py`, `pipeline.py`, `assembly.py`, `components.py`, `registry.py` | Request/output contracts, `NativeDiffusionPipeline`, component assembly, registry |
| `recipes/` | One immutable recipe per model (`wan.py`, `cosmos2p5.py`, `matrix_game.py`, `hunyuan_video.py`, `ltx.py`, …) |
| [`models/`](diffusion_model/models/README.md) | Networks, denoisers, autoencoders (VAEs), text/image encoders, initializers, upsamplers |
| `loaders/` | Checkpoint loading with full key/shape coverage |
| `runners/`, `schedulers/` | Denoising loops and samplers |
| `optimizations/` | QKV fusion, static cross-KV, caching, and other inference speedups |
| `extensions/` | Research extension points (memory, conditioning) |

A new diffusion model is added as a recipe that composes these roles, not as a new pipeline,
loader, or scheduler. See [Native diffusion](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/native-diffusion).

Users normally reach these models through [`pipelines/`](../pipelines/README.md) or
`wf run <model-id>` rather than importing them directly.
