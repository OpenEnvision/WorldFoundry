# `diffusion_model/models`

Canonical model components for the [native diffusion kernel](../README.md), grouped by the
role each one plays in a sampling run rather than by upstream repository. A recipe in
[`../recipes/`](../recipes/) picks one component per role, the assembler builds them, and a
framework runner drives them through the Protocols in [`../contracts.py`](../contracts.py).
No sampling loop, pipeline, scheduler, or checkpoint download lives here.

## Roles

| Path | Recipe slot → Protocol | Contents |
| --- | --- | --- |
| `networks/<family>/` | not bound directly | Checkpoint-compatible DiT / UNet math: module topology, tensor shapes, and the forward pass, with parameter names that match the released checkpoints |
| `denoisers/<family>.py` | `DENOISER` → `Denoiser` | Builds a network from its bound checkpoint, remaps state-dict keys, and adapts the forward call to `DenoiserInput` → `DenoiserOutput`; `graph_wrapped.py` holds the opt-in CUDA Graph and TeaCache mixins used by the Wan and SANA adapters |
| `encoders/<family>/` | `CONDITIONER` → `ConditionEncoder` | Text, image, audio, action, and geometry encoders that turn a `DiffusionRequest` into `Conditioning` (`positive` / `negative` / `shared`); `t5/` and `structured_conditioning.py` are shared |
| `autoencoders/<family>/` | `LATENT_ENCODER` → `LatentEncoder`, `DECODER` → `LatentDecoder` | Image, video, and audio latent codecs |
| `initializers/<family>` | `LATENT_INITIALIZER` → `LatentInitializer` | Initial noise and image/video-conditioned latents in each family's latent geometry; an initializer that implements `EncodedLatentInitializer` reuses the recipe's `LATENT_ENCODER` instead of loading its own VAE; `provided.py` (caller-supplied latents) and `video_conditioning.py` are shared |
| `upsamplers/<family>/` | `LATENT_PROCESSOR` → `LatentProcessor` | LTX's spatial latent processor between resolution stages, and the HunyuanVideo 1.5 spatial upsampler used by its super-resolution pass |
| `representations/<family>/` | not bound directly | Patchify and token-packing helpers shared by one family's roles (Cosmos3, LTX) |

The assembler `isinstance`-checks each built component against its slot's Protocol.
Components satisfy the Protocol structurally; there is no framework base class to inherit.

## How a component is wired

Each family module exposes `build_*` factories. A factory takes one `ComponentBuildContext`
and returns the component: checkpoints come from the names the recipe bound, and device,
dtype, offload, and attention come from `context.policy`. From `recipes/step_video.py` and
`denoisers/step_video.py` (abridged):

```python
# Recipe: bind the factory and name the checkpoint it receives.
ComponentSpec(ComponentKey(ComponentKind.DENOISER), build_step_video_denoiser, {"weights": "transformer"})


# Factory: load the network through the shared loader, then wrap it in the adapter.
def build_step_video_denoiser(context: ComponentBuildContext) -> StepVideoDenoiser:
    model = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=StepVideoModel, config=STEP_VIDEO_CONFIG, layer_container="transformer_blocks"),
        context.require_checkpoint("weights"),
        context.policy,
    )
    return StepVideoDenoiser(model)
```

Import factories from the family module (`models.denoisers.wan`), not from the role package.
The role `__init__.py` files are documentation only, apart from a small lazy export table in
`initializers/__init__.py`.

## Families

Families served by native recipes, with the model IDs registered in
[`../recipes/registry.py`](../recipes/registry.py):

| Family | Code | Model IDs |
| --- | --- | --- |
| Wan 2.1 / 2.2, VACE, SkyReels, Self-Forcing, Causal-Forcing | `networks/wan/`, `denoisers/wan.py`, `denoisers/wan_vace.py`, `encoders/wan/`, `autoencoders/wan/`, `initializers/wan/` | `wan2.1-*`, `wan2.2-*`, `wan2.1-vace`, `skyreels-v2`, `skyreels-v3`, `fastvideo-causal-wan2.2-i2v-14b`, `self-forcing`, `causal-forcing-*` |
| WorldPlay2 | `networks/wan/variants/worldplay2/`, `denoisers/worldplay2.py`, `encoders/wan/variants/worldplay2.py`, `initializers/wan/worldplay2.py` | `worldplay2`, `worldplay2-ar`, `worldplay2-bi` |
| Matrix-Game 4 | `networks/wan/variants/matrix_game4.py`, `denoisers/matrix_game4.py` | `matrix-game-4-egypt` |
| Matrix-Game 3.5 | `networks/matrix_game_3p5/` (on top of the Wan network), `denoisers/matrix_game_3p5.py` | `matrix-game-3.5-first-person`, `matrix-game-3.5-third-person` |
| Echo-Memory | `networks/echo_memory/` (on top of the Wan network), `denoisers/echo_memory*.py`, `initializers/echo_memory*.py` | `echo-memory-context-k1` |
| LTX-2 / LTX-2.3 / LTX-Video | `networks/ltx/`, `denoisers/ltx*.py`, `encoders/ltx/`, `autoencoders/ltx/` (video and audio), `initializers/ltx.py`, `representations/ltx/`, `upsamplers/ltx/` | `ltx-2-*`, `ltx-2.3-*`, `ltx-video-i2v` |
| HelixWorld | `networks/ltx/variants/helixworld.py`, `denoisers/helixworld.py`, `encoders/ltx/variants/helixworld/` | `helixworld` |
| HunyuanVideo / 1.5 | `networks/hunyuan_video/`, `denoisers/hunyuan_video*.py`, `encoders/hunyuan_video/`, `autoencoders/hunyuan_video/`, `initializers/hunyuan_video/`, `upsamplers/hunyuan_video/` | `hunyuanvideo-t2v`, `hunyuanvideo-i2v`, `hunyuanvideo-1.5-t2v`, `hunyuanvideo-1.5-i2v` |
| Cosmos Predict1 (GEN3C) | `networks/cosmos1/`, `denoisers/cosmos1.py`, `autoencoders/cosmos1/`, `initializers/cosmos1/` | `gen3c-cosmos1-7b` |
| Cosmos Predict2 | `denoisers/cosmos2.py` (on the Predict 2.5 network), `autoencoders/cosmos2p5/`, `initializers/cosmos2.py` | `cosmos-predict2-2b-video2world`, `cosmos-predict2-14b-video2world` |
| Cosmos Predict 2.5 / Transfer 2.5 | `networks/cosmos2p5/`, `denoisers/cosmos2p5.py`, `encoders/cosmos2p5/`, `autoencoders/cosmos2p5/` | `cosmos-predict2.5-2b`, `cosmos-predict2.5-14b`, `cosmos-transfer2.5-2b-controlled-video` |
| DreamDojo | `networks/cosmos2p5/action*.py`, `denoisers/dreamdojo*.py`, `encoders/dreamdojo.py`, `autoencoders/cosmos2p5/` | `dreamdojo`, `dreamdojo-student` |
| Cosmos3 | `networks/cosmos3/`, `denoisers/cosmos3.py`, `encoders/cosmos3/`, `autoencoders/cosmos3/`, `initializers/cosmos3.py`, `representations/cosmos3/` | `cosmos3-nano`, `cosmos3-super` |
| Gamma-World | `networks/gamma_world/`, `denoisers/gamma_world/`, `encoders/gamma_world/`, `autoencoders/gamma_world/`, `initializers/gamma_world/` | `gamma-world-causal-few-step`, `gamma-world-causal`, `gamma-world-bidirectional` |
| SANA | `networks/sana/`, `denoisers/sana.py`, `denoisers/sana_refiner.py` (LTX-2 refiner for streaming), `encoders/sana/`, `autoencoders/sana/` (DC-AE), `initializers/sana.py` | `sana-wm`, `sana-wm-streaming`, and the image, Sprint, ControlNet, video, and streaming IDs in `recipes/sana_variants.py` |
| Kandinsky 6 | `networks/kandinsky6/`, `denoisers/kandinsky6.py`, `encoders/kandinsky6/`, `autoencoders/kandinsky6/` | `kandinsky6-pro`, `kandinsky6-lite`, each also as `-distill` and `-pretrain` |
| Step-Video | `networks/step_video/`, `denoisers/step_video.py`, `encoders/step_video/`, `autoencoders/step_video/`, `initializers/step_video.py` | `step-video-t2v` |
| Vchitect 2 | `networks/vchitect/`, `denoisers/vchitect.py`, `encoders/vchitect.py`, `initializers/vchitect.py`, `autoencoders/sd3/` | `vchitect-2-t2v` |
| Hy-Nomas | `networks/hy_nomas.py`, `denoisers/hy_nomas.py`, `encoders/hy_nomas.py`, `initializers/hy_nomas.py`, `autoencoders/pixelumm*.py` | `hy-nomas`, `hy-nomas-low-memory` |
| T2V-Turbo (LVDM) | `networks/lvdm/`, `denoisers/t2v_turbo.py`, `encoders/t2v_turbo.py`, `autoencoders/lvdm/`, `initializers/lvdm.py` | `t2v_turbo_t2v` |
| Stable Diffusion 2, Marigold | `stable_diffusion.py` under `denoisers/`, `encoders/`, `autoencoders/`, and `initializers/` | `sd2-inpaint`, `marigold-normals-v0-1`, `wonderworld-marigold-depth` |

Recipes reuse components across families: the Wan VAE decodes SkyReels, Echo-Memory,
Matrix-Game, and 480p SANA video; the Wan text encoder serves SkyReels, Echo-Memory, and
Matrix-Game; `encoders/t5/` serves Cosmos Predict1, Cosmos Predict2, and LTX-Video.

Families whose pipelines or runtimes import components directly, without a recipe:

| Family | Code | Callers |
| --- | --- | --- |
| Wan-derived world models | `networks/wan/variants/`, `encoders/wan/variants/`, `autoencoders/wan/variants/`, `denoisers/wan_causal_world.py`, `denoisers/wan_dreamx.py` | Runtimes under `worldfoundry/synthesis/` such as `matrix_game`, `video_x_fun`, `zing`, `lingbot_world`, `dreamx_world`, `inspatio_world`, `moverse`, and `versecrafter` |
| MAGI-2 | `networks/magi2/`, `autoencoders/magi2/`, `encoders/magi2_qwen35.py`, `encoders/magi2_refiner.py` | `worldfoundry/pipelines/magi2/`, `../runners/magi2_refinement.py` |
| MiniMax H3 (with Pika and Jing variants) | `networks/minimax_h3/`, `autoencoders/minimax_h3_*`, `encoders/minimax_h3_qwen3vl.py` | `worldfoundry/pipelines/minimax/`, `synthesis/visual_generation/xgen_jing` |
| WorldCrafter, Helios | `networks/worldcrafter/`, `networks/helios/` | `../runners/worldcrafter*.py` and their synthesis runtimes |
| Evoke | `networks/evoke/` | `synthesis/visual_generation/evoke` |
| LVDM (DynamiCrafter, VideoCrafter, Vid2World, MotionCtrl) | `networks/lvdm/`, `encoders/lvdm/`, `autoencoders/lvdm/` | YAML configs under `worldfoundry/data/models/runtime/configs/` that name classes by `target:` path |
| FLUX.2 VAE | `autoencoders/flux2/` | The LARYBench evaluator |

## Conventions

- **Networks are math only.** `networks/` holds no checkpoint loading, backend selection, tokenizers, schedulers, or runtime config. A small architecture schema is fine when it fixes module topology or checkpoint-visible tensor names. Loading and key remapping belong to the denoiser factory.
- **Reuse before adding.** Attention, RoPE, RMSNorm, AdaLN, MLPs, kernels, and distributed helpers come from [`worldfoundry/core`](../../../core/README.md) (`core.attention`, `core.nn`, `core.kernels`, `core.distributed`). When an existing component's checkpoint semantics match, a recipe binds it instead of copying it.
- **Derived models are variants.** A model that extends an existing family adds only its extra layers, in that family's `variants/` folder in each role it touches (`networks/wan/variants/worldplay2/`, `encoders/ltx/variants/helixworld/`), and reuses the family's other roles.
- **Factories are pure.** A factory reads only its `ComponentBuildContext` (named checkpoints, `policy`, options). It never searches a model directory, downloads weights, or returns a runner, and importing a module never registers or loads a model.
- **Dependencies point inward.** Nothing here imports `worldfoundry.synthesis`, `pipelines`, `studio`, `evals`, or `operators`, whether through ordinary imports, lazy imports, or string targets; [`test_module_boundaries.py`](../../../../tests/base_models/diffusion_model/test_module_boundaries.py) enforces this. Diffusers model primitives are allowed; Diffusers and DiffSynth pipelines are not.
- **Inference only.** No training loops, datasets, losses, or experiment configs. Keep heavy imports lazy and keep upstream attribution and license headers on ported code.
- **Dotted paths are an interface.** The LVDM YAML configs and the `validation_imports` lists in `worldfoundry/data/models/runtime/environments/` name modules here by path, so moving or renaming a module means updating them.

## Adding a model

Reuse the family's existing roles and add only what differs: network math under
`networks/<family>/` (or its `variants/`), the adapter and its `build_*` factory under
`denoisers/`, and an encoder, codec, or initializer only when no existing one matches the
checkpoint. Bind them in a recipe and register the model ID in `../recipes/registry.py`.
Weight-free tests go under
[`tests/base_models/diffusion_model/`](../../../../tests/base_models/diffusion_model/), and
parity against the official implementation follows the
[model integration](../../../../skills/worldfoundry-model-integration/SKILL.md) and
[inference validation](../../../../skills/worldfoundry-inference-validation/SKILL.md) skills.
See [Native diffusion](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/native-diffusion)
for the full architecture.

## Known exceptions

- Some networks still read weights themselves: MAGI-2, MiniMax H3, WorldCrafter, Evoke, and
  LVDM (for their direct callers), plus SANA's ControlNet, camera-control, and null-embedding
  paths.
- Echo-Memory and Matrix-Game 3.5 build on the Wan network but keep top-level packages
  (`networks/echo_memory/`, `networks/matrix_game_3p5/`) instead of `networks/wan/variants/`.
