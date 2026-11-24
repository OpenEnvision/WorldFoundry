# `worldfoundry.synthesis`

In-tree model execution. A pipeline delegates here to actually run a model: load a checkpoint
into a native runtime, call a vendored upstream runtime, step an action policy, or submit a
hosted-API job. All runtimes are in-tree; none depend on a machine-local external checkout.

## Layout

| Path | Purpose |
| --- | --- |
| `base_synthesis.py` | `BaseSynthesis`, the base class for local runtimes and hosted APIs |
| `visual_generation/<family>/` | Video, world, and camera-controlled generation runtimes (Matrix-Game, LingBot, Cosmos, GEN3C, Lyra, …) and the hosted video client `api_video_client.py` |
| `action_generation/<policy>/` | VLA and action policies (OpenVLA, GR00T, DreamZero, Diffusion Policy, …) with `base_action_synthesis.py` |

A family directory usually contains `synthesis.py` (the `BaseSynthesis` subclass), optional
`memory` modules for streaming or multi-turn state, and a `*_runtime/` directory with the
ported upstream inference code.

## Where a fix belongs

- Input parsing → [`operators/`](../operators/README.md)
- Model execution, checkpoint selection, hosted API calls → here
- Reusable sampling, schedulers, VAEs, text encoders → [`base_models/diffusion_model`](../base_models/README.md)
- Offload, attention backends, kernels, distributed → [`core/`](../core/README.md)

VLA policies used in closed-loop embodied evaluation expose a `predict_action()` callable in
`action_generation/<policy>/runtime.py`, referenced from
`data/models/runtime/configs/vla_va_wam/<policy>.yaml`. See
[Model runtime](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/model-runtime).
