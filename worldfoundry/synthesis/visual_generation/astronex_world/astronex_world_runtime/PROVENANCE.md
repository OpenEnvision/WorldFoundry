# Astronex-World inference runtime

Adapted from Astronex-Robotics/Astronex-World, revision
`27584bf1a89da01ef35a03b48d83bc818d2adf2b` (Apache License 2.0; see root
`THIRD-PARTY-NOTICES`). Every file was modified by WorldFoundry.

- `inference/generate.py` and `inference/sample.py` are the release's front end and
  sampler, reduced to the flags the WorldFoundry runtime passes; `sample.py` starts as a
  package module.
- `inference/pipelines/` holds the causal and bidirectional multi-step samplers
  (UniPC from `base_models/diffusion_model/schedulers/flow_unipc.py`).
- `models/denoiser.py` builds the denoiser from
  `base_models/diffusion_model/models/networks/wan/variants/astronex`, which carries the
  Astronex network on the shared Wan norms, RoPE, causal head and MinWM PRoPE math.
- `models/components.py` loads the released UMT5-xxl and Wan2.2 VAE weights onto the
  native `WanTextEncoder` and `VideoVAE38_`.
- `utils/` keeps the camera trajectory parser, the action-token tables and the
  frustum-overlap history selection.
