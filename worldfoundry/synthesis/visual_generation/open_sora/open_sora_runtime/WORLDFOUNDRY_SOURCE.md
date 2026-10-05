Open-Sora v1.2.0 inference source
================================

This directory contains the `opensora` package and `scripts/inference.py` from
[`hpcaitech/Open-Sora` tag `v1.2.0`](https://github.com/hpcaitech/Open-Sora/tree/v1.2.0),
commit `17cce908b22283acc3c946816c81f46dd442a453`. The upstream repository
LICENSE is Apache-2.0; third-party notices in individual source files are
retained. Training scripts, demos, datasets, and tests are omitted.

WorldFoundry makes six single-GPU compatibility changes to the upstream
inference sources:

- `scripts/inference.py` imports ColossalAI only when torch distributed is
  initialized. Single-GPU inference does not use ColossalAI.
- `opensora/utils/ckpt_utils.py` imports ColossalAI checkpoint helpers only
  inside the sharded-checkpoint loader; its `Booster` annotation is postponed.
  It loads an official Hugging Face `model.safetensors` from a checkpoint
  directory directly, while preserving ColossalAI loading for sharded checkpoints.
- `opensora/schedulers/iddpm/respace.py` uses the current torch CUDA device
  instead of ColossalAI's device helper when constructing its timestep map.
- `opensora/models/text_encoder/t5.py` accepts a local T5 checkpoint directory
  as well as the upstream Hugging Face repository ID.
- `opensora/datasets/utils.py` writes H.264 frames through PyAV directly so
  current PyAV versions can choose the picture type during encoding.

Distributed execution and sharded checkpoints still require the upstream
ColossalAI dependency. WorldFoundry's supported route is single GPU.

The bundled `tools/convert_t5_to_safetensors.py` prepares a separate derived
DeepFloyd T5 cache for the pinned torch 2.5.1 environment. It reads trusted
published shards with `torch.load(weights_only=True)` and never changes the
source checkpoint directory.
