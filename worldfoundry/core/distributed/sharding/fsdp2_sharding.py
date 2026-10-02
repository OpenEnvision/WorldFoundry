# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Inference-only FSDP2 sharding for vendored Wan-style block models.

Scope note (TE-13): this module (and :mod:`.block_fsdp`) serves the vendored
inference paths only.  Training code must use
a training-specific sharding implementation instead — the fallback
branch in :func:`shard_model` below silently degrades to a single unsharded
device when the FSDP2 API is unavailable, which is acceptable for inference
but would be a silent single-GPU run (wrong gradients/memory profile) in
training.
"""

import gc
import os

import torch

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:  # Torch < 2.9 lacks the FSDP2 helper used upstream.
    MixedPrecisionPolicy = None
    fully_shard = None

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

# ──────────────────────────────────────────────────────────────────────────
# FSDP2 inference wrap — silent single-device fallback when API is missing
# ──────────────────────────────────────────────────────────────────────────


def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32):
    """Shard a vendored block model with FSDP2 for inference.

    Inference/vendored-only (TE-13): when the torch build lacks the FSDP2
    helpers this silently falls back to an unsharded single-device module,
    which is fine for inference but must never carry a training run — use
    a training-specific sharding implementation for training.
    """
    if fully_shard is None or MixedPrecisionPolicy is None:
        model.to(param_dtype)
        if torch.cuda.is_available():
            model.to(torch.device("cuda", torch.cuda.current_device()))
        return model

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    if os.getenv("WORLDFOUNDRY_FSDP_FORWARD_PREFETCH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        for block, next_block in zip(model.blocks, model.blocks[1:]):
            set_prefetch = getattr(block, "set_modules_to_forward_prefetch", None)
            if callable(set_prefetch):
                set_prefetch([next_block])

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    """Drop the sharded module and flush CUDA cache after inference teardown.

    FSDP2 does not expose a flat-param handle, so this cannot free storage
    the way :func:`block_fsdp.free_model` does — only the Python graph and
    the allocator cache.
    """

    del model
    gc.collect()
    torch.cuda.empty_cache()
