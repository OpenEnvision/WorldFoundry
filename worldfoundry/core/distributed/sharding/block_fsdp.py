# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Inference-only FSDP1 block wrapping for vendored Wan-style models.

Scope note (TE-13): this module (and :mod:`.fsdp2_sharding`) serves the
vendored inference paths only.  Training code must use
a training-specific sharding implementation — these wrappers configure
FSDP for memory-bounded inference (no optimizer/gradient state handling,
checkpoint format ties, or trainable-parameter audits).
"""

import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage

# ──────────────────────────────────────────────────────────────────────────
# FSDP1 wrap / teardown — inference only; training uses apply_fsdp2
# ──────────────────────────────────────────────────────────────────────────


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False,
):
    """Wrap a vendored block model with FSDP1 for inference (TE-13).

    Inference/vendored-only; training runs must use
    a training-specific sharding implementation.
    """
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype),
        device_id=device_id,
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False,
    )
    return model


def shard_model_orig_params(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    """Always wrap with ``use_orig_params=True`` so LoRA / PEFT can see real names.

    Same FSDP1 inference contract as :func:`shard_model`, but never falls back
    to flattened params. Needed when adapters index parameters by the original
    ``named_parameters`` keys.
    """

    return FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype),
        device_id=device_id,
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=sync_module_states,
        use_orig_params=True,
    )


def free_model(model):
    """Release FSDP flat-param storage, then drop the module and CUDA cache.

    ``del model`` alone leaves the flat parameter allocation alive until the
    next GC cycle, which is enough to OOM the next shard.
    """

    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
