"""Opt-in, caller-supervised Cosmos Predict inference coordination."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from worldfoundry.core.distributed.model_parallel.inference_parallel import (
    build_inference_parallel_context,
    validate_inference_parallel_policy,
    validate_parallel_degrees,
)


def build_cosmos_parallel_context(options, policy, *, attention_heads: int):
    tp, cp = validate_parallel_degrees(options.get("tensor_parallel", 1), options.get("context_parallel", 1))
    if tp * cp == 1:
        return None
    validate_inference_parallel_policy(policy)
    for name in (
        "quantization", "quantization_mode", "quantization_config", "compile", "torch_compile",
        "cuda_graph", "device_map", "runtime_options", "teacache", "approximate_attention",
    ):
        if options.get(name) not in (None, False, "none", "false", 0):
            raise ValueError(f"multi-rank Cosmos inference does not support {name}")
    if attention_heads % tp:
        raise ValueError("tensor_parallel must evenly divide Cosmos2.5 attention heads")
    parallel = build_inference_parallel_context(tensor_parallel=tp, context_parallel=cp, device=policy.device)
    return parallel


def _input_signature(value):
    """Compare request data once, before conditioning or tensor collectives."""

    if isinstance(value, torch.Tensor):
        copied = value.detach().cpu().contiguous()
        return ("tensor", tuple(copied.shape), str(copied.dtype),
                hashlib.sha256(copied.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return tuple((str(key), _input_signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_input_signature(item) for item in value)

    import numpy as np
    from PIL import Image

    if isinstance(value, (np.ndarray, Image.Image)):
        copied = np.ascontiguousarray(value)
        return ("image-array", copied.shape, str(copied.dtype), hashlib.sha256(copied.tobytes()).hexdigest())
    from pathlib import Path

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"multi-rank Cosmos request input has unsupported type {type(value).__name__}")


def agree_cosmos_request(parallel, **values) -> None:
    error = None
    try:
        signature = _input_signature(values)
    except Exception as caught:
        error = caught
        signature = ("invalid-request", type(caught).__name__, str(caught))
    parallel.agree(signature)
    if error is not None:
        raise error


__all__ = ["agree_cosmos_request", "build_cosmos_parallel_context"]
