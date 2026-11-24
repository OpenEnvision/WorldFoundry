# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Runtime-policy limits for multi-rank Cosmos 2.5 inference.

The Cosmos 2.5 denoiser build and the Cosmos pipelines both call
:func:`validate_cosmos_parallel_policy`. The module imports no network code, so a
pipeline can reject an unsupported parallel request before any weights load.
"""

from __future__ import annotations

import torch

from worldfoundry.core.model_loading.policy import OffloadMode, QuantizationMode, RuntimePolicy


def validate_cosmos_parallel_policy(policy: RuntimePolicy) -> None:
    """Reject runtime policies that multi-rank Cosmos inference has not been validated with.

    Raises:
        ValueError: when offload, a non-float32 dtype, autocast, CUDA TF32,
            quantization, compilation or runtime options are requested.
    """

    if policy.offload.mode is not OffloadMode.NONE:
        raise ValueError("multi-rank Cosmos inference requires offload_mode=none")
    if policy.dtype is not torch.float32:
        raise ValueError("multi-rank Cosmos inference currently requires torch_dtype=float32 for numerical stability")
    if torch.is_autocast_enabled(policy.device.type):
        raise ValueError("multi-rank Cosmos inference requires autocast to be disabled")
    if policy.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("multi-rank Cosmos inference requires CUDA matmul TF32 to be disabled")
    if policy.quantization.mode is not QuantizationMode.NONE or policy.compile or policy.options:
        raise ValueError("multi-rank Cosmos inference requires dense, uncompiled weights without runtime options")


__all__ = ["validate_cosmos_parallel_policy"]
