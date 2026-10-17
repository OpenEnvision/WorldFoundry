# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact, address-stable feature storage owned by a streaming request."""

import torch


def update_feature_cache(previous, current):
    """Copy a completed causal step's tail, reusing matching inference storage."""
    if not isinstance(current, torch.Tensor):
        return current
    if (
        not torch.is_grad_enabled()
        and isinstance(previous, torch.Tensor)
        and previous.shape == current.shape
        and previous.dtype == current.dtype
        and previous.device == current.device
    ):
        previous.copy_(current)
        return previous
    memory_format = torch.channels_last_3d if current.ndim == 5 and current.stride(1) == 1 else torch.contiguous_format
    return current.clone(memory_format=memory_format)


def commit_feature_cache(state, updated):
    """Commit only after the decoder has finished reading the previous history."""
    for index, value in enumerate(updated):
        state[index] = update_feature_cache(state[index], value)
