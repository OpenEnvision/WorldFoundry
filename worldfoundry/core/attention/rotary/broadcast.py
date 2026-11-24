# SPDX-FileCopyrightText: 2026 OpenEnvision
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
# Shared from the SAM and SAM2 source distributions.

from collections.abc import Sequence

import torch


def broadcat(tensors: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Broadcast all input axes before concatenating rotary position tables."""
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
