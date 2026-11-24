# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# From Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan normalization with rCM's explicit parameter reset contract."""

from ..reference_21 import WanRMSNorm as ReferenceWanRMSNorm


class WanRMSNorm(ReferenceWanRMSNorm):
    """Reference Wan normalization used by rCM's graph initializer."""

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)
