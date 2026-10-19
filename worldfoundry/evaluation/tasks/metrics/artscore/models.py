# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible ArtScore ResNet architecture.

Architecture reference: https://github.com/jchen175/ArtScore.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


def get_resnet(args: Any) -> nn.Module:
    """Construct ResNet50/101 with ArtScore's dense or single-linear regression head."""
    try:
        dropout = float(args.dropout)
    except (TypeError, ValueError, AttributeError):
        dropout = 0.0
    constructor = models.resnet101 if args.backbone_config == "resnet101" else models.resnet50
    resnet = constructor(weights=None)
    if args.no_dense_layer:
        resnet.fc = nn.Linear(2048, 1, bias=True)
    else:
        resnet.fc = nn.Sequential(
            nn.Linear(2048, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 1, bias=True),
        )
    return resnet
