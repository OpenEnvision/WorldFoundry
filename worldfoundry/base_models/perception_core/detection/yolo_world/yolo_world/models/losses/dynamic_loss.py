# Copyright (c) Tencent Inc. All rights reserved.

import torch
import torch.nn as nn
from mmdet.models.losses.mse_loss import mse_loss
from mmyolo.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class CoVMSELoss(nn.Module):
    def __init__(self, dim: int = 0, reduction: str = "mean", loss_weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        pred: Tensor,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function of loss."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        cov = pred.std(self.dim) / pred.mean(self.dim).clamp(min=self.eps)
        target = torch.zeros_like(cov)
        loss = self.loss_weight * mse_loss(cov, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
