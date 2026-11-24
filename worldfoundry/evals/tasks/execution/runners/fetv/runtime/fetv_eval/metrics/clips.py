"""
mid.metric
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

import random
from typing import Any

import torch
from torch import Tensor
from torchmetrics.metric import Metric


class ClipScore(Metric):
    r"""
    Calculates CLIP-S which is used to assess the alignment between the
    conditional texts and the generated images.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """

    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self, feature: int = 512, limit: int = 30000, avg_first: bool = True, weight=1.0, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = True
        self._dtype = torch.float64
        self.name = "CLIPScore"
        self.weight = weight

        for k in ["x", "y", "x0"]:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor, x0: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. This
        method changes the precision of features into double-precision before
        saving the features.

        Args:
            x (Tensor): tensor with the extracted real image features
            y (Tensor): tensor with the extracted text features
            x0 (Tensor): tensor with the extracted fake image features
        """
        assert x0.shape[0] == y.shape[0] and x0.shape[-1] == y.shape[-1]

        self.orig_dtype = x0.dtype
        y, x0 = [x.double() for x in [y, x0]]
        self.x_feat.append(None)
        self.y_feat.append(y)
        self.x0_feat.append(x0)

    def _modify(self, mode: Any = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with
                the condition by randomly-shuffling their counterparts.
        """
        if mode == "real":
            self.x0_feat = self.x_feat  # real text instead of fake text
        elif mode == "shuffle":
            random.shuffle(self.x0_feat)
        return self

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the CLIP-S score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0) for k in ["y", "x0"]]

        return self._compute(*feats, reduction).to(self.orig_dtype)

    def _compute(self, Y: Tensor, Z: Tensor, reduction):
        def dot(x, y):
            return (x * y).sum(dim=-1)

        excess = Z.shape[0] - self.limit
        if excess > 0:
            Y, Z = [x[:-excess] for x in [Y, Z]]

        scores = dot(Z, Y)

        if reduction:
            return self.weight * scores.float().mean()
        else:
            return self.weight * scores.float()
