# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""MoGe-2 grid and DINOv2 attention conventions used by UniWorld-View."""

from typing import Any

from torch import Tensor

from worldfoundry.core.nn.transformer.vit_qkv import QKVSelfAttention

from ..model.v2 import MoGeModel_v2


class UniWorldMoGeModel(MoGeModel_v2):
    """Preserve UniWorld-View arithmetic without changing MoGe checkpoint keys."""

    def __init__(self, **model_kwargs: Any) -> None:
        """Keep xFormers when available and the upstream eager fallback otherwise."""
        super().__init__(**model_kwargs)
        for module in self.encoder.backbone.modules():
            if isinstance(module, QKVSelfAttention):
                module.fused_attn = False

    def _token_grid_shape(self, num_tokens: int | Tensor, aspect_ratio: float) -> tuple[int, int]:
        """Match UniWorld-View@660212c4 rather than rounding each grid dimension."""
        return int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
