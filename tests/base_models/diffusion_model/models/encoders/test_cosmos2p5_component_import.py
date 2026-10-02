"""Transformers-version compatibility for the shared Reason1 encoder."""

from __future__ import annotations

import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component import (
    Cosmos25TextBackbone,
    Qwen2_5_VLRMSNorm,
)


def test_cosmos25_component_resolves_qwen_rmsnorm_export() -> None:
    assert issubclass(Qwen2_5_VLRMSNorm, nn.Module)
    assert issubclass(Cosmos25TextBackbone, nn.Module)
