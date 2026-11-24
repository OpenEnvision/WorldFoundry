# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo-WM's per-call camera, causal-cache and late-mask inputs.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from dataclasses import dataclass
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs


@dataclass(frozen=True)
class EchoWMControl:
    """Borrow the request's camera tensors and per-layer caches without retaining them on the model."""

    action_cond: dict[str, torch.Tensor] | None = None
    kv_caches: list[dict[str, Any]] | None = None
    current_video_token_start: int = 0
    current_audio_token_start: int = 0


@dataclass(frozen=True)
class EchoWMModality(Modality):
    """LTX tokens with Echo's alternate late-layer and cross-modal masks."""

    late_attention_mask: torch.Tensor | None = None
    cross_kv_mask: torch.Tensor | None = None
    cross_query_mask: torch.Tensor | None = None
    late_cross_kv_mask: torch.Tensor | None = None
    late_cross_query_mask: torch.Tensor | None = None
    v2a_grad_scale: float = 1.0
    control: EchoWMControl | None = None


@dataclass(frozen=True)
class EchoWMTransformerArgs(TransformerArgs):
    """Prepared masks and request-local camera/cache inputs for an Echo-WM block."""

    cross_attention_mask: torch.Tensor | None = None
    cross_output_mask: torch.Tensor | None = None
    late_cross_attention_mask: torch.Tensor | None = None
    late_cross_output_mask: torch.Tensor | None = None
    late_self_attention_mask: torch.Tensor | None = None
    v2a_grad_scale: float = 1.0
    control: EchoWMControl | None = None
