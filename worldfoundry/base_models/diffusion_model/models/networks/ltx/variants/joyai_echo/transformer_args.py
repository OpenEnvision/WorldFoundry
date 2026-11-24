# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo-WM mask preparation layered on the shared LTX stems.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from dataclasses import replace
from typing import cast

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgsPreprocessor
from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.model import (
    EchoCrossNoisePreprocessor,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.echo_wm import (
    EchoWMModality,
    EchoWMTransformerArgs,
)


class EchoWMArgsPreprocessor(TransformerArgsPreprocessor):
    """Add the late self-attention mask and request-local control to native stem outputs."""

    def prepare(self, modality: Modality, cross_modality: Modality | None = None) -> EchoWMTransformerArgs:
        args = super().prepare(modality, cross_modality)
        modality = cast(EchoWMModality, modality)
        return EchoWMTransformerArgs(
            **vars(args),
            late_self_attention_mask=self._prepare_self_attention_mask(
                modality.late_attention_mask, modality.latent.dtype
            ),
            v2a_grad_scale=float(modality.v2a_grad_scale),
            control=modality.control,
        )

    def _prepare_cross_attention_mask(
        self, cross_kv_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        if cross_kv_mask is None:
            return None
        mask = cross_kv_mask.to(x_dtype)
        if mask.ndim == 2:
            return (mask - 1).reshape((mask.shape[0], 1, 1, mask.shape[-1])) * torch.finfo(x_dtype).max
        if mask.ndim == 3:
            return (mask - 1).unsqueeze(1) * torch.finfo(x_dtype).max
        raise ValueError(f"Expected cross_kv_mask shape (B, K) or (B, Q, K), got {tuple(mask.shape)}")

    @staticmethod
    def _prepare_cross_output_mask(cross_query_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        if cross_query_mask is None:
            return None
        mask = cross_query_mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        return mask.to(x_dtype)


class EchoWMAVArgsPreprocessor(EchoCrossNoisePreprocessor):
    """Add early/late cross-modal KV and output masks after Echo's noise conditioning."""

    def prepare(self, modality: Modality, cross_modality: Modality | None = None) -> EchoWMTransformerArgs:
        args = cast(EchoWMTransformerArgs, super().prepare(modality, cross_modality))
        if cross_modality is None:
            return args
        modality = cast(EchoWMModality, modality)
        cross_modality = cast(EchoWMModality, cross_modality)
        preprocessor = cast(EchoWMArgsPreprocessor, self.simple_preprocessor)
        dtype = modality.latent.dtype
        return replace(
            args,
            cross_attention_mask=preprocessor._prepare_cross_attention_mask(cross_modality.cross_kv_mask, dtype),
            cross_output_mask=preprocessor._prepare_cross_output_mask(modality.cross_query_mask, dtype),
            late_cross_attention_mask=preprocessor._prepare_cross_attention_mask(
                cross_modality.late_cross_kv_mask, dtype
            ),
            late_cross_output_mask=preprocessor._prepare_cross_output_mask(modality.late_cross_query_mask, dtype),
        )
