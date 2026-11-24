# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo's cross-stream noise conditioning on the native LTX graph.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
)


class EchoCrossNoisePreprocessor(MultiModalTransformerArgsPreprocessor):
    """Use the other stream's sigma for both AV scale/shift and gate embeddings."""

    def _prepare_cross_attention_timestep(
        self,
        modality_timesteps: torch.Tensor,
        cross_modality_sigma: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._prepare_cross_attention_timestep(
            modality_timesteps=cross_modality_sigma,
            cross_modality_sigma=cross_modality_sigma,
            timestep_scale_multiplier=timestep_scale_multiplier,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )


class JoyAIEchoModel(LTXModel):
    """Native LTX topology with Echo's cross-stream noise convention."""

    def _init_preprocessors(self, cross_pe_max_pos: int | None = None) -> None:
        super()._init_preprocessors(cross_pe_max_pos)
        for name in ("video_args_preprocessor", "audio_args_preprocessor"):
            preprocessor = getattr(self, name, None)
            if isinstance(preprocessor, MultiModalTransformerArgsPreprocessor):
                # Keep the native stem bindings; only cross-noise preparation differs.
                preprocessor.__class__ = EchoCrossNoisePreprocessor

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        reference = video if video is not None else audio
        if reference is not None and perturbations is not None:
            # Echo creates guidance rules before the stem's autocast dtype is known.
            perturbations = BatchedPerturbationConfig.from_masks(
                perturbations.block_masks.to(reference.x), perturbations.block_masks_cpu
            )
        return super()._process_transformer_blocks(video, audio, perturbations)
