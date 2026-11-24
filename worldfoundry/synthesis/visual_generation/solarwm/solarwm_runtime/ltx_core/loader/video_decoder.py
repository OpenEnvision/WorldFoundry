# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""SolarWM loader operations for the canonical LTX diffusion decoder.

Adapted from Lightricks/LTX-2 d1511477; see THIRD-PARTY-NOTICES.
"""

from typing import cast

from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
    DiffusionVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.apply import apply_diffvae_mode
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import DiffVAEMode
from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import CompilationConfig

from .module_ops import ModuleOps


def _diffusion_decoder_matcher(model: nn.Module) -> bool:
    return isinstance(model, DiffusionVideoDecoder)


def build_diffvae_mode_op(mode: DiffVAEMode, compilation_config: CompilationConfig | None = None) -> ModuleOps:
    """Apply the selected decoder preset before checkpoint assignment."""

    def mutator(model: nn.Module) -> nn.Module:
        return apply_diffvae_mode(cast(DiffusionVideoDecoder, model), mode, compilation_config=compilation_config)

    return ModuleOps(name=f"diffvae_mode_{mode.value}", matcher=_diffusion_decoder_matcher, mutator=mutator)


def build_cutlass_fna_diffusion_decoder_op() -> ModuleOps:
    """Keep the model-loader operation identity for the chunked-eager preset."""
    inner = build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER)
    return ModuleOps(name="cutlass_fna_diffusion_decoder", matcher=inner.matcher, mutator=inner.mutator)
