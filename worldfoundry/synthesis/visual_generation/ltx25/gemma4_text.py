# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export for the native PyTorch Gemma-4 Unified encoder."""

from __future__ import annotations

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma4_unified import (
    Gemma4RMSNorm,
    Gemma4UnifiedAttention,
    Gemma4UnifiedDecoderLayer,
    Gemma4UnifiedMLP,
    NativeGemma4UnifiedEncoderWrapper,
    NativeGemma4UnifiedTextBackbone,
    load_native_gemma4_unified_encoder,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma4_unified import (
    _build_gemma4_tokenizer_compat as _build_gemma4_tokenizer_compat,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma4_unified import (
    _build_rope_inv_freq as _build_rope_inv_freq,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma4_unified import (
    _compute_cos_sin as _compute_cos_sin,
)

__all__ = [
    "Gemma4RMSNorm",
    "Gemma4UnifiedAttention",
    "Gemma4UnifiedDecoderLayer",
    "Gemma4UnifiedMLP",
    "NativeGemma4UnifiedEncoderWrapper",
    "NativeGemma4UnifiedTextBackbone",
    "load_native_gemma4_unified_encoder",
]
