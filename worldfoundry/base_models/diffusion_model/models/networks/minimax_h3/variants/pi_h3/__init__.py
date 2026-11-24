# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3.
"""pi-H3 pruned MiniMax-H3 variant and Sol-H3 Decoding Without a Decoder (DWD) head."""

from __future__ import annotations

from .model import (
    PI_H3_DEFAULT_DWD_HEAD_CONFIG,
    PiH3DiTModel,
    PiH3DWDAdaLayerNormOut,
    PiH3PrunedAdalnProj,
    PiH3PrunedFinalLayer,
    PiH3PrunedTimeEmbedder,
)
from .sol_dwd_head import (
    RGBColorEncoder,
    SolH3Attention,
    SolH3Block,
    SolH3DWDHead,
    SolH3SwiGLU,
)

__all__ = [
    "PI_H3_DEFAULT_DWD_HEAD_CONFIG",
    "PiH3DWDAdaLayerNormOut",
    "PiH3DiTModel",
    "PiH3PrunedAdalnProj",
    "PiH3PrunedFinalLayer",
    "PiH3PrunedTimeEmbedder",
    "RGBColorEncoder",
    "SolH3Attention",
    "SolH3Block",
    "SolH3DWDHead",
    "SolH3SwiGLU",
]
