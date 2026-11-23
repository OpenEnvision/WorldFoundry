# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License).
"""FastFrames interactive-latency autoregressive video diffusion pipeline."""

from __future__ import annotations

from .pipeline_fastframes import FastFramesPipeline, configure_fastframes_backends, pixels_to_uint8

__all__ = [
    "FastFramesPipeline",
    "configure_fastframes_backends",
    "pixels_to_uint8",
]
