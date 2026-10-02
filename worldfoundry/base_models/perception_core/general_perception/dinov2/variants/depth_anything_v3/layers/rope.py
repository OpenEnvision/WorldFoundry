# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Compatibility exports for the shared WorldFoundry 2D RoPE implementation."""

from worldfoundry.core.attention.rotary.rope_2d import PositionGetter, RotaryPositionEmbedding2D

__all__ = ["PositionGetter", "RotaryPositionEmbedding2D"]
