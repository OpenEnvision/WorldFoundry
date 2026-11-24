# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Modality geometry shared by GR00T-derived observation transforms."""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class RotationType(Enum):
    """Type of rotation representation"""

    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    EULER_ANGLES_RYP = "euler_angles_ryp"
    EULER_ANGLES_PRY = "euler_angles_pry"
    EULER_ANGLES_PYR = "euler_angles_pyr"
    EULER_ANGLES_YRP = "euler_angles_yrp"
    EULER_ANGLES_YPR = "euler_angles_ypr"


class VideoMetadata(BaseModel):
    """Metadata of the video modality"""

    resolution: tuple[int, int] = Field(..., description="Resolution of the video")
    channels: int = Field(..., description="Number of channels in the video", gt=0)
    fps: float = Field(..., description="Frames per second", gt=0)


class StateActionMetadata(BaseModel):
    absolute: bool = Field(..., description="Whether the state or action is absolute")
    rotation_type: RotationType | None = Field(None, description="Type of rotation, if any")
    shape: tuple[int, ...] = Field(..., description="Shape of the state or action")
    continuous: bool = Field(..., description="Whether the state or action is continuous")


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata] = Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = Field(..., description="Metadata of the state")
    action: dict[str, StateActionMetadata] = Field(..., description="Metadata of the action")


class NormalizationMetadata(Protocol):
    """Family-owned statistics and shared modality descriptions consumed by transforms."""

    statistics: BaseModel
    modalities: DatasetModalities
