"""Image/video tensor shape configuration shared by the codec and world model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ImageConfig(BaseModel):
    """Spatial and temporal shape of a video tensor.

    Attributes:
        height: Frame height in pixels.
        width: Frame width in pixels.
        channels: Number of colour channels (3 for RGB).
        timesteps: Number of frames per clip.
        fps: Frame rate the clip is sampled at.
    """

    model_config = ConfigDict(extra="forbid")

    height: int
    width: int
    channels: int = 3
    timesteps: int = 20
    fps: int = 10
