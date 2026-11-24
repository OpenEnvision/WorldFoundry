"""Fixed geometry for the LTX-2.5 153-frame profile."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import BackendContractError


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{name} must be a positive integer, got {value!r}")
    return value


def pixel_frames_to_latent_frames(pixel_frames: int, temporal_factor: int = 8) -> int:
    """Map an exact causal-VAE pixel grid to temporal latent slices."""

    frames = _positive_int("pixel_frames", pixel_frames)
    factor = _positive_int("temporal_factor", temporal_factor)
    if (frames - 1) % factor:
        raise BackendContractError(
            f"pixel_frames must have exact causal-VAE form 1 + factor*n; got frames={frames}, factor={factor}"
        )
    return 1 + (frames - 1) // factor


def causal_camera_pixel_indices(
    pixel_frames: int = 153,
    temporal_factor: int = 8,
) -> tuple[int, ...]:
    """Return causal interval starts ``0,1,9,...,145``."""

    latent_frames = pixel_frames_to_latent_frames(pixel_frames, temporal_factor)
    result = (0, *(1 + index * temporal_factor for index in range(latent_frames - 1)))
    if len(result) != latent_frames or result[-1] >= pixel_frames:
        raise BackendContractError("internal LTX causal camera alignment is inconsistent")
    return result


@dataclass(frozen=True)
class LTX25Geometry:
    """The supported LTX-2.5 geometry."""

    pixel_frames: int = 153
    height: int = 512
    width: int = 768
    latent_channels: int = 128
    latent_frames: int = 20
    latent_height: int = 16
    latent_width: int = 24
    temporal_factor: int = 8
    spatial_factor: int = 32
    model_fps: float = 24.0

    def __post_init__(self) -> None:
        observed = (
            self.pixel_frames,
            self.height,
            self.width,
            self.latent_channels,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.temporal_factor,
            self.spatial_factor,
            float(self.model_fps),
        )
        expected = (153, 512, 768, 128, 20, 16, 24, 8, 32, 24.0)
        if observed != expected:
            raise BackendContractError(
                f"LTX-2.5 geometry is fixed to 768x512x153f -> [128,20,16,24] at model_fps=24; got {observed!r}"
            )
        if pixel_frames_to_latent_frames(self.pixel_frames, self.temporal_factor) != self.latent_frames:
            raise BackendContractError("LTX temporal VAE geometry is inconsistent")
        if (self.height // self.spatial_factor, self.width // self.spatial_factor) != (
            self.latent_height,
            self.latent_width,
        ):
            raise BackendContractError("LTX spatial VAE geometry is inconsistent")

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return self.latent_channels, self.latent_frames, self.latent_height, self.latent_width

    @property
    def first_frame_latent_shape(self) -> tuple[int, int, int, int]:
        return self.latent_channels, 1, self.latent_height, self.latent_width

    @property
    def tokens_per_latent(self) -> int:
        return self.latent_height * self.latent_width

    @property
    def video_tokens(self) -> int:
        return self.latent_frames * self.tokens_per_latent

    @property
    def camera_pixel_indices(self) -> tuple[int, ...]:
        return causal_camera_pixel_indices(self.pixel_frames, self.temporal_factor)


STABLE_GEOMETRY = LTX25Geometry()


__all__ = [
    "STABLE_GEOMETRY",
    "LTX25Geometry",
    "causal_camera_pixel_indices",
    "pixel_frames_to_latent_frames",
]
