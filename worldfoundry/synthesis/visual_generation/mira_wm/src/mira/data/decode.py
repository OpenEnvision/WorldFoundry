"""Video decoding from raw in-tar bytes.

TorchCodec is used when its native library matches the installed torch build. PyAV is the in-tree
runtime fallback, so inference does not depend on a system FFmpeg executable or a particular
TorchCodec wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def decode_frames(
    video_bytes: bytes,
    frame_indices: list[int],
    frame_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Decode the given frame indices into a uint8 tensor of shape (T, C, H, W).

    If `frame_size=(H, W)` is given, frames are resampled to it with antialiased bilinear
    interpolation (typically a downscale). The source is decoded full-res first, then resized.
    """
    if not frame_indices:
        import torch

        return torch.empty((0, 3, *(frame_size or (0, 0))), dtype=torch.uint8)

    try:
        from torchcodec.decoders import VideoDecoder  # pyright: ignore[reportPrivateImportUsage]

        decoder = VideoDecoder(video_bytes, device="cpu")
        frames = decoder.get_frames_at(frame_indices).data  # (T, C, H, W) uint8
    except Exception as torchcodec_error:  # noqa: BLE001 - native ABI/load errors require fallback.
        frames = _decode_frames_pyav(video_bytes, frame_indices, torchcodec_error=torchcodec_error)
    if frame_size is not None:
        frames = _resize(frames, frame_size)
    return frames


def _decode_frames_pyav(
    video_bytes: bytes,
    frame_indices: list[int],
    *,
    torchcodec_error: Exception,
) -> torch.Tensor:
    """Decode selected RGB frames with PyAV while preserving caller order and duplicates."""

    import io

    import torch

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "MIRA video decoding requires either a working TorchCodec native library or PyAV; "
            f"TorchCodec failed with: {torchcodec_error}"
        ) from exc

    requested = set(int(index) for index in frame_indices)
    if min(requested) < 0:
        raise ValueError(f"frame indices must be non-negative, got {frame_indices}")
    decoded: dict[int, torch.Tensor] = {}
    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                array = frame.to_ndarray(format="rgb24")
                decoded[index] = torch.from_numpy(array).permute(2, 0, 1).contiguous()
                if len(decoded) == len(requested):
                    break

    missing = sorted(requested.difference(decoded))
    if missing:
        raise IndexError(f"video ended before requested MIRA frame indices were decoded: {missing}")
    return torch.stack([decoded[int(index)] for index in frame_indices], dim=0).to(torch.uint8)


def _resize(frames: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    """Resample (T, C, H, W) uint8 frames to `frame_size=(H, W)`, returning uint8.

    No-op when already at the target size. Antialiased bilinear interpolation runs in float;
    the result is rounded and clamped back into the valid uint8 range (antialiasing can ring
    slightly past [0, 255]).
    """
    import torch
    import torch.nn.functional as F

    if tuple(frames.shape[-2:]) == tuple(frame_size):
        return frames
    resized = F.interpolate(
        frames.float(), size=frame_size, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.round().clamp_(0, 255).to(torch.uint8)
