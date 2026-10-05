from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


try:
    from torchvision.io import write_video as _torchvision_write_video
except Exception:  # pragma: no cover - depends on torchvision build features.
    _torchvision_write_video = None


def _to_numpy(video_array: Any) -> np.ndarray:
    if isinstance(video_array, torch.Tensor):
        array = video_array.detach().cpu().numpy()
    else:
        array = np.asarray(video_array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _write_video_imageio(
    filename: str | Path,
    video_array: Any,
    *,
    fps: int | float,
    video_codec: str | None,
    options: dict[str, str] | None,
) -> None:
    """Write THWC frames without going through torchvision's PyAV shim."""
    import imageio.v2 as imageio

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_params = [item for key, value in (options or {}).items() for item in (f"-{key}", str(value))]
    writer_kwargs: dict[str, Any] = {
        "fps": float(fps),
        "macro_block_size": 1,
    }
    if video_codec:
        writer_kwargs["codec"] = video_codec
    if ffmpeg_params:
        writer_kwargs["ffmpeg_params"] = ffmpeg_params
    with imageio.get_writer(str(output_path), **writer_kwargs) as writer:
        for frame in _to_numpy(video_array):
            writer.append_data(frame)


def write_video(
    filename: str | Path,
    video_array: Any,
    fps: int | float,
    video_codec: str | None = None,
    options: dict[str, str] | None = None,
) -> None:
    """Torchvision-compatible video writer with an imageio fallback.

    Recent PyAV releases reject torchvision's string ``pict_type`` assignment
    with ``TypeError: an integer is required``.  Treat that as an encoder
    compatibility failure and retry through imageio/FFmpeg.
    """
    if _torchvision_write_video is not None:
        try:
            _torchvision_write_video(
                str(filename),
                video_array,
                fps=fps,
                video_codec=video_codec,
                options=options,
            )
            return
        except TypeError:
            Path(filename).unlink(missing_ok=True)

    _write_video_imageio(
        filename,
        video_array,
        fps=fps,
        video_codec=video_codec,
        options=options,
    )
