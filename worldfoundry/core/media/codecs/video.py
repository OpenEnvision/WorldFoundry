"""Video read/write primitives shared by inference and evaluation code.

Decode and encode stay here so model pipelines do not import cv2/decord
directly. FFmpeg is resolved as explicit path → ImageIO bundle → ``PATH``
by default. Set ``WORLDFOUNDRY_FFMPEG_PREFERENCE=system`` when a deployment
intentionally wants its system binary first.

Why not always decode the whole file: I2V / eval often need a few frames
or a resized tensor. :func:`sample_video_frames` and
:func:`load_frames_from_video` take that path; tiling for huge inputs
lives in :mod:`worldfoundry.core.media.processing.video_tiling`. Writes go through
H.264 helpers so evaluation artifacts share one color-range contract.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from worldfoundry.core.io.filesystem.storage import (
    local_path_for_uri,
    parse_uri_scheme,
    uri_to_local_path,
    write_binary_uri,
)
from worldfoundry.core.io.paths import scratch_directory
from worldfoundry.core.media.types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from worldfoundry.core.observability.nvtx import nvtx_range

# ──────────────────────────────────────────────────────────────────────────
# FFmpeg / decode — explicit path → ImageIO bundle → PATH; never import cv2 here
# ──────────────────────────────────────────────────────────────────────────

_MAX_FFMPEG_STDERR_BYTES = 1024 * 1024
_FFMPEG_STDERR_TRUNCATED_PREFIX = b"[earlier FFmpeg stderr truncated]\n"


def _imageio_ffmpeg_executable() -> str | None:
    """Return ImageIO's pinned FFmpeg binary when that optional dep exists."""

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        return None


def _resolve_ffmpeg_executable(
    explicit: str | Path | None = None,
    *,
    preference: str | None = None,
) -> str | None:
    """Resolve FFmpeg with a deterministic, overrideable priority.

    The bundled executable is the default because its encoder build is pinned
    with ImageIO and benchmarks substantially faster than several older PATH
    installations. ``preference="system"`` (or the matching environment
    variable) reverses the last two candidates without removing the portable
    fallback.
    """

    if explicit is not None:
        return str(explicit)
    requested = str(
        preference
        if preference is not None
        else os.getenv("WORLDFOUNDRY_FFMPEG_PREFERENCE", "bundled")
    ).strip().casefold()
    aliases = {
        "bundle": "bundled",
        "bundled": "bundled",
        "imageio": "bundled",
        "path": "system",
        "system": "system",
    }
    try:
        resolved_preference = aliases[requested]
    except KeyError as error:
        raise ValueError(
            "WORLDFOUNDRY_FFMPEG_PREFERENCE must be 'bundled' or 'system'"
        ) from error

    if resolved_preference == "system":
        return shutil.which("ffmpeg") or _imageio_ffmpeg_executable()
    return _imageio_ffmpeg_executable() or shutil.which("ffmpeg")


def extract_frames_from_video_url(video_url: str):
    """Decode a remote video URL and return RGB PIL frames."""

    from PIL import Image

    frames = load_video_frames(video_url)
    return [Image.fromarray(frame[..., :3]) for frame in frames]


def resize_video_tensor_to_resolution(video_tensor, resolution: Iterable[int]):
    """Resize and center-crop a ``[T,C,H,W]`` tensor to ``(target_h, target_w)``."""

    import torchvision

    target_h, target_w = [int(value) for value in resolution]
    orig_h, orig_w = int(video_tensor.shape[2]), int(video_tensor.shape[3])
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    resized = torchvision.transforms.functional.resize(video_tensor, resizing_shape)
    return torchvision.transforms.functional.center_crop(resized, [target_h, target_w])


def read_image_as_video_tensor(
    image_path: str | Path,
    resolution: Iterable[int],
    num_video_frames: int,
    *,
    resize: bool = True,
):
    """Load an image and materialize a ``[1,C,T,H,W]`` uint8 conditioning video tensor."""

    import torch
    import torchvision
    from PIL import Image

    path = Path(image_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {path.suffix}")
    if num_video_frames < 1:
        raise ValueError("num_video_frames must be at least 1")

    image = Image.open(path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    first_frame = image_tensor.unsqueeze(0)
    zero_tail = torch.zeros_like(first_frame).repeat(num_video_frames - 1, 1, 1, 1)
    video = torch.cat([first_frame, zero_tail], dim=0)
    video = (video * 255.0).to(torch.uint8)
    if resize:
        video = resize_video_tensor_to_resolution(video, resolution)
    return video.unsqueeze(0).permute(0, 2, 1, 3, 4)


def video_tensor_to_uint8_frames(
    video_tensor,
    *,
    value_range: str | tuple[float, float] = "auto",
    max_frames: int | None = None,
    from_end: bool = False,
) -> "object":
    """Convert a normalized torch video tensor to uint8 THWC frames.

    ``value_range="auto"`` treats tensors with negative values as ``[-1, 1]``
    and non-negative floating tensors as ``[0, 1]``.
    """

    import torch

    if not torch.is_tensor(video_tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(video_tensor)}")
    max_frames = _validate_frame_limit(max_frames, from_end)
    # Normalize on the source device, then transfer the compact uint8 result.
    # Moving a decoded CUDA video to CPU first used to copy four bytes per
    # channel and run clamp/scale on the host.  The final artifact only needs
    # one byte per channel, so doing the numerically identical FP32 work on the
    # accelerator cuts both PCIe traffic and public-generation latency.
    tensor = video_tensor.detach()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 5D video tensor, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor shape [C,T,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3, 4}:
        video = tensor
    elif tensor.shape[0] in {1, 3, 4}:
        video = tensor.permute(1, 2, 3, 0)
    elif tensor.shape[1] in {1, 3, 4}:
        video = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unable to infer channel layout for video tensor shape {tuple(tensor.shape)}")
    video = video.float()
    if value_range == "auto":
        low, high = (-1.0, 1.0) if float(video.min()) < 0.0 else (0.0, 1.0)
    elif value_range == "-1,1":
        low, high = -1.0, 1.0
    elif value_range == "0,1":
        low, high = 0.0, 1.0
    else:
        low, high = value_range
    if high <= low:
        raise ValueError("value_range high bound must be larger than low bound.")
    # Range inference must see the original input, but only selected frames
    # need conversion and the CUDA-to-host copy.
    if max_frames is not None:
        video = video[-max_frames:] if from_end else video[:max_frames]
    video = (
        (video.clamp(float(low), float(high)) - float(low))
        * (255.0 / (float(high) - float(low)))
    ).to(torch.uint8)
    return video.contiguous().cpu().numpy()


def _validate_frame_limit(max_frames: int | None, from_end: bool) -> int | None:
    import operator

    if not isinstance(from_end, bool):
        raise TypeError("from_end must be a bool")
    if max_frames is None:
        if from_end:
            raise ValueError("from_end requires max_frames")
        return None
    if isinstance(max_frames, bool):
        raise TypeError("max_frames must be an integer")
    max_frames = operator.index(max_frames)
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    return max_frames


def coerce_video_frames(video_input, *, max_frames: int | None = None, from_end: bool = False):
    """Normalize video inputs to uint8 THWC, optionally keeping a prefix/tail.

    File inputs use the same ImageIO decoder as the full reader. Prefix reads
    stop early; tail reads retain only the requested frame count while reading
    to EOF. Array/tensor inputs retain whole-input range detection before
    slicing, so truncation does not change automatic value normalization.
    """

    import os

    import numpy as np
    import torch
    from PIL import Image

    max_frames = _validate_frame_limit(max_frames, from_end)
    if isinstance(video_input, (str, os.PathLike)):
        if max_frames is None:
            return load_video_frames(str(video_input))
        return load_video_frames(str(video_input), max_frames=max_frames, from_end=from_end)

    if torch.is_tensor(video_input):
        if max_frames is None:
            return video_tensor_to_uint8_frames(video_input)
        return video_tensor_to_uint8_frames(video_input, max_frames=max_frames, from_end=from_end)

    if isinstance(video_input, np.ndarray):
        video = video_input
    elif isinstance(video_input, (list, tuple)):
        frames = []
        for frame in video_input:
            if torch.is_tensor(frame):
                tensor = frame.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                frame = tensor.numpy()
            elif isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
            frames.append(frame)
        video = np.stack(frames, axis=0)
    else:
        raise TypeError(f"Unsupported video input type: {type(video_input)}")

    if video.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {tuple(video.shape)}")

    if video.shape[-1] in {1, 3, 4}:
        converted = video
    elif video.shape[1] in {1, 3, 4}:
        converted = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unable to infer channel layout for video shape {tuple(video.shape)}")

    if converted.dtype != np.uint8:
        normalized_range = np.issubdtype(converted.dtype, np.floating) and converted.max() <= 1.0
        if max_frames is not None:
            converted = converted[-max_frames:] if from_end else converted[:max_frames]
        if np.issubdtype(converted.dtype, np.floating):
            if normalized_range:
                converted = np.clip(converted * 255.0, 0, 255)
            else:
                converted = np.clip(converted, 0, 255)
        else:
            converted = np.clip(converted, 0, 255)
        return converted.astype(np.uint8)
    if max_frames is None:
        return converted
    return converted[-max_frames:] if from_end else converted[:max_frames]


def load_video_frames(video_path: str | Path, *, max_frames: int | None = None, from_end: bool = False):
    """Decode a video, optionally bounding its prefix/tail without changing RGB conversion."""

    import imageio
    import numpy as np

    max_frames = _validate_frame_limit(max_frames, from_end)
    with local_path_for_uri(video_path) as local_path:
        if max_frames is None:
            frames = imageio.mimread(str(local_path), memtest=False)
        else:
            from collections import deque

            frames = deque(maxlen=max_frames)
            reader = imageio.get_reader(str(local_path), mode="I")
            try:
                for frame in reader:
                    # Some readers reuse a frame buffer on their next read.
                    frames.append(np.array(frame, copy=True))
                    if not from_end and len(frames) == max_frames:
                        break
            finally:
                reader.close()
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    arrays = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        arrays.append(array)
    return np.stack(arrays, axis=0)


def read_video_window_rgb(
    video_path: str | Path,
    *,
    start_frame: int = 0,
    frame_count: int,
):
    """Decode up to ``frame_count`` RGB frames without decoding the tail.

    Frame offsets are decoded sequentially from the beginning, preserving
    PyAV/torchvision frame ordering and RGB conversion, including variable
    frame-rate inputs. A window that reaches EOF returns its available frames;
    an empty window raises ``ValueError``. The decoder and any temporary local
    copy of a remote input are closed even when decoding fails.
    """

    import operator

    import av
    import numpy as np

    if isinstance(start_frame, bool) or isinstance(frame_count, bool):
        raise TypeError("start_frame and frame_count must be integers")
    start_frame = operator.index(start_frame)
    frame_count = operator.index(frame_count)
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    frames = []
    with local_path_for_uri(video_path) as local_path:
        with av.open(str(local_path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index < start_frame:
                    continue
                frames.append(frame.to_rgb().to_ndarray())
                if len(frames) == frame_count:
                    break
    if not frames:
        raise ValueError(f"No frames found in video window starting at {start_frame}: {video_path}")
    return np.stack(frames, axis=0)


def get_video_details(video_path: str | Path) -> tuple[int, float, float]:
    """Return ``(total_frames, fps, duration_seconds)`` for a local video."""

    from decord import VideoReader, cpu

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    if path.stat().st_size < 1024:
        raise ValueError(f"Video too short: {path}")
    reader = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
    total_frames = len(reader)
    original_fps = float(reader.get_avg_fps())
    return total_frames, original_fps, total_frames / original_fps


def sample_video_frames(
    video_path: str | Path,
    max_frames: int,
    fps: float = 1.0,
    force_sample: bool = False,
    *,
    num_threads: int = 2,
) -> tuple["object", str, float]:
    """Decode uniformly bounded RGB frames and their correct source timestamps.

    ``fps`` is the requested temporal sampling rate when ``force_sample`` is
    false. Forced sampling returns exactly ``max_frames`` entries (including
    repeated indices for very short clips), matching common video-LLM input
    contracts. The returned tuple is ``(THWC uint8 frames, timestamp text,
    duration_seconds)``.
    """

    import numpy as np
    from decord import VideoReader, cpu

    max_frames = int(max_frames)
    requested_fps = float(fps)
    num_threads = int(num_threads)
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if not math.isfinite(requested_fps) or requested_fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=num_threads)
    total_frames = len(reader)
    source_fps = float(reader.get_avg_fps())
    if total_frames < 1:
        raise ValueError(f"No frames found in video: {video_path}")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source frame rate {source_fps!r} for video: {video_path}")

    if force_sample:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int64)
    else:
        frame_step = max(int(round(source_fps / requested_fps)), 1)
        indices = np.arange(0, total_frames, frame_step, dtype=np.int64)
        if len(indices) > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int64)

    timestamps = [float(index) / source_fps for index in indices]
    timestamp_text = ",".join(f"{timestamp:.2f}s" for timestamp in timestamps)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    return frames, timestamp_text, total_frames / source_fps


def list_numbered_frame_paths(
    frame_dir: str | Path,
    *,
    prefix: str = "frames_",
    suffix: str = ".png",
) -> tuple[Path, ...]:
    """List numerically named frame files without counting unrelated directory entries."""

    root = Path(frame_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    if not suffix.startswith("."):
        raise ValueError("suffix must be a non-empty extension beginning with '.'")
    normalized_suffix = suffix.lower()
    indexed: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_file() or not path.name.startswith(prefix) or path.suffix.lower() != normalized_suffix:
            continue
        index_text = path.name[len(prefix) : -len(path.suffix)]
        try:
            frame_index = int(index_text)
        except ValueError:
            continue
        indexed.append((frame_index, path))
    return tuple(path for _, path in sorted(indexed, key=lambda item: (item[0], item[1].name)))


def extract_video_frames_to_directory(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "frames_",
    suffix: str = ".png",
    ffmpeg_path: str | Path | None = None,
    threads: int = 1,
    timeout_seconds: float | None = None,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Extract every source frame with bounded, deterministic ffmpeg settings."""

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video path not found: {source}")
    if not prefix:
        raise ValueError("prefix must not be empty")
    if not suffix.startswith("."):
        raise ValueError("suffix must be a non-empty extension beginning with '.'")
    if int(threads) < 1:
        raise ValueError("threads must be positive")
    if timeout_seconds is not None and float(timeout_seconds) <= 0:
        raise ValueError("timeout_seconds must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = list_numbered_frame_paths(root, prefix=prefix, suffix=suffix)
    if existing and not overwrite:
        return existing
    for path in existing:
        path.unlink()
    executable = _resolve_ffmpeg_executable(ffmpeg_path)
    if not executable:
        raise FileNotFoundError("ffmpeg was not found in PATH and no ImageIO-bundled executable is available")

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(int(threads)),
        "-i",
        str(source),
        "-vsync",
        "0",
        "-start_number",
        "1",
        "-y",
        str(root / f"{prefix}%d{suffix}"),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None if timeout_seconds is None else float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        for path in list_numbered_frame_paths(root, prefix=prefix, suffix=suffix):
            path.unlink()
        raise TimeoutError(f"ffmpeg frame extraction timed out after {timeout_seconds} seconds") from exc
    if completed.returncode != 0:
        for path in list_numbered_frame_paths(root, prefix=prefix, suffix=suffix):
            path.unlink()
        detail = "\n".join((completed.stderr or "").splitlines()[-20:])
        raise RuntimeError(f"ffmpeg frame extraction failed with code {completed.returncode}: {detail}")
    extracted = list_numbered_frame_paths(root, prefix=prefix, suffix=suffix)
    if not extracted:
        raise ValueError(f"ffmpeg produced no frames for video: {source}")
    return extracted


def _ffprobe_number(value: Any) -> float | None:
    """Parse an ffprobe numeric field; ``N/A`` / empty become ``None``, not 0."""

    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ffprobe_rate(value: Any) -> float | None:
    """Parse ``num/den`` frame rates; ``0/0`` is treated as missing, not infinity."""

    if value in (None, "", "0/0", "N/A"):
        return None
    text = str(value)
    if "/" not in text:
        return _ffprobe_number(text)
    numerator, denominator = text.split("/", 1)
    numerator_value = _ffprobe_number(numerator)
    denominator_value = _ffprobe_number(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return numerator_value / denominator_value


def probe_video_metadata(
    video_path: str | Path,
    *,
    ffprobe_path: str | Path | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, int | float | None]:
    """Read lightweight video metadata with one bounded ``ffprobe`` process."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video path not found: {path}")
    executable = str(ffprobe_path) if ffprobe_path is not None else shutil.which("ffprobe")
    if not executable:
        raise FileNotFoundError("ffprobe was not found in PATH")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"ffprobe timed out after {timeout_seconds:g}s for {path}") from exc
    if completed.returncode != 0:
        reason = completed.stderr.strip() or f"ffprobe exited with status {completed.returncode}"
        raise ValueError(reason)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    format_payload = payload.get("format")
    format_metadata = format_payload if isinstance(format_payload, dict) else {}
    fps = _ffprobe_rate(stream.get("avg_frame_rate")) or _ffprobe_rate(stream.get("r_frame_rate"))
    duration = _ffprobe_number(stream.get("duration"))
    if duration is None:
        duration = _ffprobe_number(format_metadata.get("duration"))
    frame_count_value = _ffprobe_number(stream.get("nb_frames"))
    frame_count = None if frame_count_value is None else int(frame_count_value)
    width = _ffprobe_number(stream.get("width"))
    height = _ffprobe_number(stream.get("height"))
    return {
        "width": None if width is None else int(width),
        "height": None if height is None else int(height),
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
    }


def load_frames_from_video(
    video_path: str | Path,
    indices: Iterable[int],
    video_decode_backend: str = "decord",
    eval_: bool = True,
):
    """Load selected RGB frames into a torch tensor using decord or OpenCV."""

    import os

    import cv2
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    path = str(video_path)
    frame_indices = [int(index) for index in indices]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".gif", ".webm"} or video_decode_backend == "opencv":
        capture = cv2.VideoCapture(path)
        frames: dict[int, np.ndarray] = {}
        max_index = max(frame_indices)
        frame_id = 0
        ok = True
        while ok and frame_id <= max_index:
            ok, frame = capture.read()
            if ok and frame_id in frame_indices:
                frames[frame_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
        capture.release()
        return torch.tensor(np.stack([frames[index] for index in frame_indices if index in frames]))

    reader = VideoReader(path) if eval_ else VideoReader(path, num_threads=1, ctx=cpu(0))
    batch = reader.get_batch(frame_indices)
    if isinstance(batch, torch.Tensor):
        return batch
    return torch.tensor(batch.asnumpy())


def read_video(video_path: str | Path, *, return_metadata: bool = True):
    """Decode a video into frames and optional metadata."""

    import imageio
    import numpy as np

    with local_path_for_uri(video_path) as local_path:
        reader = imageio.get_reader(str(local_path))
        frames = [np.asarray(frame) for frame in reader]
        metadata = reader.get_meta_data()
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    stacked = np.stack(frames, axis=0)
    return (stacked, metadata) if return_metadata else stacked


def save_video_frames(video_frames, output_path: str | Path, fps: int = 16, **kwargs) -> None:
    """Write a THWC uint8 frame array/list to a video path or URI."""

    write_video(video_frames, output_path, fps=fps, **kwargs)


def _drain_ffmpeg_stderr(
    stream: Any,
    result: list[bytes],
    *,
    max_bytes: int = _MAX_FFMPEG_STDERR_BYTES,
) -> None:
    """Drain FFmpeg stderr concurrently and retain a bounded diagnostic tail."""

    tail = bytearray()
    truncated = False
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        encoded = chunk.encode() if isinstance(chunk, str) else bytes(chunk)
        tail.extend(encoded)
        excess = len(tail) - max_bytes
        if excess > 0:
            del tail[:excess]
            truncated = True

    prefix = _FFMPEG_STDERR_TRUNCATED_PREFIX if truncated else b""
    result.append(prefix + bytes(tail))


# ──────────────────────────────────────────────────────────────────────────
# H.264 writers — one color-range contract for eval artifacts
# ──────────────────────────────────────────────────────────────────────────


@nvtx_range("worldfoundry.output.h264")
def save_video_h264(
    video_frames,
    output_path: str | Path,
    *,
    fps: float = 16.0,
    crf: int = 18,
    preset: str = "medium",
    faststart: bool = False,
) -> None:
    """Write THWC RGB frames as a local H.264/yuv420p MP4 with FFmpeg.

    ``faststart`` is opt-in because relocating MP4 metadata performs another
    file pass after encoding.  Local generation results are already complete
    when this function returns, so the extra pass adds latency without
    changing the encoded frames or their quality.
    """

    frames = coerce_video_frames(video_frames)
    if int(frames.shape[0]) == 0:
        return
    if parse_uri_scheme(output_path) != "file":
        raise ValueError("save_video_h264 currently supports local output paths only")

    target = uri_to_local_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_executable()
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH and no ImageIO-bundled executable is available; "
            "cannot encode H.264 output video"
        )

    frame_count, height, width, _ = frames.shape
    command = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
    ]
    if faststart:
        command.extend(("-movflags", "+faststart"))
    command.append(str(target))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=_drain_ffmpeg_stderr,
        args=(process.stderr, stderr_chunks),
        name="worldfoundry-ffmpeg-stderr",
        daemon=True,
    )
    stderr_thread.start()
    write_error: BrokenPipeError | None = None
    try:
        try:
            for index in range(frame_count):
                process.stdin.write(frames[index].tobytes())
        except BrokenPipeError as exc:
            write_error = exc
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError as exc:
                if write_error is None:
                    write_error = exc
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        stderr_thread.join()
        raise
    stderr_thread.join()
    message = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if write_error is not None:
        raise RuntimeError(
            f"ffmpeg closed while encoding {target} (exit code {returncode}): {message}"
        ) from write_error
    if returncode != 0:
        raise RuntimeError(f"ffmpeg H.264 encode failed for {target}: {message}")


def save_image_or_video_tensor(
    tensor,
    save_path,
    *,
    fps: int = 24,
    quality: int | None = None,
    ffmpeg_params: list[str] | None = None,
    value_range: str | tuple[float, float] = "auto",
    image_format: str = "JPEG",
    video_format: str = "mp4",
    **kwargs,
) -> str | None:
    """Save a normalized ``[C,T,H,W]`` or ``[B,C,T,H,W]`` tensor as image/video.

    A single-frame tensor is saved as an image; multi-frame tensors are saved as
    videos. Local paths and URI-like targets supported by ``worldfoundry.core.io``
    storage helpers are both accepted.
    """

    from io import BytesIO

    from PIL import Image

    frames = video_tensor_to_uint8_frames(tensor, value_range=value_range)
    target = save_path
    is_file_obj = hasattr(target, "write")

    if frames.shape[0] == 1:
        image = Image.fromarray(frames[0][..., :3]).convert("RGB")
        if is_file_obj:
            image.save(target, format=image_format, quality=quality or 85, **kwargs)
            return None

        path_text = str(target)
        if not Path(path_text).suffix:
            path_text = f"{path_text}.jpg"
        if parse_uri_scheme(path_text) == "file":
            output_path = uri_to_local_path(path_text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format=image_format, quality=quality or 85, **kwargs)
        else:
            buffer = BytesIO()
            image.save(buffer, format=image_format, quality=quality or 85, **kwargs)
            write_binary_uri(path_text, buffer)
        return path_text

    if is_file_obj:
        suffix = f".{video_format.lstrip('.')}"
        video_kwargs = dict(kwargs)
        if ffmpeg_params is not None:
            video_kwargs["ffmpeg_params"] = ffmpeg_params
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            write_video(
                frames,
                handle.name,
                fps=fps,
                quality=quality,
                format=video_format,
                **video_kwargs,
            )
            handle.seek(0)
            target.write(handle.read())
        return None

    path_text = str(target)
    if not Path(path_text).suffix:
        path_text = f"{path_text}.mp4"
    video_kwargs = dict(kwargs)
    if ffmpeg_params is not None:
        video_kwargs["ffmpeg_params"] = ffmpeg_params
    write_video(
        frames,
        path_text,
        fps=fps,
        quality=quality,
        format=video_format,
        **video_kwargs,
    )
    return path_text


def write_video_torchvision(
    filename: str | Path,
    video_array: Any,
    fps: float,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Write RGB video frames with a ``torchvision.io.write_video``-compatible signature."""

    try:
        from torchvision.io import write_video as torchvision_write_video
    except (ImportError, AttributeError, RuntimeError):
        torchvision_write_video = None

    if torchvision_write_video is not None:
        try:
            torchvision_write_video(str(filename), video_array, fps, *args, **kwargs)
            return
        except TypeError as exc:
            # torchvision<=0.20 assigns the string ``"NONE"`` to
            # ``VideoFrame.pict_type``.  Newer PyAV releases require the enum's
            # integer value instead, so encoding fails after inference has
            # already completed.  Preserve the torchvision-compatible entry
            # point while using the in-tree FFmpeg encoder for this version
            # mismatch.
            if "integer is required" not in str(exc):
                raise

            options = kwargs.get("options")
            try:
                crf = int(options.get("crf", 18)) if isinstance(options, dict) else 18
            except (TypeError, ValueError):
                crf = 18
            # Match torchvision's input contract exactly here: write_video
            # casts THWC inputs directly to uint8.  coerce_video_frames()
            # treats non-negative torch tensors as normalized [0, 1] data,
            # which turns the common float [0, 255] inputs used by official
            # runtimes into almost entirely white frames.
            import torch

            frames = torch.as_tensor(video_array, dtype=torch.uint8).detach().cpu().numpy()
            save_video_h264(frames, filename, fps=fps, crf=crf)
            return

    import cv2
    import numpy as np

    frames = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else np.asarray(video_array)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("write_video_torchvision expects frames shaped [T, H, W, C]")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_video(
    video_frames,
    output_path: str | Path,
    *,
    fps: int = 16,
    quality: int | None = None,
    format: str | None = None,
    **kwargs,
) -> None:
    """Write a THWC video array/list to a local path or remote URI.

    The ordinary local MP4 path streams RGB bytes straight to FFmpeg.  This
    avoids importing and initializing ImageIO's plugin after inference while
    retaining its effective default x264 CRF (23).  Custom ImageIO options,
    explicit quality settings, non-MP4 containers, and remote targets keep the
    general ImageIO path.
    """

    frames = coerce_video_frames(video_frames)

    if parse_uri_scheme(output_path) == "file":
        target = uri_to_local_path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized_format = None if format is None else str(format).lstrip(".").casefold()
        if (
            target.suffix.casefold() == ".mp4"
            and normalized_format in {None, "mp4"}
            and quality is None
            and not kwargs
        ):
            save_video_h264(frames, target, fps=fps, crf=23)
            return

        import imageio

        write_kwargs = {"fps": fps, "macro_block_size": 1, **kwargs}
        if quality is not None:
            write_kwargs["quality"] = quality
        imageio.mimsave(str(target), frames, format=format, **write_kwargs)
        return

    import imageio

    write_kwargs = {"fps": fps, "macro_block_size": 1, **kwargs}
    if quality is not None:
        write_kwargs["quality"] = quality
    suffix = Path(str(output_path)).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        imageio.mimsave(handle.name, frames, format=format, **write_kwargs)
        handle.seek(0)
        write_binary_uri(output_path, handle.read())


def materialize_video_input(
    video_input,
    output_dir: Optional[str] = None,
    filename: str = "input.mp4",
    fps: int = 24,
) -> str:
    """Return a local video path, materializing in-memory inputs when needed."""

    import os

    if isinstance(video_input, (str, os.PathLike)):
        candidate = uri_to_local_path(video_input) if parse_uri_scheme(video_input) == "file" else None
        if (
            candidate is not None
            and candidate.exists()
            and candidate.is_file()
            and candidate.suffix.lower() in VIDEO_EXTENSIONS
        ):
            return str(candidate.resolve())

    if output_dir is None:
        output_dir = scratch_directory("worldfoundry_video_")
    output_path = Path(output_dir).expanduser().resolve() / filename
    frames = coerce_video_frames(video_input)
    save_video_frames(frames, str(output_path), fps=fps)
    return str(output_path)


def save_videos_grid(videos, path: str, rescale=False, n_rows=6, fps=8):
    """Save a batch of BCTHW video tensors as a single grid video."""

    import os

    import imageio
    import numpy as np
    import torchvision
    from einops import rearrange

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


__all__ = [
    "VIDEO_EXTENSIONS",
    "coerce_video_frames",
    "extract_video_frames_to_directory",
    "extract_frames_from_video_url",
    "get_video_details",
    "list_numbered_frame_paths",
    "load_frames_from_video",
    "load_video_frames",
    "materialize_video_input",
    "probe_video_metadata",
    "read_image_as_video_tensor",
    "read_video",
    "read_video_window_rgb",
    "resize_video_tensor_to_resolution",
    "sample_video_frames",
    "save_image_or_video_tensor",
    "save_video_frames",
    "save_videos_grid",
    "video_tensor_to_uint8_frames",
    "write_video",
    "write_video_torchvision",
]
