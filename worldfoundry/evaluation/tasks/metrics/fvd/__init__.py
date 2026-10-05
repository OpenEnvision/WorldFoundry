"""Fréchet Video Distance (FVD) metric."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

METRIC_ID = "fvd"
ALIASES = ("frechet-video-distance",)
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "video_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Fréchet Video Distance between reference and generated video sets.",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


@lru_cache(maxsize=1)
def _fvd_core() -> Any:
    from worldfoundry.evaluation.tasks.metrics.fvd import fvd_core as _module

    return _module


def compute_fvd(*args: Any, **kwargs: Any) -> float:
    return _fvd_core().compute_fvd(*args, **kwargs)


def compute_fvd_from_numpy(
    real_videos: np.ndarray,
    generated_videos: np.ndarray,
    *,
    device: str = "cuda",
    i3d_checkpoint: str | Path | None = None,
    batch_size: int = 8,
) -> float:
    return compute_fvd(
        real_videos,
        generated_videos,
        device=device,
        i3d_checkpoint=i3d_checkpoint,
        batch_size=batch_size,
    )


def compute_fvd_from_frame_dirs(
    reference_frame_dirs: Sequence[str | Path],
    generated_frame_dirs: Sequence[str | Path],
    *,
    device: str = "cuda",
    i3d_checkpoint: str | Path | None = None,
    max_frames: int = 16,
) -> float:
    import numpy as np
    from PIL import Image

    def _load_video(frames_dir: str | Path, limit: int) -> np.ndarray:
        paths = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        if len(paths) > limit:
            indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
            paths = [paths[i] for i in indices]
        frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
        return np.stack(frames, axis=0)[None, ...]

    real = np.concatenate([_load_video(path, max_frames) for path in reference_frame_dirs], axis=0)
    gen = np.concatenate([_load_video(path, max_frames) for path in generated_frame_dirs], axis=0)
    return compute_fvd_from_numpy(real, gen, device=device, i3d_checkpoint=i3d_checkpoint)


compute = compute_fvd_from_numpy

__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_fvd",
    "compute_fvd_from_frame_dirs",
    "compute_fvd_from_numpy",
]
