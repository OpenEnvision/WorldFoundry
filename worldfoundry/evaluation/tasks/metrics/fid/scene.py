"""SceneFID protocol: FID on object crops (OC-GAN style)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from worldfoundry.evaluation.tasks.metrics._shared.bbox import bbox_xyxy
from worldfoundry.evaluation.tasks.metrics.fid.compute import compute_fid

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def _load_bboxes(path: Path) -> dict[str, list[list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bboxes JSON must be an object mapping image keys to bbox lists")
    normalized: dict[str, list[list[float]]] = {}
    for key, boxes in payload.items():
        if not isinstance(boxes, list):
            raise ValueError(f"bbox list for {key!r} must be a list")
        normalized[str(key)] = [list(map(float, box)) for box in boxes]
    return normalized


def _resolve_image_path(image_root: Path, key: str) -> Path:
    candidate = image_root / key
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"could not resolve image for bbox key {key!r} under {image_root}")


def _save_crop(image: Image.Image, box: Sequence[float], dest: Path, *, min_crop_size: int) -> bool:
    width, height = image.size
    x1, y1, x2, y2 = bbox_xyxy(box)
    left = max(0, min(width, int(round(min(x1, x2)))))
    top = max(0, min(height, int(round(min(y1, y2)))))
    right = max(0, min(width, int(round(max(x1, x2)))))
    bottom = max(0, min(height, int(round(max(y1, y2)))))
    if right - left < min_crop_size or bottom - top < min_crop_size:
        return False
    crop = image.crop((left, top, right, bottom))
    crop.save(dest)
    return True


def extract_object_crops(
    image_root: str | Path,
    bboxes_json: str | Path,
    output_dir: str | Path,
    *,
    min_crop_size: int = 32,
) -> Path:
    """Extract crops using ``(x1, y1, x2, y2)`` or ``(x, y, w, h, extra)`` boxes."""
    return _extract_object_crops(
        Path(image_root), _load_bboxes(Path(bboxes_json)), Path(output_dir), min_crop_size=min_crop_size
    )


def _extract_object_crops(
    image_root: Path,
    bboxes: dict[str, list[list[float]]],
    output: Path,
    *,
    min_crop_size: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    crop_index = 0
    for key, boxes in bboxes.items():
        image_path = _resolve_image_path(image_root, key)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            for box in boxes:
                dest = output / f"crop_{crop_index:06d}.png"
                if _save_crop(rgb, box, dest, min_crop_size=min_crop_size):
                    crop_index += 1
    if crop_index == 0:
        raise ValueError(f"no object crops extracted from {image_root}")
    return output


def _resolve_crop_dir(
    image_root: Path,
    *,
    crop_dir: str | Path | None,
    bboxes_json: str | Path | None,
    scratch_root: Path,
    min_crop_size: int,
    bbox_source_root: Path | None = None,
) -> Path:
    if crop_dir is not None:
        path = Path(crop_dir)
        if not path.is_dir() or not _iter_images(path):
            raise ValueError(f"crop directory has no images: {path}")
        return path
    if bboxes_json is None:
        raise ValueError(
            "SceneFID requires pre-extracted crop directories or a bboxes JSON manifest "
            "(object crop + FID protocol)."
        )
    bboxes = _load_bboxes(Path(bboxes_json))
    if bbox_source_root is not None:
        bboxes = {
            str(Path(key).relative_to(bbox_source_root)) if Path(key).is_absolute() else key: boxes
            for key, boxes in bboxes.items()
        }
    return _extract_object_crops(image_root, bboxes, scratch_root, min_crop_size=min_crop_size)


def compute_scene_fid(
    reference: str | Path,
    generated: str | Path,
    *,
    reference_crops: str | Path | None = None,
    generated_crops: str | Path | None = None,
    reference_bboxes_json: str | Path | None = None,
    generated_bboxes_json: str | Path | None = None,
    min_crop_size: int = 32,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> float:
    """Compute crop FID, matching shared manifest keys relative to each image root."""
    ref_root = Path(reference).absolute()
    gen_root = Path(generated).absolute()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if reference_crops is None or generated_crops is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="worldfoundry-scene-fid-")
            scratch = Path(temp_dir.name)
            ref_crops = _resolve_crop_dir(
                ref_root,
                crop_dir=reference_crops,
                bboxes_json=reference_bboxes_json,
                scratch_root=scratch / "reference_crops",
                min_crop_size=min_crop_size,
            )
            gen_crops = _resolve_crop_dir(
                gen_root,
                crop_dir=generated_crops,
                bboxes_json=generated_bboxes_json or reference_bboxes_json,
                scratch_root=scratch / "generated_crops",
                min_crop_size=min_crop_size,
                bbox_source_root=ref_root if generated_bboxes_json is None else None,
            )
        else:
            ref_crops = _resolve_crop_dir(
                ref_root,
                crop_dir=reference_crops,
                bboxes_json=None,
                scratch_root=Path(reference_crops),
                min_crop_size=min_crop_size,
            )
            gen_crops = _resolve_crop_dir(
                gen_root,
                crop_dir=generated_crops,
                bboxes_json=None,
                scratch_root=Path(generated_crops),
                min_crop_size=min_crop_size,
            )
        return compute_fid(
            ref_crops,
            gen_crops,
            batch_size=batch_size,
            cuda=cuda,
            feature_extractor=feature_extractor,
            **kwargs,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


__all__ = ["compute_scene_fid", "extract_object_crops"]
