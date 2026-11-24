"""Discover persisted geometry assets eligible for Splats versus generic point previews."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def _npz_contains_geometry(path: Path) -> bool:
    """Return whether an NPZ has explicit XYZ or a depth camera bundle."""

    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
    except Exception:
        return False
    point_keys = {"world_points", "points", "xyz", "point_cloud", "pts3d", "point_map"}
    return bool(keys & point_keys) or {"depth", "intrinsics", "extrinsics"} <= keys


def normalize_output_relative(path_text: str, output_dir: str | Path) -> str:
    """Return ``path_text`` trimmed; if absolute under ``output_dir`` make it posix relative."""

    text = Path(path_text).as_posix()
    outp = Path(output_dir).resolve()
    maybe = Path(text)
    if maybe.is_absolute():
        try:
            rel = maybe.resolve().relative_to(outp)
            return rel.as_posix()
        except ValueError:
            return text
    return text.strip().lstrip("./")


def first_geometry_point_candidate(
    artifact_paths: Sequence[str],
    output_dir: str | Path,
    *,
    gs_ply_predicate,
) -> str | None:
    """Pick the first on-disk generic point/depth bundle eligible for Viser."""

    ordered = sorted({str(p) for p in artifact_paths if p})
    for raw in ordered:
        path = Path(raw)
        if not path.exists():
            continue
        suf = path.suffix.lower()
        if suf == ".ply":
            if gs_ply_predicate(path):
                continue
            return normalize_output_relative(str(path.resolve()), output_dir)
        if suf in {".pcd", ".xyz"}:
            return normalize_output_relative(str(path.resolve()), output_dir)
        if suf == ".npz" and _npz_contains_geometry(path):
            return normalize_output_relative(str(path.resolve()), output_dir)
    scanned = sorted(Path(output_dir).glob("**/*.ply"))
    for cand in scanned:
        if cand.is_file() and not gs_ply_predicate(cand):
            return cand.relative_to(Path(output_dir).resolve()).as_posix()
    return None


def first_splat_asset(artifact_paths: Iterable[str], *, gs_ply_predicate=None) -> tuple[str | None, str | None]:
    """Return `(path_text, format_hint)` prioritizing Gaussian-friendly extensions."""

    exts_priority = (
        ".spz",
        ".ksplat",
        ".splat",
        ".sog",
    )
    seen = sorted({Path(p).resolve() for p in artifact_paths if p})
    buckets: dict[str, Path | None] = {ext.strip("."): None for ext in exts_priority}
    gaussian_ply: Path | None = None
    for path in seen:
        if not path.exists() or path.is_dir():
            continue
        suf = path.suffix.lower()
        if suf == ".ply" and gs_ply_predicate is not None and gaussian_ply is None and gs_ply_predicate(path):
            gaussian_ply = path
            continue
        fmt = suf.strip(".") or None
        if fmt in buckets and buckets[fmt] is None:
            buckets[fmt] = path

    ordered_keys = tuple(ext.strip(".") for ext in exts_priority)
    for key in ordered_keys:
        resolved = buckets.get(key)
        if resolved:
            return resolved.as_posix(), key
    if gaussian_ply is not None:
        return gaussian_ply.as_posix(), "ply_gaussian"
    return None, None
