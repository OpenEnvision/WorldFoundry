"""Raw H3 decoding helpers for offline preencoding."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...errors import DataContractError


def _invert_se3(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float32)
    rotation = np.swapaxes(matrices[..., :3, :3], -1, -2)
    output = np.zeros_like(matrices)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation, matrices[..., :3, 3])
    output[..., 3, 3] = 1
    return output


def _load_intrinsics_candidate(
    archive: Any,
    intrinsics_bytes: bytes | None,
) -> np.ndarray | None:
    if intrinsics_bytes:
        loaded = np.load(io.BytesIO(intrinsics_bytes), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                for key in ("camera_K", "K", "intrinsics"):
                    if key in loaded.files:
                        return np.asarray(loaded[key])
                raise DataContractError("H3 intrinsics NPZ requires camera_K, K, or intrinsics")
            finally:
                loaded.close()
        return np.asarray(loaded)
    files = set(archive.files)
    for key in ("camera_K", "K", "intrinsics"):
        if key in files:
            return np.asarray(archive[key])
    if {"fx", "fy"} <= files:
        fx, fy = np.asarray(archive["fx"]), np.asarray(archive["fy"])
        cx = np.asarray(archive["cx"]) if "cx" in files else 0.5
        cy = np.asarray(archive["cy"]) if "cy" in files else 0.5
        return np.stack(np.broadcast_arrays(fx, fy, cx, cy), axis=-1)
    return None


def _normalise_K(
    candidate: np.ndarray | None,
    source_indices: Sequence[int],
    transform: Mapping[str, int],
    *,
    enforce_focal_guard: bool = True,
) -> np.ndarray:
    count = len(source_indices)
    if candidate is None:
        output = np.zeros((count, 3, 3), dtype=np.float32)
        output[:, 0, 0] = 969.6969696969696 / (960.0 * 2.0)
        output[:, 1, 1] = 969.6969696969696 / (540.0 * 2.0)
        output[:, 0, 2] = output[:, 1, 2] = 0.5
        output[:, 2, 2] = 1
        return output
    value = np.asarray(candidate, dtype=np.float64)
    maximum = max(source_indices)
    if value.shape == (3, 3):
        matrices = np.broadcast_to(value, (count, 3, 3)).copy()
    elif value.ndim == 3 and value.shape[1:] == (3, 3):
        if value.shape[0] > maximum:
            matrices = value[np.asarray(source_indices)].copy()
        elif value.shape[0] == 1:
            matrices = np.broadcast_to(value, (count, 3, 3)).copy()
        else:
            raise DataContractError("H3 camera K cannot align to source indices")
    elif value.shape == (4,) or (value.ndim == 2 and value.shape[1] == 4):
        vectors = value.reshape(1, 4) if value.shape == (4,) else value
        if vectors.shape[0] > maximum:
            vectors = vectors[np.asarray(source_indices)]
        elif vectors.shape[0] == 1:
            vectors = np.broadcast_to(vectors, (count, 4))
        else:
            raise DataContractError("H3 camera intrinsics vectors cannot align")
        matrices = np.zeros((count, 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = vectors[:, 0]
        matrices[:, 1, 1] = vectors[:, 1]
        matrices[:, 0, 2] = vectors[:, 2]
        matrices[:, 1, 2] = vectors[:, 3]
        matrices[:, 2, 2] = 1
    else:
        raise DataContractError(f"unsupported H3 camera K shape {value.shape}")
    normalized = bool(
        np.nanmax(np.abs(matrices[:, (0, 1), (0, 1)])) <= 4 and np.nanmax(np.abs(matrices[:, (0, 1), (2, 2)])) <= 4
    )
    if normalized:
        matrices[:, 0, :] *= transform["source_w"]
        matrices[:, 1, :] *= transform["source_h"]
    matrices[:, 0, 0] *= transform["resized_w"] / transform["source_w"]
    matrices[:, 1, 1] *= transform["resized_h"] / transform["source_h"]
    matrices[:, 0, 2] = matrices[:, 0, 2] * transform["resized_w"] / transform["source_w"] - transform["crop_left"]
    matrices[:, 1, 2] = matrices[:, 1, 2] * transform["resized_h"] / transform["source_h"] - transform["crop_top"]
    matrices[:, 0, :] /= transform["target_w"]
    matrices[:, 1, :] /= transform["target_h"]
    matrices[:, 2, :] = (0, 0, 1)
    focal = matrices[:, (0, 1), (0, 1)]
    if not np.isfinite(matrices).all() or (enforce_focal_guard and (np.any(focal <= 0) or np.any(focal > 4))):
        raise DataContractError("H3 camera K exceeds the normalized PRoPE guards")
    return matrices.astype(np.float32)


def decode_camera(
    camera_bytes: bytes,
    source_indices: Sequence[int],
    transform: Mapping[str, int],
    *,
    intrinsics_bytes: bytes | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select absolute C2W and normalized K rows for one exact H3 window."""

    with np.load(io.BytesIO(camera_bytes), allow_pickle=False) as archive:
        files = set(archive.files)
        if "vipe_status" in files:
            status = str(np.asarray(archive["vipe_status"]).reshape(-1)[0]).strip().lower()
            if status != "ok":
                raise DataContractError(f"VIPE camera status is {status!r}")
        if "vipe_c2w" in files:
            c2w_full = np.asarray(archive["vipe_c2w"], dtype=np.float32)
        elif "vipe_w2c" in files:
            c2w_full = _invert_se3(archive["vipe_w2c"])
        elif "camera_c2w" in files or "c2w" in files:
            key = "camera_c2w" if "camera_c2w" in files else "c2w"
            c2w_full = np.asarray(archive[key], dtype=np.float32)
        elif "camera_w2c" in files or "w2c" in files:
            key = "camera_w2c" if "camera_w2c" in files else "w2c"
            c2w_full = _invert_se3(archive[key])
        else:
            raise DataContractError("H3 camera NPZ has no explicit c2w/w2c array")
        K = _load_intrinsics_candidate(archive, intrinsics_bytes)
    if c2w_full.ndim != 3 or c2w_full.shape[1:] != (4, 4):
        raise DataContractError("H3 camera pose array must be [N,4,4]")
    if c2w_full.shape[0] <= max(source_indices):
        raise DataContractError("H3 camera track is shorter than the selected window")
    c2w = c2w_full[np.asarray(source_indices)].copy()
    if not np.isfinite(c2w).all():
        raise DataContractError("H3 camera c2w contains non-finite values")
    return c2w.astype(np.float32), _normalise_K(K, source_indices, transform)


__all__ = [
    "decode_camera",
]
