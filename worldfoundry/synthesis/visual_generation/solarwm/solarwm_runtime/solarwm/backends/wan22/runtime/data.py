"""Transport-neutral raw Wan sample decoding and logical-DP iteration."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ....data.camera import CameraGuards, absolute_c2w, camera_audit_prefix_frames, invert_se3, load_camera_npz
from ....errors import DataContractError

FX_NORM = 969.6969696969696 / (960.0 * 2)
FY_NORM = 969.6969696969696 / (540.0 * 2)
CAMERA_CONVENTION = "authoritative_source_c2w_no_axis_flip"
_CAMERA_ARRAY_KEYS = frozenset({"c2w", "w2c", "vipe_c2w", "vipe_w2c", "relative_w2c"})


def latent_aligned_pixel_indices(pixel_frames: int) -> np.ndarray:
    """Return Wan's causal VAE alignment: 0, 1, 5, 9, ..."""

    if pixel_frames < 1 or (pixel_frames - 1) % 4:
        raise DataContractError("Wan pixel frames must be 1 + 4*(latent_frames-1)")
    latent_frames = 1 + (pixel_frames - 1) // 4
    return np.asarray([0, *(1 + 4 * index for index in range(latent_frames - 1))])


def _relative_c2w_fp32(
    matrices: np.ndarray,
    storage: str,
) -> np.ndarray:
    """Build first-frame-relative C2W with stable FP32 operation order."""

    if storage == "relative_w2c":
        relative = invert_se3(matrices)
        relative[0] = np.eye(4, dtype=np.float32)
        return relative
    c2w = absolute_c2w(matrices, storage)
    w2c = invert_se3(c2w)
    target_c2w = np.eye(4, dtype=np.float32)
    absolute_to_relative = target_c2w @ w2c[0]
    relative = np.empty_like(c2w)
    relative[0] = target_c2w
    for index in range(1, c2w.shape[0]):
        relative[index] = absolute_to_relative @ c2w[index]
    return relative


def _torch_invert_se3_fp32(matrices: Any) -> Any:
    """Invert SE3 matrices with the camera-token FP32 operation order."""

    import torch

    rotation = matrices[..., :3, :3]
    rotation_t = rotation.transpose(-1, -2)
    result = torch.zeros_like(matrices)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_t,
        matrices[..., :3, 3],
    )
    result[..., 3, 3] = 1.0
    return result


def decode_video(
    payload: bytes,
    frame_indices: Sequence[int],
    *,
    height: int,
    width: int,
) -> Any:
    """Decode selected frames and apply the canonical bilinear-center-crop path."""

    try:
        import decord
        import torch
    except ImportError as exc:
        raise DataContractError("raw Wan decoding requires torch and decord") from exc

    decord.bridge.set_bridge("torch")
    try:
        reader = decord.VideoReader(io.BytesIO(payload), num_threads=1)
        frames = reader.get_batch([int(value) for value in frame_indices])
    except Exception as exc:
        raise DataContractError(f"cannot decode selected video frames: {exc}") from exc
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    return _preprocess_video_frames(frames, height=height, width=width)


def _preprocess_video_frames(frames: Any, *, height: int, width: int) -> Any:
    """Bound full-resolution FP32 temporaries without changing pixel arithmetic."""

    import torch
    import torch.nn.functional as functional

    _, source_height, source_width, _ = frames.shape
    resized_height, resized_width = source_height, source_width
    if (source_height, source_width) != (height, width):
        scale = max(height / source_height, width / source_width)
        resized_height = round(source_height * scale)
        resized_width = round(source_width * scale)
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    processed = []
    # Resize is independent per frame. Keep the same layout, FP32 operation
    # order, interpolation and crop, but do not expand a whole 4K clip at once.
    for chunk in frames.split(8, dim=0):
        chunk = chunk.permute(0, 3, 1, 2).contiguous().float() / 255.0
        if (resized_height, resized_width) != (source_height, source_width):
            chunk = functional.interpolate(
                chunk,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
        chunk = chunk[:, :, top : top + height, left : left + width]
        processed.append((chunk * 2.0 - 1.0).to(torch.bfloat16))
    return torch.cat(processed, dim=0)


def _camera_array_key(manifest: Mapping[str, Any], configured: str) -> str:
    configured = str(configured).strip()
    if configured not in _CAMERA_ARRAY_KEYS:
        raise DataContractError(
            f"data.camera_array_key must be one of {sorted(_CAMERA_ARRAY_KEYS)}, got {configured!r}"
        )
    camera = manifest.get("camera", {})
    declared = ""
    if isinstance(camera, Mapping):
        declared = str(camera.get("array_key", "")).strip()
    declared = declared or str(manifest.get("camera_array_key", "")).strip()
    if declared and declared not in _CAMERA_ARRAY_KEYS:
        raise DataContractError(f"unsupported manifest camera array key {declared!r}")
    if declared and declared != configured:
        raise DataContractError(
            f"manifest camera array key conflicts with data.camera_array_key: {declared!r} != {configured!r}"
        )
    return configured


def _validate_camera_manifest(
    manifest: Mapping[str, Any],
    *,
    guards: CameraGuards,
) -> None:
    camera = manifest.get("camera", {})
    if not isinstance(camera, Mapping):
        raise DataContractError("raw Wan manifest.camera must be a mapping")
    if str(camera.get("convention", "")) != CAMERA_CONVENTION:
        raise DataContractError(f"raw Wan camera convention must be {CAMERA_CONVENTION!r}")
    if str(camera.get("dtype", "")) != "float32":
        raise DataContractError("raw Wan manifest camera dtype must be float32")
    if camera.get("finite") is not True:
        raise DataContractError("raw Wan manifest must attest camera finite=true")
    expected_floats = {
        "magnitude_audit_seconds": 10.0,
        "max_camera_abs": guards.max_camera_abs,
        "max_rel_translation": guards.max_rel_translation,
    }
    for field, expected in expected_floats.items():
        try:
            actual = float(camera.get(field))
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"raw Wan manifest camera {field} must be numeric") from exc
        if expected is None or actual != float(expected):
            raise DataContractError(f"raw Wan manifest camera {field}={actual} does not match {expected}")
    shape = camera.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 3
        or list(shape[1:]) != [4, 4]
    ):
        raise DataContractError("raw Wan manifest camera shape must be [T,4,4]")
    try:
        if int(shape[0]) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise DataContractError("raw Wan manifest camera shape[0] must be positive") from exc


def build_camera_tokens(
    payload: bytes,
    source_frame_indices: Sequence[int],
    manifest: Mapping[str, Any],
    *,
    source_fps: float,
    output_fps: float,
    frame_sequence_length: int,
    guards: CameraGuards,
    configured_array_key: str,
    manifest_guards: CameraGuards | None = None,
) -> Mapping[str, Any]:
    """Select, rebase, guard, and spatially expand authoritative cameras."""

    import torch

    # The manifest records the guards used when the sample was published.
    # Validation may override runtime guards without changing that record.
    _validate_camera_manifest(manifest, guards=manifest_guards or guards)
    matrices, storage = load_camera_npz(
        payload,
        _camera_array_key(manifest, configured_array_key),
    )
    matrices = np.asarray(matrices, dtype=np.float32)
    declared_shape = tuple(int(value) for value in manifest["camera"]["shape"])
    if matrices.shape != declared_shape:
        raise DataContractError(
            f"camera NPZ shape does not match manifest.camera.shape: {matrices.shape} != {declared_shape}"
        )
    source_indices = np.asarray(source_frame_indices, dtype=np.int64)
    if source_indices.min(initial=0) < 0 or source_indices.max(initial=0) >= len(matrices):
        raise DataContractError("selected camera frames are outside the source trajectory")
    latent_pixel = latent_aligned_pixel_indices(len(source_indices))
    selected = matrices[source_indices[latent_pixel]]
    # Preserve the camera-token FP32 operation order. This
    # deliberately differs from the shared canonical relative-W2C shortcut.
    relative_c2w = _relative_c2w_fp32(selected, storage)
    audit_pixels = camera_audit_prefix_frames(len(source_indices), output_fps if output_fps > 0 else source_fps)
    audit_latents = int(np.searchsorted(latent_pixel, audit_pixels, side="left"))
    audit_latents = max(1, min(audit_latents, len(relative_c2w)))
    relative_c2w = guards.apply(relative_c2w, audit_rows=audit_latents)

    viewmats = _torch_invert_se3_fp32(torch.as_tensor(relative_c2w, dtype=torch.float32))
    viewmats = viewmats.unsqueeze(1).expand(-1, int(frame_sequence_length), -1, -1).reshape(-1, 4, 4).contiguous()
    intrinsics = torch.zeros((1, 3, 3), dtype=torch.float32)
    intrinsics[:, 0, 0] = FX_NORM
    intrinsics[:, 1, 1] = FY_NORM
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    intrinsics = intrinsics.expand(viewmats.shape[0], -1, -1).contiguous()
    return {"viewmats": viewmats, "K": intrinsics}


__all__ = [
    "build_camera_tokens",
    "decode_video",
    "latent_aligned_pixel_indices",
]
