from __future__ import annotations

import numpy as np

from wrbench.adapters.operators.trajectory_utils import make_yaw_w2c


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def hydra_smoothstep_goreturn(peak_deg: float, num_frames: int = 77) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return np.asarray([0.0], dtype=np.float32)

    forward_count = max(2, num_frames // 2)
    return_count = max(2, num_frames - forward_count)
    forward = np.asarray(peak_deg, dtype=np.float32) * _smoothstep01(
        np.linspace(0.0, 1.0, forward_count, dtype=np.float32)
    )
    backward = np.asarray(peak_deg, dtype=np.float32) * (
        1.0 - _smoothstep01(np.linspace(0.0, 1.0, return_count, dtype=np.float32))
    )
    return np.concatenate([forward, backward], axis=0).astype(np.float32)


def hydra_yaw_target_c2w(peak_deg: float, num_frames: int = 77) -> np.ndarray:
    yaw_deg = hydra_smoothstep_goreturn(float(peak_deg), num_frames)
    w2c = np.asarray(make_yaw_w2c(yaw_deg), dtype=np.float32)
    return np.linalg.inv(w2c).astype(np.float32)


def hydra_pretransform_c2w(c2w: np.ndarray) -> np.ndarray:
    mats = np.asarray(c2w, dtype=np.float32)
    out = np.empty_like(mats)
    out[..., :, 0] = mats[..., :, 2]
    out[..., :, 1] = mats[..., :, 0]
    out[..., :, 2] = -mats[..., :, 1]
    out[..., :, 3] = mats[..., :, 3]
    out[..., :3, 3] *= 100.0
    return out.astype(np.float32)


def hydra_camera_payload_dict(c2w: np.ndarray) -> dict[str, list[list[float]]]:
    mats = hydra_pretransform_c2w(c2w)
    return {str(i): mats[i].tolist() for i in range(len(mats))}
