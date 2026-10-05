"""Core Fréchet Video Distance utilities (adapted from MiraBench / StyleGAN-V / TF-GAN)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

MAX_BATCH = 8
TARGET_RESOLUTION = (224, 224)


def preprocess_videos_uint8(videos: np.ndarray, target_resolution: tuple[int, int] = TARGET_RESOLUTION) -> torch.Tensor:
    """Preprocess uint8 videos ``(N, T, H, W, C)`` to I3D input ``(N, C, T, H, W)`` in [-1, 1]."""
    b, t, h, w, c = videos.shape
    all_frames = torch.as_tensor(videos, dtype=torch.float32).flatten(end_dim=1)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(all_frames, size=target_resolution, mode="bilinear", align_corners=False)
    resized = resized.view(b, t, c, *target_resolution)
    output = resized.transpose(1, 2).contiguous()
    return 2.0 * output / 255.0 - 1.0


def _symmetric_matrix_square_root(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


def trace_sqrt_product(sigma: torch.Tensor, sigma_v: torch.Tensor) -> torch.Tensor:
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def cov(m: torch.Tensor, rowvar: bool = False) -> torch.Tensor:
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
    trace = torch.trace(sigma + sigma_w) - 2.0 * trace_sqrt_product(sigma, sigma_w)
    mean = torch.sum((m - m_w) ** 2)
    return trace + mean


def frechet_distance_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute FVD/FID from precomputed mean and covariance."""
    diff = mu1 - mu2
    covmean, _ = np.linalg.eig(np.dot(sigma1, sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.sum(np.sqrt(np.maximum(covmean, 0)))
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def resolve_i3d_checkpoint(
    explicit: str | Path | None = None,
    *,
    extra_candidates: tuple[str | Path, ...] = (),
) -> str:
    """Resolve the I3D checkpoint path, failing loudly when it is not staged.

    ``extra_candidates`` lets benchmark runtimes prepend their official-protocol
    staging locations without duplicating this search logic.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"I3D checkpoint not found: {path}")
    candidates = [
        os.environ.get("WORLDFOUNDRY_FVD_I3D_CKPT"),
        os.environ.get("WORLDFOUNDRY_MIRABENCH_FVD_I3D_CKPT"),
        *(str(candidate) for candidate in extra_candidates),
        str(Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/huggingface/hub")).expanduser() / "i3d_pretrained_400.pt"),
        str(Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/huggingface/hub")).expanduser() / "MiraBench/fvd/i3d_pretrained_400.pt"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates if path)
    raise FileNotFoundError(
        "FVD requires i3d_pretrained_400.pt. Set WORLDFOUNDRY_FVD_I3D_CKPT.\n"
        f"Searched:\n{searched}"
    )


def load_fvd_i3d(device: torch.device, checkpoint: str | Path | None = None) -> Any:
    """Load Inception I3D model used by FVD."""
    from worldfoundry.evaluation.tasks.metrics.fvd.pytorch_i3d import InceptionI3d

    ckpt = resolve_i3d_checkpoint(checkpoint)
    model = InceptionI3d(400, in_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()
    return model


@torch.no_grad()
def extract_i3d_features(
    videos: np.ndarray,
    i3d: Any,
    device: torch.device,
    batch_size: int = MAX_BATCH,
) -> torch.Tensor:
    """Extract I3D features from uint8 videos ``(N, T, H, W, C)``."""
    videos_t = preprocess_videos_uint8(videos)
    logits: list[torch.Tensor] = []
    for start in range(0, videos_t.shape[0], batch_size):
        batch = videos_t[start : start + batch_size].to(device)
        logits.append(i3d(batch))
    return torch.cat(logits, dim=0)


def compute_fvd(
    real_videos: np.ndarray,
    generated_videos: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    i3d_checkpoint: str | Path | None = None,
    batch_size: int = MAX_BATCH,
) -> float:
    """Compute Fréchet Video Distance between two video sets."""
    device_t = torch.device(device)
    i3d = load_fvd_i3d(device_t, i3d_checkpoint)
    real_embed = extract_i3d_features(real_videos, i3d, device_t, batch_size=batch_size)
    gen_embed = extract_i3d_features(generated_videos, i3d, device_t, batch_size=batch_size)
    return float(frechet_distance(real_embed, gen_embed).item())


__all__ = [
    "compute_fvd",
    "extract_i3d_features",
    "frechet_distance",
    "frechet_distance_from_stats",
    "load_fvd_i3d",
    "preprocess_videos_uint8",
    "resolve_i3d_checkpoint",
]
