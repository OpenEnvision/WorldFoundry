"""Temporal index helpers for Matrix inference memory."""

import torch


def _compute_mosaic_frame_indices(noisy_count, interval, device=None):
    noisy_count = int(noisy_count)
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"mosaic_interval must be >= 1, got {interval}")
    if noisy_count % interval:
        raise ValueError(f"Noisy latent length {noisy_count} must be divisible by mosaic_interval {interval}")
    return torch.arange(
        interval - 1,
        noisy_count,
        interval,
        dtype=torch.long,
        device=device,
    )
