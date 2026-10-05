"""Small tensor helpers shared by Matrix inference orchestration."""

from __future__ import annotations


class _InferenceTensorMixin:
    @staticmethod
    def _ensure_batched_latents(latents):
        if latents.ndim == 4:
            return latents.unsqueeze(0)
        if latents.ndim == 5:
            return latents
        raise ValueError(f"Expected latent tensor with 4 or 5 dims, got {tuple(latents.shape)}")


__all__ = ["_InferenceTensorMixin"]
