"""mira.inference: offline autoregressive rollout and checkpoint loading.

Public API:
    rollout — autoregressive denoising of a clip into latents, without the final decode
    measure_rollout_speed — time :func:`rollout` over a fixed number of latent frames
    load_world_model — instantiate the right world-model class from a checkpoint dir
"""

from __future__ import annotations

from .loading import load_world_model
from .rollout import measure_rollout_speed, rollout

__all__ = [
    "load_world_model",
    "measure_rollout_speed",
    "rollout",
]
