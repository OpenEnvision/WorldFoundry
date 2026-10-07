"""Choose the DiT class and reject sampler settings the factory cannot combine."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.dit import DiffusionTransformer3D
from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.piflow_dit import PiFlowDiffusionTransformer3D

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel


class PiFlowConfig(ConfigModel):
    """Configuration embedded next to a distilled base-model checkpoint."""

    enabled: bool = False
    dx_num_grid_points: int = Field(default=10, ge=2)
    eps: float = Field(default=1e-6, gt=0)
    final_step_size_scale: float = Field(default=0.5, gt=0, le=1)
    num_policy_substeps: int = Field(default=128, ge=1)
    shift: float = Field(default=5.0, gt=0)


def dit_class(piflow_enabled: bool) -> type[nn.Module]:
    if piflow_enabled:
        return PiFlowDiffusionTransformer3D
    return DiffusionTransformer3D


def reject_incompatible_cache(cfg: Any, cache_mode: str | None) -> None:
    """PiFlow and MagCache/NaviCache are mutually exclusive. Checked once, at apply time."""
    if not cfg.piflow.enabled:
        return
    if not cfg.dit.is_multimodal:
        raise ValueError("piflow.enabled requires dit.is_multimodal=true")
    if cfg.generation.guidance_weight != 1.0:
        raise ValueError("piflow.enabled requires generation.guidance_weight=1.0")
    if cache_mode not in (None, "none"):
        raise ValueError("piflow.enabled is incompatible with MagCache/NaviCache")
