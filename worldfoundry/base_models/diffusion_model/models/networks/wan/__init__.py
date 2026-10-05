"""Shared Wan 2.1 / 2.2 diffusion-transformer building blocks.

This package is the checkpoint-shaped DiT used by native WorldFoundry Wan
recipes.  :mod:`.model` is the canonical packed-tensor graph (patchify,
timestep AdaLN, 3D RoPE, self/cross-attention processors, unpatchify).
:mod:`.adapter` adds the optional spatial/camera control residual.
:mod:`.vace` / :mod:`.vace_core` add VACE hint branches.  :mod:`.mixins`
holds state-neutral helpers reused by research variants.

:mod:`.reference_21` and :mod:`.reference_22` keep the official Alibaba
list-of-videos forward (variable-length packing).  :mod:`.variants`
contains recipe-specific forks: action, camera, causal, linear attention,
TeaCache-friendly processors, DualCamCtrl, Echo-Infinity memory, forcing,
LingBot, DreamZero, MagicWorld, MoVerse, MinWM, and others.

``WanModelStateDictConverter`` is imported lazily from the denoiser layer
so this package can stay a pure network definition.
"""

from importlib import import_module

from .adapter import ResidualBlock, SimpleAdapter
from .model import (
    AttentionModule,
    MLP,
    CrossAttention,
    GateModule,
    Head,
    WanModel,
)
from .variants import WanControlNet


def __getattr__(name):
    """Resolve ``WanModelStateDictConverter`` without importing the denoiser at module load."""
    if name != "WanModelStateDictConverter":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("worldfoundry.base_models.diffusion_model.models.denoisers.wan").WanModelStateDictConverter
    globals()[name] = value
    return value

__all__ = [
    "CrossAttention",
    "AttentionModule",
    "GateModule",
    "Head",
    "MLP",
    "ResidualBlock",
    "SimpleAdapter",
    "WanModel",
    "WanModelStateDictConverter",
    "WanControlNet",
]
