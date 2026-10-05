"""DDIM samplers for latent video diffusion recipes.

Legacy LVDM / VideoCrafter solvers used by T2V-Turbo and vid2world
variants.  They predate the stateless ``DiffusionScheduler`` Protocol;
recipes that still need DDIM import these types rather than going through
Diffusers.  Numerical internals stay in the sibling modules.
"""

from .ddim import DDIMSampler

__all__ = ["DDIMSampler"]
