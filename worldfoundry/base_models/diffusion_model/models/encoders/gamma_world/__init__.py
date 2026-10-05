"""Gamma-World prompt + action :class:`~...contracts.ConditionEncoder`.

Package entry for the multi-player world conditioner.
:class:`~.component.GammaWorldConditioner` reuses the Cosmos
Reason1 backbone and renames ``context`` to the tensor names
the Gamma denoisers expect.

Also normalizes frame-aligned action tracks from
``DiffusionRequest``.  Text encoding is not reimplemented here.

Wan-family video latents are :mod:`~...autoencoders.gamma_world`;
this package is prompt/action only.
"""

from .component import GammaWorldConditioner, build_gamma_world_conditioner

__all__ = ["GammaWorldConditioner", "build_gamma_world_conditioner"]
