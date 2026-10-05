"""Cosmos Predict1 / GEN3C :class:`~...contracts.EncodedLatentInitializer`.

Re-exports :class:`.component.Cosmos1Gen3CInitializer`, which encodes a
reference still, estimates depth, and builds the Video2World pose / warp
cache consumed by :class:`~...denoisers.cosmos1.Cosmos1Gen3CDenoiser`.
"""

from .component import Cosmos1Gen3CInitializer, build_cosmos1_gen3c_initializer

__all__ = ["Cosmos1Gen3CInitializer", "build_cosmos1_gen3c_initializer"]
