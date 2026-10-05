"""Family :class:`~...contracts.ConditionEncoder` adapters.

Each encoder implements ``encode(DiffusionRequest, device, dtype) ->
Conditioning`` with ``positive`` / ``negative`` / ``shared`` tensor maps
consumed by the matching denoiser.  This package is a documentation
surface; recipes import sibling factories, not this ``__init__``.
"""

__all__: list[str] = []
