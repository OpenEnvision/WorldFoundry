"""LTX audio-video diffusion transformer.

Public surface for the Lightricks LTX DiT: :class:`LTXModel` (velocity
backbone), :class:`X0Model` (x0 wrapper), and :class:`Modality` (per-stream
``B T D`` packed tokens).  Video tokens are already patchified latents
(not ``B C T H W``); audio is a 1-D temporal token stream.  Dual-stream
blocks optionally cross-attend A↔V.
"""

from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel, X0Model

__all__ = [
    "LTXModel",
    "Modality",
    "X0Model",
]
