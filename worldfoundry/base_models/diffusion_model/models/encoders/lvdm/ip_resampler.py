"""Compatibility names for IP-Adapter style LVDM resamplers.

Re-exports the Perceiver-style classes from :mod:`.resampler`
under the names some VideoCrafter / IP-Adapter checkpoints
expect.

No new math lives here.  The implementation is
:class:`~.resampler.Resampler`.

LVDM family only; Wan IP-Adapter paths are elsewhere.
"""

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.resampler import (
    FeedForward,
    ImageProjModel,
    PerceiverAttention,
    Resampler,
    reshape_tensor,
)

__all__ = [
    "FeedForward",
    "ImageProjModel",
    "PerceiverAttention",
    "Resampler",
    "reshape_tensor",
]
