"""SD3-family per-frame 2-D VAE decoder role.

Package entry for SD3 / Vchitect frame decode.
:class:`~.component.SD3FrameDecoder` folds ``[B, F, C, H, W]``
through :class:`~worldfoundry.core.nn.latent.vae2d.NativeVAE2DDecoder`.

Official scale/shift is ``1.5305`` / ``0.0609``.  Output is BCTHW
RGB in ``[-1, 1]``.  Encode is not implemented here (T2V noise is
drawn directly).  ``return_latent`` skips pixel decode.

Sibling of the Vchitect denoiser / conditioner, not a video CNN.
"""

from .component import SD3FrameDecoder, build_sd3_frame_decoder

__all__ = ["SD3FrameDecoder", "build_sd3_frame_decoder"]
