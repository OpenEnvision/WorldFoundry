"""StepVideo 64-channel video VAE decoder role.

Package entry for the StepVideo latent codec.
:class:`~.component.StepVideoDecoder` wraps
:class:`~.model.AutoencoderKL` (``z_channels=64``).

The VAE returns ``[B, F, C, H, W]``; the adapter permutes to
BCTHW and clamps ``[-1, 1]``.  ``return_latent`` skips decode.
Encode is not on this role (T2V draws noise with 17-frame packing).

This is StepVideo's own VAE, not SD / Wan / Hunyuan.
"""

from .model import AutoencoderKL
from .component import StepVideoDecoder, build_step_video_decoder

__all__ = ["AutoencoderKL", "StepVideoDecoder", "build_step_video_decoder"]
