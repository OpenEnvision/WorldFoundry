"""LVDM / VideoCrafter frame-wise SD AutoencoderKL codec.

Package entry for VideoCrafter / LVDM latents.
:class:`~.component.LVDMVideoDecoder` decodes each frame through
an SD AutoencoderKL (``scale_factor=0.18215``).

Input latents are BCTHW, 4 channels; there is no temporal VAE.
Weights come from the VideoCrafter ``first_stage_model.*`` prefix.
The CNN is :class:`~.model.AutoencoderKL`.

This is the SD-image-VAE family, shared with T2V-Turbo recipes,
not a causal 3-D codec like Wan or Hunyuan.
"""

from .component import LVDMVideoDecoder, build_lvdm_video_decoder
from .model import AutoencoderKL

__all__ = ["AutoencoderKL", "LVDMVideoDecoder", "build_lvdm_video_decoder"]
