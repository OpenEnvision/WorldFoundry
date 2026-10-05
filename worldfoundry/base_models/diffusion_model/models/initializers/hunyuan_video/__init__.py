"""HunyuanVideo / HunyuanVideo 1.5 :class:`~...contracts.LatentInitializer`.

Re-exports :class:`.component.HunyuanVideoLatentInitializer`.  T2V draws
``[B,C,T,H/s,W/s]`` noise.  I2V modes implement
:class:`~...contracts.EncodedLatentInitializer` and either concat a
first-frame latent+mask (original I2V) or freeze the first latent frame
(1.5).
"""

from .component import HunyuanVideoLatentInitializer, build_hunyuan_video_latent_initializer

__all__ = ["HunyuanVideoLatentInitializer", "build_hunyuan_video_latent_initializer"]
