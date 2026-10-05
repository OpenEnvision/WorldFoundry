"""Wan conditioning variants (action CLIP, S2V audio, motion).

Namespace for checkpoint-shaped prompt / control encoders
that keep the Wan ``context`` contract but match a specific
recipe: action CLIP, speech-to-video audio, motion
controller, Animate pose, DreamX UMT5, Fun-Control CLIP,
and T5 compatibility shims.

Default T2V/I2V recipes should use
:class:`~..component.WanTextConditioner` /
:class:`WanImageTextConditioner`.

These are ConditionEncoder-side variants, not VAE variants
(:mod:`~...autoencoders.wan.variants`).
"""

from .motion_controller import WanMotionControllerModel, WanMotionControllerModelDictConverter
from .s2v_audio import WanS2VAudioEncoder, WanS2VAudioEncoderStateDictConverter, get_sample_indices

__all__ = [
    "WanMotionControllerModel",
    "WanMotionControllerModelDictConverter",
    "WanS2VAudioEncoder",
    "WanS2VAudioEncoderStateDictConverter",
    "get_sample_indices",
]
