"""Shared native T5 :class:`~...contracts.ConditionEncoder` role.

Package entry for the generic T5 encoder used by several
image/video recipes.  :class:`~.component.T5EncoderConditioner`
tokenizes prompts and returns ``context`` / mask tensors.

:func:`convert_t5_encoder_state_dict` remaps Hugging Face
keys onto :class:`T5EncoderModule`.  Wan's UMT5 is a
separate graph (:mod:`~..wan.model`).

This is the shared T5 family, not Gemma or OpenCLIP.
"""

from .component import (
    T5EncoderConditioner,
    T5EncoderModule,
    build_t5_encoder_conditioner,
    convert_t5_encoder_state_dict,
)

__all__ = [
    "T5EncoderConditioner",
    "T5EncoderModule",
    "build_t5_encoder_conditioner",
    "convert_t5_encoder_state_dict",
]
