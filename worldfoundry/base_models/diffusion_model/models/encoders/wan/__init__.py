"""Wan UMT5 / CLIP :class:`~...contracts.ConditionEncoder` roles.

Package entry for Wan prompt (and optional image) conditioning.
:class:`~.component.WanTextConditioner` encodes prompts to
``Conditioning(positive={"context": ...}, negative=...)``.
:class:`WanImageTextConditioner` adds CLIP visual tokens to
``shared``.

Factories load UMT5 + tokenizer (and optional CLIP) through
the native module loader.  Chinese / multilingual prompt
strings pass through unchanged.  Variants live under
:mod:`.variants`.
"""

from .component import (
    WanImageTextConditioner,
    WanTextConditioner,
    WanUMT5PromptEncoder,
    build_diffusers_wan_text_conditioner,
    build_wan_image_text_conditioner,
    build_wan_text_conditioner,
    convert_diffusers_umt5_encoder_state_dict,
    convert_wan_clip_vision_state_dict,
)
from .environment import (
    WanVGGTEnvironmentEncoder,
    load_vggt_1b_backbone,
    load_wan_vggt_environment_encoder,
    resolve_vggt_1b_root,
)
from .image import WanImageEncoder, WanImageEncoderStateDictConverter
from .model import HuggingfaceTokenizer, WanTextEncoder, WanTextEncoderStateDictConverter
from .prompter import WanPrompter

__all__ = [
    "HuggingfaceTokenizer",
    "WanImageEncoder",
    "WanImageEncoderStateDictConverter",
    "WanImageTextConditioner",
    "WanPrompter",
    "WanTextConditioner",
    "WanTextEncoder",
    "WanTextEncoderStateDictConverter",
    "WanUMT5PromptEncoder",
    "WanVGGTEnvironmentEncoder",
    "build_diffusers_wan_text_conditioner",
    "build_wan_image_text_conditioner",
    "build_wan_text_conditioner",
    "convert_diffusers_umt5_encoder_state_dict",
    "convert_wan_clip_vision_state_dict",
    "load_vggt_1b_backbone",
    "load_wan_vggt_environment_encoder",
    "resolve_vggt_1b_root",
]
