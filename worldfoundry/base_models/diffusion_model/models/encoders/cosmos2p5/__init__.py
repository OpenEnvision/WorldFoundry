"""Cosmos Predict 2.5 Reason1 :class:`~...contracts.ConditionEncoder`.

Package entry for the Qwen2.5-VL text tower used by Predict 2.5.
:class:`~.component.Cosmos25PromptConditioner` tokenizes prompts
and returns ``Conditioning.positive["context"]``.

Does not rewrite user prompt strings.  Optional English system
prefix lives on the component.  Weight remap is
:func:`convert_reason1_text_state_dict`.

Sibling of Cosmos3 tokenizer-only :mod:`~..cosmos3` and the
Gamma-World rename layer :mod:`~..gamma_world`.
"""

from .component import (
    Cosmos25PromptConditioner,
    Cosmos25TextBackbone,
    CosmosReason1TextEncoder,
    CosmosReason1TextEncoderConfig,
    build_cosmos25_prompt_conditioner,
    convert_reason1_text_state_dict,
    load_cosmos_reason1_prompt_encoder,
)

__all__ = [
    "Cosmos25PromptConditioner",
    "Cosmos25TextBackbone",
    "CosmosReason1TextEncoder",
    "CosmosReason1TextEncoderConfig",
    "build_cosmos25_prompt_conditioner",
    "convert_reason1_text_state_dict",
    "load_cosmos_reason1_prompt_encoder",
]
