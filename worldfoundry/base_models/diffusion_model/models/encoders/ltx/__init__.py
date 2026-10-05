"""LTX-2 Gemma :class:`~...contracts.ConditionEncoder` role.

Package entry for Lightricks prompt conditioning.
:class:`~.component.LTXPromptConditioner` runs Gemma-3, then
the embeddings connector / processor so video and audio
caption tokens match DiT cross-attention width.

``encode`` returns :class:`~...contracts.Conditioning` with
video/audio context tensors.  19B graphs project captions
inside the transformer; 22B projects here.

Tokenizer / connector internals live in this package.
"""

from .component import LTXPromptConditioner, build_ltx_prompt_conditioner

__all__ = [
    "LTXPromptConditioner",
    "build_ltx_prompt_conditioner",
]
