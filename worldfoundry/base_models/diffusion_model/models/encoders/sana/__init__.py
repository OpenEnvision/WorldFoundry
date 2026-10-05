"""Sana Gemma-2 :class:`~...contracts.ConditionEncoder` role.

Package entry for Sana image prompt conditioning.
:class:`~.component.SanaPromptConditioner` tokenizes prompts
with Gemma-2 and returns ``Conditioning`` maps for the Sana
DiT.

Text-only; image latents are
:mod:`~...autoencoders.sana`.  Does not rewrite user prompt
strings.

Sibling of LTX Gemma-3 (:mod:`~..ltx`) but a different
connector and hidden-size contract.
"""

from .component import SanaPromptConditioner, build_sana_prompt_conditioner
from .refiner import SanaWMRefinerConditioner, build_sana_wm_refiner_conditioner

__all__ = [
    "SanaPromptConditioner",
    "SanaWMRefinerConditioner",
    "build_sana_prompt_conditioner",
    "build_sana_wm_refiner_conditioner",
]
