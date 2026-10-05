"""Cosmos3 tokenizer-only :class:`~...contracts.ConditionEncoder`.

Package entry for Cosmos3 prompt ids.  The omni transformer
embeds ``input_ids`` itself — there is no separate text tower.
:class:`~.component.Cosmos3PromptConditioner.encode` returns
``input_ids`` and optional ``action_domain_id``.

System / duration / FPS templates are English; user prompt
strings are not rewritten.

Distinct from Predict 2.5 Reason1 (:mod:`~..cosmos2p5`).
"""

from .component import Cosmos3PromptConditioner, build_cosmos3_prompt_conditioner

__all__ = ["Cosmos3PromptConditioner", "build_cosmos3_prompt_conditioner"]
