"""Framework-neutral prompt preprocessing hooks used by inference components.

``torch`` is imported lazily inside the methods that need ``no_grad``, so
``from worldfoundry.core import PromptProcessor`` does not load an accelerator
runtime. This class does not own weight loading: callers inject refiners and
extenders through ``from_model_manager``.
"""

from __future__ import annotations

from collections.abc import Sequence


class PromptProcessor:
    """Apply optional prompt refiners and extenders without owning model loading.

    ``refiners`` rewrite a string or list in place; ``extenders`` take a
    ``{"prompt": ...}`` mapping and return an expanded dict. Both run under
    ``torch.no_grad()``.
    """

    def __init__(self) -> None:
        """Start with empty hook lists; callers append via ``load_prompt_*``."""
        self.refiners = []
        self.extenders = []

    def load_prompt_refiners(self, model_source, refiner_classes: Sequence[type] = ()) -> None:
        """Instantiate each refiner with ``from_model_manager(model_source)`` and append it."""
        for refiner_class in refiner_classes:
            self.refiners.append(refiner_class.from_model_manager(model_source))

    def load_prompt_extenders(self, model_source, extender_classes: Sequence[type] = ()) -> None:
        """Instantiate each extender with ``from_model_manager(model_source)`` and append it."""
        for extender_class in extender_classes:
            self.extenders.append(extender_class.from_model_manager(model_source))

    def process_prompt(self, prompt, positive: bool = True):
        """Run refiners in order on one prompt or a list; lists are processed recursively."""
        import torch

        with torch.no_grad():
            if isinstance(prompt, list):
                return [self.process_prompt(item, positive=positive) for item in prompt]
            for refiner in self.refiners:
                prompt = refiner(prompt, positive=positive)
            return prompt

    def extend_prompt(self, prompt: str, positive: bool = True):
        """Wrap ``prompt`` as ``{"prompt": ...}`` and run extenders in order.

        ``positive`` is kept only to match the refiner API; this implementation
        does not use it.
        """
        import torch

        del positive
        with torch.no_grad():
            extended = {"prompt": prompt}
            for extender in self.extenders:
                extended = extender(extended)
            return extended


__all__ = ["PromptProcessor"]
