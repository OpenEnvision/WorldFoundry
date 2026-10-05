"""Shared synthesis facade for the Yume runtime family.

Yume 1.0 and Yume 1.5 expose the same wrapper contract -- a
``(model, device, weight_dtype)`` constructor, an ``fsdp``-aware
``from_pretrained``, and per-interaction generation -- and differ only in which
runtime class they resolve. Subclasses supply that class through
:meth:`~RuntimeFacadeSynthesis._runtime_cls`.
"""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeFacadeSynthesis


class YumeFacadeSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over a Yume-family runtime.

    Each subclass resolves its runtime lazily so that importing the adapter does
    not pull in the bundled Yume checkout.
    """

    def __init__(self, model: Any = None, device: Any = None, weight_dtype: Any = None, *, runtime: Any = None) -> None:
        """Wrap ``runtime``, or build one for ``model`` on ``device``."""
        super().__init__(runtime=runtime, model=model, device=device, weight_dtype=weight_dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device,
        weight_dtype,
        fsdp,
        **runtime_kwargs: Any,
    ):
        """Load the runtime from pretrained weights and wrap it."""
        return cls(
            runtime=cls._runtime_cls().from_pretrained(
                pretrained_model_path=pretrained_model_path,
                device=device,
                weight_dtype=weight_dtype,
                fsdp=fsdp,
                **runtime_kwargs,
            )
        )

    def predict_per_interaction(self, *args: Any, **kwargs: Any):
        """Generate one clip per interaction step through the runtime."""
        return self.runtime.predict_per_interaction(*args, **kwargs)


__all__ = ["YumeFacadeSynthesis"]
