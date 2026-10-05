"""Shared bases for synthesis wrappers that delegate to a model-owned runtime.

Every model directory owns its own runtime code, so the synthesis layer above it
is pure plumbing: construct the runtime, forward attribute lookups, forward
``predict``. Two shapes of that plumbing recur across the model adapters and are
captured here so each adapter only spells out what genuinely differs.

:class:`RuntimeFacadeSynthesis`
    Builds its runtime on demand and forwards every unknown attribute to it.
    Suits models whose runtime already exposes the public surface.

:class:`RuntimeAdapterSynthesis`
    Wraps an already-built runtime and pins the six-argument ``predict``
    contract that the Studio call path uses. Suits models whose runtime is
    produced by ``from_pretrained`` and never constructed bare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Sequence

from ..base_synthesis import BaseSynthesis


class DelegatedPredictMixin:
    """Forwards the Studio ``predict`` contract to ``self.runtime``.

    Split out from :class:`RuntimeAdapterSynthesis` so adapters that must
    inherit a different base (a runtime-profile subclass, say) can still share
    the one canonical signature.
    """

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate through the owned runtime and return its result mapping."""
        return self.runtime.predict(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )


class RuntimeFacadeSynthesis(BaseSynthesis):
    """Attribute-forwarding facade over a runtime that owns the real model code.

    Subclasses name their runtime through :attr:`RUNTIME_CLS` -- or override
    :meth:`_runtime_cls` when the import has to stay lazy -- and inherit
    construction, delegation, ``from_pretrained`` and ``predict``.
    """

    #: Runtime type this facade wraps. Left unset when :meth:`_runtime_cls` is
    #: overridden to defer the import.
    RUNTIME_CLS: ClassVar[type | None] = None

    def __init__(self, runtime: Any = None, **runtime_kwargs: Any) -> None:
        """Wrap ``runtime``, or build one from ``runtime_kwargs`` when omitted."""
        super().__init__()
        self.runtime = runtime if runtime is not None else self._runtime_cls()(**runtime_kwargs)

    @classmethod
    def _runtime_cls(cls) -> type:
        """Resolve the runtime type, importing it lazily if a subclass says so."""
        if cls.RUNTIME_CLS is None:
            raise NotImplementedError(f"{cls.__name__} must set RUNTIME_CLS or override _runtime_cls()")
        return cls.RUNTIME_CLS

    def __getattr__(self, name: str) -> Any:
        # Read ``runtime`` out of __dict__ rather than through self: on a
        # half-built instance the attribute is absent, and self.runtime would
        # route straight back here forever.
        runtime = self.__dict__.get("runtime")
        if runtime is None:
            raise AttributeError(name)
        return getattr(runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        """Load the runtime from pretrained weights and wrap it."""
        return cls(runtime=cls._runtime_cls().from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any):
        """Forward the call to the owned runtime."""
        return self.runtime.predict(*args, **kwargs)


class RuntimeAdapterSynthesis(DelegatedPredictMixin, BaseSynthesis):
    """Adapter around a runtime built by ``from_pretrained``.

    Mirrors the runtime's identity attributes onto the adapter so callers can
    read them without reaching through ``.runtime``. Adapters whose runtime
    also plans its own execution add a ``runtime_plan`` passthrough themselves;
    it is deliberately absent here because not every runtime offers one.
    """

    #: Runtime type this adapter drives.
    RUNTIME_CLS: ClassVar[type]

    #: Runtime attributes copied onto the adapter at construction time.
    MIRRORED_RUNTIME_ATTRS: ClassVar[tuple[str, ...]] = (
        "model_id",
        "model_name",
        "generation_type",
        "device",
    )

    def __init__(self, runtime: Any) -> None:
        """Wrap ``runtime`` and mirror :attr:`MIRRORED_RUNTIME_ATTRS` from it."""
        super().__init__()
        self.runtime = runtime
        for attr in self.MIRRORED_RUNTIME_ATTRS:
            setattr(self, attr, getattr(runtime, attr))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ):
        """Build the runtime from pretrained assets and wrap it.

        Args:
            pretrained_model_path: Asset path, or a mapping of runtime options.
            args: Accepted for loader compatibility and ignored.
            device: Device selector handed to the runtime.
            model_id: Profile identifier; falls back to the class ``MODEL_ID``.
            kwargs: Further runtime options.
        """
        del args
        runtime = cls.RUNTIME_CLS.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id or getattr(cls, "MODEL_ID", None),
            **kwargs,
        )
        return cls(runtime)


__all__ = [
    "DelegatedPredictMixin",
    "RuntimeAdapterSynthesis",
    "RuntimeFacadeSynthesis",
]
