"""WorldArena GRiT wrapper backed by the shared base-model implementation."""

from __future__ import annotations

from typing import Any


class DenseCaptioning:
    """Lazy proxy for the shared GRiT DenseCaptioning implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            from worldfoundry.base_models.perception_core.captioning.grit.model import DenseCaptioning as Impl
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "WorldArena GRiT metrics require the optional detectron2/GRiT runtime. "
                "Install the benchmark metric environment before running dense-caption metrics."
            ) from exc
        self._impl = Impl(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


__all__ = ["DenseCaptioning"]
