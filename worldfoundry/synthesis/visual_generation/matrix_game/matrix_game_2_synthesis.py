"""Synthesis wrapper for the MatrixGame2 runtime.

Also re-exports ``process_video``: the runtime module looks the helper up by
name at generation time, and :meth:`MatrixGame2Synthesis.predict` installs it
there before delegating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from worldfoundry.core.media.artifacts import process_game_control_video as process_video

from ..runtime_facade import RuntimeFacadeSynthesis

if TYPE_CHECKING:
    from .matrix_game_2_runtime.worldfoundry_runtime import MatrixGame2Runtime


def _runtime_module():
    """Import the runtime module on first use, not at adapter import time."""
    from .matrix_game_2_runtime import worldfoundry_runtime

    return worldfoundry_runtime


class MatrixGame2Synthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the MatrixGame2 runtime."""

    @classmethod
    def _runtime_cls(cls) -> type:
        return _runtime_module().MatrixGame2Runtime

    def __init__(
        self,
        pipeline: Any = None,
        vae: Any = None,
        weight_dtype: Any = None,
        mode: str = "universal",
        device: str = "cuda",
        *,
        runtime: "MatrixGame2Runtime | None" = None,
    ) -> None:
        """Wrap ``runtime``, or build one from the given pipeline components."""
        super().__init__(
            runtime=runtime,
            pipeline=pipeline,
            vae=vae,
            weight_dtype=weight_dtype,
            mode=mode,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        mode: str = "universal",
        device=None,
        weight_dtype: Any = None,
        **kwargs,
    ) -> "MatrixGame2Synthesis":
        """Load the MatrixGame2 runtime from pretrained weights and wrap it."""
        return cls(
            runtime=cls._runtime_cls().from_pretrained(
                pretrained_model_path=pretrained_model_path,
                mode=mode,
                device=device,
                weight_dtype=weight_dtype,
                **kwargs,
            )
        )

    @staticmethod
    def _resolve_checkpoint_path(model_root: str, mode: str, checkpoint_path: str | None = None) -> str:
        """Resolve a checkpoint path through the runtime's own resolution rules."""
        return _runtime_module().MatrixGame2Runtime._resolve_checkpoint_path(
            model_root=model_root,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )

    def predict(self, *args, **kwargs):
        """Publish ``process_video`` into the runtime module, then delegate."""
        _runtime_module().process_video = process_video
        return self.runtime.predict(*args, **kwargs)


__all__ = ["MatrixGame2Synthesis", "process_video"]
