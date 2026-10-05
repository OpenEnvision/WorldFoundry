"""WorldFoundry synthesis facades for independent Echo-Memory models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis


class _LazyEchoMemoryRuntime:
    """Delay native Wan component imports until a synthesis model runs."""

    def __new__(cls, **kwargs: Any) -> Any:
        from .runtime import EchoMemoryRuntime

        return EchoMemoryRuntime(**kwargs)


class EchoMemorySynthesis(RuntimeVideoSynthesis):
    """Shared implementation; concrete subclasses pin one public model ID."""

    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = _LazyEchoMemoryRuntime
    PRIMARY_PATH_KEY = "checkpoint_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/echo_memory/runtime_defaults.yaml"

    @classmethod
    def build_runtime_kwargs(
        cls,
        pretrained_model_path: Any = None,
        generator_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        overrides = dict(generator_overrides or {})
        overrides.update(kwargs)
        requested = overrides.pop("model_id", cls.MODEL_NAME)
        if requested != cls.MODEL_NAME:
            raise ValueError(f"{cls.__name__} is permanently bound to {cls.MODEL_NAME!r}, got {requested!r}")
        if "recipe_id" in overrides:
            raise ValueError(
                f"{cls.__name__} has an immutable memory recipe; select another Echo model ID "
                "instead of passing recipe_id"
            )
        runtime_kwargs = super().build_runtime_kwargs(
            pretrained_model_path=pretrained_model_path,
            generator_overrides=overrides,
        )
        runtime_kwargs["model_id"] = cls.MODEL_NAME
        return runtime_kwargs

    def _prediction_runtime_overrides(
        self,
        kwargs: Mapping[str, Any],
        *,
        fps: int | None,
    ) -> dict[str, Any]:
        overrides = super()._prediction_runtime_overrides(kwargs, fps=fps)
        aliases = {
            "chunks": "num_chunks",
            "rounds": "num_chunks",
            "trajectory": "camera_trajectory",
            "camera": "camera_trajectory",
            "sampling_steps": "sample_steps",
        }
        direct = {
            "num_chunks",
            "camera_trajectory",
            "camera_translation_step",
            "camera_rotation_step_degrees",
            "negative_prompt",
            "width",
            "height",
        }
        for key, value in kwargs.items():
            canonical = aliases.get(key, key)
            if value is not None and canonical in direct:
                overrides[canonical] = value
        return overrides

    def _apply_prediction_runtime_overrides(self, overrides: Mapping[str, Any]) -> None:
        changed = any(self.runtime_kwargs.get(key) != value for key, value in overrides.items())
        if changed and self.generator is not None:
            close = getattr(self.generator, "close", None)
            if callable(close):
                close()
        super()._apply_prediction_runtime_overrides(overrides)

    def close(self) -> None:
        generator = self.generator
        self.generator = None
        if generator is not None:
            close = getattr(generator, "close", None)
            if callable(close):
                close()


class EchoMemoryContextK1Synthesis(EchoMemorySynthesis):
    MODEL_NAME = "echo-memory-context-k1"


__all__ = [
    "EchoMemoryContextK1Synthesis",
    "EchoMemorySynthesis",
]
