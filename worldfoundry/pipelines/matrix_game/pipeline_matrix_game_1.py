"""Matrix Game 1 visual generation pipeline module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...synthesis.visual_generation.matrix_game.matrix_game_1_synthesis import (
    MatrixGame1Synthesis,
)
from ..pipeline_utils import PipelineABC


class MatrixGame1Pipeline(PipelineABC):
    """WorldFoundry Matrix-Game-1 pipeline backed by the in-tree official runner."""

    MODEL_ID = "matrix-game-1"
    generation_type = "image_to_video"

    def __init__(
        self,
        synthesis_model: MatrixGame1Synthesis | None = None,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        """Initialize a Matrix-Game-1 wrapper without loading heavy model weights.

        Args:
            synthesis_model: Lightweight synthesis/preflight wrapper.
            device: Target device label preserved in runtime metadata.
            model_id: Optional catalog/runtime identifier override.
        """
        self.model_id = model_id or self.MODEL_ID
        self.synthesis_model = synthesis_model or MatrixGame1Synthesis(device=device)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> MatrixGame1Pipeline:
        """Create the Matrix-Game-1 wrapper and preserve local resource paths.

        Args:
            model_path: Optional checkpoint path or mapping from dispatch/model-zoo.
            required_components: Optional component paths such as ``conda_dir``.
            device: Target device label preserved in metadata.
            model_id: Optional catalog/runtime identifier override.
            **kwargs: Additional preflight metadata.
        """
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
            checkpoint_dir = (
                options.pop("checkpoint_dir", None)
                or options.pop("pretrained_model_path", None)
                or options.pop("model_path", None)
            )
        else:
            checkpoint_dir = model_path
        options.update(required_components or {})
        options.update(kwargs)
        synthesis_model = MatrixGame1Synthesis.from_pretrained(
            checkpoint_dir,
            device=device,
            checkpoint_dir=checkpoint_dir,
            conda_dir=options.get("conda_dir") or options.get("python_env_dir"),
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            model_id=model_id or str(options.get("model_id") or cls.MODEL_ID),
        )

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        image_path: str | Path | None = None,
        input_path: str | Path | None = None,
        source_image_path: str | Path | None = None,
        video: Any = None,
        interactions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Normalize Matrix-Game-1 inputs for the official runner.

        Args:
            prompt: Optional text prompt.
            images: Optional conditioning image.
            image_path: Optional filesystem conditioning image path.
            input_path: Optional Studio input path fallback.
            source_image_path: Optional source image path fallback.
            video: Optional conditioning video.
            interactions: Optional keyboard or mouse tokens.
            **kwargs: Additional metadata preserved in the plan.
        """
        resolved_image_path = (
            image_path
            or input_path
            or source_image_path
            or kwargs.pop("image_path", None)
            or kwargs.pop("input_path", None)
            or kwargs.pop("source_image_path", None)
        )
        if resolved_image_path is None and isinstance(images, (str, Path)):
            resolved_image_path = images
        return {
            "prompt": prompt or "",
            "images": images,
            "image_path": None if resolved_image_path is None else str(resolved_image_path),
            "input_path": None if resolved_image_path is None else str(resolved_image_path),
            "video": video,
            "interactions": tuple(interactions or ()),
            **kwargs,
        }

    def preflight(self) -> dict[str, Any]:
        """Return Matrix-Game-1 readiness without importing heavy dependencies.

        Args:
            None.
        """
        if self.synthesis_model is None:
            raise RuntimeError("MatrixGame1Pipeline synthesis_model is not initialized.")
        return self.synthesis_model.preflight()

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        image_path: str | Path | None = None,
        input_path: str | Path | None = None,
        source_image_path: str | Path | None = None,
        video: Any = None,
        interactions: Sequence[str] | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Run Matrix-Game-1 or raise a clear runtime error when assets are incomplete.

        Args:
            prompt: Optional text prompt.
            images: Optional conditioning image.
            image_path: Optional filesystem conditioning image path.
            input_path: Optional Studio input path fallback.
            source_image_path: Optional source image path fallback.
            video: Optional conditioning video.
            interactions: Optional keyboard or mouse tokens.
            output_path: Optional execution report JSON path.
            fps: Optional requested output FPS.
            return_dict: Return the full plan payload when true.
            **kwargs: Additional metadata preserved in the plan.
        """
        if self.synthesis_model is None:
            raise RuntimeError("MatrixGame1Pipeline synthesis_model is not initialized.")
        processed = self.process(
            prompt=prompt,
            images=images,
            image_path=image_path,
            input_path=input_path,
            source_image_path=source_image_path,
            video=video,
            interactions=interactions,
            **kwargs,
        )
        result = self.synthesis_model.predict(
            output_path=output_path,
            fps=fps,
            **processed,
        )
        if return_dict:
            return result
        artifact_path = result.get("artifact_path") if isinstance(result, dict) else None
        if artifact_path:
            return artifact_path
        return result["status"]

    def stream(
        self,
        images: Any = None,
        image_path: str | Path | None = None,
        input_path: str | Path | None = None,
        source_image_path: str | Path | None = None,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Expose Matrix-Game-1 execution through the stream API.

        Args:
            images: Optional conditioning image.
            image_path: Optional filesystem conditioning image path.
            input_path: Optional Studio input path fallback.
            source_image_path: Optional source image path fallback.
            interactions: Optional keyboard or mouse tokens.
            prompt: Optional text prompt.
            output_path: Optional execution report JSON path.
            fps: Optional requested output FPS.
            return_dict: Return the full plan payload when true.
            **kwargs: Additional metadata preserved in the plan.
        """
        return self(
            prompt=prompt,
            images=images,
            image_path=image_path,
            input_path=input_path,
            source_image_path=source_image_path,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = ["MatrixGame1Pipeline"]
