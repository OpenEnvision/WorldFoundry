"""WorldFoundry pipeline for ABot-World inference."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

from ..pipeline_utils import PipelineABC


class ABotWorldPipeline(PipelineABC):
    """Image-and-action to interactive world-video pipeline."""

    MODEL_ID = "abot-world-0-5b-lf"
    MODEL_PATH_OPTION = "checkpoint_dir"

    def __init__(
        self,
        runtime: Any,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            model_id=model_id,
            synthesis_model=runtime,
            memory_module=None,
            device=device,
        )
        self.runtime = runtime

    @staticmethod
    def _checkpoint_path(value: Any) -> Any:
        if isinstance(value, Mapping):
            return (
                value.get("path")
                or value.get("local_path")
                or value.get("checkpoint_dir")
                or value.get("checkpoint_source")
            )
        return value

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "ABotWorldPipeline":
        from worldfoundry.synthesis.visual_generation.abot_world import ABotWorldInference

        options = cls._runtime_options(model_path, required_components, kwargs)
        checkpoint_candidates = (
            options.pop("checkpoint_dir", None),
            options.pop("checkpoint_source", None),
            options.pop("repo_root", None),
        )
        checkpoint = cls._checkpoint_path(next((value for value in checkpoint_candidates if value), None))
        if checkpoint is None:
            raise ValueError("ABotWorldPipeline.from_pretrained requires a local checkpoint directory")
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        cls._strip_framework_loading_options(options)
        parameters = inspect.signature(ABotWorldInference).parameters
        runtime_options = {
            key: value for key, value in options.items() if key in parameters
        }
        runtime = ABotWorldInference(
            checkpoint,
            device=device,
            **runtime_options,
        )
        return cls(
            runtime,
            model_id=resolved_model_id,
            device=device,
        )

    def __call__(
        self,
        images: Any,
        prompt: str = "",
        interactions: Any = None,
        reference_images: Any = None,
        num_frames: int | None = 57,
        num_blocks: int | None = None,
        seed: int = 42,
        fps: int = 16,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **_: Any,
    ) -> Any:
        video = self.runtime.generate(
            images,
            prompt=prompt,
            actions=interactions,
            reference_images=reference_images,
            seed=seed,
            num_frames=num_frames,
            num_blocks=num_blocks,
        )
        artifact_path = None
        if output_path is not None:
            artifact_path = save_image_or_video_tensor(
                video.permute(1, 0, 2, 3),
                output_path,
                fps=int(fps),
                value_range=(0.0, 1.0),
            )
        result = {
            "video": video,
            "artifact_path": artifact_path,
            "fps": int(fps),
            "model_id": self.model_id,
        }
        if return_dict:
            return result
        return artifact_path or video

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        reference_images: Any = None,
        seed: int = 42,
        **_: Any,
    ) -> dict[str, Any]:
        return self.runtime.configure(
            images,
            prompt=prompt,
            reference_images=reference_images,
            seed=seed,
        )

    def stream_realtime(
        self,
        interactions: Sequence[str] | Mapping[str, Any] | str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        frames = self.runtime.step(interactions)
        return {"frames": frames, "fps": 16, "model_id": self.model_id}

    def reset_realtime(self) -> None:
        self.runtime.reset()


__all__ = ["ABotWorldPipeline"]
