"""Native WorldFoundry pipelines for the released Matrix-Game 3.5 base models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_synthesis import (
    MatrixGame35FirstPersonSynthesis,
    MatrixGame35Synthesis,
    MatrixGame35ThirdPersonSynthesis,
)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, str) and value == ""):
            return value
    return None


class MatrixGame35Pipeline(PipelineABC):
    """Shared native input contract; subclasses permanently select one checkpoint."""

    ABSTRACT_PIPELINE = True
    MODEL_ID: ClassVar[str]
    SYNTHESIS_CLS: ClassVar[type[MatrixGame35Synthesis]]
    generation_type = "camera_trajectory_image_to_video"

    def __init__(
        self,
        *,
        synthesis_model: MatrixGame35Synthesis | None = None,
        device: str = "cuda",
    ) -> None:
        if not getattr(self, "MODEL_ID", None):
            raise TypeError("Instantiate a model-specific MatrixGame35Pipeline subclass")
        super().__init__(model_id=self.MODEL_ID, synthesis_model=synthesis_model, device=device)
        self.model_name = self.MODEL_ID

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> "MatrixGame35Pipeline":
        options: dict[str, Any] = {}
        checkpoint_path = model_path
        if isinstance(model_path, Mapping):
            options.update(model_path)
            checkpoint_path = None
        elif isinstance(model_path, str) and not model_path.strip():
            checkpoint_path = None
        nested_components = options.pop("required_components", None)
        if nested_components is not None:
            if not isinstance(nested_components, Mapping):
                raise TypeError("required_components must be a mapping")
            options.update(nested_components)
        options.update(required_components or {})
        options.update(kwargs)

        requested_model = str(options.get("model_id") or model_id or cls.MODEL_ID)
        if requested_model != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} is bound to {cls.MODEL_ID!r}, got {requested_model!r}")
        options["model_id"] = cls.MODEL_ID
        synthesis = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=checkpoint_path,
            device=device,
            lazy=lazy,
            generator_overrides=options,
        )
        return cls(synthesis_model=synthesis, device=device)

    def preflight(self) -> dict[str, Any]:
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.MODEL_ID} synthesis is not initialized")
        return self.synthesis_model.preflight()

    def process(
        self,
        *,
        prompt: str | None = None,
        images: Any = None,
        image: Any = None,
        image_path: str | Path | None = None,
        input_path: str | Path | None = None,
        video: Any = None,
        interactions: Any = None,
        camera_path: Any = None,
        camera: Any = None,
        trajectory_npz: Any = None,
        refs: Any = None,
        subject_refs: Any = None,
        ref_image_path: Any = None,
        caption_path: str | Path | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if video is not None:
            raise ValueError("Matrix-Game 3.5 consumes an anchor image, not an input video")
        merged = dict(operator_kwargs or {})
        merged.update(kwargs)
        if isinstance(interactions, Mapping):
            for key, value in interactions.items():
                merged.setdefault(str(key), value)

        anchor = _first_present(
            images,
            image,
            image_path,
            input_path,
            merged.pop("image", None),
            merged.pop("image_path", None),
            merged.pop("input_path", None),
            merged.pop("source_image_path", None),
        )
        resolved_camera = _first_present(
            camera_path,
            camera,
            trajectory_npz,
            merged.pop("camera_path", None),
            merged.pop("camera", None),
            merged.pop("trajectory_npz", None),
            merged.pop("camera_npz", None),
            merged.pop("trajectory_path", None),
            merged.pop("pose_path", None),
        )
        if resolved_camera is None and isinstance(interactions, (str, Path)):
            resolved_camera = interactions
        resolved_refs = _first_present(
            refs,
            subject_refs,
            ref_image_path,
            merged.pop("refs", None),
            merged.pop("subject_refs", None),
            merged.pop("ref_image_path", None),
            merged.pop("reference_images", None),
            merged.pop("protagonist_refs", None),
        )
        resolved_caption = _first_present(
            caption_path,
            merged.pop("caption_path", None),
            merged.pop("caption_json", None),
        )

        if anchor is None:
            raise ValueError(f"{self.MODEL_ID} requires an anchor image")
        if resolved_camera is None:
            raise ValueError(f"{self.MODEL_ID} requires camera_path/trajectory_npz with extrinsics and intrinsics")
        if not prompt and resolved_caption is None:
            raise ValueError(f"{self.MODEL_ID} requires prompt or caption_path")
        return {
            "prompt": str(prompt or ""),
            "images": anchor,
            "camera_path": resolved_camera,
            "refs": resolved_refs,
            "caption_path": resolved_caption,
            **merged,
        }

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        image: Any = None,
        image_path: str | Path | None = None,
        input_path: str | Path | None = None,
        video: Any = None,
        interactions: Any = None,
        camera_path: Any = None,
        camera: Any = None,
        trajectory_npz: Any = None,
        refs: Any = None,
        subject_refs: Any = None,
        ref_image_path: Any = None,
        caption_path: str | Path | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        num_blocks: int | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        camera_convention: str | None = None,
        keep_workspace: bool | None = None,
        timeout_seconds: float | None = None,
        extra_args: Any = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.MODEL_ID} synthesis is not initialized")
        processed = self.process(
            prompt=prompt,
            images=images,
            image=image,
            image_path=image_path,
            input_path=input_path,
            video=video,
            interactions=interactions,
            camera_path=camera_path,
            camera=camera,
            trajectory_npz=trajectory_npz,
            refs=refs,
            subject_refs=subject_refs,
            ref_image_path=ref_image_path,
            caption_path=caption_path,
            operator_kwargs=operator_kwargs,
            num_blocks=num_blocks,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            camera_convention=camera_convention,
            keep_workspace=keep_workspace,
            timeout_seconds=timeout_seconds,
            extra_args=extra_args,
            **kwargs,
        )
        result = self.synthesis_model.predict(
            output_path=output_path,
            fps=fps,
            **processed,
        )
        return result if return_dict else result["artifact_path"]

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Expose multi-block official rollout through the common stream surface."""

        return self(*args, **kwargs)

    def close(self) -> None:
        if self.synthesis_model is not None:
            self.synthesis_model.close()


class MatrixGame35FirstPersonPipeline(MatrixGame35Pipeline):
    MODEL_ID = "matrix-game-3.5-first-person"
    SYNTHESIS_CLS = MatrixGame35FirstPersonSynthesis


class MatrixGame35ThirdPersonPipeline(MatrixGame35Pipeline):
    MODEL_ID = "matrix-game-3.5-third-person"
    SYNTHESIS_CLS = MatrixGame35ThirdPersonSynthesis


__all__ = [
    "MatrixGame35FirstPersonPipeline",
    "MatrixGame35Pipeline",
    "MatrixGame35ThirdPersonPipeline",
]
