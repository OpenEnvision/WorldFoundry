"""Longvie visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...operators.longvie_operator import LongVieOperator
from ...synthesis.visual_generation.longvie.longvie_synthesis import LongVieSynthesis
from ..pipeline_utils import PipelineABC


class LongViePipeline(PipelineABC):
    """WorldFoundry pipeline for LongVie controllable long-video generation."""

    MODEL_ID = "longvie-1"

    def __init__(
        self,
        model_id: str | None = None,
        operator: LongVieOperator | None = None,
        synthesis_model: LongVieSynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.operator = operator or LongVieOperator()
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LongViePipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            options.update(model_path)
        elif model_path is not None:
            options["longvie_weight_dir"] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(options.pop("model_id", None) or model_id or cls.MODEL_ID)
        synthesis_model = LongVieSynthesis.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
            model_id=resolved_model_id,
        )
        return cls(
            model_id=resolved_model_id,
            operator=LongVieOperator(),
            synthesis_model=synthesis_model,
            device=device,
        )

    def process(
        self,
        *,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        ref_image_path: str | Path | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        self.operator.get_interaction(interactions)
        try:
            processed_interaction = self.operator.process_interaction()["processed_interactions"]
        finally:
            self.operator.delete_last_interaction()
        perception = self.operator.process_perception(
            images=images,
            video=video,
            interactions=processed_interaction,
            ref_image_path=ref_image_path,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        return {
            "prompt": prompt or kwargs.get("text") or kwargs.get("instruction") or "",
            **perception,
        }

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = 16,
        return_dict: bool = False,
        ref_image_path: str | Path | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("LongVie synthesis model is not loaded. Use from_pretrained() first.")
        processed = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            ref_image_path=ref_image_path,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["input_image"],
            video={
                "dense_video": processed["dense_video"],
                "sparse_video": processed["sparse_video"],
            },
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result.get("video") or result.get("artifact_path") or result

    def stream(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = 16,
        return_dict: bool = False,
        ref_image_path: str | Path | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        *,
        reset_memory: bool = False,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate the next queued 81-frame segment using resident state.

        LongVie is a full diffusion segment generator, not a low-latency key
        controller. The previous final frame, eight-frame history, and noise
        are reused automatically; callers provide new dense/sparse control
        videos for every continuation segment.
        """
        if self.synthesis_model is None:
            raise RuntimeError("LongVie synthesis model is not loaded. Use from_pretrained() first.")
        if reset_memory:
            self.synthesis_model.reset_memory()
        elif self.synthesis_model.last_frame is None:
            raise RuntimeError(
                "LongVie EXTEND requires a completed segment in this resident pipeline. "
                "Run the first segment before extending; a newly started process cannot "
                "reconstruct the official eight-frame/noise continuation state."
            )
        kwargs["continue_from_memory"] = not reset_memory
        return self(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            return_dict=return_dict,
            ref_image_path=ref_image_path,
            operator_kwargs=operator_kwargs,
            num_frames=num_frames,
            **kwargs,
        )

    def reset_memory(self) -> None:
        """Reset memory for LongViePipeline."""
        if self.synthesis_model is not None:
            self.synthesis_model.reset_memory()

    def stream_state_ready(self) -> bool:
        """Return whether an honest history/noise continuation is available."""

        return bool(self.synthesis_model is not None and self.synthesis_model.last_frame is not None)


class LongVie1Pipeline(LongViePipeline):
    """Pipeline implementation for LongVie1 visual generation."""

    MODEL_ID = "longvie-1"


class LongVie2Pipeline(LongViePipeline):
    """LongVie 2 adapter using the shared LongVie official runtime port."""

    MODEL_ID = "longvie-2"
