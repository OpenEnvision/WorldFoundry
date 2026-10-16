# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Public DreamDojo distilled student loading, generation, and resident streaming."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class DreamDojoStudentPipeline(ComponentPipeline):
    """Distilled offline generation and resident normalized-action streaming."""

    MODEL_ID = "dreamdojo-student"
    MODEL_PATH_OPTION = "student_checkpoint_path"
    OPERATOR_TARGET = "worldfoundry.operators.dreamdojo_operator:DreamDojoOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.dreamdojo:DreamDojoSynthesis"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> DreamDojoStudentPipeline:
        """Configure explicit student assets; actual weight loading remains lazy."""
        return cast(
            DreamDojoStudentPipeline,
            super().from_pretrained(
                model_path,
                required_components,
                device,
                model_id,
                **{**kwargs, "variant": "student"},
            ),
        )

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        action_file: str | Path | None = None,
        output_path: str | Path | None = None,
        height: int = 480,
        width: int = 640,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        fps: int = 10,
        seed: int = 42,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Export an isolated student rollout from normalized actions or an action file."""
        if action_file is not None:
            kwargs["action_file"] = action_file
        return super().__call__(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            seed=seed,
            return_dict=return_dict,
            **kwargs,
        )

    def stream(
        self,
        prompt: str = "",
        images: Any = None,
        interactions: Any = None,
        action_file: str | Path | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield generated 13/4-frame chunks from an isolated normalized-action rollout."""
        if action_file is not None:
            kwargs["action_file"] = action_file
        return self.synthesis_model.stream_student(
            prompt=prompt,
            images=images,
            interactions=interactions,
            **kwargs,
        )

    def prepare_realtime(self) -> dict[str, Any]:
        """Load the resident student and advertise its 13/4-frame cadence."""
        return self.synthesis_model.prepare_realtime()

    def configure_realtime(self, images: Any, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        """Seed the student session without generating or recording synthetic controls."""
        return self.synthesis_model.configure_realtime(images, prompt, **kwargs)

    def stream_realtime(self, interactions: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Advance using normalized robot actions; keyboard controls are not robot actions."""
        return self.synthesis_model.stream_realtime(interactions, **kwargs)

    def realtime_next_output_frames(self) -> int:
        """Return the native output budget for the next STEP."""
        return self.synthesis_model.realtime_next_output_frames()

    def reset_realtime(self) -> None:
        """Discard session history while retaining student weights."""
        self.synthesis_model.reset_realtime()
