"""DualCamCtrl operator for image/depth/camera conditioned video generation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class DualCamCtrlOperator(BaseOperator):
    """Normalize DualCamCtrl prompt, image, depth image, and camera path inputs."""

    MODEL_ID = "dualcamctrl"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["camera_trajectory", "depth_image"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["camera_path", "trajectory_file"]
        self.interaction_template_init()

    @staticmethod
    def _trajectories(interaction: Any) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, (str, Path)):
            text = str(interaction).strip()
            return [text] if text else []
        if isinstance(interaction, Iterable):
            return [str(item).strip() for item in interaction if str(item).strip()]
        raise TypeError(f"DualCamCtrl interaction must be a camera path or sequence, got {type(interaction).__name__}.")

    def check_interaction(self, interaction: Any) -> bool:
        self._trajectories(interaction)
        return True

    def get_interaction(self, interaction: Any) -> None:
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        trajectories = self._trajectories(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(trajectories)
        result: dict[str, Any] = {"actions": trajectories}
        if trajectories:
            result["camera_path"] = trajectories[0]
            result["trajectory_file"] = trajectories[0]
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        text = prompt if prompt is not None else kwargs.get("caption")
        return {"prompt": "" if text is None else str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if video is not None:
            raise ValueError(
                "DualCamCtrl consumes a still image, a depth image, and a camera trajectory, not input video."
            )
        depth_image = kwargs.pop("depth_image", None) or kwargs.pop("depth_path", None)
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "depth_image": str(depth_image) if depth_image is not None else None,
            "extra_inputs": dict(kwargs),
        }


__all__ = ["DualCamCtrlOperator"]
