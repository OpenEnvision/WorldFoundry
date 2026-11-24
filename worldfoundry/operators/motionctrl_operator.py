"""Module for the MotionCtrl operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class MotionCtrlOperator(BaseOperator):
    """Operator for MotionCtrl camera/object motion-conditioned generation."""

    MODEL_ID = "motionctrl"
    CAMERA_MOTION_ALIASES: ClassVar[set[str]] = {
        "backward",
        "camera",
        "camera-motion",
        "camera_motion",
        "down",
        "forward",
        "left",
        "orbit",
        "pan",
        "right",
        "rotate",
        "rotate-left",
        "rotate-right",
        "tilt",
        "up",
    }
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["camera_motion", "object_motion"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["camera_motion", "object_motion", "both"]
        self.interaction_template_init()

    @staticmethod
    def _normalize_motion(item: str) -> str:
        key = item.strip().replace("_", "-").casefold()
        if key in MotionCtrlOperator.CAMERA_MOTION_ALIASES:
            return "camera_motion"
        return item

    @staticmethod
    def _motions(interaction) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, str):
            text = interaction.strip()
            return [MotionCtrlOperator._normalize_motion(text)] if text else []
        if isinstance(interaction, Iterable):
            return [
                MotionCtrlOperator._normalize_motion(str(item).strip()) for item in interaction if str(item).strip()
            ]
        raise TypeError(
            f"MotionCtrl interaction must be a condition type or sequence, got {type(interaction).__name__}."
        )

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        allowed = {"camera_motion", "object_motion", "both"}
        for item in self._motions(interaction):
            if item not in allowed and not Path(item).suffix:
                raise ValueError(f"Unsupported MotionCtrl interaction {item!r}; expected condtype/path values.")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        motions = self._motions(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(motions)
        result: dict[str, Any] = {"actions": motions}
        if motions:
            first = motions[0]
            if first in {"camera_motion", "object_motion", "both"}:
                result["condtype"] = first
            else:
                result["cond_dir"] = first
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        text = prompt if prompt is not None else kwargs.get("caption")
        return {"prompt": "" if text is None else str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if video is not None:
            raise ValueError(
                "MotionCtrl official inference uses motion conditions and optional references, not an input video."
            )
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
