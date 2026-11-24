"""Module for the LongVie operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base_operator import BaseOperator


class LongVieOperator(BaseOperator):
    """Normalize LongVie prompt, first-frame, and control-video inputs."""

    def __init__(self, operation_types=None) -> None:
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=operation_types or ["textual_instruction", "visual_instruction"])
        self.interaction_template = ["text_prompt", "dense_video", "sparse_video"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction is None:
            return True
        if isinstance(interaction, (str, list, tuple, Mapping)):
            return True
        raise TypeError(f"Unsupported LongVie interaction type: {type(interaction).__name__}")

    def process_interaction(self):
        """Process the recorded interactions and return the generated actions."""
        if not self.current_interaction:
            return {"processed_interactions": None}
        interaction = self.current_interaction[-1]
        self.interaction_history.append(interaction)
        return {"processed_interactions": interaction}

    @staticmethod
    def _pick(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def process_perception(
        self,
        *,
        images=None,
        video=None,
        interactions=None,
        ref_image_path: str | Path | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        payload = dict(operator_kwargs or {})
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        interaction_map = interactions if isinstance(interactions, Mapping) else {}
        video_map = video if isinstance(video, Mapping) else {}
        input_image = self._first_present(
            self._pick(payload, "input_image", "image", "first_frame"),
            images,
            ref_image_path,
        )
        dense_video = self._first_present(
            self._pick(payload, "dense_video", "depth_video", "depth"),
            self._pick(video_map, "dense_video", "depth_video", "depth"),
            self._pick(interaction_map, "dense_video", "depth_video", "depth"),
            # Studio's conditioned-video workbench has one primary video
            # upload. LongVie assigns that explicit slot to the dense depth
            # control; the sparse pointmap/track path is supplied separately.
            video if video is not None and not isinstance(video, Mapping) else None,
        )
        sparse_video = self._first_present(
            self._pick(payload, "sparse_video", "track_video", "track", "pointmap_video"),
            self._pick(video_map, "sparse_video", "track_video", "track", "pointmap_video"),
            self._pick(interaction_map, "sparse_video", "track_video", "track", "pointmap_video"),
        )
        return {
            "input_image": input_image,
            "dense_video": dense_video,
            "sparse_video": sparse_video,
        }
