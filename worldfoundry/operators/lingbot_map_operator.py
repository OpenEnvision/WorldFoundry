"""Module for the LingBotMap operator implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base_operator import BaseOperator


class LingBotMapOperator(BaseOperator):
    """Input normalizer for LingBot-Map 3D reconstruction."""

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=operation_types or ["visual_instruction", "reconstruction_mode"])
        self.interaction_template = interaction_template or [
            "streaming_reconstruction",
            "windowed_reconstruction",
            "camera_pose_estimation",
            "depth_estimation",
            "point_cloud_generation",
        ]
        self.interaction_template_init()

    def process_perception(self, input_signal: Any):
        """Process perception inputs like images, videos, and reference frames."""
        if isinstance(input_signal, (str, os.PathLike)):
            path = Path(input_signal).expanduser()
            if path.is_dir():
                image_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
                return sorted(str(item) for item in path.iterdir() if item.suffix.lower() in image_exts)
            if path.is_file() and path.suffix.lower() == ".txt":
                return [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        return input_signal

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction {interaction!r} not in interaction_template: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if interaction is None:
            return
        if isinstance(interaction, (list, tuple)):
            for item in interaction:
                if item is not None:
                    self.get_interaction(str(item))
            return
        self.check_interaction(str(interaction))
        self.current_interaction.append(str(interaction))

    def process_interaction(self, **kwargs):
        """Process the recorded interactions and return the generated actions."""
        mode = kwargs.get("mode")
        if self.current_interaction:
            latest = self.current_interaction[-1]
            self.interaction_history.append(latest)
            if latest == "windowed_reconstruction":
                mode = "windowed"
            elif latest == "streaming_reconstruction":
                mode = "streaming"
        return {"mode": mode} if mode else {}

    def delete_last_interaction(self):
        """Clear the entire current interaction list (not just the last item).

        ``BaseOperator.delete_last_interaction`` pops one entry and raises
        ``ValueError`` when the list is empty. This override instead empties
        the list: ``LingBotMapPipeline.process`` records one or more template
        tokens in a single ``get_interaction`` call, then always invokes this
        method in ``finally`` to reset per-request state. Raising on empty, or
        popping only the last token, would leak leftover modes across calls
        or break the cleanup path. An already-empty list is a no-op.
        """
        if self.current_interaction:
            self.current_interaction = []
