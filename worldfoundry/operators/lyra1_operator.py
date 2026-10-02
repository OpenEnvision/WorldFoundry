"""Module for the Lyra1 operator implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from worldfoundry.core.io.image import load_pil_image

from .base_operator import BaseOperator


class Lyra1Operator(BaseOperator):
    """Collapse WorldFoundry actions into a single Lyra-1 trajectory."""

    OFFICIAL_TRAJECTORIES = [
        "left",
        "right",
        "up",
        "down",
        "zoom_in",
        "zoom_out",
        "clockwise",
        "counterclockwise",
    ]

    ACTION_TO_TRAJECTORY = {
        "forward": "zoom_in",
        "backward": "zoom_out",
        "left": "left",
        "right": "right",
        "camera_l": "counterclockwise",
        "camera_r": "clockwise",
        "camera_up": "up",
        "camera_down": "down",
        "camera_zoom_in": "zoom_in",
        "camera_zoom_out": "zoom_out",
    }

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types
            or [
                "textual_instruction",
                "action_instruction",
                "visual_instruction",
            ]
        )
        self.interaction_template = interaction_template or sorted(
            set(self.OFFICIAL_TRAJECTORIES) | set(self.ACTION_TO_TRAJECTORY.keys())
        )
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        candidate = self._normalize_interaction_candidate(interaction)
        if candidate["trajectory"] is None:
            raise ValueError(
                f"{candidate['action']} not in template. Available: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, (list, tuple)):
            normalized = [self._normalize_interaction(item) for item in interaction]
        else:
            normalized = [self._normalize_interaction(interaction)]
        self.current_interaction.append(normalized)

    def process_interaction(self, prompt: str = "") -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        latest_interaction = self.current_interaction[-1]
        if len(latest_interaction) == 0:
            raise ValueError("Interaction list is empty.")
        self.interaction_history.append(latest_interaction)

        selected_trajectory = self._select_trajectory(latest_interaction)
        actions = [item["action"] for item in latest_interaction]
        mapped_trajectories = [item["trajectory"] for item in latest_interaction]

        captions = [
            item["caption"]
            for item in latest_interaction
            if item["caption"] and item["caption"].strip()
        ]
        selected_caption = captions[-1] if captions else (prompt or "")

        return {
            "actions": actions,
            "mapped_trajectories": mapped_trajectories,
            "trajectory": selected_trajectory,
            "trajectory_prompt": selected_caption,
        }

    def process_perception(
        self,
        images=None,
        videos=None,
        mode: str = "static",
    ):
        """Process perception inputs like images, videos, and reference frames."""
        mode = str(mode).lower()
        if mode == "static":
            if images is None:
                raise ValueError("Lyra-1 static mode expects an input image.")
            return load_pil_image(images)
        if mode == "dynamic":
            if videos is None:
                raise ValueError("Lyra-1 dynamic mode expects an input video or VIPE path.")
            return videos
        raise ValueError(f"Unsupported Lyra-1 mode: {mode}")

    def _normalize_interaction_candidate(self, interaction: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize interaction candidate implementation."""
        if isinstance(interaction, str):
            action = interaction
            caption = ""
            explicit_trajectory = (
                action
                if action in self.OFFICIAL_TRAJECTORIES and action not in self.ACTION_TO_TRAJECTORY
                else None
            )
        elif isinstance(interaction, dict):
            explicit_trajectory = interaction.get("trajectory")
            action = (
                interaction.get("action")
                or interaction.get("signal")
                or interaction.get("interaction")
                or explicit_trajectory
            )
            caption = (
                interaction.get("caption")
                or interaction.get("prompt")
                or interaction.get("text_prompt")
                or ""
            )
        else:
            raise TypeError(f"Unsupported interaction type: {type(interaction)}")

        explicit_trajectory = (
            explicit_trajectory if explicit_trajectory in self.OFFICIAL_TRAJECTORIES else None
        )
        trajectory = explicit_trajectory or self.ACTION_TO_TRAJECTORY.get(action)
        return {
            "action": action,
            "caption": caption,
            "trajectory": trajectory,
            "explicit": explicit_trajectory is not None,
        }

    def _normalize_interaction(self, interaction: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize interaction implementation."""
        candidate = self._normalize_interaction_candidate(interaction)
        self.check_interaction(interaction)
        return candidate

    def _select_trajectory(self, items: Sequence[Dict[str, Any]]) -> str:
        """Select trajectory implementation."""
        explicit = [item["trajectory"] for item in items if item["explicit"]]
        if explicit:
            return explicit[0]

        counts: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        for idx, item in enumerate(items):
            trajectory = item["trajectory"]
            counts[trajectory] = counts.get(trajectory, 0) + 1
            first_seen.setdefault(trajectory, idx)

        ranked = sorted(
            counts.items(),
            key=lambda pair: (-pair[1], first_seen[pair[0]]),
        )
        return ranked[0][0]
