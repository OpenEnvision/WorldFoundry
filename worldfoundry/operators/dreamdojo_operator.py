"""Module for the DreamDojo operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .base_operator import BaseOperator


class DreamDojoOperator(BaseOperator):
    """Operator for NVIDIA DreamDojo action-conditioned robot world generation."""

    MODEL_ID = "dreamdojo"
    WORLD_MODEL_FAMILY = "action_conditioned_robot_world_model"
    ACTION_REPRESENTATION = "384d_unified_robot_action_or_latent_action_chunk"
    OBSERVATION_LAYOUT = "egocentric_robot_video_or_initial_frame_with_actions"
    DEFAULT_INPUT_SCHEMA = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": False,
        "actions": [
            "robot_action",
            "unified_384d_action",
            "action_file",
            "latent_action",
            "dataset_action_sequence",
        ],
    }
    ACTION_SPACE_DIM = 384
    ACTION_SLICES = {
        "fourier_gr1": [0, 29],
        "manus_retargeted_gr1": [29, 58],
        "unitree_g1": [58, 101],
        "bimanual_yam": [101, 147],
        "agibot": [147, 169],
        "reserved": [169, 220],
        "mano": [220, 352],
        "latent": [352, 384],
    }

    def __init__(self, input_schema: Dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["robot_action_sequence", "unified_384d_action", "dataset_action_sequence"]
        self.interaction_template_init()

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Is number implementation."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _is_numeric_vector(cls, value: Any) -> bool:
        """Is numeric vector implementation."""
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and all(cls._is_number(item) for item in value)
        )

    @classmethod
    def _normalise_action_item(cls, item: Any) -> Any:
        """Normalise action item implementation."""
        if isinstance(item, Path):
            return str(item)
        tolist = getattr(item, "tolist", None)
        if callable(tolist):
            return cls._normalise_action_item(tolist())
        if isinstance(item, Mapping):
            return {str(key): value for key, value in item.items()}
        if cls._is_numeric_vector(item):
            return list(item)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [cls._normalise_action_item(value) for value in item]
        if isinstance(item, str):
            text = item.strip()
            return text
        return item

    @classmethod
    def _extract_actions(cls, interaction: Any) -> list[Any]:
        """Extract actions implementation."""
        if interaction is None:
            return []
        if isinstance(interaction, Path):
            return [str(interaction)]
        if isinstance(interaction, str):
            text = interaction.strip()
            return [text] if text else []
        if isinstance(interaction, Mapping):
            for key in (
                "actions",
                "action",
                "action_sequence",
                "robot_action_sequence",
                "robot_actions",
                "unified_384d_action",
                "latent_action_sequence",
                "dataset_action_sequence",
            ):
                if key in interaction:
                    return cls._extract_actions(interaction[key])
            for key in ("action_file", "actions_path", "trajectory_file", "dataset_path"):
                if interaction.get(key):
                    return [str(interaction[key])]
            return [dict(interaction)]
        if cls._is_numeric_vector(interaction):
            return [list(interaction)]
        if isinstance(interaction, Iterable):
            actions: list[Any] = []
            for item in interaction:
                normalised = cls._normalise_action_item(item)
                if normalised in ("", None):
                    continue
                actions.append(normalised)
            return actions
        raise TypeError(
            f"DreamDojo interaction must be robot actions, action-file paths, or controls; got {type(interaction).__name__}."
        )

    @staticmethod
    def _controls(interaction: Any) -> dict[str, Any]:
        """Controls implementation."""
        return {str(key): value for key, value in interaction.items()} if isinstance(interaction, Mapping) else {}

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._extract_actions(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = self._extract_actions(raw)
        controls = self._controls(raw)
        self.interaction_history.append(actions)
        return {
            "actions": actions,
            "robot_action_sequence": actions,
            "world_model_controls": controls,
            "action_contract": {
                "kind": "dreamdojo_unified_robot_action_space",
                "dimension": self.ACTION_SPACE_DIM,
                "slices": self.ACTION_SLICES,
                "default_robot": controls.get("robot") or controls.get("embodiment") or "gr1",
            },
            "operator_metadata": {
                "model_id": self.MODEL_ID,
                "world_model_family": self.WORLD_MODEL_FAMILY,
                "action_representation": self.ACTION_REPRESENTATION,
                "observation_layout": self.OBSERVATION_LAYOUT,
            },
        }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        text = prompt if prompt is not None else kwargs.get("instruction", kwargs.get("task", kwargs.get("caption")))
        prompt_text = "" if text is None else str(text)
        return {
            "prompt": prompt_text,
            "prompt_channels": {
                "language_instruction": prompt_text,
                "task": kwargs.get("task"),
                "caption": kwargs.get("caption"),
            },
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        observation = {
            "initial_image": images,
            "video_context": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "dataset_path": kwargs.get("dataset_path"),
            "episode_id": kwargs.get("episode_id"),
            "camera_layout": kwargs.get("camera_layout", "egocentric"),
        }
        return {
            "images": images,
            "video": video,
            "ref_image_path": observation["ref_image_path"],
            "dreamdojo_observation": observation,
            "extra_inputs": dict(kwargs),
        }
