"""WorldFoundry synthesis facade for FastWAM."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class FastWAMSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct FastWAM action generation."""

    MODEL_ID = "fastwam"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key in (
            "state",
            "proprio",
            "agent_pos",
            "joint_state",
            "robot_state",
            "joint_action",
            "images",
            "combined_image",
            "model_image",
            "image",
            "full_image",
            "agentview_image",
            "wrist_image",
            "robot0_eye_in_hand_image",
            "head_camera",
            "left_camera",
            "right_camera",
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
            "context",
            "context_mask",
            "text_embeddings",
            "text_attention_mask",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["FastWAMSynthesis"]
