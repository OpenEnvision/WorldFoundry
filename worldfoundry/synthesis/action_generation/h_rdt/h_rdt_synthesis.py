"""WorldFoundry synthesis facade for H-RDT."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class HRDTSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct H-RDT action generation."""

    MODEL_ID = "h-rdt"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key in (
            "state",
            "agent_pos",
            "proprio",
            "robot_state",
            "joint_state",
            "image_tokens",
            "vision_tokens",
            "language_tokens",
            "lang_tokens",
            "language_attention_mask",
            "head_cam",
            "right_cam",
            "left_cam",
            "image0",
            "image1",
            "image2",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["HRDTSynthesis"]
