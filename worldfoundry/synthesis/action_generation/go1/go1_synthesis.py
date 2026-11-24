"""WorldFoundry synthesis facade for GO-1."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class GO1Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct GO-1 action generation."""

    MODEL_ID = "go1"

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
            "images",
            "cam_high",
            "cam_right_wrist",
            "cam_left_wrist",
            "top",
            "right",
            "left",
            "image",
            "wrist_image",
            "control_frequency",
            "ctrl_freq",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["GO1Synthesis"]
