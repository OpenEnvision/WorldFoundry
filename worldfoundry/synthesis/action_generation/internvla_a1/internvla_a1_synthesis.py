"""WorldFoundry synthesis facade for InternVLA-A1."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class InternVLAA1Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct InternVLA-A1 action generation."""

    MODEL_ID = "internvla-a1"

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
            "cam_left_wrist",
            "cam_right_wrist",
            "head_camera",
            "left_camera",
            "right_camera",
            "reset",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["InternVLAA1Synthesis"]
