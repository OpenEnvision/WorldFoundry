"""WorldFoundry synthesis facade for Galaxea G0Plus."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class GalaxeaVLASynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct G0Plus action generation."""

    MODEL_ID = "galaxea-vla"

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
            "vision",
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
            "image",
            "wrist_image",
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["GalaxeaVLASynthesis"]
