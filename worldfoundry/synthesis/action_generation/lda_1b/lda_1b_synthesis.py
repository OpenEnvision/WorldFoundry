"""WorldFoundry synthesis facade for LDA-1B."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class LDA1BSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct LDA-1B action generation."""

    MODEL_ID = "lda-1b"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key in (
            "state",
            "proprio",
            "joint_state",
            "robot_state",
            "images",
            "vision",
            "ego_view",
            "video.ego_view",
            "image",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["LDA1BSynthesis"]
