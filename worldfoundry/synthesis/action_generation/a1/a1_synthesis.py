"""WorldFoundry synthesis facade for the in-tree A1 policy."""

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class A1Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    MODEL_ID = "a1"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key, value in kwargs.items():
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None


__all__ = ["A1Synthesis"]
