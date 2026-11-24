"""WorldFoundry synthesis facade for Mem-0."""

from collections.abc import Mapping

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class Mem0Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    MODEL_ID = "mem-0"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key, value in kwargs.items():
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None


__all__ = ["Mem0Synthesis"]
