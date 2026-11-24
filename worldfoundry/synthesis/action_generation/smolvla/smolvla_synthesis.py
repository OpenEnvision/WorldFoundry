"""WorldFoundry synthesis facade for SmolVLA."""

from collections.abc import Mapping

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class SmolVLASynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed SmolVLA action generation."""

    MODEL_ID = "smolvla"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key, value in kwargs.items():
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None


__all__ = ["SmolVLASynthesis"]
