"""WorldFoundry synthesis facade for Spirit-v1.5."""

from collections.abc import Mapping

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class SpiritV15Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed Spirit-v1.5 action generation."""

    MODEL_ID = "spirit-v1.5"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key, value in kwargs.items():
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None


__all__ = ["SpiritV15Synthesis"]
