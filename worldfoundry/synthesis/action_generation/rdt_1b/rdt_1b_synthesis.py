"""WorldFoundry synthesis facade for RDT-1B."""

from collections.abc import Mapping

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class RDT1BSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed RDT-1B action generation."""

    MODEL_ID = "rdt-1b"

    @staticmethod
    def _select_observation(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key, value in kwargs.items():
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None


__all__ = ["RDT1BSynthesis"]
