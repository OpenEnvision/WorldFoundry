"""Minimal policy interface for in-process inference."""

import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations."""

    def reset(self) -> None:  # noqa: B027 - Optional hook with a no-op default.
        """Reset the policy to its initial state."""
