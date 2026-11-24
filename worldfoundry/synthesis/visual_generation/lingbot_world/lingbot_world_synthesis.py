"""
Provides a synthesis adapter for the LingBot runtime, allowing integration into
a broader synthesis framework.

This module primarily defines the `LingBotSynthesis` class, which wraps a
`LingBotWorldRuntime` instance to provide a consistent interface for
performing synthesis tasks, such as predictions. It also re-exports
key components and constants from the `lingbot_world` module for convenience.
"""

from typing import Any

from ...base_synthesis import BaseSynthesis
from .runtime import (
    DEFAULT_LINGBOT_ACT_REPO,
    DEFAULT_LINGBOT_BASE_REPO,
    DEFAULT_LINGBOT_FAST_REPO,
    DEFAULT_LINGBOT_HFD_ROOT,
    SUPPORTED_LINGBOT_TASKS,
    LingBotRuntime,
    LingBotWorldRuntime,
    lingbot_runtime_root,
    load_runtime,
)


class LingBotSynthesis(BaseSynthesis):
    """Thin synthesis adapter around the canonical base-model LingBot runtime."""

    def __init__(self, runtime: LingBotWorldRuntime):
        """
        Initializes the LingBotSynthesis adapter with a LingBotWorldRuntime instance.

        Args:
            runtime: An initialized instance of LingBotWorldRuntime, which handles
                     the core LingBot model and its configuration.
        """
        super().__init__()
        self.runtime = runtime
        self.core_model = runtime.core_model
        self.config = runtime.config
        self.default_offload_model = runtime.default_offload_model

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "LingBotSynthesis":
        """
        Factory method to create a LingBotSynthesis instance by loading a pretrained
        LingBot runtime.

        This method acts as a convenience wrapper around `load_runtime` from the
        `lingbot_world` module, which handles the loading and initialization of
        the underlying `LingBotWorldRuntime`.

        Args:
            *args: Positional arguments to pass to `load_runtime`.
            **kwargs: Keyword arguments to pass to `load_runtime`.

        Returns:
            A new instance of LingBotSynthesis, initialized with the loaded runtime.
        """
        return cls(load_runtime(*args, **kwargs))

    @classmethod
    def runtime_plan(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Retrieves the runtime plan for a LingBotWorldRuntime instance.

        This method delegates to the static `runtime_plan` method of the
        `LingBotWorldRuntime` class, providing information about how the
        runtime would be constructed given certain parameters.

        Args:
            *args: Positional arguments to pass to `LingBotWorldRuntime.runtime_plan`.
            **kwargs: Keyword arguments to pass to `LingBotWorldRuntime.runtime_plan`.

        Returns:
            A dictionary representing the runtime plan.
        """
        return LingBotWorldRuntime.runtime_plan(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying LingBotWorldRuntime instance.

        If an attribute is not found directly on `LingBotSynthesis`, this method
        attempts to retrieve it from the encapsulated `self.runtime` object.
        This allows `LingBotSynthesis` to act as a transparent proxy for many
        runtime-specific attributes and methods.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the internal runtime.

        Raises:
            AttributeError: If the attribute is not found on either `LingBotSynthesis`
                            or its `runtime` object.
        """
        # Delegate unknown attribute lookups to the internal LingBotWorldRuntime instance.
        return getattr(self.runtime, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a prediction using the underlying LingBot runtime.

        This method directly calls the `predict` method of the encapsulated
        `LingBotWorldRuntime` instance, passing all arguments along.

        Args:
            *args: Positional arguments to pass to `self.runtime.predict`.
            **kwargs: Keyword arguments to pass to `self.runtime.predict`.

        Returns:
            The result of the prediction from the LingBot runtime.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = [
    "DEFAULT_LINGBOT_ACT_REPO",
    "DEFAULT_LINGBOT_BASE_REPO",
    "DEFAULT_LINGBOT_FAST_REPO",
    "DEFAULT_LINGBOT_HFD_ROOT",
    "SUPPORTED_LINGBOT_TASKS",
    "LingBotRuntime",
    "LingBotSynthesis",
    "LingBotWorldRuntime",
    "lingbot_runtime_root",
    "load_runtime",
]
