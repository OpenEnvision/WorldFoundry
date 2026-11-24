"""
Module for integrating the Lyra-2 synthesis runtime with a base synthesis interface.

This module provides the Lyra2Synthesis class, which acts as an adapter around the Lyra2Runtime
to conform to the BaseSynthesis interface, allowing for consistent interaction with different
synthesis models. It also re-exports key components from the runtime module.
"""

from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis
from .runtime import (
    DEFAULT_DA3_MODEL_NAME,
    DEFAULT_WEIGHT_DTYPE,
    Lyra2Runtime,
    load_runtime,
)


class Lyra2Synthesis(BaseSynthesis):
    """
    Synthesis adapter around the Lyra-2 runtime.

    This class provides a unified interface for interacting with the Lyra-2 model,
    wrapping the core Lyra2Runtime and integrating it with the BaseSynthesis system.
    It proxies method calls to the underlying runtime instance.
    """

    MODEL_ID = Lyra2Runtime.MODEL_ID
    DISPLAY_NAME = Lyra2Runtime.DISPLAY_NAME
    BLOCKED_REASONS = Lyra2Runtime.BLOCKED_REASONS

    def __init__(self, runtime: Lyra2Runtime):
        """
        Initializes the Lyra2Synthesis adapter with a given Lyra2Runtime instance.

        Args:
            runtime (Lyra2Runtime): The Lyra-2 runtime instance to adapt.
        """
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Lyra2Synthesis:
        """
        Class method to create a Lyra2Synthesis instance by loading a pretrained Lyra-2 runtime.

        This method acts as a factory, passing all arguments to the underlying `load_runtime`
        function to initialize the Lyra2Runtime, and then wraps it in a Lyra2Synthesis instance.

        Args:
            *args: Positional arguments passed directly to `load_runtime`.
            **kwargs: Keyword arguments passed directly to `load_runtime`.

        Returns:
            Lyra2Synthesis: A new instance of Lyra2Synthesis with the loaded runtime.
        """
        return cls(load_runtime(*args, **kwargs))

    def api_init(self, *, api_key: str, endpoint: str) -> None:
        """
        Initializes the API for the underlying runtime.

        This method currently serves as a no-op as API key and endpoint handling
        might be managed at a different level or not directly applicable to
        the Lyra-2 runtime's initialization via this interface.

        Args:
            api_key (str): The API key (currently unused).
            endpoint (str): The API endpoint (currently unused).

        Returns:
            None
        """
        del api_key, endpoint  # Mark parameters as intentionally unused
        return None

    def load_runtime(self, strict: bool = True) -> Lyra2Synthesis:
        """
        Loads or reloads the underlying Lyra-2 runtime.

        Delegates the loading process to the internal `Lyra2Runtime` instance.

        Args:
            strict (bool): Whether to strictly enforce loading. Defaults to `True`.

        Returns:
            Lyra2Synthesis: The current instance, allowing for method chaining.
        """
        self.runtime.load_runtime(strict=strict)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Performs a prediction using the underlying Lyra-2 runtime.

        All arguments are directly forwarded to the `predict` method of the
        internal `Lyra2Runtime` instance.

        Args:
            *args: Positional arguments for the runtime's predict method.
            **kwargs: Keyword arguments for the runtime's predict method.

        Returns:
            dict[str, Any]: The prediction results from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Provides proxy access to attributes of the underlying Lyra2Runtime instance.

        If an attribute is not found directly on the Lyra2Synthesis object, this method
        attempts to retrieve it from the `self.runtime` object.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The attribute value from the `runtime` object.

        Raises:
            AttributeError: If the attribute does not exist on the `Lyra2Synthesis`
                            object itself and cannot be found on the proxied `runtime` object,
                            or if `runtime` has not yet been set (e.g., during initialization
                            before `__init__` completes).
        """
        # Attempt to get the 'runtime' attribute from the instance's dictionary.
        # This avoids recursive __getattr__ calls if 'runtime' itself is not set yet.
        runtime = self.__dict__.get("runtime")
        if runtime is None:
            # If 'runtime' isn't initialized yet, raise an AttributeError immediately.
            # This handles cases where an attribute is accessed before __init__ is complete.
            raise AttributeError(name)
        # If 'runtime' exists, attempt to get the requested attribute from it.
        return getattr(runtime, name)


__all__ = [
    "DEFAULT_DA3_MODEL_NAME",
    "DEFAULT_WEIGHT_DTYPE",
    "Lyra2Runtime",
    "Lyra2Synthesis",
    "load_runtime",
]
