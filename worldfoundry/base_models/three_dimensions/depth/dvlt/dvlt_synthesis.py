"""
This module provides the DVLTSynthesis class, which acts as a synthesis adapter for the
DVLT (Déjà View Looping Transformer) reconstruction runtime.

It allows for easy integration and usage of the DVLT model by inheriting from BaseSynthesis
and delegating calls to a DVLTRuntime instance. It also re-exports important constants
and functions from the underlying DVLT runtime module.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.base_models.three_dimensions.depth.dvlt.runtime import (
    DEFAULT_DVLT_CHECKPOINT,
    DEFAULT_DVLT_IMAGE_SIZE,
    DEFAULT_DVLT_INFERENCE_STEPS,
    DEFAULT_DVLT_PATCH_SIZE,
    DVLTRuntime,
    load_runtime,
)

from worldfoundry.synthesis.base_synthesis import BaseSynthesis


class DVLTSynthesis(BaseSynthesis):
    """
    Synthesis adapter around the base-model DVLT runtime.

    This class extends BaseSynthesis and acts as a wrapper, delegating core
    functionality like prediction to an internal DVLTRuntime instance.
    It simplifies interaction with the DVLT model by providing a consistent
    synthesis interface.
    """

    MODEL_ID = DVLTRuntime.MODEL_ID
    DISPLAY_NAME = DVLTRuntime.DISPLAY_NAME

    def __init__(self, runtime: DVLTRuntime):
        """
        Initializes the DVLTSynthesis adapter with a DVLTRuntime instance.

        Args:
            runtime: An initialized instance of DVLTRuntime, which handles
                     the actual model loading and inference.
        """
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "DVLTSynthesis":
        """
        Factory method to create a DVLTSynthesis instance by loading a pretrained
        DVLT runtime.

        This method forwards its arguments to `worldfoundry.base_models.three_dimensions.depth.dvlt.runtime.load_runtime`
        to instantiate the underlying DVLTRuntime.

        Args:
            *args: Positional arguments passed directly to `load_runtime`.
            **kwargs: Keyword arguments passed directly to `load_runtime`.

        Returns:
            A new instance of DVLTSynthesis, initialized with the loaded runtime.
        """
        return cls(load_runtime(*args, **kwargs))

    def api_init(self, *, api_key: str, endpoint: str) -> None:
        """
        Initializes any API-specific configurations.

        For DVLTSynthesis, this method is a no-op as the DVLT model typically runs
        locally or its API handling is managed at a lower level within the runtime.
        The `api_key` and `endpoint` parameters are consumed but not used.

        Args:
            api_key: An API key for authentication (not used in this implementation).
            endpoint: The API endpoint URL (not used in this implementation).
        """
        del api_key, endpoint
        return None

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Generates predictions using the underlying DVLT runtime.

        This method acts as a proxy, forwarding all arguments to the `predict`
        method of the internal DVLTRuntime instance.

        Args:
            *args: Positional arguments passed directly to `self.runtime.predict`.
            **kwargs: Keyword arguments passed directly to `self.runtime.predict`.

        Returns:
            A dictionary containing the prediction results from the DVLT model.
        """
        return self.runtime.predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Custom attribute access method to delegate undefined attributes to the
        underlying DVLTRuntime instance.

        This allows direct access to methods and properties of `self.runtime`
        through the `DVLTSynthesis` instance, effectively making it behave
        like a proxy for the runtime.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying `self.runtime` object.

        Raises:
            AttributeError: If the attribute does not exist on either `DVLTSynthesis`
                            or its `runtime` instance.
        """
        # Attempt to retrieve the runtime instance from the instance dictionary
        runtime = self.__dict__.get("runtime")
        # If runtime is not yet set (e.g., during object initialization before __init__ completes),
        # or if it's explicitly None, raise an AttributeError
        if runtime is None:
            raise AttributeError(name)
        # Delegate attribute access to the underlying runtime object
        return getattr(runtime, name)


__all__ = [
    "DEFAULT_DVLT_CHECKPOINT",
    "DEFAULT_DVLT_IMAGE_SIZE",
    "DEFAULT_DVLT_INFERENCE_STEPS",
    "DEFAULT_DVLT_PATCH_SIZE",
    "DVLTRuntime",
    "DVLTSynthesis",
    "load_runtime",
]
