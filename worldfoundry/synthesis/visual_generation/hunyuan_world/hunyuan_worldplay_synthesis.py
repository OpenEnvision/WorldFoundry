"""This module serves as an adapter and proxy for the Hunyuan World Play visual generation runtime.

It provides the `HunyuanWorldPlaySynthesis` class, which conforms to the `BaseSynthesis`
interface, and re-exports key components from the underlying runtime module
for direct access, enabling lazy loading of the runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.runtime import (
        HunyuanWorldPlayRuntime,
    )


_RUNTIME_MODULE = "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.runtime"
_RUNTIME_EXPORTS = {
    "HunyuanVideoPipelineOutput",
    "HunyuanWorldPlayRuntime",
    "_HunyuanWorldPlayInternalPipeline",
    "load_runtime",
}


def _runtime_module():
    """Lazily imports and returns the Hunyuan World Play runtime module.

    This function ensures that the potentially heavy runtime module is only loaded
    when one of its components is explicitly requested or used.

    Returns:
        ModuleType: The imported Hunyuan World Play runtime module.
    """
    return import_module(_RUNTIME_MODULE)


def load_runtime(*args: Any, **kwargs: Any):
    """Loads the Hunyuan World Play runtime by calling its `load_runtime` function.

    This function acts as a direct proxy to the `load_runtime` function
    exported by the Hunyuan World Play runtime module.

    Args:
        *args: Positional arguments to pass to the underlying `load_runtime` function.
        **kwargs: Keyword arguments to pass to the underlying `load_runtime` function.

    Returns:
        Any: The loaded Hunyuan World Play runtime instance.
    """
    return _runtime_module().load_runtime(*args, **kwargs)


class HunyuanWorldPlaySynthesis(BaseSynthesis):
    """A synthesis adapter that wraps the `HunyuanWorldPlayRuntime` to conform to the `BaseSynthesis` interface.

    This class provides a convenient way to integrate the Hunyuan World Play runtime
    into a broader synthesis framework, delegating operations like initialization
    and prediction to the underlying runtime object.
    """

    def __init__(self, runtime: HunyuanWorldPlayRuntime):
        """Initializes the synthesis adapter with a given runtime instance.

        Args:
            runtime: An initialized instance of `HunyuanWorldPlayRuntime`.
        """
        super().__init__()
        self.runtime = runtime
        self.model = runtime.model

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> HunyuanWorldPlaySynthesis:
        """Creates a new `HunyuanWorldPlaySynthesis` instance by loading the runtime from pretrained weights.

        This method acts as a convenience constructor, forwarding its arguments
        to the module's `load_runtime` function to obtain a runtime instance.

        Args:
            *args: Positional arguments to pass to `load_runtime`.
            **kwargs: Keyword arguments to pass to `load_runtime`.

        Returns:
            HunyuanWorldPlaySynthesis: A new instance of the synthesis adapter.
        """
        return cls(load_runtime(*args, **kwargs))

    def api_init(self, *, api_key: str, endpoint: str) -> Any:
        """Initializes the API for the underlying Hunyuan World Play runtime.

        Args:
            api_key: The API key for authentication.
            endpoint: The API endpoint URL.

        Returns:
            Any: The result from the runtime's `api_init` method.
        """
        return self.runtime.api_init(api_key=api_key, endpoint=endpoint)

    @torch.no_grad()
    def predict(self, *, data: dict[str, Any]) -> Any:
        """Performs a prediction using the underlying runtime model.

        This method delegates the prediction task to the wrapped `HunyuanWorldPlayRuntime`
        and ensures that gradients are not computed during the process.

        Args:
            data: A dictionary containing the input data required for the prediction.

        Returns:
            Any: The prediction output from the `HunyuanWorldPlayRuntime`.
        """
        return self.runtime.predict(data=data)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allows the synthesis instance to be called directly, forwarding the call to the underlying runtime.

        This enables the `HunyuanWorldPlaySynthesis` object to behave like the
        wrapped `HunyuanWorldPlayRuntime` for direct invocation.

        Args:
            *args: Positional arguments to pass to the runtime's `__call__` method.
            **kwargs: Keyword arguments to pass to the runtime's `__call__` method.

        Returns:
            Any: The result of the runtime's `__call__` method.
        """
        return self.runtime(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the wrapped `runtime` object if the attribute is not found on `HunyuanWorldPlaySynthesis` itself.

        This allows direct access to attributes and methods of the underlying
        `HunyuanWorldPlayRuntime` instance through the synthesis adapter.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the `runtime` object.

        Raises:
            AttributeError: If `runtime` is not initialized or the attribute does not exist on `runtime`.
        """
        # Access 'runtime' directly from __dict__ to avoid infinite recursion during attribute lookup
        runtime = self.__dict__.get("runtime")
        if runtime is None:
            # Raise AttributeError if 'runtime' itself hasn't been initialized
            raise AttributeError(name)
        return getattr(runtime, name)


def __getattr__(name: str) -> Any:
    """Lazily imports and re-exports symbols from the specified runtime module.

    This function is part of Python's module-level attribute access customization.
    When an attribute `name` is requested from this module that is not explicitly
    defined, this function attempts to load it from the `_RUNTIME_MODULE`
    if `name` is in `_RUNTIME_EXPORTS`.

    Args:
        name: The name of the attribute being accessed from this module.

    Returns:
        Any: The value of the attribute from the runtime module.

    Raises:
        AttributeError: If the requested attribute `name` is not in `_RUNTIME_EXPORTS`.
    """
    if name in _RUNTIME_EXPORTS:
        value = getattr(_runtime_module(), name)
        # Cache the imported attribute in the module's globals to avoid re-importing on subsequent access
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "HunyuanVideoPipelineOutput",  # noqa: F822 - Provided by the module-level __getattr__.
    "HunyuanWorldPlayRuntime",
    "HunyuanWorldPlaySynthesis",
    "_HunyuanWorldPlayInternalPipeline",  # noqa: F822 - Provided by the module-level __getattr__.
    "load_runtime",
]
