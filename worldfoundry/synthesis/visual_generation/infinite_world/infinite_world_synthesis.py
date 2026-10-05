"""
This module provides a synthesis adapter for the WorldFoundry framework,
integrating the `InfiniteWorldRuntime` for visual generation tasks.

It wraps the core functionality of `InfiniteWorldRuntime` to conform to
the `BaseSynthesis` interface, allowing it to be used within the WorldFoundry
synthesis pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime import (
    DEFAULT_NEGATIVE_PROMPT,
    InfiniteWorldRuntime,
)

from ...base_synthesis import BaseSynthesis


class InfiniteWorldSynthesis(BaseSynthesis):
    """Thin WorldFoundry synthesis adapter over the base Infinite-World runtime."""

    chunk_frames = InfiniteWorldRuntime.chunk_frames
    chunk_stride = InfiniteWorldRuntime.chunk_stride

    def __init__(self, runtime: InfiniteWorldRuntime):
        """
        Initializes the InfiniteWorldSynthesis adapter with a given runtime instance.

        Args:
            runtime: An instance of InfiniteWorldRuntime to be wrapped.
        """
        super().__init__()
        self.runtime = runtime

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the wrapped InfiniteWorldRuntime instance.

        This method is called when an attribute is not found in the InfiniteWorldSynthesis
        instance itself. It attempts to retrieve the attribute from the 'runtime' object.

        Args:
            name: The name of the attribute to retrieve.

        Returns:
            The value of the attribute from the wrapped runtime instance.

        Raises:
            AttributeError: If the attribute is not found in the runtime instance.
        """
        # Safely get the 'runtime' attribute from the instance's dictionary
        # to prevent infinite recursion if 'runtime' itself is not yet set.
        runtime = self.__dict__.get("runtime")
        if runtime is not None:
            return getattr(runtime, name)
        raise AttributeError(name)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Loads a pre-trained InfiniteWorldRuntime instance and wraps it.

        This class method acts as a factory, delegating the loading logic
        to `InfiniteWorldRuntime.from_pretrained`.

        Args:
            *args: Positional arguments to pass to `InfiniteWorldRuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `InfiniteWorldRuntime.from_pretrained`.

        Returns:
            An instance of InfiniteWorldSynthesis wrapping the loaded runtime,
            or a dictionary if the runtime loading returns a configuration dict.
        """
        runtime = InfiniteWorldRuntime.from_pretrained(*args, **kwargs)
        # If the underlying runtime returns a configuration dictionary instead of an instance,
        # propagate that dictionary directly.
        if isinstance(runtime, dict):
            return runtime
        return cls(runtime)

    @classmethod
    def plan(
        cls,
        pretrained_model_path=None,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates a plan for the InfiniteWorldRuntime based on provided configurations.

        This class method delegates to `InfiniteWorldRuntime.plan` to generate
        a configuration or plan dictionary without instantiating the full runtime.

        Args:
            pretrained_model_path: Path to a pre-trained model.
            device: The device to use for computation (e.g., "cuda", "cpu").
            config_path: Path to a configuration file.
            **kwargs: Additional keyword arguments for planning.

        Returns:
            A dictionary representing the generated plan or configuration.
        """
        return InfiniteWorldRuntime.plan(
            pretrained_model_path=pretrained_model_path,
            device=device,
            config_path=config_path,
            **kwargs,
        )

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes the prediction logic using the wrapped InfiniteWorldRuntime instance.

        This method directly delegates all arguments to the `predict` method of
        the underlying runtime.

        Args:
            *args: Positional arguments to pass to `runtime.predict`.
            **kwargs: Keyword arguments to pass to `runtime.predict`.

        Returns:
            A dictionary containing the prediction results from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)


# ``DEFAULT_NEGATIVE_PROMPT`` is re-exported for the package __init__ to publish.
__all__ = ["DEFAULT_NEGATIVE_PROMPT", "InfiniteWorldSynthesis"]