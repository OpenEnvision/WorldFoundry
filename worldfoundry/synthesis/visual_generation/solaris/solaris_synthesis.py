"""
This module provides a synthesis facade for the Solaris visual generation runtime.

It defines the `SolarisSynthesis` class, which acts as a wrapper around the
`worldfoundry_runtime.SolarisRuntime` to integrate it into the `BaseSynthesis`
framework. This allows for easy instantiation and direct delegation of
synthesis operations like `predict`.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.solaris.worldfoundry_runtime import (
    SolarisRuntime,
)

from ...base_synthesis import BaseSynthesis


class SolarisSynthesis(BaseSynthesis):
    """
    A synthesis facade that wraps the `SolarisRuntime` for visual generation tasks.

    This class provides a convenient interface to the underlying Solaris model,
    handling its initialization and delegating all synthesis-related operations
    to the `SolarisRuntime` instance. It can either be initialized with an
    existing `SolarisRuntime` or construct one from provided path arguments.
    """

    def __init__(
        self,
        runtime_root: str | None = None,
        pretrained_model_dir: str | None = None,
        eval_data_dir: str | None = None,
        output_dir: str | None = None,
        checkpoint_dir: str | None = None,
        jax_cache_dir: str | None = None,
        model_weights_path: str | None = None,
        *,
        runtime: SolarisRuntime | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SolarisSynthesis facade.

        This constructor can either take an already initialized `SolarisRuntime`
        instance or a set of required path arguments to construct a new one.
        If path arguments are provided, they are validated to ensure all
        necessary components for `SolarisRuntime` instantiation are present.

        Args:
            runtime_root: Data/work base for relative asset paths. Required if `runtime` is not provided.
            pretrained_model_dir: Directory containing pretrained model components. Required if `runtime` is not provided.
            eval_data_dir: Directory for evaluation data. Required if `runtime` is not provided.
            output_dir: Directory for model outputs. Required if `runtime` is not provided.
            checkpoint_dir: Directory for model checkpoints. Required if `runtime` is not provided.
            jax_cache_dir: Directory for JAX caching. Required if `runtime` is not provided.
            model_weights_path: Path to the model weights file. Required if `runtime` is not provided.
            runtime: An optional pre-initialized `SolarisRuntime` instance.
                     If provided, other path arguments are ignored.
            **kwargs: Additional keyword arguments to pass to the `SolarisRuntime` constructor
                      if it is being initialized by this class.

        Raises:
            ValueError: If `runtime` is None and any of the required path arguments are missing.
        """
        super().__init__()
        if runtime is None:
            # Collect all potentially required path arguments for SolarisRuntime.
            required = {
                "runtime_root": runtime_root,
                "pretrained_model_dir": pretrained_model_dir,
                "eval_data_dir": eval_data_dir,
                "output_dir": output_dir,
                "checkpoint_dir": checkpoint_dir,
                "jax_cache_dir": jax_cache_dir,
                "model_weights_path": model_weights_path,
            }
            # Identify any missing arguments from the collected set.
            missing = [name for name, value in required.items() if value is None]
            if missing:
                # Raise an error if any required arguments are not provided when instantiating runtime.
                raise ValueError("Missing Solaris runtime components: " + ", ".join(missing))
            # All required arguments are present, instantiate SolarisRuntime.
            runtime = SolarisRuntime(
                runtime_root=str(runtime_root),
                pretrained_model_dir=str(pretrained_model_dir),
                eval_data_dir=str(eval_data_dir),
                output_dir=str(output_dir),
                checkpoint_dir=str(checkpoint_dir),
                jax_cache_dir=str(jax_cache_dir),
                model_weights_path=str(model_weights_path),
                **kwargs,
            )
        self.runtime = runtime

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying `SolarisRuntime` instance.

        This method allows attributes of `SolarisRuntime` (like specific model
        properties or methods) to be accessed directly through the `SolarisSynthesis`
        instance. It prevents infinite recursion if `runtime` itself is requested.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the encapsulated `SolarisRuntime` instance.

        Raises:
            AttributeError: If the requested attribute is `runtime` itself, or
                            if the attribute does not exist on the `SolarisRuntime` instance.
        """
        if name == "runtime":
            # Prevent infinite recursion if __getattr__ is called for 'runtime' attribute.
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> SolarisSynthesis:
        """
        Creates a `SolarisSynthesis` instance by loading a pretrained `SolarisRuntime`.

        This is a convenience factory method that mirrors the `from_pretrained`
        method of `SolarisRuntime`, allowing for easy instantiation of the
        synthesis facade from a pretrained model.

        Args:
            *args: Positional arguments to pass to `SolarisRuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `SolarisRuntime.from_pretrained`.

        Returns:
            A new `SolarisSynthesis` instance initialized with the pretrained runtime.
        """
        return cls(runtime=SolarisRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any):
        """
        Performs a prediction using the underlying `SolarisRuntime` instance.

        This method acts as a direct pass-through to the `predict` method of
        the encapsulated `SolarisRuntime`, executing the visual generation task.

        Args:
            *args: Positional arguments to pass to `SolarisRuntime.predict`.
            **kwargs: Keyword arguments to pass to `SolarisRuntime.predict`.

        Returns:
            The result of the `SolarisRuntime.predict` method.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["SolarisSynthesis"]
