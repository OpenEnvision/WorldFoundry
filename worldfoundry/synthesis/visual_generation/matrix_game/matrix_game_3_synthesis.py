"""
This module provides a synthesis facade for the Matrix-Game-3 base runtime.

It allows for the creation, configuration, and interaction with the MatrixGame3Runtime
without directly importing the runtime module, acting as an abstraction layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.worldfoundry_runtime import (
        MatrixGame3Runtime,
    )


def _runtime_module():
    """
    Dynamically imports and returns the worldfoundry_runtime module for Matrix-Game-3.

    This function defers the import of the runtime module until it's actually needed,
    avoiding potential circular dependencies or unnecessary imports at module load time.
    """
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime import (
        worldfoundry_runtime,
    )

    return worldfoundry_runtime


def _runtime_cls():
    """
    Returns the MatrixGame3Runtime class from the dynamically imported runtime module.
    """
    return _runtime_module().MatrixGame3Runtime


#: Default aliases configuration for Matrix-Game-3 models.
DEFAULT_MATRIX_GAME3_ALIASES = _runtime_module().DEFAULT_MATRIX_GAME3_ALIASES
#: Default checkpoint directory for Matrix-Game-3 models.
DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR = _runtime_module().DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR


class MatrixGame3Synthesis(BaseSynthesis):
    """
    A synthesis facade for the Matrix-Game-3 base runtime.

    This class acts as a thin wrapper around the MatrixGame3Runtime,
    providing a consistent interface for model loading and prediction
    while abstracting away the direct interaction with the runtime implementation.
    """

    def __init__(
        self,
        runtime_path: str | None = None,
        checkpoint_dir: str | None = None,
        device: str = "cuda",
        defaults: dict | None = None,
        *,
        runtime: MatrixGame3Runtime | None = None,
    ):
        """
        Initializes a new instance of MatrixGame3Synthesis.

        This constructor allows either providing an already instantiated runtime
        object or providing parameters to dynamically create one.

        Args:
            runtime_path (str | None): The path to the runtime configuration or model.
                                       Required if 'runtime' is not provided.
            checkpoint_dir (str | None): The directory for model checkpoints.
                                         Required if 'runtime' is not provided.
            device (str): The device to use for computation (e.g., "cuda", "cpu").
                          Defaults to "cuda".
            defaults (Optional[dict]): Optional dictionary of default parameters for the runtime.
            runtime (MatrixGame3Runtime | None): An already instantiated MatrixGame3Runtime object.
                                                 If provided, 'runtime_path' and 'checkpoint_dir'
                                                 are ignored.
        Raises:
            ValueError: If 'runtime' is None and either 'runtime_path' or 'checkpoint_dir' is None.
        """
        super().__init__()
        if runtime is None:
            # If no runtime object is provided, ensure necessary parameters for creating one are present.
            if runtime_path is None or checkpoint_dir is None:
                raise ValueError("runtime_path and checkpoint_dir are required when runtime is not provided.")
            # Instantiate the MatrixGame3Runtime using the provided parameters.
            runtime = _runtime_cls()(
                runtime_path=runtime_path,
                checkpoint_dir=checkpoint_dir,
                device=device,
                defaults=defaults,
            )
        self.runtime = runtime

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying runtime object.

        This method allows MatrixGame3Synthesis to expose methods and properties
        of the MatrixGame3Runtime directly, making the facade transparent.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the internal runtime object.

        Raises:
            AttributeError: If the attribute is 'runtime' itself, to prevent recursion.
                            Otherwise, raises AttributeError if the attribute is not found
                            on the internal runtime object.
        """
        # Prevent infinite recursion if 'runtime' itself is accessed via __getattr__
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any,
        args: Any = None,
        device: str | None = None,
        checkpoint_dir: str | None = None,
        **kwargs: Any,
    ) -> MatrixGame3Synthesis:
        """
        Creates a MatrixGame3Synthesis instance by loading a pretrained model.

        This class method delegates the model loading to the underlying runtime's
        `from_pretrained` method and then wraps the resulting runtime object.

        Args:
            pretrained_model_path (Any): The path or identifier for the pretrained model.
            args (Any): Additional arguments to pass to the runtime's from_pretrained method.
            device (str | None): The device to load the model onto (e.g., "cuda", "cpu").
            checkpoint_dir (str | None): The directory for model checkpoints.
            **kwargs (Any): Arbitrary keyword arguments to pass to the runtime's
                            from_pretrained method.

        Returns:
            MatrixGame3Synthesis: A new instance of MatrixGame3Synthesis with the loaded model.
        """
        # Delegate the pretrained model loading to the actual runtime class.
        runtime = _runtime_cls().from_pretrained(
            pretrained_model_path=pretrained_model_path,
            args=args,
            device=device,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )
        # Create and return a new Synthesis instance, wrapping the loaded runtime.
        return cls(runtime=runtime)

    @staticmethod
    def resolve_checkpoint_dir(path_value: str | None) -> str:
        """
        Resolves the full path for a checkpoint directory using the runtime's logic.

        Args:
            path_value (str | None): The path value to resolve.

        Returns:
            str: The resolved absolute path to the checkpoint directory.
        """
        return _runtime_module().resolve_checkpoint_dir(path_value)

    @staticmethod
    def build_subprocess_env(device: str | None = None) -> dict:
        """
        Builds the environment variables necessary for a subprocess using the runtime's logic.

        Args:
            device (str | None): The device string (e.g., "cuda", "cpu") to include in the environment.

        Returns:
            dict: A dictionary of environment variables suitable for a subprocess.
        """
        return _runtime_module().build_subprocess_env(device)

    def predict(self, *args: Any, **kwargs: Any):
        """
        Delegates the prediction call to the underlying runtime object.

        This method forwards all arguments and keyword arguments directly
        to the `predict` method of the internal MatrixGame3Runtime instance.

        Args:
            *args (Any): Positional arguments to pass to the runtime's predict method.
            **kwargs (Any): Keyword arguments to pass to the runtime's predict method.

        Returns:
            Any: The result of the prediction from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = [
    "DEFAULT_MATRIX_GAME3_ALIASES",
    "DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR",
    "MatrixGame3Synthesis",
]
