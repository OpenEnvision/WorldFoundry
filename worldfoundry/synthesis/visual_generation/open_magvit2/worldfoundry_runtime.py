"""
This module provides a runtime interface for the Open-MAGVIT2 model, facilitating class-conditional image generation
within the WorldFoundry framework. It handles model loading, configuration resolution, and image generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.core.io.paths import package_module_root as package_root
from worldfoundry.evaluation.utils import worldfoundry_data_path

# Root directory for default Open-MAGVIT2 runtime configurations within worldfoundry data.
DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "open_magvit2")
# Default configuration file for Open-MAGVIT2 model.
DEFAULT_OPEN_MAGVIT2_CONFIG = DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT / "imagenet_conditional_llama_L.yaml"


def _resolve_hfd_root() -> Path:
    """
    Resolves the Hugging Face Downloader (HFD) root path.

    Returns:
        Path: The root path where HFD downloads models.
    """
    return hfd_root_path()


# Root directory for default Open-MAGVIT2 checkpoints downloaded via HFD.
DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT = _resolve_hfd_root() / "TencentARC--Open-MAGVIT2"


class OpenMAGVIT2Runtime:
    """
    A runtime interface for the Open-MAGVIT2 model, integrated with the in-tree runtime package.

    This class provides methods to initialize, configure, and run the Open-MAGVIT2 model
    for class-conditional image generation.
    """

    MODEL_ID = "open-magvit2"
    DISPLAY_NAME = "Open-MAGVIT2"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        class_id: int = 207,
        batch_size: int = 1,
    ) -> None:
        """
        Initializes the OpenMAGVIT2Runtime.

        Args:
            model_id (str): Identifier for the model. Defaults to "open-magvit2".
            device (str): The device to run the model on (e.g., "cuda", "cpu"). Defaults to "cuda".
            checkpoint_path (str | Path | None): Path to the model checkpoint file. If None,
                                                 a default path will be resolved.
            config_path (str | Path | None): Path to the model configuration file. If None,
                                             a default path will be resolved.
            class_id (int): Default class ID for generation when not specified in predict.
                            Defaults to 207 (flamingo).
            batch_size (int): Default batch size for generation. Defaults to 1.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "class_conditional_image_generation"
        self.device = device
        # Resolve and store checkpoint path, expanding user directory if necessary.
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser())
        # Resolve and store configuration path, expanding user directory if necessary.
        self.config_path = None if config_path is None else str(Path(config_path).expanduser())
        self.class_id = int(class_id)
        self.batch_size = int(batch_size)
        self._model = None  # Stores the loaded model instance.
        self._config = None  # Stores the loaded model configuration.

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "OpenMAGVIT2Runtime":
        """
        Initializes OpenMAGVIT2Runtime from a pretrained model path or configuration.

        This factory method allows flexibility in how model parameters are provided,
        supporting dictionaries, direct paths, and keyword arguments.

        Args:
            pretrained_model_path (Any): A path (str or Path) or a dictionary of options
                                         containing `checkpoint_path` and `config_path`.
            args (Any): Placeholder argument, currently unused.
            device (str | None): Device to use (e.g., "cuda", "cpu"). Overrides options.
            model_id (str | None): Model identifier. Overrides options.
            **kwargs (Any): Additional keyword arguments to override or set runtime parameters.

        Returns:
            OpenMAGVIT2Runtime: An initialized instance of the runtime.
        """
        del args
        # Parse pretrained_model_path: if it's a mapping, use it as options; otherwise, treat it as a checkpoint path.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        
        # Update options with any additional keyword arguments provided.
        options.update(kwargs)
        
        # Initialize the runtime instance, resolving parameters from multiple potential sources.
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path") or options.get("model_path"),
            config_path=options.get("config_path") or options.get("config"),
            class_id=int(options.get("class_id", 207)),
            batch_size=int(options.get("batch_size", 1)),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """
        Returns the root path of the in-tree Open-MAGVIT2 runtime package.

        This path is used to dynamically add the package to `sys.path` to enable
        local imports of the model's inference logic.

        Returns:
            Path: The `Path` object pointing to the runtime package directory.
        """
        return package_root("worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime")

    def _resolve_config(self) -> Path:
        """
        Resolves the configuration file path for the Open-MAGVIT2 model.

        If a `config_path` is specified during initialization, it is used. Otherwise,
        the `DEFAULT_OPEN_MAGVIT2_CONFIG` is returned. Relative paths are resolved
        against `DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT`.

        Returns:
            Path: The absolute path to the resolved configuration file.

        Raises:
            FileNotFoundError: If the resolved config file does not exist. (Implicit from Path.resolve())
        """
        if self.config_path:
            path = Path(self.config_path).expanduser()
            # If the path is relative, resolve it against the default config root.
            return path if path.is_absolute() else (DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT / path).resolve()
        return DEFAULT_OPEN_MAGVIT2_CONFIG

    def _resolve_checkpoint(self) -> Path:
        """
        Resolves the model checkpoint file path for Open-MAGVIT2.

        If a `checkpoint_path` is specified during initialization, it is used.
        Otherwise, it searches for preferred checkpoint names in `DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT`,
        then falls back to globbing for any `.ckpt` or `.pth` files.

        Returns:
            Path: The absolute path to the resolved checkpoint file.

        Raises:
            FileNotFoundError: If no suitable checkpoint file is found.
        """
        if self.checkpoint_path:
            return Path(self.checkpoint_path).expanduser().resolve()
        
        # Define a list of preferred checkpoint filenames.
        preferred_names = (
            "AR_256_L.ckpt",
            "AR_256_XL.ckpt",
            "AR_256_B.ckpt",
            "imagenet_256_L.ckpt",
            "imagenet_256_B.ckpt",
            "imagenet_128_L.ckpt",
            "imagenet_128_B.ckpt",
        )
        # Create a list of potential paths from preferred names in the default root.
        candidates = tuple(DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT / name for name in preferred_names)
        
        # Filter existing paths from preferred names, then add any other .ckpt and .pth files.
        candidates = (
            tuple(path for path in candidates if path.exists())
            + tuple(sorted(DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT.glob("*.ckpt")))
            + tuple(sorted(DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT.glob("*.pth")))
        )
        
        # If any candidate checkpoint is found, return the first one.
        if candidates:
            return candidates[0].resolve()
        
        # Raise an error if no checkpoint could be found.
        raise FileNotFoundError(
            "Open-MAGVIT2 checkpoint not found. Pass checkpoint_path/ckpt_path pointing to the transformer checkpoint."
        )

    def _ensure_model(self):
        """
        Ensures the Open-MAGVIT2 model and configuration are loaded.

        If the model and config are not already loaded, this method dynamically adds the
        runtime's path to `sys.path` and imports the `load_model` function to load
        the model and its configuration based on the resolved paths.

        Returns:
            tuple: A tuple containing the loaded model instance and its configuration.
        """
        if self._model is not None:
            return self._model, self._config
        
        import sys

        runtime_root = self._runtime_root()
        runtime_path = str(runtime_root)
        
        # Add the runtime package path to sys.path if not already present,
        # ensuring dynamic imports from the in-tree package work.
        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)
            
        # These imports are dynamic and depend on `runtime_path` being in `sys.path`.
        from worldfoundry.synthesis.visual_generation.open_magvit2 import open_magvit2_runtime  # noqa: F401
        from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.inference import load_model

        # Load the model and configuration using the resolved paths and specified device.
        self._model, self._config, self.device = load_model(
            self._resolve_config(),
            self._resolve_checkpoint(),
            device=self.device,
        )
        return self._model, self._config

    @staticmethod
    def _pair(value: Any, cast: type, name: str) -> tuple[Any, Any]:
        """
        Parses a value into a two-element tuple, typically for parameters that accept
        separate values for image and video (though Open-MAGVIT2 currently only handles images).

        Args:
            value (Any): The input value, which can be a comma-separated string, a sequence,
                         or a single value to be duplicated.
            cast (type): The type to cast each part of the pair to (e.g., float, int).
            name (str): The name of the parameter being parsed, used for error messages.

        Returns:
            tuple[Any, Any]: A two-element tuple of the specified cast type.

        Raises:
            ValueError: If the input value cannot be parsed into exactly two parts.
        """
        if isinstance(value, str):
            # Split string by comma if it's a string.
            parts = value.split(",")
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            # If it's a sequence (list, tuple), use its elements directly.
            parts = list(value)
        else:
            # If it's a single value, duplicate it to form a pair.
            parts = [value, value]
        
        # Ensure that exactly two parts were obtained.
        if len(parts) != 2:
            raise ValueError(f"Open-MAGVIT2 {name} expects two comma-separated values.")
        
        # Cast each part to the target type.
        return cast(parts[0]), cast(parts[1])

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generates a class-conditional image using the Open-MAGVIT2 model.

        Note: Open-MAGVIT2 is a class-conditional image generation model and does not
        accept image or video inputs directly for conditioning.

        Args:
            prompt (str): Placeholder, currently unused.
            images (Any): Input images, not supported for this model. Will raise ValueError if provided.
            video (Any): Input video, not supported for this model. Will raise ValueError if provided.
            interactions (Sequence[str]): A list of interaction types (e.g., "debug", "benchmark").
            output_path (str | Path | None): The desired path to save the generated image.
                                              If None, defaults to "open_magvit2.png" in the current directory.
            fps (int | None): Placeholder, currently unused.
            **kwargs (Any): Additional generation parameters which can override defaults:
                            `class_id` (int), `batch_size` (int), `steps` (int),
                            `temperature` (tuple[float, float] or str), `top_k` (tuple[int, int] or str),
                            `top_p` (tuple[float, float] or str), `cfg_scale` (tuple[float, float] or str).
                            Parameters like temperature, top_k, top_p, and cfg_scale can be
                            passed as a comma-separated string (e.g., "1.0,1.0") or a 2-element sequence.

        Returns:
            dict[str, Any]: A dictionary containing generation results, including status, model ID,
                            artifact path, and SHA256 hash of the generated image.

        Raises:
            ValueError: If `images` or `video` inputs are provided.
        """
        del prompt, fps
        # Validate that no unsupported image or video inputs are provided.
        if images is not None or video is not None:
            raise ValueError("Open-MAGVIT2 wrapper is class-conditional and does not accept image or video inputs.")
        
        # Ensure the model and configuration are loaded before prediction.
        model, config = self._ensure_model()
        
        # Dynamically import the image saving function from the runtime.
        from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.inference import (
            save_class_image,
        )

        # Determine the output path for the generated image.
        target = Path(output_path) if output_path is not None else Path.cwd() / "open_magvit2.png"
        
        # Call the actual image generation function with resolved parameters.
        target = save_class_image(
            model,
            config,
            target,
            class_id=int(kwargs.get("class_id", self.class_id)),
            batch_size=int(kwargs.get("batch_size", self.batch_size)),
            steps=None if kwargs.get("steps") is None else int(kwargs["steps"]),
            temperature=self._pair(kwargs.get("temperature", (1.0, 1.0)), float, "temperature"),
            top_k=self._pair(kwargs.get("top_k", (0, 0)), int, "top_k"),
            top_p=self._pair(kwargs.get("top_p", (0.96, 0.96)), float, "top_p"),
            cfg_scale=self._pair(kwargs.get("cfg_scale", (4.0, 4.0)), float, "cfg_scale"),
        )
        
        # Return a dictionary of results, including metadata and the SHA256 hash of the generated artifact.
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_image",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "runtime": "worldfoundry.open_magvit2.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "class_id": int(kwargs.get("class_id", self.class_id)),
            "interactions": list(interactions),
        }


__all__ = [
    "DEFAULT_OPEN_MAGVIT2_CHECKPOINT_ROOT",
    "DEFAULT_OPEN_MAGVIT2_CONFIG",
    "DEFAULT_OPEN_MAGVIT2_CONFIG_ROOT",
    "OpenMAGVIT2Runtime",
]
