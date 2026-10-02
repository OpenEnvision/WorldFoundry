"""Module for abstracting video synthesis runtimes, providing a common interface for model inference.

This module defines a base class `RuntimeVideoSynthesis` which serves as a wrapper
around various model-specific video generation runtimes. It handles common tasks
such as loading configuration, preparing input arguments, converting generated
frames to a standardized format, and saving output videos. It also includes
utility functions for path resolution and data type conversions.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from worldfoundry.core.io import load_serialized, resolve_data_path
from worldfoundry.core.media.codecs.image import load_pil_image, materialize_image_input
from worldfoundry.runtime.assets import expand_worldfoundry_path

from ..base_synthesis import BaseSynthesis


def _frames_to_uint8_array(frames) -> np.ndarray:
    """Convert generated frames into a uint8 video array.

    This function accepts various input types for video frames (PyTorch tensors,
    NumPy arrays, lists of PIL images or NumPy arrays) and converts them into
    a standardized 4D NumPy array of shape (N, H, W, C) with `uint8` data type,
    where N is the number of frames, H and W are height and width, and C is
    the number of channels (3 for RGB).

    Args:
        frames: Tensor, array, PIL frame list, or numpy-compatible frame sequence.

    Returns:
        A 4D NumPy array representing the video frames, with `uint8` data type.

    Raises:
        ValueError: If the input frames have an unsupported shape or contain NaN/infinite values.
        TypeError: If the input frames type is not supported.
    """
    if torch.is_tensor(frames):
        # Detach tensor from graph, move to CPU, and convert to NumPy array.
        tensor = frames.detach().cpu()
        # Permute dimensions if tensor is in (N, C, H, W) format.
        if tensor.ndim == 4 and tensor.shape[1] in {1, 3, 4}:
            tensor = tensor.permute(0, 2, 3, 1)
        # Validate tensor shape.
        elif tensor.ndim != 4 or tensor.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Unsupported tensor frame shape: {tuple(tensor.shape)}")
        # Convert bfloat16 to float before numpy conversion, as numpy does not support bfloat16 directly.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.numpy()
    elif isinstance(frames, np.ndarray):
        array = frames
    elif isinstance(frames, (list, tuple)):
        arrays = []
        # Iterate through list/tuple of frames, converting each to a NumPy array.
        for frame in frames:
            if torch.is_tensor(frame):
                frame = frame.detach().cpu()
                if frame.dtype == torch.bfloat16:
                    frame = frame.float()
                frame = frame.numpy()
                # Permute dimensions if frame is in (C, H, W) format.
                if frame.ndim == 3 and frame.shape[0] in {1, 3, 4}:
                    frame = np.transpose(frame, (1, 2, 0))
            elif hasattr(frame, "convert"):  # Handle PIL Image objects.
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
            arrays.append(frame)
        # Stack individual frame arrays into a single 4D array.
        array = np.stack(arrays, axis=0)
    else:
        raise TypeError(f"Unsupported generated frame type: {type(frames)}")

    # Ensure the resulting array is 4D (N, H, W, C).
    if array.ndim != 4:
        raise ValueError(f"Expected 4D video array, got shape {tuple(array.shape)}")

    # Convert array to uint8 data type and clip values.
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Generated video contains NaN or infinite values.")
            # Scale float values to 0-255 if they are in 0-1 range, otherwise just clip.
            array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255)
        else:
            # Clip integer-like types to 0-255 range.
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)

    # Convert grayscale (1 channel) to RGB (3 channels).
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    # Remove alpha channel if present (4 channels).
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _expand_worldfoundry_tokens(value: Any) -> Any:
    """Recursively expand WorldFoundry path tokens in runtime kwargs.

    This function searches for WorldFoundry environment tokens (e.g., "$WORLDFOUNDRY_")
    within string values and expands them using the `expand_worldfoundry_path` utility.
    It applies this expansion recursively to values within dictionaries, lists, and tuples.

    Args:
        value: Runtime kwarg value that may contain a WorldFoundry environment token.

    Returns:
        The value with any WorldFoundry tokens expanded.
    """

    # Check if the value is a string containing WorldFoundry tokens.
    if isinstance(value, str) and ("$WORLDFOUNDRY_" in value or "${WORLDFOUNDRY_" in value):
        return str(expand_worldfoundry_path(value))
    # Recursively apply to dictionary values.
    if isinstance(value, dict):
        return {key: _expand_worldfoundry_tokens(item) for key, item in value.items()}
    # Recursively apply to list elements.
    if isinstance(value, list):
        return [_expand_worldfoundry_tokens(item) for item in value]
    # Recursively apply to tuple elements.
    if isinstance(value, tuple):
        return tuple(_expand_worldfoundry_tokens(item) for item in value)
    return value


def _runtime_defaults_from_data(config_path: str, config_key: str | None = None) -> Dict[str, Any]:
    """Load runtime kwargs from a package data config.

    This function reads a serialized configuration file from the package data,
    which is expected to contain default runtime arguments. It supports two formats:
    either a top-level `runtime_kwargs` key, or a `defaults` mapping where a
    specific `config_key` points to the desired runtime arguments.

    Args:
        config_path: The path to the configuration file within the package data.
        config_key: Optional key to use if the config file defines a 'defaults' mapping.

    Returns:
        A dictionary containing the loaded runtime default arguments.

    Raises:
        ValueError: If the config file format is incorrect or a required key is missing.
        KeyError: If `config_key` is provided but not found in the defaults.
    """

    # Resolve the data path and load the serialized configuration payload.
    payload = load_serialized(resolve_data_path(*Path(config_path).parts))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Runtime defaults config must be a mapping: {config_path}")

    # Prioritize 'runtime_kwargs' key if present.
    if "runtime_kwargs" in payload:
        value = payload["runtime_kwargs"]
    else:
        # Otherwise, look for 'defaults' mapping and use 'config_key'.
        defaults = payload.get("defaults")
        if not isinstance(defaults, Mapping):
            raise ValueError(f"Runtime defaults config must define 'defaults': {config_path}")
        if not config_key:
            raise ValueError(f"{config_path} requires a runtime defaults key")
        if config_key not in defaults:
            raise KeyError(f"Runtime defaults key {config_key!r} not found in {config_path}")
        value = defaults[config_key]

    # Ensure the retrieved value is a mapping.
    if not isinstance(value, Mapping):
        raise ValueError(f"Runtime defaults entry must be a mapping: {config_path}:{config_key or 'runtime_kwargs'}")
    return dict(value)


def default_ckpt_path(*parts: str) -> str:
    """Build a checkpoint path from open-source-safe runtime tokens.

    This function constructs a path to a model checkpoint by joining the provided
    parts with the `$WORLDFOUNDRY_CKPT_DIR` environment token, ensuring the path
    is correctly expanded by the WorldFoundry runtime.

    Args:
        *parts: Path components to be appended to the base checkpoint directory.

    Returns:
        An expanded string path to the checkpoint directory.
    """

    return str(expand_worldfoundry_path(str(Path("$WORLDFOUNDRY_CKPT_DIR").joinpath(*parts))))


def default_hfd_path(repo_dir: str) -> str:
    """Build a staged Hugging Face checkpoint path.

    This function constructs a path to a Hugging Face staged directory by joining
    the provided repository directory name with the `$WORLDFOUNDRY_HFD_ROOT` environment
    token, ensuring the path is correctly expanded by the WorldFoundry runtime.

    Args:
        repo_dir: Local HFD directory name such as ``Wan-AI--Wan2.1-T2V-1.3B``.

    Returns:
        An expanded string path to the Hugging Face directory.
    """

    return str(expand_worldfoundry_path(str(Path("$WORLDFOUNDRY_HFD_ROOT") / repo_dir)))


def _filter_supported_constructor_kwargs(runtime_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter keyword arguments to only include those supported by a class constructor.

    This utility inspects the `__init__` signature of `runtime_cls` and returns
    a new dictionary containing only the key-value pairs from `kwargs` that
    correspond to parameters defined in the constructor. If the constructor
    accepts `**kwargs` (i.e., `inspect.Parameter.VAR_KEYWORD`), all original
    kwargs are returned without filtering.

    Args:
        runtime_cls: The class whose constructor signature will be inspected.
        kwargs: A dictionary of keyword arguments to filter.

    Returns:
        A dictionary containing only the kwargs supported by `runtime_cls`'s constructor.
    """
    signature = inspect.signature(runtime_cls)
    parameters = signature.parameters
    # If the constructor accepts **kwargs, return all provided kwargs.
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    # Otherwise, filter kwargs to only include those explicitly defined in the signature.
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }


class RuntimeVideoSynthesis(BaseSynthesis):
    """Base class for one-model video runtime synthesis implementations.

    This abstract class provides a standardized interface for interacting with
    various model-specific video generation runtimes. It manages the lifecycle
    of the underlying generator, handles configuration loading, argument
    translation, and output processing. Concrete subclasses are expected to
    define specific attributes like `MODEL_NAME`, `GENERATION_TYPE`,
    `RUNTIME_CLS`, `RUNTIME_CONFIG_PATH`, and `PRIMARY_PATH_KEY`.
    """

    def __init__(
        self,
        model_name: str,
        generation_type: str,
        runtime_cls,
        runtime_kwargs: Dict[str, Any],
        lazy: bool = True,
    ) -> None:
        """Initializes the RuntimeVideoSynthesis instance.

        Args:
            model_name: The canonical identifier for the model.
            generation_type: The type of generation (e.g., "video", "image").
            runtime_cls: The actual model-specific runtime class to be instantiated.
            runtime_kwargs: Constructor keyword arguments for `runtime_cls`.
            lazy: If True, the `runtime_cls` instance (generator) is created
                  only when `predict` is called for the first time.
        """
        super().__init__()
        self.model_name = model_name
        self.generation_type = generation_type
        self.runtime_cls = runtime_cls
        self.runtime_kwargs = runtime_kwargs
        self.lazy = lazy
        self.generator = None
        # Instantiate the generator immediately if not lazy.
        if not lazy:
            self._ensure_generator()

    @classmethod
    def default_runtime_kwargs(cls) -> Dict[str, Any]:
        """Return runtime kwargs from data-backed defaults.

        This class method retrieves the default runtime configuration for the
        specific model implementation. It expects the class to define
        `RUNTIME_CONFIG_PATH` and optionally `RUNTIME_CONFIG_KEY` (or `MODEL_NAME`).

        Returns:
            A dictionary of default runtime keyword arguments.

        Raises:
            AttributeError: If `RUNTIME_CONFIG_PATH` is not defined for the class.
        """

        config_path = getattr(cls, "RUNTIME_CONFIG_PATH", None)
        if not config_path:
            raise AttributeError(f"{cls.__name__} must define RUNTIME_CONFIG_PATH")
        # Use RUNTIME_CONFIG_KEY or MODEL_NAME as the key in the defaults file.
        config_key = getattr(cls, "RUNTIME_CONFIG_KEY", None) or getattr(cls, "MODEL_NAME", None)
        return _runtime_defaults_from_data(config_path, config_key)

    @classmethod
    def build_runtime_kwargs(
        cls,
        pretrained_model_path=None,
        generator_overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build runtime kwargs from class defaults and caller overrides.

        This method consolidates runtime configuration. It starts with the
        class's default arguments, then applies overrides from `generator_overrides`
        and `kwargs`. It also handles a `pretrained_model_path` by mapping it
        to a specific key defined by `PRIMARY_PATH_KEY`. Finally, it expands
        any WorldFoundry path tokens present in the final arguments.

        Args:
            pretrained_model_path: Optional asset path mapped to `PRIMARY_PATH_KEY`.
            generator_overrides: Explicit runtime kwargs passed by pipeline loaders.
            kwargs: Additional runtime kwargs passed through `from_pretrained`.

        Returns:
            A dictionary of consolidated and expanded runtime keyword arguments.

        Raises:
            ValueError: If `runtime_root` is passed as an override, as it's not supported.
        """
        runtime_kwargs = cls.default_runtime_kwargs()
        # Combine pipeline-specific overrides and direct kwargs.
        overrides = dict(generator_overrides or {})
        overrides.update(kwargs)
        if "runtime_root" in overrides:
            raise ValueError(
                f"{cls.__name__} does not accept runtime_root. Runtime code is owned by its model directory."
            )
        # Apply pretrained_model_path if a primary path key is defined and not already overridden.
        primary_path_key = getattr(cls, "PRIMARY_PATH_KEY", None)
        if pretrained_model_path is not None and primary_path_key and primary_path_key not in overrides:
            overrides[primary_path_key] = pretrained_model_path
        runtime_kwargs.update(overrides)
        # Expand any WorldFoundry environment tokens in the final runtime kwargs.
        runtime_kwargs = {
            key: _expand_worldfoundry_tokens(value)
            for key, value in runtime_kwargs.items()
        }
        return runtime_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        lazy: bool = True,
        generator_overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Create this model's synthesis wrapper from local defaults.

        This is the primary factory method for instantiating the synthesis wrapper.
        It builds the runtime keyword arguments, handles device selection, and
        creates an instance of `RuntimeVideoSynthesis`.

        Args:
            pretrained_model_path: Optional external weight or checkpoint path.
            args: Reserved for compatibility with pipeline loaders (ignored).
            device: Reserved device selector; concrete runtimes own device behavior.
            lazy: Delay runtime construction until first prediction.
            generator_overrides: Explicit runtime kwargs passed by pipeline loaders.
            kwargs: Additional runtime kwargs.

        Returns:
            An instance of the concrete `RuntimeVideoSynthesis` subclass.
        """
        del args  # This argument is typically unused and reserved for compatibility.
        # Build the complete set of runtime keyword arguments.
        runtime_kwargs = cls.build_runtime_kwargs(
            pretrained_model_path=pretrained_model_path,
            generator_overrides=generator_overrides,
            **kwargs,
        )
        # Set 'device' in kwargs if provided and not already present.
        if device is not None:
            runtime_kwargs.setdefault("device", device)
        return cls(
            model_name=cls.MODEL_NAME,
            generation_type=cls.GENERATION_TYPE,
            runtime_cls=cls.RUNTIME_CLS,
            runtime_kwargs=runtime_kwargs,
            lazy=lazy,
        )

    def _ensure_generator(self):
        """Instantiate this model's owned runtime when needed.

        If the `generator` attribute is `None`, this method creates an instance
        of `self.runtime_cls` using the `runtime_kwargs` stored in the object.
        It also filters the `runtime_kwargs` to only include those supported
        by the constructor's signature.

        Returns:
            The instantiated generator object.
        """
        if self.generator is None:
            # Filter kwargs to only pass arguments supported by the runtime_cls constructor.
            self.generator = self.runtime_cls(**_filter_supported_constructor_kwargs(self.runtime_cls, self.runtime_kwargs))
        return self.generator

    def _prediction_runtime_overrides(self, kwargs: Mapping[str, Any], *, fps: Optional[int]) -> Dict[str, Any]:
        """Map Studio call kwargs onto runtime constructor kwargs.

        This method translates a generic set of prediction-time keyword arguments
        into specific arguments expected by the underlying model runtime's
        `generate_video` method or its constructor. It handles argument aliases
        and filters for supported parameters based on the runtime's signature.

        Args:
            kwargs: Per-call options collected by the pipeline.
            fps: Output fps argument passed explicitly to `predict`.

        Returns:
            A dictionary of runtime-specific keyword arguments to be applied as overrides.
        """

        # Defines aliases for common parameters that might have different names.
        aliases = {
            "fps": ("fps", "frame_rate"),
            "frame_rate": ("frame_rate", "fps"),
            "num_frames": ("frames", "num_frames", "video_length"),
            "frame_num": ("frames", "num_frames", "video_length"),
            "max_frames": ("frames", "num_frames", "video_length"),
            "video_length": ("frames", "num_frames", "video_length"),
            "steps": ("sample_steps", "num_inference_steps", "ddim_steps"),
            "num_steps": ("sample_steps", "num_inference_steps", "ddim_steps"),
            "num_inference_steps": ("sample_steps", "num_inference_steps", "ddim_steps"),
            "infer_steps": ("sample_steps", "num_inference_steps", "ddim_steps"),
            "sampling_steps": ("sample_steps", "num_inference_steps", "ddim_steps"),
            "guidance_scale": (
                "sample_guide_scale",
                "guidance_scale",
                "unconditional_guidance_scale",
            ),
            "cfg_scale": (
                "sample_guide_scale",
                "guidance_scale",
                "unconditional_guidance_scale",
            ),
            "seed": ("base_seed", "seed"),
            "shift": ("sample_shift", "time_shift"),
            "time_shift": ("sample_shift", "time_shift"),
        }
        # Defines direct mapping keys that don't need aliases.
        direct_keys = {
            "task",
            "size",
            "frames",
            "num_frames",
            "fps",
            "frame_rate",
            "height",
            "width",
            "sample_steps",
            "num_inference_steps",
            "sample_shift",
            "time_shift",
            "sample_solver",
            "sample_guide_scale",
            "guidance_scale",
            "base_seed",
            "seed",
            "negative_prompt",
            "motion_gs",
            "use_motion_cond",
            "percentage",
            "lcm_origin_steps",
            "offload_model",
            "t5_cpu",
            "t5_fsdp",
            "dit_fsdp",
            "ulysses_size",
            "ring_size",
            "ckpt_dir",
        }
        # Inspect the runtime class constructor to determine supported parameters.
        signature = inspect.signature(self.runtime_cls)
        parameters = signature.parameters
        # Check if the runtime constructor accepts arbitrary keyword arguments (**kwargs).
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        overrides: Dict[str, Any] = {}

        def add_supported(candidates: tuple[str, ...], value: Any) -> None:
            """Helper to add a value to overrides if any candidate key is supported."""
            supported_candidates = [candidate for candidate in candidates if candidate in parameters]
            if supported_candidates:
                # Use the first supported candidate key.
                for candidate in supported_candidates:
                    overrides[candidate] = value
            elif supports_var_kwargs and candidates:
                # If **kwargs is supported, use the first candidate key as a fallback.
                overrides[candidates[0]] = value

        # Apply explicit 'fps' override.
        if fps is not None:
            add_supported(("fps", "frame_rate"), fps)
        # Process other keyword arguments.
        for key, value in kwargs.items():
            if value is None:
                continue
            mapped = aliases.get(key)
            if mapped:
                add_supported(mapped, value)
                continue
            if key in direct_keys:
                add_supported((key,), value)
                continue
        return overrides

    def _apply_prediction_runtime_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Update runtime kwargs and rebuild the generator when constructor args changed.

        This method applies prediction-specific overrides to the `runtime_kwargs`.
        If any of these overrides change a value that affects the generator's
        constructor arguments, the `generator` instance is reset to `None`,
        forcing a re-instantiation on the next prediction call.

        Args:
            overrides: A mapping of new runtime keyword argument values.
        """

        changed = False
        for key, value in overrides.items():
            if self.runtime_kwargs.get(key) != value:
                self.runtime_kwargs[key] = value
                changed = True
        # If any runtime_kwargs were changed, the generator needs to be rebuilt.
        if changed:
            self.generator = None

    def predict(
        self,
        prompt: str,
        images=None,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Generate a video through the model-owned runtime.

        This is the main inference method. It takes a text prompt and optional
        image input, processes prediction-time overrides, ensures the generator
        is instantiated, calls the underlying model's `generate_video` method,
        converts the output frames, and optionally saves the video to a file.

        Args:
            prompt: Text prompt for generation.
            images: Optional image input for image-to-video models (e.g., a PIL Image or path).
            output_path: Optional path for writing the generated video file.
            fps: Optional output frames per second for video saving.
            return_dict: If True, returns a dictionary with video data and metadata;
                         otherwise, returns the raw video array.
            kwargs: Additional write options or runtime-specific arguments.

        Returns:
            A NumPy array of the generated video (uint8, 4D) or a dictionary
            containing the video and its metadata, depending on `return_dict`.
        """
        # Translate and apply prediction-time overrides to runtime_kwargs.
        runtime_overrides = self._prediction_runtime_overrides(kwargs, fps=fps)
        if runtime_overrides:
            self._apply_prediction_runtime_overrides(runtime_overrides)
        # Ensure the generator is instantiated before making a prediction.
        generator = self._ensure_generator()
        # Materialized inputs are needed only for the synchronous runtime call;
        # clean them immediately afterwards so long-lived Studio workers do not
        # accumulate one directory per prediction below /tmp.
        if images is not None:
            with tempfile.TemporaryDirectory(prefix=f"{self.model_name}_") as temp_dir:
                image_path = materialize_image_input(
                    load_pil_image(images),
                    temp_dir,
                    filename="input.png",
                )
                frames = generator.generate_video(prompt=prompt, image_path=image_path)
        else:
            frames = generator.generate_video(prompt=prompt, image_path=None)
        # Convert the generated frames to a standardized uint8 NumPy array.
        video = _frames_to_uint8_array(frames)
        save_path = None
        # Determine the target output path for saving the video.
        target_path = output_path or kwargs.get("save_path")
        if target_path is not None:
            save_path = Path(target_path).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import imageio
            # Save the video using imageio, prioritizing explicit fps, then kwargs fps, then generator's fps, then a default.
            imageio.mimsave(str(save_path), video, fps=fps or kwargs.get("fps") or getattr(generator, "fps", 16) or 16)

        # Prepare the result dictionary.
        result = {
            "video": video,
            "generated_video_path": str(save_path) if save_path is not None else None,
            "prompt": prompt,
            "generation_type": self.generation_type,
            "model_name": self.model_name,
        }
        if return_dict:
            return result
        return video
