"""
This module provides a synthesis adapter for the WorldFM model.

It wraps the underlying WorldFM runtime to provide a standardized interface
for generating visual content based on frame conditions, integrating it
into a broader synthesis framework.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from worldfoundry.synthesis.visual_generation.worldfm.runtime import (
    WorldFMRuntime,
    load_worldfm_runtime,
)
from worldfoundry.synthesis.visual_generation.worldfm.worldfm_runtime import DEFAULT_WORLDFM_REPO

from ...base_synthesis import BaseSynthesis


class WorldFMSynthesis(BaseSynthesis):
    """
    A synthesis adapter for the WorldFM model.

    This class provides a high-level interface to interact with the WorldFM runtime
    for generating visual content. It extends `BaseSynthesis` to fit into a broader
    synthesis framework, simplifying integration and usage.
    """

    def __init__(self, runtime: WorldFMRuntime) -> None:
        """
        Initializes the WorldFMSynthesis adapter with a WorldFM runtime instance.

        Args:
            runtime: An initialized instance of WorldFMRuntime, which handles the
                     core model loading and inference logic.
        """
        super().__init__()
        self.runtime = runtime
        # Expose key properties from the underlying runtime for direct access
        self.service = runtime.service
        self.checkpoint_path = runtime.checkpoint_path
        self.vae_path = runtime.vae_path
        self.step = runtime.step
        self.image_size = runtime.image_size
        self.version = runtime.version
        self.cfg_scale = runtime.cfg_scale
        self.device = runtime.device

    @classmethod
    def _resolve_assets(
        cls,
        checkpoint_source: str | None,
        vae_source: str | None,
        *,
        step: int,
        checkpoint_filename: str | None,
    ) -> tuple[str, str]:
        """
        Resolves asset paths for the WorldFM model.

        This class method delegates asset resolution to the underlying WorldFMRuntime.
        It determines the actual paths for the checkpoint and VAE model files
        based on the provided sources and parameters.

        Args:
            checkpoint_source: The source path or identifier for the model checkpoint.
                               Can be a local path, Hugging Face ID, etc.
            vae_source: The source path or identifier for the VAE model.
            step: The training step associated with the checkpoint, used for resolution.
            checkpoint_filename: The specific filename of the checkpoint to load if
                                 multiple exist at the source.

        Returns:
            A tuple containing the resolved path for the checkpoint and the VAE model.
        """
        return WorldFMRuntime.resolve_assets(
            checkpoint_source,
            vae_source,
            step=step,
            checkpoint_filename=checkpoint_filename,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_WORLDFM_REPO,
        args: Any = None,
        device: str | None = None,
        vae_path: str | None = None,
        checkpoint_filename: str | None = None,
        step: int = 2,
        image_size: int = 512,
        version: str = "sigma",
        cfg_scale: float = 4.5,
        weight_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> WorldFMSynthesis:
        """
        Loads a pre-trained WorldFM model and initializes the synthesis adapter.

        This factory method simplifies the process of creating a WorldFMSynthesis instance
        by loading the underlying WorldFM runtime from specified pre-trained assets.

        Args:
            pretrained_model_path: The path or Hugging Face repository ID for the
                                   pre-trained WorldFM model. Defaults to `DEFAULT_WORLDFM_REPO`.
            args: Legacy argument, currently ignored and will be removed.
            device: The device to load the model onto (e.g., "cuda", "cpu"). If None,
                    it will default based on `torch.cuda.is_available()`.
            vae_path: The path to the VAE model. If None, a default VAE will be used.
            checkpoint_filename: The specific filename of the checkpoint within the
                                 `pretrained_model_path` to load.
            step: The model checkpoint's training step.
            image_size: The desired output image size (width and height).
            version: The model version to use (e.g., "sigma").
            cfg_scale: Classifier-free guidance scale. Higher values encourage
                       stronger adherence to the prompt.
            weight_dtype: The data type to use for model weights (e.g., torch.float16).
            **kwargs: Additional keyword arguments to pass to the underlying
                      `load_worldfm_runtime` function.

        Raises:
            ValueError: If 'plan_only' or 'output_path' are present in kwargs,
                        as these options are no longer supported.

        Returns:
            An instance of WorldFMSynthesis initialized with the loaded model.
        """
        del args  # Remove legacy argument
        # Check for deprecated arguments that are no longer supported
        if "plan_only" in kwargs:
            raise ValueError("WorldFM no longer supports plan_only; provide local assets and run the real runtime.")
        if "output_path" in kwargs:
            raise ValueError("WorldFM no longer writes runtime plans; use output_dir during predict().")

        # Load the WorldFM runtime with the specified parameters
        loaded = load_worldfm_runtime(
            pretrained_model_path=pretrained_model_path,
            device=device,
            vae_path=vae_path,
            checkpoint_filename=checkpoint_filename,
            step=step,
            image_size=image_size,
            version=version,
            cfg_scale=cfg_scale,
            weight_dtype=weight_dtype,
        )
        return cls(loaded)

    def predict(
        self,
        frame_conditions: Sequence[dict[str, Any]],
        output_dir: str | Path | None = None,
        scene_name: str = "worldfm_scene",
        save_mode: str = "video",
        fps: int = 30,
        return_dict: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """
        Generates visual content based on a sequence of frame conditions.

        This method delegates the actual prediction task to the underlying WorldFM runtime.
        It can generate images or videos based on the provided conditions and saving preferences.

        Args:
            frame_conditions: A sequence of dictionaries, each describing the conditions
                              for a single frame (e.g., prompt, style, camera settings, etc.).
            output_dir: The directory where the generated output (video/images) should be saved.
                        If None, output might be returned in memory depending on `save_mode`.
            scene_name: A name for the scene, used for naming output files/folders.
            save_mode: Specifies how to save the output. Can be "video", "images", etc.
            fps: Frames per second for video output, if `save_mode` is "video".
            return_dict: If True, returns results as a dictionary (e.g., {"video_path": ...}).
                         If False, might return a list of generated frames or paths.
            seed: An optional random seed for reproducible generation.
            **kwargs: Additional keyword arguments to pass to the underlying runtime's predict method.

        Returns:
            The prediction results, which can be a dictionary containing output paths/data
            or a list of generated assets, depending on `return_dict` and `save_mode`.
        """
        # Delegate the prediction call to the internal WorldFMRuntime instance
        return self.runtime.predict(
            frame_conditions,
            output_dir=output_dir,
            scene_name=scene_name,
            save_mode=save_mode,
            fps=fps,
            return_dict=return_dict,
            seed=seed,
            **kwargs,
        )
