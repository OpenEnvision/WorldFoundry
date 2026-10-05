"""
This module defines the AnimateDiffSynthesis class, an adapter for integrating
AnimateDiff video generation capabilities within the worldfoundry runtime
synthesis framework.

It extends RuntimeProfileSynthesis to provide a specialized interface for
loading AnimateDiff models and performing predictions, handling various
configuration options and fallback mechanisms.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.visual_generation.animatediff.worldfoundry_runtime import (
    DEFAULT_ANIMATEDIFF_HF_HUB_CACHE,
    DEFAULT_ANIMATEDIFF_INFERENCE_CONFIG,
    DEFAULT_ANIMATEDIFF_INTEGRATED_ROOT,
    DEFAULT_ANIMATEDIFF_MOTION_MODULE,
    DEFAULT_ANIMATEDIFF_REALISTIC_VISION,
    DEFAULT_SD15_ROOT,
    AnimateDiffRuntime,
)
from worldfoundry.evaluation.models.runtime.profiles import RuntimeProfileSynthesis


def _resolve_sd15_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    for candidate in (
        path.parent / "stable-diffusion-v1-5--stable-diffusion-v1-5",
        path.parent / "runwayml--stable-diffusion-v1-5",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return path


class AnimateDiffSynthesis(RuntimeProfileSynthesis):
    """
    AnimateDiffSynthesis serves as an adapter to integrate AnimateDiff video generation
    into the worldfoundry synthesis runtime.

    It extends RuntimeProfileSynthesis, providing a specialized interface for
    loading AnimateDiff models, configuring their various components (like
    motion modules and base models), and performing video generation tasks.
    """

    MODEL_ID = "animatediff"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "AnimateDiffSynthesis":
        """
        Initializes an AnimateDiffSynthesis instance from a pretrained model path or configuration.

        This method handles the parsing of various configuration options,
        including paths for motion modules, base models, DreamBooth models,
        and runtime settings.

        Args:
            pretrained_model_path: The path to the pretrained model or a dictionary
                                   of options. Can be a string, Path object, or Mapping.
            args: Additional arguments to pass to the superclass `from_pretrained` method.
            device: The device to run the model on (e.g., "cuda", "cpu").
            model_id: An optional ID for the model. Defaults to `cls.MODEL_ID`.
            **kwargs: Arbitrary keyword arguments that can override or provide
                      additional configuration options.

        Returns:
            An initialized AnimateDiffSynthesis instance.
        """
        # Convert pretrained_model_path to a dictionary of options if it's a mapping, otherwise initialize an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a string or Path, treat it as the motion adapter path.
        if isinstance(pretrained_model_path, (str, Path)):
            options["motion_adapter_path"] = str(pretrained_model_path)
        # Update options with any additional keyword arguments provided.
        options.update(kwargs)

        # Call the superclass's from_pretrained to handle base profile initialization.
        instance = super().from_pretrained(
            options,
            args=args,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

        # Extract checkpoints from the profile, ensuring they are mutable dictionaries.
        checkpoints = [dict(item) for item in instance.profile.checkpoints]
        # Attempt to retrieve the motion module path from options.
        motion_module_path = options.get("motion_module_path") or options.get("motion_module")

        # If motion_module_path is not explicitly provided, try to infer it from checkpoints.
        if not motion_module_path:
            for checkpoint in checkpoints:
                local_dir = str(checkpoint.get("local_dir") or "")
                repo_id = str(checkpoint.get("repo_id") or "")
                # Check for specific identifiers related to 'guoyww/animatediff' to locate the motion module.
                if repo_id == "guoyww/animatediff" or local_dir.endswith("guoyww--animatediff"):
                    motion_module_path = str(Path(local_dir) / "mm_sd_v15_v2.ckpt")
                    break

        # Set various instance attributes, prioritizing explicit options, then inferred values, then defaults.
        instance.motion_module_path = str(motion_module_path or DEFAULT_ANIMATEDIFF_MOTION_MODULE)
        instance.base_model_path = str(
            _resolve_sd15_path(
                options.get("sd15_path") or options.get("base_model_path") or DEFAULT_SD15_ROOT
            )
        )
        instance.dreambooth_model_path = str(
            options.get("dreambooth_model_path")
            or options.get("dreambooth_path")
            or DEFAULT_ANIMATEDIFF_REALISTIC_VISION
        )
        instance.official_python = str(options.get("official_python") or sys.executable)
        instance.hf_hub_cache = str(options.get("hf_hub_cache") or DEFAULT_ANIMATEDIFF_HF_HUB_CACHE)
        instance.integrated_runtime_root = str(
            options.get("integrated_runtime_root") or DEFAULT_ANIMATEDIFF_INTEGRATED_ROOT
        )
        instance.inference_config = str(options.get("inference_config") or DEFAULT_ANIMATEDIFF_INFERENCE_CONFIG)
        instance.negative_prompt = str(options.get("negative_prompt", ""))

        # Initialize the AnimateDiffRuntime using the current synthesis instance.
        instance.runtime = AnimateDiffRuntime.from_synthesis(instance)
        return instance

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
        Generates video or images using the configured AnimateDiff model based on the provided prompt and inputs.

        This method attempts to use the internal AnimateDiffRuntime for prediction.
        If an error occurs and `allow_validation_fallback` is True, it falls back
        to the superclass's `predict` method for planning without actual execution.

        Args:
            prompt: The text prompt to guide video generation.
            images: Optional input images for image-to-video generation or control.
            video: Optional input video for video-to-video generation.
            interactions: A sequence of interaction strings, if applicable.
            output_path: The path where the generated output (video/images) should be saved.
            fps: Frames per second for video generation.
            **kwargs: Additional keyword arguments to pass to the underlying runtime.
                      Includes `allow_validation_fallback` which determines fallback behavior.

        Returns:
            A dictionary containing the prediction results, typically including the output path.

        Raises:
            Exception: If an error occurs during prediction and `allow_validation_fallback` is False.
        """
        # Determine if a fallback to validation-only prediction is allowed.
        allow_validation_fallback = bool(kwargs.pop("allow_validation_fallback", False))
        runtime_kwargs = dict(kwargs)

        try:
            # Attempt to perform prediction using the specialized AnimateDiff runtime.
            return self.runtime.predict(
                prompt=prompt,
                images=images,
                video=video,
                interactions=interactions,
                output_path=output_path,
                fps=fps,
                **runtime_kwargs,
            )
        except Exception:
            # If an exception occurs and fallback is not allowed, re-raise the exception.
            if not allow_validation_fallback:
                raise
            # If fallback is allowed, call the superclass's predict method with `plan_only=True`
            # to perform validation without actual execution.
            return super().predict(
                prompt=prompt,
                images=images,
                video=video,
                interactions=interactions,
                output_path=output_path,
                fps=fps,
                plan_only=True,
                **kwargs,
            )
