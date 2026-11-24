"""
This module provides a synthesis adapter for the WorldCam runtime, enabling integration
with a broader synthesis framework. It defines `WorldCamSynthesis` as a thin wrapper
around `WorldCamRuntime`, delegating most operations to the underlying runtime object.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.visual_generation.worldcam.worldfoundry_runtime import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_SHARED_HFD_ROOT,
    DEFAULT_WAN_MODEL_DIR,
    DEFAULT_WAN_REPO,
    DEFAULT_WEIGHT_DTYPE,
    DEFAULT_WORLDCAM_CHECKPOINT,
    DEFAULT_WORLDCAM_CKPT_DIR,
    DEFAULT_WORLDCAM_REPO,
    OFFICIAL_SOURCE_REPO,
    WorldCamRuntime,
)

try:
    from ...base_synthesis import BaseSynthesis
except ModuleNotFoundError as exc:
    # If the module is not found, check if it's due to torch not being installed.
    # If it's a different module error, re-raise it.
    if exc.name != "torch":
        raise

    class BaseSynthesis:  # type: ignore[no-redef]
        """
        A placeholder BaseSynthesis class used when the full `base_synthesis`
        module cannot be imported (e.g., if torch is not available).
        """

        def __init__(self):
            """Initializes the placeholder BaseSynthesis."""


class WorldCamSynthesis(BaseSynthesis):
    """
    Thin synthesis adapter for the WorldCam runtime.

    This class wraps a `WorldCamRuntime` instance, providing a `BaseSynthesis`-compatible
    interface while delegating actual model operations (like planning and prediction)
    to the underlying runtime. It also supports `from_pretrained` for easy instantiation.
    """

    def __init__(self, runtime: WorldCamRuntime):
        """
        Initializes the WorldCamSynthesis adapter with a pre-configured WorldCamRuntime instance.

        Args:
            runtime: An initialized instance of `WorldCamRuntime`.
        """
        super().__init__()
        self.runtime = runtime

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying WorldCamRuntime instance.

        This allows direct access to methods and attributes of the `self.runtime` object
        as if they were part of `WorldCamSynthesis` itself.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The attribute from the underlying `WorldCamRuntime` instance.
        """
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = DEFAULT_WAN_REPO,
        worldcam_ckpt_path: str | Path = DEFAULT_WORLDCAM_REPO,
        device: str = "cuda",
        weight_dtype: Any = DEFAULT_WEIGHT_DTYPE,
        checkpoint_name: str = DEFAULT_WORLDCAM_CHECKPOINT,
        prompt_prefix: str = "<A first-person shooter CS game> ",
        height: int = 480,
        width: int = 832,
        load_model: bool = True,
        allow_download: bool = False,
        **kwargs: Any,
    ) -> WorldCamSynthesis:
        """
        Instantiates WorldCamSynthesis by loading a WorldCamRuntime from pretrained weights.

        This class method wraps the `WorldCamRuntime.from_pretrained` method,
        providing a convenient way to set up the synthesis adapter.

        Args:
            pretrained_model_path: Path to the pretrained model or Hugging Face repository ID for WAN.
            worldcam_ckpt_path: Path to the WorldCam checkpoint or Hugging Face repository ID.
            device: The device to load the model onto (e.g., "cuda", "cpu").
            weight_dtype: The data type for model weights (e.g., torch.float16, torch.float32).
            checkpoint_name: The specific checkpoint file name within `worldcam_ckpt_path`.
            prompt_prefix: A prefix to be added to all user prompts for generation.
            height: The height of the generated video frames.
            width: The width of the generated video frames.
            load_model: If True, the model weights are loaded immediately.
            allow_download: If True, allows downloading models from Hugging Face if not found locally.
            **kwargs: Additional keyword arguments passed directly to `WorldCamRuntime.from_pretrained`.

        Returns:
            An instance of `WorldCamSynthesis` initialized with the loaded runtime.
        """
        runtime = WorldCamRuntime.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            worldcam_ckpt_path=worldcam_ckpt_path,
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_name=checkpoint_name,
            prompt_prefix=prompt_prefix,
            height=height,
            width=width,
            load_model=load_model,
            allow_download=allow_download,
            **kwargs,
        )
        return cls(runtime)

    @staticmethod
    def _runtime_root() -> Path:
        """
        Returns the root directory where runtime-related assets are stored.

        This is a static method that delegates to `WorldCamRuntime.runtime_root()`.

        Returns:
            A Path object representing the runtime root directory.
        """
        return WorldCamRuntime.runtime_root()

    def plan(
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
        Generates a plan for video synthesis based on the provided inputs.

        This method delegates the planning operation to the underlying `WorldCamRuntime` instance.

        Args:
            prompt: The text prompt describing the desired video content.
            images: Optional input images to condition the generation.
            video: Optional input video to condition the generation.
            interactions: A sequence of interaction prompts for multi-turn generation.
            output_path: Optional path to save the generated output.
            fps: Optional frames per second for the output video.
            **kwargs: Additional keyword arguments passed directly to `WorldCamRuntime.plan`.

        Returns:
            A dictionary containing the planning results, as returned by the runtime.
        """
        return self.runtime.plan(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )

    def predict(
        self,
        prompt: str,
        condition_video,
        intrinsics: Any,
        extrinsics: Any,
        num_ar_steps: int,
        negative_prompt: str | None = None,
        cfg_scale: float = 4.0,
        seed: int = 0,
        long_term_memory_start_step: int = 30,
        long_term_memory_num_clips: int = 4,
        long_term_memory_ref_indices: list[int] | None = None,
        attention_sink_inference: bool = False,
        trim_conditioning: bool = True,
        num_inference_steps: int = 50,
        tiled: bool = True,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """
        Generates a video based on the provided prompt and conditioning information.

        This method delegates the prediction (video generation) operation to the
        underlying `WorldCamRuntime` instance.

        Args:
            prompt: The main text prompt for video generation.
            condition_video: The video to use as conditioning input.
            intrinsics: Camera intrinsic parameters for the scene.
            extrinsics: Camera extrinsic parameters (pose) for the scene.
            num_ar_steps: Number of auto-regressive steps to perform.
            negative_prompt: An optional negative prompt to guide generation away from certain concepts.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            long_term_memory_start_step: The step at which long-term memory processing begins.
            long_term_memory_num_clips: Number of clips to use for long-term memory.
            long_term_memory_ref_indices: Specific indices of reference frames for long-term memory.
            attention_sink_inference: Whether to use attention sink inference for improved consistency.
            trim_conditioning: If True, trims the conditioning video to fit the generation length.
            num_inference_steps: Number of diffusion inference steps.
            tiled: If True, uses tiled processing for larger resolutions or reduced memory usage.
            return_dict: If True, returns a dictionary of results; otherwise, returns a raw output.
            **kwargs: Additional keyword arguments passed directly to `WorldCamRuntime.predict`.

        Returns:
            The prediction output, which can be a video tensor, a dictionary, or other
            format depending on the `return_dict` flag and the runtime's implementation.
        """
        return self.runtime.predict(
            prompt=prompt,
            condition_video=condition_video,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_ar_steps=num_ar_steps,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            seed=seed,
            long_term_memory_start_step=long_term_memory_start_step,
            long_term_memory_num_clips=long_term_memory_num_clips,
            long_term_memory_ref_indices=long_term_memory_ref_indices,
            attention_sink_inference=attention_sink_inference,
            trim_conditioning=trim_conditioning,
            num_inference_steps=num_inference_steps,
            tiled=tiled,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_SHARED_HFD_ROOT",
    "DEFAULT_WAN_MODEL_DIR",
    "DEFAULT_WAN_REPO",
    "DEFAULT_WEIGHT_DTYPE",
    "DEFAULT_WORLDCAM_CHECKPOINT",
    "DEFAULT_WORLDCAM_CKPT_DIR",
    "DEFAULT_WORLDCAM_REPO",
    "OFFICIAL_SOURCE_REPO",
    "WorldCamSynthesis",
]
