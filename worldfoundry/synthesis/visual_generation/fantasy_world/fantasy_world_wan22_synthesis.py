"""
This module provides a wrapper for the FantasyWorld Wan2.2 video generation model,
allowing for synthesis of videos from an initial image, optional end image,
and text prompt, guided by camera motion parameters.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...base_synthesis import BaseSynthesis
from .runtime_env import (
    DEFAULT_FANTASY_WORLD_MOGE2_REPO,
    DEFAULT_FANTASY_WORLD_WAN22_REPO,
    resolve_fantasy_world_wan22_base_dir,
    resolve_fantasy_world_wan22_checkpoint_dir,
    resolve_fantasy_world_wan22_lora_dir,
)
from .wan22_runner import build_wan22_runner
from .worldfoundry_runtime import (
    build_camera_params,
    save_prediction_artifacts,
)


class FantasyWorldWan22Synthesis(BaseSynthesis):
    """
    A wrapper class for the FantasyWorld Wan2.2 video synthesis model.

    This class provides an interface to load a pre-trained Wan2.2 model
    and generate videos based on input images, text prompts, and camera parameters.
    It manages the underlying model runner and artifact saving.
    """

    def __init__(
        self,
        runner,
        *,
        checkpoint_dir: str,
        base_model_path: str,
        lora_path: str,
        moge_pretrained: str,
        device: str = "cuda",
    ) -> None:
        """
        Initializes the FantasyWorldWan22Synthesis instance.

        Args:
            runner: The pre-configured Wan2.2 model runner instance.
            checkpoint_dir (str): The directory containing model checkpoints.
            base_model_path (str): The path to the base diffusion model.
            lora_path (str): The path to the LoRA weights directory.
            moge_pretrained (str): The path or name of the pre-trained MoGE model.
            device (str): The device (e.g., "cuda" or "cpu") where the model will run.
        """
        super().__init__()
        self.runner = runner
        self.checkpoint_dir = checkpoint_dir
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.moge_pretrained = moge_pretrained
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_FANTASY_WORLD_WAN22_REPO,
        args=None,
        device: str | None = None,
        wan_model_path: str | None = None,
        lora_path: str | None = None,
        moge_path: str | None = None,
        moge_pretrained: str | None = None,
        sample_steps: int = 50,
        cfg_scale: float = 5.0,
        timestep_boundary: int = 900,
        frames: int = 81,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        base_seed: int = 1024,
        high_model_device: str | None = None,
        low_model_device: str | None = None,
        moge_device: str | None = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        model_ckpt_high: str | None = None,
        model_ckpt_low: str | None = None,
        **kwargs,
    ) -> FantasyWorldWan22Synthesis:
        """
        Loads a pre-trained FantasyWorld Wan2.2 synthesis model.

        This class method resolves model paths, validates checkpoints, and
        configures the underlying Wan2.2 runner based on the provided parameters.

        Args:
            pretrained_model_path (str): Path to the pre-trained Wan2.2 model repository.
            args: Optional additional arguments (not directly used here, but for compatibility).
            device (Optional[str]): The primary device for the model (e.g., "cuda", "cpu").
            wan_model_path (Optional[str]): Path to the base diffusion model for Wan2.2.
            lora_path (Optional[str]): Path to the LoRA weights directory.
            moge_path (Optional[str]): Path to the MoGE model.
            moge_pretrained (Optional[str]): Identifier for a pre-trained MoGE model.
            sample_steps (int): Number of sampling steps for video generation.
            cfg_scale (float): Classifier-free guidance scale.
            timestep_boundary (int): Timestep boundary for noise models.
            frames (int): Number of frames to generate in the video.
            fps (int): Frames per second for the generated video.
            height (int): Height of the generated video frames.
            width (int): Width of the generated video frames.
            base_seed (int): Base random seed for reproducibility.
            high_model_device (Optional[str]): Specific device for the high-noise model.
            low_model_device (Optional[str]): Specific device for the low-noise model.
            moge_device (Optional[str]): Specific device for the MoGE model.
            weight_dtype (torch.dtype): Data type for model weights (e.g., torch.bfloat16).
            model_ckpt_high (Optional[str]): Path to the high-noise model checkpoint.
            model_ckpt_low (Optional[str]): Path to the low-noise model checkpoint.
            **kwargs: Additional keyword arguments.

        Returns:
            FantasyWorldWan22Synthesis: An initialized instance of the synthesis model.

        Raises:
            FileNotFoundError: If required model checkpoints are not found.
        """
        resolved_device = device or "cuda"

        # Resolve paths for various model components, using defaults if not provided.
        checkpoint_dir = resolve_fantasy_world_wan22_checkpoint_dir(pretrained_model_path)
        base_model_dir = resolve_fantasy_world_wan22_base_dir(wan_model_path)
        lora_dir = resolve_fantasy_world_wan22_lora_dir(lora_path)
        resolved_moge_pretrained = moge_pretrained or DEFAULT_FANTASY_WORLD_MOGE2_REPO

        # Determine the specific paths for high and low noise model checkpoints.
        # If not provided, default to paths within the resolved checkpoint_dir.
        high_ckpt = (
            Path(model_ckpt_high).expanduser().resolve() if model_ckpt_high else checkpoint_dir / "high_noise_model.pth"
        )
        low_ckpt = (
            Path(model_ckpt_low).expanduser().resolve() if model_ckpt_low else checkpoint_dir / "low_noise_model.pth"
        )

        # Validate that the necessary checkpoint files exist.
        if not high_ckpt.is_file():
            raise FileNotFoundError(f"FantasyWorld Wan2.2 high checkpoint not found: {high_ckpt}")
        if not low_ckpt.is_file():
            raise FileNotFoundError(f"FantasyWorld Wan2.2 low checkpoint not found: {low_ckpt}")

        # Build the Wan2.2 runner with all resolved paths and parameters.
        runner = build_wan22_runner(
            base_dir=str(base_model_dir),
            lora_dir=str(lora_dir),
            model_ckpt_high=str(high_ckpt),
            model_ckpt_low=str(low_ckpt),
            moge_path=moge_path,
            moge_pretrained=resolved_moge_pretrained,
            base_seed=base_seed,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            timestep_boundary=timestep_boundary,
            frames=frames,
            fps=fps,
            height=height,
            width=width,
            device=resolved_device,
            high_model_device=high_model_device,
            low_model_device=low_model_device,
            moge_device=moge_device,
            weight_dtype=weight_dtype,
        )

        # Return an instance of the class with the configured runner and paths.
        return cls(
            runner=runner,
            checkpoint_dir=str(checkpoint_dir),
            base_model_path=str(base_model_dir),
            lora_path=str(lora_dir),
            moge_pretrained=str(resolved_moge_pretrained),
            device=resolved_device,
        )

    def predict(
        self,
        *,
        image,
        end_image=None,
        prompt: str,
        camera_source,
        K=None,
        output_dir: str | None = None,
        scene_name: str = "fantasyworld_wan22_scene",
        neg_prompt: str | None = None,
        fps: int = 16,
        using_scale: bool = True,
        conf_threshold: float = 1.5,
        stride: int = 4,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Generates a video based on an initial image, optional end image, text prompt, and camera parameters.

        Args:
            image: The initial PIL Image for the video.
            end_image: Optional PIL Image for the end frame of the video.
            prompt (str): The text prompt guiding the video generation.
            camera_source: Camera trajectory source (e.g., a path to a JSON file or a dictionary).
            K: Optional intrinsic camera matrix.
            output_dir (Optional[str]): Directory to save generated artifacts. If None, artifacts are not saved.
            scene_name (str): Name for the generated scene, used for artifact naming.
            neg_prompt (Optional[str]): Negative text prompt to guide generation away from certain concepts.
            fps (int): Frames per second for the output video.
            using_scale (bool): Whether to use a specific scaling factor in generation.
            conf_threshold (float): Confidence threshold for point cloud generation.
            stride (int): Stride for processing frames when generating point cloud.
            return_dict (bool): If True, returns a dictionary of results; otherwise, returns frames directly.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[List[PIL.Image.Image], Dict[str, Any]]: The generated video frames
            (list of PIL Images) or a dictionary containing frames, video path,
            point cloud path, and other metadata.
        """
        # Convert input images to RGB format if they are not already.
        image_rgb = image.convert("RGB")
        end_image_rgb = end_image.convert("RGB") if end_image is not None else None

        # Build camera parameters from the source, adapting to the model's expected input size.
        camera_params = build_camera_params(
            camera_source,
            image_size=(self.runner.height, self.runner.width),
            K=K,
            model_name="FantasyWorld Wan2.2",
        )

        # Generate the video frames and prediction data using the configured runner.
        frames, prediction = self.runner.generate_video(
            image=image_rgb,
            end_image=end_image_rgb,
            prompt=prompt or "",  # Ensure prompt is a string, even if empty.
            neg_prompt=neg_prompt,
            camera_params=camera_params,
            using_scale=using_scale,
        )

        generated_video_path = None
        pointcloud_path = None
        saved_output_dir = None

        # If an output directory is provided, save the generated artifacts.
        if output_dir is not None:
            artifacts = save_prediction_artifacts(
                frames=frames,
                prediction=prediction,
                output_dir=output_dir,
                scene_name=scene_name,
                fps=fps,
                conf_threshold=conf_threshold,
                stride=stride,
                mask_operator=">",  # Operator used for creating masks, typically for thresholding.
            )
            generated_video_path = artifacts["generated_video_path"]
            pointcloud_path = artifacts["pointcloud_path"]
            saved_output_dir = artifacts["output_dir"]

        # Compile results into a dictionary.
        result = {
            "frames": frames,
            "video": frames,  # Alias for frames for backward compatibility/clarity.
            "prediction": prediction,
            "camera_params": camera_params,
            "generated_video_path": generated_video_path,
            "pointcloud_path": pointcloud_path,
            "scene_name": scene_name,
            "output_dir": saved_output_dir,
            "fps": fps,
            "variant": "wan22",
        }

        # Return the full result dictionary or just the frames based on the flag.
        if return_dict:
            return result
        return result["frames"]
