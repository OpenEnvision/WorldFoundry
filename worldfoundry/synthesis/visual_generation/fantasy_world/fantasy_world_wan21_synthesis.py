"""
Module for synthesizing 3D scenes and videos using the FantasyWorld Wan2.1 model.

This module provides a wrapper (`FantasyWorldWan21Synthesis`) around the
FantasyWorld Wan2.1 model, allowing users to generate videos and 3D point clouds
from an input image and a text prompt. It handles model loading, camera
parameter setup, video generation, and artifact saving.
"""

from __future__ import annotations

import torch

from ...base_synthesis import BaseSynthesis
from .runtime_env import (
    DEFAULT_FANTASY_WORLD_MOGE2_REPO,
    DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT,
    DEFAULT_FANTASY_WORLD_WAN21_REPO,
    resolve_fantasy_world_wan21_base_dir,
    resolve_fantasy_world_wan21_checkpoint,
)
from .wan21_runner import FantasyWorldWan21Runner
from .worldfoundry_runtime import (
    build_camera_params,
    save_prediction_artifacts,
)


class FantasyWorldWan21Synthesis(BaseSynthesis):
    """FantasyWorld Wan2.1 synthesis wrapper.

    This class provides an interface to interact with the FantasyWorld Wan2.1 model
    for generating 3D scenes and videos from an input image and a text prompt.
    It manages the model runner and configuration.
    """

    def __init__(
        self,
        runner: FantasyWorldWan21Runner,
        *,
        checkpoint_path: str,
        wan_model_path: str,
        moge_pretrained: str,
        device: str = "cuda",
    ) -> None:
        """Initializes the FantasyWorldWan21Synthesis instance.

        Args:
            runner (FantasyWorldWan21Runner): The underlying runner responsible for model inference.
            checkpoint_path (str): Path to the Wan2.1 model checkpoint.
            wan_model_path (str): Path to the base directory of the Wan2.1 model.
            moge_pretrained (str): Path to the pre-trained MOGE model repository.
            device (str, optional): The device to run the model on (e.g., "cuda", "cpu"). Defaults to "cuda".
        """
        super().__init__()
        self.runner = runner
        self.checkpoint_path = checkpoint_path
        self.wan_model_path = wan_model_path
        self.moge_pretrained = moge_pretrained
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_FANTASY_WORLD_WAN21_REPO,
        args=None,
        device: str | None = None,
        wan_model_path: str | None = None,
        moge_path: str | None = None,
        moge_pretrained: str | None = None,
        sample_steps: int = 50,
        sample_guide_scale: float = 5.0,
        frames: int = 81,
        fps: int = 16,
        height: int = 336,
        width: int = 592,
        start_index: int = 16,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> FantasyWorldWan21Synthesis:
        """Factory method to create a FantasyWorldWan21Synthesis instance from a pretrained model.

        This method handles resolving model paths and initializing the
        FantasyWorldWan21Runner with specified parameters.

        Args:
            pretrained_model_path (str, optional): Path to the pretrained Wan2.1 model repository.
                Defaults to DEFAULT_FANTASY_WORLD_WAN21_REPO.
            args (Any, optional): Additional arguments, currently unused. Defaults to None.
            device (Optional[str], optional): The device to run the model on (e.g., "cuda", "cpu").
                If None, defaults to "cuda".
            wan_model_path (Optional[str], optional): Custom path to the Wan2.1 base model directory.
                If None, it will be resolved from `pretrained_model_path`.
            moge_path (Optional[str], optional): Custom path to the MOGE model checkpoint. Defaults to None.
            moge_pretrained (Optional[str], optional): Path to the pre-trained MOGE model repository.
                If None, defaults to DEFAULT_FANTASY_WORLD_MOGE2_REPO.
            sample_steps (int, optional): Number of sampling steps for the generation process. Defaults to 50.
            sample_guide_scale (float, optional): Guidance scale for sampling. Defaults to 5.0.
            frames (int, optional): Number of frames to generate in the video. Defaults to 81.
            fps (int, optional): Frames per second for the generated video. Defaults to 16.
            height (int, optional): Height of the generated video frames. Defaults to 336.
            width (int, optional): Width of the generated video frames. Defaults to 592.
            start_index (int, optional): Starting index for frame generation. Defaults to 16.
            weight_dtype (torch.dtype, optional): Data type for model weights (e.g., torch.bfloat16).
                Defaults to torch.bfloat16.
            **kwargs: Additional keyword arguments to pass to the runner.

        Returns:
            FantasyWorldWan21Synthesis: An initialized instance of the synthesis wrapper.
        """
        resolved_device = device or "cuda"
        # Resolve model paths based on provided arguments or defaults
        checkpoint_path = resolve_fantasy_world_wan21_checkpoint(pretrained_model_path)
        resolved_wan_model_path = resolve_fantasy_world_wan21_base_dir(wan_model_path)
        resolved_moge_pretrained = moge_pretrained or DEFAULT_FANTASY_WORLD_MOGE2_REPO

        # Initialize the core model runner with all necessary configurations
        runner = FantasyWorldWan21Runner(
            ckpt_dir=str(resolved_wan_model_path),
            model_ckpt=str(checkpoint_path),
            moge_path=moge_path,
            moge_pretrained=resolved_moge_pretrained,
            sample_steps=sample_steps,
            sample_guide_scale=sample_guide_scale,
            frames=frames,
            fps=fps,
            height=height,
            width=width,
            start_index=start_index,
            device=resolved_device,
            weight_dtype=weight_dtype,
        )
        return cls(
            runner=runner,
            checkpoint_path=str(checkpoint_path),
            wan_model_path=str(resolved_wan_model_path),
            moge_pretrained=str(resolved_moge_pretrained),
            device=resolved_device,
        )

    def predict(
        self,
        *,
        image,
        prompt: str,
        camera_source,
        K=None,
        output_dir: str | None = None,
        scene_name: str = "fantasyworld_wan21_scene",
        neg_prompt: str | None = None,
        fps: int = 16,
        seed: int = 1024,
        using_scale: bool = True,
        conf_threshold: float = 1.0,
        stride: int = 4,
        return_dict: bool = False,
        **kwargs,
    ):
        """Generates a video and 3D point cloud from an input image and text prompt.

        Args:
            image: The input image (PIL Image or similar) to condition the generation.
            prompt (str): The positive text prompt guiding the scene generation.
            camera_source: Parameters defining the camera perspective.
            K (Any, optional): Intrinsic camera matrix. If None, it will be derived. Defaults to None.
            output_dir (Optional[str], optional): Directory to save generated artifacts (video, point cloud).
                If None, artifacts are not saved to disk. Defaults to None.
            scene_name (str, optional): Name to use for the output scene files. Defaults to "fantasyworld_wan21_scene".
            neg_prompt (Optional[str], optional): The negative text prompt to guide generation away from.
                If None, uses DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT. Defaults to None.
            fps (int, optional): Frames per second for the output video. Defaults to 16.
            seed (int, optional): Random seed for reproducible generation. Defaults to 1024.
            using_scale (bool, optional): Whether to use scale in the generation process. Defaults to True.
            conf_threshold (float, optional): Confidence threshold for point cloud generation. Defaults to 1.0.
            stride (int, optional): Stride for processing or artifact saving. Defaults to 4.
            return_dict (bool, optional): If True, returns a dictionary of results. Otherwise, returns frames.
                Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[List[PIL.Image.Image], Dict[str, Any]]: The generated video frames as a list of PIL Images,
            or a dictionary containing frames, video path, point cloud path, and other metadata if `return_dict` is True.
        """
        image_rgb = image.convert("RGB")
        # Build camera parameters based on source and image size for the model
        camera_params = build_camera_params(
            camera_source,
            image_size=(self.runner.height, self.runner.width),
            K=K,
            model_name="FantasyWorld Wan2.1",
        )
        # Generate video frames and 3D prediction using the runner
        frames, prediction = self.runner.generate_video(
            image=image_rgb,
            camera_params=camera_params,
            prompt=prompt or "",
            neg_prompt=neg_prompt if neg_prompt is not None else DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT,
            using_scale=using_scale,
            seed=seed,
        )

        generated_video_path = None
        pointcloud_path = None
        saved_output_dir = None
        if output_dir is not None:
            # Save the generated video frames and 3D prediction artifacts to the specified output directory
            artifacts = save_prediction_artifacts(
                frames=frames,
                prediction=prediction,
                output_dir=output_dir,
                scene_name=scene_name,
                fps=fps,
                conf_threshold=conf_threshold,
                stride=stride,
                mask_operator=">=",
            )
            generated_video_path = artifacts["generated_video_path"]
            pointcloud_path = artifacts["pointcloud_path"]
            saved_output_dir = artifacts["output_dir"]

        result = {
            "frames": frames,
            "video": frames,  # Alias for frames for backward compatibility or clarity
            "prediction": prediction,
            "camera_params": camera_params,
            "generated_video_path": generated_video_path,
            "pointcloud_path": pointcloud_path,
            "scene_name": scene_name,
            "output_dir": saved_output_dir,
            "fps": fps,
            "variant": "wan21",
        }
        if return_dict:
            return result
        return result["frames"]
