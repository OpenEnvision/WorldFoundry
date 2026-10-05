"""
This module provides the CameraCtrlRuntime class, which encapsulates the
inference logic for the CameraCtrl text-to-video generation model.

It defines the runtime for generating videos based on text prompts and camera
trajectory specifications, leveraging a Stable Diffusion 1.5 base model
and specialized CameraCtrl components.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from worldfoundry.core.io import file_sha256
from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.runtime.env import resolve_hfd_root

DEFAULT_CAMERACTRL_CONFIG = worldfoundry_data_path(
    "models",
    "runtime", "configs",
    "camera_control",
    "cameractrl_256_384.yaml",
)
DEFAULT_SHARED_HFD_ROOT = resolve_hfd_root()
DEFAULT_SD15_ROOT = Path(
    os.environ.get(
        "WORLDFOUNDRY_SD15_ROOT",
        str(DEFAULT_SHARED_HFD_ROOT / "stable-diffusion-v1-5--stable-diffusion-v1-5"),
    )
)
DEFAULT_CAMERACTRL_CKPT = DEFAULT_SHARED_HFD_ROOT / "hehao13--CameraCtrl" / "CameraCtrl.ckpt"
DEFAULT_CAMERACTRL_IMAGE_LORA = DEFAULT_SHARED_HFD_ROOT / "hehao13--CameraCtrl" / "RealEstate10K_LoRA.ckpt"
DEFAULT_CAMERACTRL_MOTION_ADAPTER = DEFAULT_SHARED_HFD_ROOT / "guoyww--animatediff" / "v3_sd15_adapter.ckpt"


class CameraCtrlRuntime:
    """
    CameraCtrl inference runtime for text-to-video generation with camera control.

    This class provides an interface to load the CameraCtrl model,
    prepare inputs (like trajectory embeddings), and generate videos
    based on a text prompt and a specified camera path.
    """

    MODEL_ID = "cameractrl"
    DISPLAY_NAME = "CameraCtrl"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        sd15_path: str | Path = DEFAULT_SD15_ROOT,
        pose_adaptor_ckpt: str | Path = DEFAULT_CAMERACTRL_CKPT,
        model_config: str | Path = DEFAULT_CAMERACTRL_CONFIG,
        motion_module_ckpt: str | Path | None = None,
        motion_adapter_ckpt: str | Path | None = DEFAULT_CAMERACTRL_MOTION_ADAPTER,
        image_lora_ckpt: str | Path | None = DEFAULT_CAMERACTRL_IMAGE_LORA,
        image_lora_rank: int = 2,
        unet_subfolder: str = "unet",
        personalized_base_model: str | Path | None = None,
    ) -> None:
        """
        Initializes the CameraCtrlRuntime.

        Args:
            model_id: Identifier for the model.
            device: The device to run the model on (e.g., "cuda", "cpu").
            sd15_path: Path to the Stable Diffusion 1.5 model.
            pose_adaptor_ckpt: Path to the CameraCtrl pose adaptor checkpoint.
            model_config: Path to the OmegaConf model configuration file.
            motion_module_ckpt: Optional path to a motion module checkpoint.
            image_lora_ckpt: Optional path to an image LoRA checkpoint for fine-tuning.
            image_lora_rank: Rank of the image LoRA, if used.
            unet_subfolder: Subfolder within the UNet model directory.
            personalized_base_model: Optional path to a personalized base model.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "camera_control_video"
        self.device = device
        self.sd15_path = str(Path(sd15_path).expanduser())
        self.pose_adaptor_ckpt = str(Path(pose_adaptor_ckpt).expanduser())
        self.model_config = str(Path(model_config).expanduser())
        self.motion_module_ckpt = None if motion_module_ckpt is None else str(Path(motion_module_ckpt).expanduser())
        self.motion_adapter_ckpt = (
            None if motion_adapter_ckpt is None else str(Path(motion_adapter_ckpt).expanduser())
        )
        self.image_lora_ckpt = None if image_lora_ckpt is None else str(Path(image_lora_ckpt).expanduser())
        self.image_lora_rank = int(image_lora_rank)
        self.unet_subfolder = unet_subfolder
        self.personalized_base_model = (
            None if personalized_base_model is None else str(Path(personalized_base_model).expanduser())
        )
        self._pipeline = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CameraCtrlRuntime":
        """
        Builds a lazy CameraCtrl runtime instance from pretrained model options.

        This class method allows for flexible initialization, accepting either
        a direct path or a mapping of options, and overriding defaults with
        provided keyword arguments.

        Args:
            pretrained_model_path: Path to a pretrained model or a dictionary
                                   of configuration options.
            device: Overrides the default device if provided.
            model_id: Overrides the default model_id if provided.
            **kwargs: Additional keyword arguments to override instance
                      initialization parameters.

        Returns:
            An instance of CameraCtrlRuntime, with its pipeline lazily loaded.
        """
        # Parse pretrained_model_path, allowing it to be a dict of options or a direct path
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["pose_adaptor_ckpt"] = str(pretrained_model_path)
        options.update(kwargs)  # Merge with kwargs, giving kwargs precedence
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            sd15_path=options.get("sd15_path") or options.get("ori_model_path") or DEFAULT_SD15_ROOT,
            pose_adaptor_ckpt=(
                options.get("pose_adaptor_ckpt")
                or options.get("checkpoint_path")
                or options.get("ckpt_path")
                or DEFAULT_CAMERACTRL_CKPT
            ),
            model_config=options.get("model_config") or options.get("config") or DEFAULT_CAMERACTRL_CONFIG,
            motion_module_ckpt=options.get("motion_module_ckpt"),
            motion_adapter_ckpt=options.get(
                "motion_adapter_ckpt",
                DEFAULT_CAMERACTRL_MOTION_ADAPTER,
            ),
            image_lora_ckpt=options.get("image_lora_ckpt", DEFAULT_CAMERACTRL_IMAGE_LORA),
            image_lora_rank=int(options.get("image_lora_rank", 2)),
            unet_subfolder=str(options.get("unet_subfolder") or "unet"),
            personalized_base_model=options.get("personalized_base_model"),
        )

    def _ensure_pipeline(self):
        """
        Ensures the inference pipeline is loaded and ready.

        This method implements lazy loading for the underlying CameraCtrl
        pipeline. It checks if the pipeline is already loaded; if not,
        it initializes it using the configuration parameters provided during
        CameraCtrlRuntime instantiation. It also validates CUDA availability.

        Raises:
            RuntimeError: If the device is set to 'cpu' as CameraCtrl requires CUDA.

        Returns:
            The initialized CameraCtrl inference pipeline.
        """
        if self._pipeline is not None:
            return self._pipeline
        # Determine the actual device to use, falling back to CPU if CUDA is not available
        device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError("CameraCtrl requires CUDA because the official inference path moves modules to CUDA.")
        model_configs = OmegaConf.load(self.model_config)
        unet_additional_kwargs = (
            model_configs["unet_additional_kwargs"] if "unet_additional_kwargs" in model_configs else None
        )
        from .cameractrl_runtime.inference import get_pipeline

        self._pipeline = get_pipeline(
            self.sd15_path,
            self.unet_subfolder,
            self.image_lora_rank,
            self.image_lora_ckpt,
            unet_additional_kwargs,
            self.motion_module_ckpt,
            self.motion_adapter_ckpt,
            model_configs["pose_encoder_kwargs"],
            model_configs["attention_processor_kwargs"],
            model_configs["noise_scheduler_kwargs"],
            self.pose_adaptor_ckpt,
            self.personalized_base_model,
            device,
        )
        self.device = device  # Update device in case it changed (e.g., from cuda:0 to cuda)
        return self._pipeline

    @staticmethod
    def trajectory_embedding(
        trajectory_file: str | Path,
        *,
        image_height: int,
        image_width: int,
        original_pose_width: int,
        original_pose_height: int,
        device: str,
    ) -> torch.Tensor:
        """
        Converts a CameraCtrl trajectory text file into Plucker ray conditioning.

        This static method reads camera parameters from a specified trajectory file,
        adjusts them based on target image dimensions, and computes a Plucker
        ray embedding suitable for the CameraCtrl model.

        Args:
            trajectory_file: Path to the text file containing camera trajectory data.
            image_height: The target height of the output image frames.
            image_width: The target width of the output image frames.
            original_pose_width: The original width used when generating the trajectory file.
            original_pose_height: The original height used when generating the trajectory file.
            device: The torch device to move the final embedding to (e.g., "cuda").

        Returns:
            A torch.Tensor representing the Plucker ray conditioning for the trajectory.
        """

        from einops import rearrange

        from .cameractrl_runtime.camera_geometry import Camera, get_relative_pose, ray_condition

        poses = Path(trajectory_file).read_text(encoding="utf-8").splitlines()
        # Parse camera parameters from the trajectory file, skipping the header line
        cam_params = [[float(value) for value in pose.strip().split(" ")] for pose in poses[1:] if pose.strip()]
        cameras = [Camera(cam_param) for cam_param in cam_params]

        # Adjust camera intrinsics based on aspect ratio difference between original pose and target image
        sample_wh_ratio = image_width / image_height
        pose_wh_ratio = original_pose_width / original_pose_height
        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = image_height * pose_wh_ratio
            for camera in cameras:
                camera.fx = resized_ori_w * camera.fx / image_width
        else:
            resized_ori_h = image_width / pose_wh_ratio
            for camera in cameras:
                camera.fy = resized_ori_h * camera.fy / image_height

        # Construct intrinsic matrices for all cameras
        intrinsic = np.asarray(
            [
                [
                    camera.fx * image_width,
                    camera.fy * image_height,
                    camera.cx * image_width,
                    camera.cy * image_height,
                ]
                for camera in cameras
            ],
            dtype=np.float32,
        )
        k_matrix = torch.as_tensor(intrinsic)[None]
        # Get relative poses (camera-to-world transformations)
        c2ws = torch.as_tensor(get_relative_pose(cameras))[None]
        # Compute Plucker ray embedding from intrinsics, extrinsics, and image dimensions
        plucker_embedding = ray_condition(k_matrix, c2ws, image_height, image_width, device="cpu")[0]
        # Reshape and move to target device for the model
        plucker_embedding = plucker_embedding.permute(0, 3, 1, 2).contiguous()[None].to(device)
        return rearrange(plucker_embedding, "b f c h w -> b c f h w")

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
        Generates a video based on a text prompt and a camera trajectory.

        Args:
            prompt: The text prompt describing the desired video content.
            images: Not used by CameraCtrl; will be ignored.
            video: Not used by CameraCtrl; will raise an error if provided.
            interactions: A sequence of strings, where the first element can be
                          interpreted as the `trajectory_file` path.
            output_path: The path where the generated video will be saved.
                         Defaults to 'cameractrl.mp4' in the current working directory.
            fps: Frames per second for the output video.
            **kwargs: Additional parameters for video generation, including:
                - trajectory_file (str | Path): Path to the trajectory file.
                                                Overrides `interactions[0]`.
                - seed (int): Random seed for reproducibility (default: 42).
                - height (int): Height of the generated video frames (default: 256).
                - width (int): Width of the generated video frames (default: 384).
                - original_pose_width (int): Original width used for trajectory
                                             calculation (default: 1280).
                - original_pose_height (int): Original height used for trajectory
                                              calculation (default: 720).
                - num_frames (int): Number of frames in the output video (default: 16).
                - infer_steps (int): Number of inference steps (default: 25).
                - cfg_scale (float): Classifier-free guidance scale (default: 14.0).
                - negative_prompt (str): A prompt to guide generation away from.

        Returns:
            A dictionary containing metadata about the generated video, including
            its path, SHA256 hash, and parameters used.

        Raises:
            ValueError: If an input video is provided or no trajectory file is specified.
        """
        del images  # CameraCtrl does not accept input images for generation.
        if video is not None:
            raise ValueError("CameraCtrl text-to-video trajectory generation does not accept input video.")

        # Determine the trajectory file path from kwargs or interactions
        trajectory_file = kwargs.get("trajectory_file") or (interactions[0] if interactions else None)
        if trajectory_file is None:
            raise ValueError("CameraCtrl requires trajectory_file or a trajectory path in interactions.")

        pipeline = self._ensure_pipeline()
        device = self.device
        seed = int(kwargs.get("seed", 42))
        generator = torch.Generator(device=device).manual_seed(seed)
        height = int(kwargs.get("height", kwargs.get("image_height", 256)))
        width = int(kwargs.get("width", kwargs.get("image_width", 384)))

        # Generate the pose embedding from the trajectory file
        pose_embedding = self.trajectory_embedding(
            trajectory_file,
            image_height=height,
            image_width=width,
            original_pose_width=int(kwargs.get("original_pose_width", 1280)),
            original_pose_height=int(kwargs.get("original_pose_height", 720)),
            device=device,
        )

        # Perform video generation using the CameraCtrl pipeline
        sample = pipeline(
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt"),
            pose_embedding=pose_embedding,
            video_length=int(kwargs.get("num_frames", kwargs.get("video_length", 16))),
            height=height,
            width=width,
            num_inference_steps=int(kwargs.get("infer_steps", kwargs.get("num_inference_steps", 25))),
            guidance_scale=float(kwargs.get("cfg_scale", kwargs.get("guidance_scale", 14.0))),
            generator=generator,
        ).videos

        # Determine output path and ensure parent directories exist
        target = Path(output_path) if output_path is not None else Path.cwd() / "cameractrl.mp4"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Save the generated video frames as a grid
        from worldfoundry.core.media.codecs.video import save_videos_grid

        save_videos_grid(sample, str(target), fps=fps or int(kwargs.get("fps", 8)))

        # Return comprehensive metadata about the generation
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "video_sha256": file_sha256(target),
            "runtime": "worldfoundry.cameractrl.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "trajectory_file": str(trajectory_file),
            "seed": seed,
        }
