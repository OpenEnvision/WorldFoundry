"""
This module provides the ReCamMasterSynthesis class for generating videos based on text prompts,
source videos, and camera trajectories using the ReCamMaster model integrated with the Wan model pipeline.
It handles model loading, initialization, and video generation through a standardized interface.
"""
import os

import torch
import torch.nn as nn
from torchvision.transforms import v2

from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict
from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.model_manager import ModelManager
from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.pipelines.wan_video_recammaster import (
    WanVideoReCamMasterPipeline,
)

from ...base_synthesis import BaseSynthesis


class ReCamMasterSynthesis(BaseSynthesis):
    """
    A class for performing video synthesis using the ReCamMaster model.

    This class extends BaseSynthesis and integrates the WanVideoReCamMasterPipeline
    to generate videos conditioned on text prompts, source videos, and camera trajectories.
    It includes preprocessing steps for input frames.
    """
    def __init__(self,
                 pipeline,
                 height=480,
                 width=832,
                 device="cuda"):
        """
        Initializes the ReCamMasterSynthesis instance.

        Args:
            pipeline (WanVideoReCamMasterPipeline): The pre-configured ReCamMaster video generation pipeline.
            height (int, optional): The target height for video frames. Defaults to 480.
            width (int, optional): The target width for video frames. Defaults to 832.
            device (str, optional): The device to run the model on (e.g., "cuda", "cpu"). Defaults to "cuda".
        """
        super(ReCamMasterSynthesis, self).__init__()
        self.pipeline = pipeline
        self.device = device

        # Define the image transformation pipeline for input frames
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),  # Crop the input frame to the specified dimensions
            v2.Resize(size=(height, width), antialias=True),  # Resize to ensure consistency, applying antialiasing
            v2.ToTensor(),  # Convert the image to a PyTorch Tensor (HWC to CHW, values in [0, 1])
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize pixel values to the range [-1, 1]
        ])

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path="Wan-AI/Wan2.1-T2V-1.3B",
                        recammaster_ckpt_path="KlingTeam/ReCamMaster-Wan2.1",
                        device="cuda",
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        """
        Loads a pre-trained ReCamMaster model and its components.

        This class method handles the loading of the base Wan model,
        initializes ReCamMaster-specific modules, and loads the ReCamMaster checkpoint.

        Args:
            pretrained_model_path (str, optional): Path to the local directory containing the Wan model checkpoints.
                                                    Defaults to "Wan-AI/Wan2.1-T2V-1.3B".
            recammaster_ckpt_path (str, optional): Path to the local directory containing the ReCamMaster checkpoint.
                                                    Defaults to "KlingTeam/ReCamMaster-Wan2.1".
            device (str, optional): The device to load the models on (e.g., "cuda", "cpu"). Defaults to "cuda".
            weight_dtype (torch.dtype, optional): The data type for model weights (e.g., torch.bfloat16).
                                                  Defaults to torch.bfloat16.
            **kwargs: Additional keyword arguments (currently not used).

        Returns:
            ReCamMasterSynthesis: An initialized instance of ReCamMasterSynthesis with the loaded models.

        Raises:
            FileNotFoundError: If `pretrained_model_path` is not a valid local directory.
        """
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            # Enforce local checkpoint loading as runtime downloads are disabled for strict execution
            raise FileNotFoundError(
                "ReCamMaster requires a local Wan checkpoint directory. "
                f"Runtime downloads are disabled for strict in-tree execution: {pretrained_model_path}"
            )
        
        # Initialize ModelManager to handle loading of base Wan model components
        model_manager = ModelManager(torch_dtype=weight_dtype, device="cpu")
        model_manager.load_models([
            os.path.join(model_root, "diffusion_pytorch_model.safetensors"),  # Main diffusion model weights
            os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),    # T5 text encoder weights
            os.path.join(model_root, "Wan2.1_VAE.pth"),                     # VAE model weights
        ])
        # Create the WanVideoReCamMasterPipeline instance from the loaded models
        pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device)

        # Initialize additional modules introduced in ReCamMaster within each DIT block of the pipeline
        # These modules enhance the DIT's ability to incorporate camera trajectory information.
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0] # Infer the hidden dimension from the DIT block's attention layer
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim) # Linear layer to encode 12-dim camera trajectory features
            block.projector = nn.Linear(dim, dim)  # Linear layer for feature projection within the block
            # Initialize weights and biases for the new modules
            block.cam_encoder.weight.data.zero_()     # Zero-initialize cam_encoder weights for a neutral starting point
            block.cam_encoder.bias.data.zero_()        # Zero-initialize cam_encoder bias
            block.projector.weight = nn.Parameter(torch.eye(dim))  # Initialize projector weight as an identity matrix
            block.projector.bias = nn.Parameter(torch.zeros(dim))  # Zero-initialize projector bias

        # Load ReCamMaster specific checkpoint state dictionary
        recammaster_ckpt_path = os.path.join(recammaster_ckpt_path, "step20000.ckpt")
        state_dict = load_tensor_state_dict(recammaster_ckpt_path)
        pipe.dit.load_state_dict(state_dict, strict=True) # Load checkpoint into the DIT model with strict key matching
        pipe.to(device) # Move the entire pipeline to the specified device
        pipe.to(dtype=weight_dtype) # Set the data type for the pipeline's tensors
        return cls(pipeline=pipe, device=device)

    @torch.no_grad()
    def predict(self,
                input_text,
                input_video,
                camera_trajotry,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                cfg_scale=5.0):
        """
        Generates a video based on a text prompt, an input video, and a camera trajectory.

        Args:
            input_text (str): The text prompt describing the desired video content.
            input_video (torch.Tensor or similar): The source video input (e.g., a latent representation or raw frames).
            camera_trajotry (torch.Tensor or similar): The camera trajectory information to guide video generation.
            num_frames (int, optional): The number of frames to generate in the output video. Defaults to 81.
            height (int, optional): The height of the output video frames. Defaults to 480.
            width (int, optional): The width of the output video frames. Defaults to 832.
            cfg_scale (float, optional): Classifier-free guidance scale. Higher values encourage
                                         adherence to the prompt, lower values allow more creativity. Defaults to 5.0.

        Returns:
            torch.Tensor: The generated video tensor.
        """
        video = self.pipeline(
            prompt=input_text,
            # A fixed negative prompt designed to mitigate common generation artifacts and undesired visual qualities.
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=input_video,
            target_camera=camera_trajotry,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=0, tiled=True
        )
        return video
