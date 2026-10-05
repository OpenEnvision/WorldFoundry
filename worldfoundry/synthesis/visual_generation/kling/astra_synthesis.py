"""
This module provides the AstraSynthesis class, an interface for the Astra video synthesis model.

AstraSynthesis integrates with the BaseSynthesis abstract class and leverages the WanVideoAstraPipeline
for video generation, including functionalities like model loading, patching, frame encoding,
diffusion prediction, and video decoding.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from ...base_synthesis import BaseSynthesis
from worldfoundry.core import maybe_download_hf_repo_on_rank0

from worldfoundry.synthesis.visual_generation.kling.astra_runtime.pipelines.wan_video_astra import (
    WanVideoAstraPipeline,
)
from worldfoundry.synthesis.visual_generation.kling.astra_runtime.models.model_manager import ModelManager
import numpy as np

from worldfoundry.synthesis.visual_generation.kling.astra_runtime.astra_utils import (
    replace_dit_model_in_manager, add_framepack_components, add_moe_components
)

class AstraSynthesis(BaseSynthesis):
    """
    AstraSynthesis provides an interface to the Astra video synthesis pipeline.

    This class handles the initialization, configuration, and execution of the
    WanVideoAstraPipeline, including model loading, component patching,
    video encoding/decoding, and the diffusion denoising process.
    """
    def __init__(self, pipe):
        """
        Initializes the AstraSynthesis instance with a pre-configured pipeline.

        Args:
            pipe (WanVideoAstraPipeline): The Astra video synthesis pipeline instance.
        """
        self.pipe = pipe
        self.device = pipe.device
        self.tiler_kwargs = {"tiled": True, "tile_size": (34, 34), "tile_stride": (18, 16)}

    @staticmethod
    def _looks_like_hf_repo_id(path_or_id):
        parts = str(path_or_id).split("/")
        return (
            len(parts) == 2
            and all(parts)
            and not str(path_or_id).startswith(("/", "./", "../", "~", "http://", "https://"))
        )

    @staticmethod
    def _resolve_path(path_or_id):
        """
        Resolves and validates a local path or Hugging Face repository id.

        Local paths are returned unchanged. Repository ids are materialized with
        Hugging Face's native cache/download semantics, guarded by the shared
        WorldFoundry rank-0 download lock.

        Args:
            path_or_id (str): Local path or HF repo id.

        Returns:
            str: Local snapshot/path if it exists or can be downloaded.

        Raises:
            FileNotFoundError: If the provided value is neither an existing
                path nor a resolvable HF repo id.
        """
        if not path_or_id:
            raise FileNotFoundError("Astra requires a model path or Hugging Face repo id.")
        candidate = Path(str(path_or_id)).expanduser()
        if candidate.exists():
            return str(candidate)
        if AstraSynthesis._looks_like_hf_repo_id(str(path_or_id)):
            maybe_download_hf_repo_on_rank0(str(path_or_id))
            return snapshot_download(str(path_or_id), local_files_only=True)
        raise FileNotFoundError(
            "Astra requires an existing local model path or a Hugging Face repo id. "
            f"Could not resolve: {path_or_id}"
        )

    @classmethod
    def from_pretrained(cls, config, device="cuda"): # the config is the AstraConfig from the pipeline
        """
        Loads and configures the Astra video synthesis pipeline from pretrained models.

        This class method performs several critical steps:
        1. Resolves local paths for WAN and Astra models.
        2. Initializes and patches the Diffusion Transformer (DiT) model.
        3. Loads base WAN models (DiT, T5, VAE) into the ModelManager.
        4. Adds custom components (camera encoders, framepack, Mixture-of-Experts)
           to the DiT architecture.
        5. Loads specific Astra weights into the patched DiT.
        6. Configures the scheduler and moves the pipeline to the specified device.

        Args:
            config (AstraConfig): Configuration object containing model paths and other settings.
            device (str, optional): The device to load the models onto (e.g., "cuda", "cpu").
                                    Defaults to "cuda".

        Returns:
            AstraSynthesis: An initialized instance of AstraSynthesis with the loaded pipeline.

        Raises:
            FileNotFoundError: If any required model path does not exist.
        """
        print("Resolving model paths...")
        # Resolve local paths for the base WAN models and the Astra-specific weights.
        resolved_wan_path = cls._resolve_path(config.wan_model_path)
        resolved_astra_path = cls._resolve_path(config.astra_path)

        # 1. Model initialization & Patching
        # Replace the default DiT model implementation in the manager with a patched version
        # to support Astra's custom components.
        replace_dit_model_in_manager()
        
        # Initialize the ModelManager to load shared models with bfloat16 precision on CPU initially.
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        # Load the core WAN models: the diffusion model, text encoder (T5), and VAE.
        model_manager.load_models([
            os.path.join(resolved_wan_path, "diffusion_pytorch_model.safetensors"),
            os.path.join(resolved_wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(resolved_wan_path, "Wan2.1_VAE.pth"),
        ])
        # Create the pipeline from the loaded models and move it to the target device.
        pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device=device)

        # 2. Add components
        # Determine the hidden dimension of the DiT model.
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        # Add a camera encoder and projector to each DiT block for incorporating camera pose information.
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            # Initialize projector as an identity mapping.
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        # Add framepack components to the DiT model for efficient handling of contextual frames.
        add_framepack_components(pipe.dit)
        
        # Configure and add Mixture-of-Experts (MoE) components to the DiT model
        # to handle different input modalities (Sekai, Nuscenes, OpenX).
        moe_config = {
            "num_experts": config.moe_num_experts,
            "top_k": config.moe_top_k,
            "hidden_dim": config.moe_hidden_dim or dim * 2,
            "sekai_input_dim": 13,
            "nuscenes_input_dim": 8,
            "openx_input_dim": 13
        }
        add_moe_components(pipe.dit, moe_config)
        
        # 3. Load weights
        print(f"Loading DiT weights from: {resolved_astra_path} ...")
        # Construct the full path to the Astra-specific DiT checkpoint.
        astra_dit_path = Path(resolved_astra_path)
        if astra_dit_path.is_dir():
            candidates = [
                astra_dit_path / "diffusion_pytorch_model.ckpt",
                astra_dit_path / "models/Astra/checkpoints/diffusion_pytorch_model.ckpt",
            ]
            astra_dit_path = next((path for path in candidates if path.is_file()), candidates[-1])
        if not astra_dit_path.is_file():
            raise FileNotFoundError(f"Astra checkpoint not found: {astra_dit_path}")
        # Load the Astra DiT state dictionary, mapping to CPU first to avoid GPU memory issues.
        dit_state_dict = torch.load(astra_dit_path, map_location="cpu", weights_only=True)
        # The released Astra checkpoint includes the camera, FramePack and MoE weights.
        pipe.dit.load_state_dict(dit_state_dict, strict=True)
        # Move the entire pipeline to the specified device.
        pipe = pipe.to(device)
        
        # Ensure the clean_x_embedder (if present) is cast to the model's dtype.
        model_dtype = next(pipe.dit.parameters()).dtype
        if hasattr(pipe.dit, 'clean_x_embedder'):
            pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
        
        # Set the number of inference steps for the scheduler based on the config.
        pipe.scheduler.set_timesteps(max(1, int(getattr(config, "num_inference_steps", 50))))
        
        return cls(pipe)
    
    def encode_frames(self, frames):
        """
        Encodes raw video frames into latent representations.

        The input frames are expected in [C, T, H, W] format, typically from an operator.
        They are batched, moved to the device, and then processed by the pipeline's VAE
        encoder with tiling for memory efficiency.

        Args:
            frames (torch.Tensor): Input video frames with shape [C, T, H, W].

        Returns:
            torch.Tensor: The encoded latent representations, moved to CPU.
        """
        # Add a batch dimension and move frames to the appropriate device and dtype.
        frames = frames.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            # Encode the video using the pipeline's VAE, applying tiling as configured.
            latents = self.pipe.encode_video(frames, **self.tiler_kwargs)[0]
        # Remove the batch dimension if it was added and is now redundant.
        if latents.dim() == 5 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        return latents.cpu()

    def load_encoded_video_from_pth(self, pth_path, start_frame=0, num_frames=10):
        """
        Loads pre-encoded video latents and associated data from a .pth file.

        Args:
            pth_path (str): Path to the .pth file containing encoded video data.
            start_frame (int, optional): The starting frame index to load. Defaults to 0.
            num_frames (int, optional): The number of frames to load. Defaults to 10.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing:
                - condition_latents (torch.Tensor): The selected range of encoded latents.
                - encoded_data (dict): The full dictionary loaded from the .pth file.

        Raises:
            ValueError: If the requested frame range exceeds the available frames in the file.
        """
        print(f"Loading encoded video from {pth_path}")
        # Load the encoded data from the .pth file, ensuring it's on CPU.
        encoded_data = torch.load(pth_path, weights_only=True, map_location="cpu")
        full_latents = encoded_data['latents']
        # Validate that the requested frame range is within the bounds of available latents.
        if start_frame + num_frames > full_latents.shape[1]:
            raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
        # Select the specific slice of latents for conditioning.
        condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
        return condition_latents, encoded_data

    @torch.no_grad()
    def predict(self, framepack_data, current_generation, prompt_emb_pos, prompt_emb_neg, args, camera_embedding_uncond=None, generator=None):
        """
        Runs the diffusion denoising loop to generate new video latents.

        This method takes initial conditions and iteratively refines a noisy latent
        representation using the diffusion model and a scheduler. It supports
        classifier-free guidance (CFG) for both camera and text conditions.

        Args:
            framepack_data (dict): Dictionary containing various conditioning latents
                                   (clean_latents, clean_latents_2x, etc.),
                                   camera embeddings, modality type, and latent indices.
            current_generation (int): The number of frames to generate in the current step.
            prompt_emb_pos (dict): Positive prompt embeddings for conditional guidance.
            prompt_emb_neg (dict): Negative prompt embeddings for unconditional guidance.
            args (object): An object containing generation arguments like
                           num_inference_steps, use_camera_cfg, camera_guidance_scale,
                           and text_guidance_scale.
            camera_embedding_uncond (torch.Tensor, optional): Unconditional camera embedding
                                                               for camera CFG. Defaults to None.

        Returns:
            torch.Tensor: The final denoised latent representation of the generated video.
        """
        device = self.device
        model_dtype = next(self.pipe.dit.parameters()).dtype
        # Set the number of inference steps for the scheduler.
        num_inference_steps = max(1, int(getattr(args, "num_inference_steps", 50)))
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        
        # Prepare input tensors by extracting them from framepack_data and adding batch dimensions.
        clean_latents = framepack_data['clean_latents'].unsqueeze(0)
        clean_latents_2x = framepack_data['clean_latents_2x'].unsqueeze(0)
        clean_latents_4x = framepack_data['clean_latents_4x'].unsqueeze(0)
        camera_embedding = framepack_data['camera_embedding'].unsqueeze(0)
        modality_type = framepack_data['modality_type']
        
        modality_inputs = {modality_type: camera_embedding}
        
        modality_inputs_uncond = None
        camera_embedding_uncond_batch = None
        # Prepare unconditional camera embeddings if camera CFG is enabled.
        if args.use_camera_cfg and camera_embedding_uncond is not None:
            camera_embedding_uncond_batch = camera_embedding_uncond[:camera_embedding.shape[1], :].unsqueeze(0)
            modality_inputs_uncond = {modality_type: camera_embedding_uncond_batch}
            
        latent_indices = framepack_data['latent_indices'].unsqueeze(0).cpu()
        clean_latent_indices = framepack_data['clean_latent_indices'].unsqueeze(0).cpu()
        clean_latent_2x_indices = framepack_data['clean_latent_2x_indices'].unsqueeze(0).cpu()
        clean_latent_4x_indices = framepack_data['clean_latent_4x_indices'].unsqueeze(0).cpu()

        # Determine dimensions of the latents.
        C = clean_latents.shape[1] 
        H = clean_latents.shape[3]
        W = clean_latents.shape[4]
        
        # Initialize new latents with random noise, which will be denoised iteratively.
        new_latents = torch.randn(1, C, current_generation, H, W, device=device, dtype=model_dtype, generator=generator)
        # Prepare any additional input required by the pipeline.
        extra_input = self.pipe.prepare_extra_input(new_latents)
        timesteps = self.pipe.scheduler.timesteps
        
        # Main diffusion denoising loop.
        for i, timestep in enumerate(timesteps):
            if i % 10 == 0: print(f"  Denoising step {i+1}/{len(timesteps)}")
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            # --- CFG LOGIC ---
            # Apply classifier-free guidance for camera conditioning.
            if args.use_camera_cfg and args.camera_guidance_scale > 1.0:
                 # Predict noise for the conditional (with camera embedding) input.
                 noise_pred_cond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
                 # Predict noise for the unconditional (without camera embedding) input.
                 noise_pred_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding_uncond_batch, modality_inputs=modality_inputs_uncond, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **(prompt_emb_neg if prompt_emb_neg else prompt_emb_pos), **extra_input)
                 # Combine conditional and unconditional predictions using the camera guidance scale.
                 noise_pred = noise_pred_uncond + args.camera_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                 
                 # Further apply text-based classifier-free guidance if both camera and text CFG are active.
                 if args.text_guidance_scale > 1.0 and prompt_emb_neg:
                     # Predict noise using the negative text prompt but with conditional camera.
                     noise_pred_text_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_neg, **extra_input)
                     # Apply text guidance on top of the camera-guided noise prediction.
                     noise_pred = noise_pred_text_uncond + args.text_guidance_scale * (noise_pred - noise_pred_text_uncond)
            # Apply only text-based classifier-free guidance if enabled and a negative prompt is provided.
            elif args.text_guidance_scale > 1.0 and prompt_emb_neg:
                 # Predict noise for the conditional (positive text) input.
                 noise_pred_cond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
                 # Predict noise for the unconditional (negative text) input.
                 noise_pred_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_neg, **extra_input)
                 # Combine conditional and unconditional predictions using the text guidance scale.
                 noise_pred = noise_pred_uncond + args.text_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # If no CFG is used, perform a single noise prediction with the positive prompt.
            else:
                 noise_pred, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
            
            # Take a single denoising step using the scheduler.
            new_latents = self.pipe.scheduler.step(noise_pred, timestep, new_latents)
        
        return new_latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        Decodes latent representations back into video frames.

        Args:
            latents (torch.Tensor): The latent representations of the video.
            tiled (bool, optional): Whether to use tiled decoding. Defaults to True.
            tile_size (tuple[int, int], optional): The size of tiles for decoding. Defaults to (34, 34).
            tile_stride (tuple[int, int], optional): The stride of tiles for decoding. Defaults to (18, 16).

        Returns:
            numpy.ndarray: The decoded video frames as a NumPy array,
                           scaled to [0, 255] and of type uint8.
        """
        # Decode the latents into video frames using the pipeline's VAE decoder.
        decoded_video = self.pipe.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        # Convert the decoded video tensor to a NumPy array, permuting dimensions
        # from (B, T, C, H, W) to (T, H, W, C) for common video formats.
        video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
        # Scale pixel values from [-1, 1] to [0, 1] and clip.
        video_np = (video_np * 0.5 + 0.5).clip(0, 1)
        # Scale to [0, 255] and convert to unsigned 8-bit integer type.
        video_np = (video_np * 255).astype(np.uint8)
        return video_np
