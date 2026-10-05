"""Wan-specific staged execution built on StagedDiffusionPipeline.

Research-oriented path for custom Wan conditioning (I2V, ControlNet, VACE,
TeaCache, USP). Not selected by NativeDiffusionAssembler; native Wan recipes
use strategy + NativeDiffusionRunner instead.

Must not: cast scheduler timesteps to BF16 before the sinusoidal embedding
(see the denoise loop); register this pipeline as a recipe runner factory.
"""

import types
from ..models.networks.wan.model import WanModel, RMSNorm
from ..models.networks.wan.vace import VaceWanModel
from ..models.encoders.wan.model import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.encoders.wan import WanImageEncoder, WanPrompter
from ..models.encoders.wan.variants import WanMotionControllerModel
from ..models.autoencoders.wan.model import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from worldfoundry.core.nn import FlowMatchScheduler
from .staged import StagedDiffusionPipeline
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from worldfoundry.core.vram import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from worldfoundry.core.nn import sinusoidal_embedding_1d


def _prepare_scheduler_timestep(timestep: torch.Tensor, *, device: str) -> torch.Tensor:
    """Move a Wan scheduler timestep without reducing its floating precision."""

    return timestep.unsqueeze(0).to(device=device)



class WanStagedPipeline(StagedDiffusionPipeline):
    """Research-oriented Wan execution with custom staged conditioning."""

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        """Init.

        Args:
            device: The device.
            torch_dtype: The torch dtype.
            tokenizer_path: The tokenizer path.
        """
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller', 'vace']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False

    @classmethod
    def from_components(cls, components, *, device="cuda", torch_dtype=torch.float16):
        """Bind explicit native Wan roles to the shared staged runner.

        ``components`` is intentionally duck-typed so research variants can
        share text/VAE roles while providing their own denoiser role without a
        model-manager backend.
        """

        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_path=str(components.tokenizer_path),
        )
        pipe.dit = components.dit
        pipe.text_encoder = components.text_encoder
        pipe.vae = components.vae
        pipe.image_encoder = getattr(components, "image_encoder", None)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.model_names = [
            name
            for name in ("text_encoder", "dit", "vae", "image_encoder")
            if getattr(pipe, name, None) is not None
        ]
        pipe.height_division_factor = pipe.vae.upsampling_factor * pipe.dit.patch_size[1]
        pipe.width_division_factor = pipe.vae.upsampling_factor * pipe.dit.patch_size[2]
        return pipe


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        """Enable vram management.

        Args:
            num_persistent_param_in_dit: The num persistent param in dit.
        """
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            vram_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
            },
            vram_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_vram_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            vram_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_source):
        """Fetch models.

        Args:
            model_manager: The model manager.
        """
        text_encoder_model_and_path = model_source.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_source.fetch_model("wan_video_dit")
        self.vae = model_source.fetch_model("wan_video_vae")
        self.image_encoder = model_source.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_source.fetch_model("wan_video_motion_controller")
        self.vace = model_source.fetch_model("wan_video_vace")


    @staticmethod
    def from_model_source(model_source, torch_dtype=None, device=None, use_usp=False):
        """From model manager.

        Args:
            model_manager: The model manager.
            torch_dtype: The torch dtype.
            device: The device.
            use_usp: The use usp.
        """
        if device is None: device = model_source.device
        if torch_dtype is None: torch_dtype = model_source.torch_dtype
        pipe = WanStagedPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_source)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from worldfoundry.core.attention.parallel.patch_xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        """Denoising model."""
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        """Encode prompt.

        Args:
            prompt: The prompt.
            positive: The positive.
        """
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode image.

        Args:
            image: The image.
            end_image: The end image.
            num_frames: The num frames.
            height: The height.
            width: The width.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = None
        if self.image_encoder is not None:
            clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if clip_context is not None and self.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        result = {"y": y}
        if clip_context is not None:
            result["clip_feature"] = clip_context.to(dtype=self.torch_dtype, device=self.device)
        return result
    
    
    def encode_control_video(self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode control video.

        Args:
            control_video: The control video.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        control_video = self.preprocess_images(control_video)
        control_video = torch.stack(control_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
        latents = self.encode_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
        return latents
    
    
    def prepare_reference_image(self, reference_image, height, width):
        """Prepare reference image.

        Args:
            reference_image: The reference image.
            height: The height.
            width: The width.
        """
        if reference_image is not None:
            self.load_models_to_device(["vae"])
            reference_image = reference_image.resize((width, height))
            reference_image = self.preprocess_images([reference_image])
            reference_image = torch.stack(reference_image, dim=2).to(dtype=self.torch_dtype, device=self.device)
            reference_latents = self.vae.encode(reference_image, device=self.device)
            return {"reference_latents": reference_latents}
        else:
            return {}
    
    
    def prepare_controlnet_kwargs(self, control_video, num_frames, height, width, clip_feature=None, y=None, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Prepare controlnet kwargs.

        Args:
            control_video: The control video.
            num_frames: The num frames.
            height: The height.
            width: The width.
            clip_feature: The clip feature.
            y: The y.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if control_video is not None:
            control_latents = self.encode_control_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=self.torch_dtype, device=self.device)
                y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=self.torch_dtype, device=self.device)
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


    def tensor2video(self, frames):
        """Tensor2video.

        Args:
            frames: The frames.
        """
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        """Prepare extra input.

        Args:
            latents: The latents.
        """
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode video.

        Args:
            input_video: The input video.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode video.

        Args:
            latents: The latents.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        model_dtype = next(iter(self.vae.parameters())).dtype
        latents = latents.to(dtype=model_dtype)
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        """Prepare unified sequence parallel."""
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    
    def prepare_motion_bucket_id(self, motion_bucket_id):
        """Prepare motion bucket id.

        Args:
            motion_bucket_id: The motion bucket id.
        """
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=self.torch_dtype, device=self.device)
        return {"motion_bucket_id": motion_bucket_id}
    
    
    def prepare_vace_kwargs(
        self,
        latents,
        vace_video=None, vace_mask=None, vace_reference_image=None, vace_scale=1.0,
        height=480, width=832, num_frames=81,
        seed=None, rand_device="cpu",
        tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        """Prepare vace kwargs.

        Args:
            latents: The latents.
            vace_video: The vace video.
            vace_mask: The vace mask.
            vace_reference_image: The vace reference image.
            vace_scale: The vace scale.
            height: The height.
            width: The width.
            num_frames: The num frames.
            seed: The seed.
            rand_device: The rand device.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if vace_video is not None or vace_mask is not None or vace_reference_image is not None:
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=self.torch_dtype, device=self.device)
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = self.encode_video(inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            reactive = self.encode_video(reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_images([vace_reference_image])
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = self.encode_video(vace_reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
                
                noise = self.generate_noise((1, 16, 1, latents.shape[3], latents.shape[4]), seed=seed, device=rand_device, dtype=torch.float32)
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                latents = torch.concat((noise, latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return latents, {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return latents, {"vace_context": None, "vace_scale": vace_scale}


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        control_video=None,
        reference_image=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        """Call.

        Args:
            prompt: The prompt.
            negative_prompt: The negative prompt.
            input_image: The input image.
            end_image: The end image.
            input_video: The input video.
            control_video: The control video.
            reference_image: The reference image.
            vace_video: The vace video.
            vace_video_mask: The vace video mask.
            vace_reference_image: The vace reference image.
            vace_scale: The vace scale.
            denoising_strength: The denoising strength.
            seed: The seed.
            rand_device: The rand device.
            height: The height.
            width: The width.
            num_frames: The num frames.
            cfg_scale: The cfg scale.
            num_inference_steps: The num inference steps.
            sigma_shift: The sigma shift.
            motion_bucket_id: The motion bucket id.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
            tea_cache_l1_thresh: The tea cache l1 thresh.
            tea_cache_model_id: The tea cache model id.
            progress_bar_cmd: The progress bar cmd.
            progress_bar_st: The progress bar st.
        """
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        needs_vae_condition = (
            self.dit is not None
            and self.dit.require_vae_embedding
            and self.dit.in_dim != self.dit.out_dim
        )
        if input_image is not None and (self.image_encoder is not None or needs_vae_condition):
            image_roles = ["vae"]
            if self.image_encoder is not None:
                image_roles.append("image_encoder")
            self.load_models_to_device(image_roles)
            image_emb = self.encode_image(input_image, end_image, num_frames, height, width, **tiler_kwargs)
        else:
            image_emb = {}
            
        # Reference image
        reference_image_kwargs = self.prepare_reference_image(reference_image, height, width)
            
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents, vace_video, vace_video_mask, vace_reference_image, vace_scale,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, **tiler_kwargs
        )
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit", "motion_controller", "vace"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Keep scheduler timesteps in their reference floating precision.
            # Casting values such as the shifted FlowMatch schedule to BF16
            # before the sinusoidal embedding quantizes the time condition and
            # visibly changes SVI/Wan generations.
            timestep = _prepare_scheduler_timestep(timestep, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller, vace=self.vace,
                x=latents, timestep=timestep,
                **prompt_emb_posi, **image_emb, **extra_input,
                **tea_cache_posi, **usp_kwargs, **motion_kwargs, **vace_kwargs, **reference_image_kwargs,
            )
            # Classic CFG: ε_neg + cfg_scale * (ε_pos - ε_neg). Skip the negative pass at scale 1.
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega, **image_emb, **extra_input,
                    **tea_cache_nega, **usp_kwargs, **motion_kwargs, **vace_kwargs, **reference_image_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames



class TeaCache:
    """Skip redundant DiT blocks when modulated time input barely changes.

    coefficients_dict is per official Wan model_id. check() accumulates a
    polynomial-rescaled relative L1; below rel_l1_thresh, update() replays the
    previous residual instead of running blocks. First and last steps always compute.
    Unknown model_id raises :exc:`ValueError`.
    """
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        """Init.

        Args:
            num_inference_steps: The num inference steps.
            rel_l1_thresh: The rel l1 thresh.
            model_id: The model id.
        """
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        """Return True when the block loop can be skipped (replay previous residual)."""
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        """Remember residual = hidden_states - input snapshot after a full block pass."""
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        """Replay the stored residual instead of running DiT blocks."""
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Single Wan DiT forward used by WanStagedPipeline.

    Official path: sin/cos from full-precision timestep, then cast only the
    finished embedding to the latent dtype. TeaCache may skip the block loop.
    VACE hints are added on mapped layers. USP chunks sequence dim across ranks.
    """
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    # The official Wan path computes sin/cos from the full-precision scheduler
    # value, then casts only the finished embedding to the latent/model dtype.
    t = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=x.dtype)
    )
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
    elif dit.require_vae_embedding:
        raise ValueError("Wan denoising requires VAE condition latents")

    if dit.has_image_input:
        if clip_feature is None:
            raise ValueError("Wan image-conditioned denoising requires a CLIP feature")
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)
    add_condition = kwargs.get("add_condition")
    if add_condition is not None:
        x = x + add_condition
    
    # Reference image
    if reference_latents is not None:
        reference_latents = dit.ref_conv(reference_latents[:, :, 0]).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


__all__ = ["TeaCache", "WanStagedPipeline", "model_fn_wan_video"]
