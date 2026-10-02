"""Native LongVie inference on the shared WorldFoundry Wan infrastructure."""

from __future__ import annotations

import os
import types
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    CausalConv3d,
    RMS_norm,
    Upsample,
    WanVideoVAE,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
    WanImageEncoder,
    WanPrompter,
    WanTextEncoder,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import (
    T5LayerNorm,
    T5RelativeEmbedding,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import (
    WanMotionControllerModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import (
    WanModelDualControl,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.vace import VaceWanModel
from worldfoundry.base_models.diffusion_model.runners.staged import (
    InferenceStage,
    InferenceStageRunner,
    StagedDiffusionPipeline,
)
from worldfoundry.core.distributed.sequence_parallel.ops import gather_forward
from worldfoundry.core.model_loading import GeneralLoRALoader, ModelConfig, load_state_dict
from worldfoundry.core.nn import FlowMatchScheduler, RMSNorm, sinusoidal_embedding_1d
from worldfoundry.core.vram import (
    AutoWrappedLinear,
    AutoWrappedModule,
    WanAutoCastLayerNorm,
    enable_vram_management,
    move_direct_tensors_to_device,
)


def _sequence_parallel_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _sequence_parallel_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _sequence_parallel_all_gather(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    return gather_forward(value, dim) if _sequence_parallel_world_size() > 1 else value


class LongViePipeline(StagedDiffusionPipeline):
    """Long vie pipeline implementation."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        """Init.

        Args:
            device: The device.
            torch_dtype: The torch dtype.
            tokenizer_path: The tokenizer path.
        """
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.dual_controller: WanModelDualControl = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.animate_adapter: Any = None
        self.audio_encoder: Any = None
        self.in_iteration_models = ("dit", "dual_controller", "motion_controller", "vace", "animate_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter")
        self.model_names = (
            "text_encoder",
            "image_encoder",
            "dit",
            "dit2",
            "vae",
            "dual_controller",
            "motion_controller",
            "vace",
            "vace2",
            "animate_adapter",
            "audio_encoder",
        )
        self.unit_runner = InferenceStageRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_HistoryVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_LongVieControlEmbedder(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]

        self.model_fn = model_fn_wan_video
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        """Load lora.

        Args:
            module: The module.
            lora_config: The lora config.
            alpha: The alpha.
            hotload: The hotload.
            state_dict: The state dict.
        """
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
        
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        """Enable vram management.

        Args:
            num_persistent_param_in_dit: The num persistent param in dit.
            vram_limit: The vram limit.
            vram_buffer: The vram buffer.
        """
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
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
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
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
                vram_limit=vram_limit,
            )
        if self.dual_controller is not None:
            dtype = next(iter(self.dual_controller.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dual_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
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
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                vram_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
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
                vram_limit=vram_limit,
            )
        if self.vae is not None:
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
            # Fine-grained wrapping moves Conv/Linear/LayerNorm computation to
            # CUDA but intentionally leaves parameters registered directly on
            # their parent alone. Wan CLIP keeps class/position embeddings on
            # the VisionTransformer root, so they must follow its wrapped
            # patch projection to avoid a CPU/CUDA concat at first inference.
            move_direct_tensors_to_device(
                self.image_encoder.model.visual,
                device=self.device,
                dtype=dtype,
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
            device = "cpu" if vram_limit is not None else self.device
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
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
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
            
            
    def initialize_usp(self, ring_degree=2, ulysses_degree=4):
        """Initialize usp.

        Args:
            ring_degree: The ring degree.
            ulysses_degree: The ulysses degree.
        """
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        if ring_degree != 1 or ulysses_degree != world_size:
            raise ValueError(
                "The in-tree LongVie USP runtime supports full-world Ulysses only: "
                f"ring_degree=1 and ulysses_degree=world_size={world_size}; got "
                f"ring_degree={ring_degree}, ulysses_degree={ulysses_degree}."
            )


    def enable_usp(self):
        """Enable usp."""
        from worldfoundry.core.attention.parallel.patch_xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        for block in self.dual_controller.control_blocks_dense:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        for block in self.dual_controller.control_blocks_sparse:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        # self.dual_controller.forward = types.MethodType(usp_dit_forward, self.dual_controller)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = _sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @classmethod
    def from_components(
        cls,
        *,
        text_encoder: WanTextEncoder,
        image_encoder: WanImageEncoder,
        dit: WanModel,
        vae: WanVideoVAE,
        tokenizer_path: str,
        control_weight_path: str,
        dit_weight_path: str | None = None,
        control_layers: int = 12,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        use_usp: bool = False,
        ring_degree: int = 1,
        ulysses_degree: int = 1,
    ) -> "LongViePipeline":
        """Assemble LongVie from explicitly loaded native Wan components."""

        pipe = cls(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        pipe.text_encoder = text_encoder
        pipe.image_encoder = image_encoder
        pipe.dit = dit
        pipe.vae = vae
        pipe.dual_controller = WanModelDualControl(
            dim=5120,
            ffn_dim=13824,
            eps=1e-6,
            num_heads=40,
            num_layers=40,
            has_image_input=True,
            control_layers=control_layers,
        ).to(dtype=torch_dtype, device="cpu")

        control_state = load_state_dict(control_weight_path, torch_dtype=torch_dtype, device="cpu")
        control_state = {
            name.removeprefix("pipe.dual_controller."): value
            for name, value in control_state.items()
        }
        pipe.dual_controller.load_state_dict(control_state, strict=False)
        if dit_weight_path:
            attention_state = load_state_dict(dit_weight_path, torch_dtype=torch_dtype, device="cpu")
            pipe.dit.load_state_dict(attention_state, strict=False)

        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_path)
        if use_usp:
            pipe.initialize_usp(ring_degree=ring_degree, ulysses_degree=ulysses_degree)
            pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        dense_video=None,
        sparse_video=None,
        history=[],
        noise=None,
    ):
        """Call.

        Args:
            prompt: The prompt.
            negative_prompt: The negative prompt.
            input_image: The input image.
            end_image: The end image.
            input_video: The input video.
            denoising_strength: The denoising strength.
            input_audio: The input audio.
            audio_embeds: The audio embeds.
            audio_sample_rate: The audio sample rate.
            s2v_pose_video: The s2v pose video.
            s2v_pose_latents: The s2v pose latents.
            motion_video: The motion video.
            control_video: The control video.
            reference_image: The reference image.
            camera_control_direction: The camera control direction.
            camera_control_speed: The camera control speed.
            camera_control_origin: The camera control origin.
            vace_video: The vace video.
            vace_video_mask: The vace video mask.
            vace_reference_image: The vace reference image.
            vace_scale: The vace scale.
            animate_pose_video: The animate pose video.
            animate_face_video: The animate face video.
            animate_inpaint_video: The animate inpaint video.
            animate_mask_video: The animate mask video.
            seed: The seed.
            rand_device: The rand device.
            height: The height.
            width: The width.
            num_frames: The num frames.
            cfg_scale: The cfg scale.
            cfg_merge: The cfg merge.
            switch_DiT_boundary: The switch dit boundary.
            num_inference_steps: The num inference steps.
            sigma_shift: The sigma shift.
            motion_bucket_id: The motion bucket id.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
            sliding_window_size: The sliding window size.
            sliding_window_stride: The sliding window stride.
            tea_cache_l1_thresh: The tea cache l1 thresh.
            tea_cache_model_id: The tea cache model id.
            progress_bar_cmd: The progress bar cmd.
            dense_video: The dense video.
            sparse_video: The sparse video.
            history: The history.
            noise: The noise.
        """
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "history_video": history,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "dense_video": dense_video,
            "sparse_video": sparse_video,
            "dense_image": dense_video[0],
            "sparse_image": sparse_video[0],
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        if noise is not None:
            inputs_shared["latents"] = noise
            sigma = 0.925926
            inputs_shared["latents"][:,:,:1] = (1 - sigma) * inputs_shared["history_latents"][:,:,-1:] + sigma * noise[:,:,:1]
            
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode
        self.load_models_to_device(['vae'])
        # history_token = inputs_shared["latents"][:,-4:].clone()
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video, inputs_shared["noise"]


class WanVideoUnit_ShapeChecker(InferenceStage):
    """Wan video unit shape checker implementation."""
    def __init__(self):
        """Init."""
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: LongViePipeline, height, width, num_frames):
        """Process.

        Args:
            pipe: The pipe.
            height: The height.
            width: The width.
            num_frames: The num frames.
        """
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(InferenceStage):
    """Wan video unit noise initializer implementation."""
    def __init__(self):
        """Init."""
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: LongViePipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        """Process.

        Args:
            pipe: The pipe.
            height: The height.
            width: The width.
            num_frames: The num frames.
            seed: The seed.
            rand_device: The rand device.
            vace_reference_image: The vace reference image.
        """
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    

class WanVideoUnit_InputVideoEmbedder(InferenceStage):
    """Wan video unit input video embedder implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LongViePipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        """Process.

        Args:
            pipe: The pipe.
            input_video: The input video.
            noise: The noise.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
            vace_reference_image: The vace reference image.
        """
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_HistoryVideoEmbedder(InferenceStage):
    """Wan video unit history video embedder implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("history_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LongViePipeline, history_video, tiled, tile_size, tile_stride):
        """Process.

        Args:
            pipe: The pipe.
            history_video: The history video.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if history_video == []:
            return {"history_latents": None}
        pipe.load_models_to_device(["vae"])
        history_video = pipe.preprocess_video(history_video)
        input_latents = pipe.vae.encode(history_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"history_latents": input_latents}


class WanVideoUnit_PromptEmbedder(InferenceStage):
    """Wan video unit prompt embedder implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: LongViePipeline, prompt, positive) -> dict:
        """Process.

        Args:
            pipe: The pipe.
            prompt: The prompt.
            positive: The positive.

        Returns:
            The return value.
        """
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(InferenceStage):
    """
    Deprecated
    """
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: LongViePipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        """Process.

        Args:
            pipe: The pipe.
            input_image: The input image.
            end_image: The end image.
            num_frames: The num frames.
            height: The height.
            width: The width.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_ImageEmbedderCLIP(InferenceStage):
    """Wan video unit image embedder clip implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: LongViePipeline, input_image, end_image, height, width):
        """Process.

        Args:
            pipe: The pipe.
            input_image: The input image.
            end_image: The end image.
            height: The height.
            width: The width.
        """
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(InferenceStage):
    """Wan video unit image embedder vae implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "input_image_token"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LongViePipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride, input_image_token):
        """Process.

        Args:
            pipe: The pipe.
            input_image: The input image.
            end_image: The end image.
            num_frames: The num frames.
            height: The height.
            width: The width.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
            input_image_token: The input image token.
        """
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        # if input_image_token is not None:
        #     y = input_image_token[0]
        # else:
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # print(y.size(),msk.size())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(InferenceStage):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LongViePipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        """Process.

        Args:
            pipe: The pipe.
            input_image: The input image.
            latents: The latents.
            height: The height.
            width: The width.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_LongVieControlEmbedder(InferenceStage):
    """Wan video unit long vie control embedder implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            input_params=("dense_video", "dense_image", "sparse_video", "sparse_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae")
        )

    def process(self, pipe: LongViePipeline, dense_video, dense_image, sparse_video, sparse_image, num_frames, height, width, tiled, tile_size, tile_stride):
        """Process.

        Args:
            pipe: The pipe.
            dense_video: The dense video.
            dense_image: The dense image.
            sparse_video: The sparse video.
            sparse_image: The sparse image.
            num_frames: The num frames.
            height: The height.
            width: The width.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        if dense_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        dense_video = pipe.preprocess_video(dense_video)
        dense_latents = pipe.vae.encode(dense_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        dense_latents = dense_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        sparse_video = pipe.preprocess_video(sparse_video)
        sparse_latents = pipe.vae.encode(sparse_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        sparse_latents = sparse_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        dense_image = pipe.preprocess_image(dense_image.resize((width, height))).to(pipe.device)
        sparse_image = pipe.preprocess_image(sparse_image.resize((width, height))).to(pipe.device)

        vae_input_dense = torch.concat([dense_image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(dense_image.device)], dim=1)
        vae_input_sparse = torch.concat([sparse_image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(sparse_image.device)], dim=1)
        dense = pipe.vae.encode([vae_input_dense.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        dense = dense.to(dtype=pipe.torch_dtype, device=pipe.device)
        dense = torch.concat([msk, dense])
        dense = dense.unsqueeze(0)
        dense = dense.to(dtype=pipe.torch_dtype, device=pipe.device)

        sparse = pipe.vae.encode([vae_input_sparse.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        sparse = sparse.to(dtype=pipe.torch_dtype, device=pipe.device)
        sparse = torch.concat([msk, sparse])
        sparse = sparse.unsqueeze(0)
        sparse = sparse.to(dtype=pipe.torch_dtype, device=pipe.device)

        dense = torch.cat([dense_latents, dense],dim=1)
        sparse = torch.cat([sparse_latents, sparse],dim=1)
        return {"dense": dense, "sparse": sparse}





class WanVideoUnit_UnifiedSequenceParallel(InferenceStage):
    """Wan video unit unified sequence parallel implementation."""
    def __init__(self):
        """Init."""
        super().__init__(input_params=())

    def process(self, pipe: LongViePipeline):
        """Process.

        Args:
            pipe: The pipe.
        """
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(InferenceStage):
    """Wan video unit tea cache implementation."""
    def __init__(self):
        """Init."""
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: LongViePipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        """Process.

        Args:
            pipe: The pipe.
            num_inference_steps: The num inference steps.
            tea_cache_l1_thresh: The tea cache l1 thresh.
            tea_cache_model_id: The tea cache model id.
        """
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(InferenceStage):
    """Wan video unit cfg merger implementation."""
    def __init__(self):
        """Init."""
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: LongViePipeline, inputs_shared, inputs_posi, inputs_nega):
        """Process.

        Args:
            pipe: The pipe.
            inputs_shared: The inputs shared.
            inputs_posi: The inputs posi.
            inputs_nega: The inputs nega.
        """
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TeaCache:
    """Tea cache implementation."""
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
        """Check.

        Args:
            dit: The dit.
            x: The x.
            t_mod: The t mod.
        """
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
        """Store.

        Args:
            hidden_states: The hidden states.
        """
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        """Update.

        Args:
            hidden_states: The hidden states.
        """
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    """Temporal tiler bcthw implementation."""
    def __init__(self):
        """Init."""
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        """Build 1d mask.

        Args:
            length: The length.
            left_bound: The left bound.
            right_bound: The right bound.
            border_width: The border width.
        """
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        """Build mask.

        Args:
            data: The data.
            is_bound: The is bound.
            border_width: The border width.
        """
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        """Run.

        Args:
            model_fn: The model fn.
            sliding_window_size: The sliding window size.
            sliding_window_stride: The sliding window stride.
            computation_device: The computation device.
            computation_dtype: The computation dtype.
            model_kwargs: The model kwargs.
            tensor_names: The tensor names.
            batch_size: The batch size.
        """
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    dual_controller: WanModelDualControl = None,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: Any = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    dense=None,
    sparse=None,
    history_latents=None,
    **kwargs,
):
    """Model fn wan video.

    Args:
        dit: The dit.
        dual_controller: The dual controller.
        motion_controller: The motion controller.
        vace: The vace.
        animate_adapter: The animate adapter.
        latents: The latents.
        timestep: The timestep.
        context: The context.
        clip_feature: The clip feature.
        y: The y.
        reference_latents: The reference latents.
        vace_context: The vace context.
        vace_scale: The vace scale.
        audio_embeds: The audio embeds.
        motion_latents: The motion latents.
        s2v_pose_latents: The s2v pose latents.
        drop_motion_frames: The drop motion frames.
        tea_cache: The tea cache.
        use_unified_sequence_parallel: The use unified sequence parallel.
        motion_bucket_id: The motion bucket id.
        pose_latents: The pose latents.
        face_pixel_values: The face pixel values.
        sliding_window_size: The sliding window size.
        sliding_window_stride: The sliding window stride.
        cfg_merge: The cfg merge.
        use_gradient_checkpointing: The use gradient checkpointing.
        use_gradient_checkpointing_offload: The use gradient checkpointing offload.
        control_camera_latents_input: The control camera latents input.
        fuse_vae_embedding_in_latents: The fuse vae embedding in latents.
        dense: The dense.
        sparse: The sparse.
        history_latents: The history latents.
    """
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            dual_controller=dual_controller,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if use_unified_sequence_parallel:
        if not dist.is_initialized():
            raise RuntimeError("LongVie USP requires an initialized torch.distributed process group")

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, _sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    def patch_volume(value, camera_control=None):
        value = dit.patch_embedding(value)
        if dit.control_adapter is not None and camera_control is not None:
            camera_value = dit.control_adapter(camera_control)
            value = [base + control for base, control in zip(value, camera_value)][0].unsqueeze(0)
        return value

    x = patch_volume(x, control_camera_latents_input)

    if history_latents is not None:
        ones = torch.ones(
            history_latents.size(0),
            20,
            history_latents.size(2),
            history_latents.size(3),
            history_latents.size(4),
            device=history_latents.device,
            dtype=history_latents.dtype
        )
        history_latents = torch.cat([ones,history_latents], dim=1)
        history = patch_volume(history_latents)
    dense = patch_volume(dense)
    sparse = patch_volume(sparse)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    dense = rearrange(dense, 'b c f h w -> b (f h w) c').contiguous()
    sparse = rearrange(sparse, 'b c f h w -> b (f h w) c').contiguous()
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if history_latents is not None:
        f_h, h_h, w_h = history.shape[2:]
        history = rearrange(history, 'b c f h w -> b (f h w) c').contiguous()
        history_token = history.shape[1]
        x = torch.cat([history, x], dim=1)
        f += f_h

    dense = dual_controller.control_initial_combine_linear_dense(dense)
    sparse = dual_controller.control_initial_combine_linear_sparse(sparse)
    control_context = dual_controller.control_text_linear(context)
    control_t_mod = dual_controller.control_t_mod(t_mod)

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs_history = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False


    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x_original_len = x.size(1)
            dense_original_len = dense.size(1)
            chunks = torch.chunk(x, _sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[_sequence_parallel_rank()]

            dense_chunks = torch.chunk(dense, _sequence_parallel_world_size(), dim=1)
            dense_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, dense_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in dense_chunks]
            dense = dense_chunks[_sequence_parallel_rank()]
            
            sparse_chunks = torch.chunk(sparse, _sequence_parallel_world_size(), dim=1)
            sparse_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, sparse_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in sparse_chunks]
            sparse = sparse_chunks[_sequence_parallel_rank()]
            

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """
            def custom_forward(*inputs):
                """Custom forward."""
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs_history,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs_history,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs_history)

            if block_id < dual_controller.control_layers:
                dense_block = dual_controller.control_blocks_dense[block_id]
                sparse_block = dual_controller.control_blocks_sparse[block_id]
                
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        dense = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(dense_block),
                            dense, control_context, control_t_mod, freqs,
                            use_reentrant=False,
                        )
                        sparse = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(sparse_block),
                            sparse, control_context, control_t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    dense = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(dense_block),
                        dense, control_context, control_t_mod, freqs,
                        use_reentrant=False,
                    )
                    sparse = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(sparse_block),
                        sparse, control_context, control_t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    dense = dense_block(dense, control_context, control_t_mod, freqs)
                    sparse = sparse_block(sparse, control_context, control_t_mod, freqs)
                # for history context, we should gather x and dense to apply the indice of them
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1 and history_latents is not None:
                    x_full = _sequence_parallel_all_gather(x, dim=1)
                    x_full = x_full[:, :x_original_len]
                    
                    dense_full = _sequence_parallel_all_gather(dense, dim=1)
                    dense_full = dense_full[:, :dense_original_len]
                    
                    sparse_full = _sequence_parallel_all_gather(sparse, dim=1)
                    sparse_full = sparse_full[:, :dense_original_len]

                    x_full[:, -dense_original_len:] += dual_controller.control_combine_linears[block_id](dense_full + sparse_full)

                    x_chunks = torch.chunk(x_full, _sequence_parallel_world_size(), dim=1)
                    x_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, x_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in x_chunks]
                    x = x_chunks[_sequence_parallel_rank()]
                else:
                    if history_latents is not None:
                        x[:,-dense.size(1):] += dual_controller.control_combine_linears[block_id](dense + sparse)
                    else:
                        x += dual_controller.control_combine_linears[block_id](dense + sparse)


        if tea_cache is not None:
            tea_cache.store(x)

    if history_latents is not None:
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            x_full = _sequence_parallel_all_gather(x, dim=1)
            x_full = x_full[:, :x_original_len]
            x_full = x_full[:, -dense_original_len:]
            
            x_chunks = torch.chunk(x_full, _sequence_parallel_world_size(), dim=1)
            current_pad_shape = x_chunks[0].shape[1] - x_chunks[-1].shape[1]
            x_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, x_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in x_chunks]
            x = x_chunks[_sequence_parallel_rank()]
        else:
            x = x[:, -dense.size(1):]
        f -= f_h
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = _sequence_parallel_all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x
