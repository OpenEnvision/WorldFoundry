"""DualCamCtrl official camera/depth dual-branch Wan pipeline."""

import torch
import os
import types
import numpy as np
from PIL import Image
from einops import rearrange, repeat, reduce
from typing import Optional, Union
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
    T5RelativeEmbedding,
    T5LayerNorm,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import (
    WanMotionControllerModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.vace import VaceWanModel
from worldfoundry.base_models.diffusion_model.runners.staged import (
    InferenceStage,
    InferenceStageRunner,
    StagedDiffusionPipeline,
)
from worldfoundry.core.model_loading import GeneralLoRALoader, load_state_dict
from worldfoundry.core.nn import FlowMatchScheduler, RMSNorm, sinusoidal_embedding_1d
from worldfoundry.core.vram import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)


class WanVideoCameraPipeline(StagedDiffusionPipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(
            shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.model_names = ("text_encoder", "image_encoder", "dit", "vae", "motion_controller", "vace")
        self.unit_runner = InferenceStageRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedder(),
            WanVideoUnit_ControlEmbedder(),
            # WanVideoUnit_FunControl(),
            # WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            # WanVideoUnit_SpeedControl(),
            # WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            # WanVideoUnit_TeaCache(),
            # WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(
            torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(
            path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                vram_config=dict(
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_vram_config=dict(
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                vram_config=dict(
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                vram_config=dict(
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                vram_config=dict(
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
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from worldfoundry.core.attention.parallel.patch_xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @classmethod
    def from_components(
        cls,
        *,
        config_path: str,
        text_encoder: WanTextEncoder,
        base_dit: WanModel,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
        tokenizer_path: str,
        motion_controller: WanMotionControllerModel | None = None,
        vace: VaceWanModel | None = None,
        copy_control_weights: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        use_usp: bool = False,
    ) -> "WanVideoCameraPipeline":
        """Assemble DualCamCtrl around explicitly loaded native Wan roles."""

        pipe = cls(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        if use_usp:
            pipe.initialize_usp()
        pipe.text_encoder = text_encoder
        pipe.vae = vae
        pipe.image_encoder = image_encoder
        pipe.motion_controller = motion_controller
        pipe.vace = vace

        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import WanControlNet
        from omegaconf import OmegaConf

        conf = OmegaConf.load(config_path)
        controlnet = WanControlNet(**conf["dit"], **conf).to(
            dtype=next(iter(base_dit.parameters())).dtype,
            device="cpu",
        )
        state = controlnet.load_state_dict(base_dit.state_dict(), strict=False)
        if state.unexpected_keys:
            raise RuntimeError(f"DualCamCtrl base Wan produced unexpected keys: {state.unexpected_keys}")

        if copy_control_weights:
            controlnet.copy_weights_from_main_branch()
        pipe.dit = controlnet
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_path)

        if use_usp:
            pipe.enable_usp()
        pipe.eval()
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
        # TODO Uncomment this
        extra_images: Optional[list[Image.Image]] = None,
        extra_image_frame_index: Optional[list[int]] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        input_control: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        return_control_latents: Optional[bool] = False,
        # Camera control
        plucker_embedding: Optional[torch.Tensor] = None,
        camera_control_direction: Optional[
            Literal[
                "Left",
                "Right",
                "Up",
                "Down",
                "LeftUp",
                "LeftDown",
                "RightUp",
                "RightDown",
            ]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        batch_size: Optional[int] = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=41,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
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
        # t2v or i2v
        t2v: Optional[bool] = False,
        stage='naive',
    ):
        # Scheduler
        try:
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
                methods=stage,
            )
        except TypeError as exc:
            if "methods" not in str(exc):
                raise
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
            )

        # Inputs
        inputs_posi = {
            "prompt_num": batch_size,
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "prompt_num": batch_size,
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "batch_size": batch_size,
            "plucker_embedding": plucker_embedding,
            "input_image": input_image,
            "input_control": input_control,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "t2v": t2v,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "return_control_latents": return_control_latents,
        }
        _cfg_scale = inputs_shared["cfg_scale"]
        # print(f"cfg scale is {_cfg_scale} ")

        for unit in self.units:
            # print(f"Process unit {unit.__class__.__name__} ...")
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name)
                  for name in self.in_iteration_models}

        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            # Inference
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep
            )
            noise_pred_nega = self.model_fn(
                **models, **inputs_shared, **inputs_nega, timestep=timestep
            )

            if isinstance(noise_pred_posi, tuple):
                assert isinstance(noise_pred_nega, tuple)
                main_noise_pred_posi = noise_pred_posi[0]
                control_noise_pred_posi = noise_pred_posi[1]

                main_noise_pred_nega = noise_pred_nega[0]
                control_noise_pred_nega = noise_pred_nega[1]

                main_noise_pred = main_noise_pred_posi + cfg_scale * (
                    main_noise_pred_posi - main_noise_pred_nega
                )
                control_noise_pred = control_noise_pred_posi + cfg_scale * (
                    control_noise_pred_posi - control_noise_pred_nega
                )
                # Scheduler
                inputs_shared["latents"] = self.scheduler.step(
                    main_noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"],
                )
                inputs_shared["control_latents"] = self.scheduler.step(
                    control_noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["control_latents"],
                )
            else:
                main_noise_pred = noise_pred_posi + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
                inputs_shared["latents"] = self.scheduler.step(
                    main_noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"],
                )

        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(["vae"])
        video, control_video = None, None
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)

        if return_control_latents:
            control_video = self.vae.decode(
                inputs_shared["control_latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            control_video = self.vae_output_to_video(control_video)

        # print(f"Video shape after VAE decode: {video.shape}")
        self.load_models_to_device([])
        return {"images": video, "control_video": control_video}


class WanVideoUnit_ShapeChecker(InferenceStage):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoCameraPipeline, height, width, num_frames):

        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )

        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(InferenceStage):
    def __init__(self):
        super().__init__(
            input_params=(
                "batch_size",
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
                "vace_reference_image",
            )
        )

    def process(
        self,
        pipe: WanVideoCameraPipeline,
        batch_size,
        height,
        width,
        num_frames,
        seed,
        rand_device,
        vace_reference_image,
    ):
        length = (num_frames - 1) // 4 + 1
        noise = pipe.generate_noise(
            (batch_size, 16, length, height // 8, width // 8),
            seed=seed,
            rand_device=rand_device,
        )

        # For debug TODO
        control_noise = noise
        return {"noise": noise, "control_noise": control_noise}


class WanVideoUnit_InputVideoEmbedder(InferenceStage):
    def __init__(self):
        super().__init__(
            input_params=(
                "batch_size",
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoCameraPipeline,
        batch_size,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
    ):
        if input_video is not None:
            raise ValueError("DualCamCtrl native inference does not accept input_video")
        return {"latents": noise}


class WanVideoUnit_PromptEmbedder(InferenceStage):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "prompt": "prompt",
                "positive": "positive",
                "prompt_num": "prompt_num",
            },
            input_params_nega={
                "prompt": "negative_prompt",
                "positive": "positive",
                "prompt_num": "prompt_num",
            },
            onload_model_names=("text_encoder",),
        )

    def process(
        self, pipe: WanVideoCameraPipeline, prompt, positive, prompt_num
    ) -> dict:
        # pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = []
        # print(f"Encoding prompt: {prompt}")
        if isinstance(prompt, str):
            prompt = [prompt] * prompt_num
        for _prompt in prompt:
            # print(f"Prompt: {_prompt}")
            _prompt_emb = pipe.prompter.encode_prompt(
                _prompt, positive=positive, device=pipe.device
            )
            # print(f"_prompt embedding shape: {_prompt_emb.shape}")
            prompt_emb.append(_prompt_emb)

        prompt_emb = torch.cat(prompt_emb, dim=0)
        prompt_emb = prompt_emb.to(dtype=pipe.torch_dtype, device=pipe.device)
        # print(f"Prompt embedding shape: {prompt_emb.shape}")
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(InferenceStage):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "batch_size",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoCameraPipeline,
        batch_size,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None:
            return {}
        # pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image).to(pipe.device)

        clip_context = pipe.image_encoder.encode_image([image])

        clip_context = clip_context.to(
            dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ControlEmbedder(InferenceStage):
    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "batch_size",
                "control_noise",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoCameraPipeline,
        batch_size,
        control_video,
        control_noise,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if control_video is not None:
            raise ValueError("DualCamCtrl native inference does not accept a separate control video")
        return {"control_latents": control_noise}


class WanVideoUnit_FunCameraControl(InferenceStage):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_control_direction",
                "camera_control_speed",
                "camera_control_origin",
                "latents",
                "input_image",
                "input_control",
                "plucker_embedding",
                "t2v",
            )
        )

    def process(
        self,
        pipe: WanVideoCameraPipeline,
        height,
        width,
        num_frames,
        plucker_embedding,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
        input_control,
        t2v,
    ):

        control_camera_video = plucker_embedding
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        # print(f"control_camera_latents shape: {control_camera_latents.shape}")
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        input_image = input_image.unsqueeze(1)  # B 1 C H W
        input_control = input_control.unsqueeze(1)  # B 1 C H W
        # print(
        #     f"input image shape: {input_image.shape}, input control shape: {input_control.shape}"
        # )
        control_input_latents = None
        input_latents = None
        if not t2v:
            input_latents = []
            for _input_image in input_image:
                _vae_input = pipe.preprocess_video(
                    _input_image)  # 1 C H W -> 1 C T H W
                _vae_input = _vae_input.to(
                    dtype=pipe.torch_dtype, device=pipe.device)
                # print(f"_vae_input shape: {_vae_input.shape}")
                _input_latents = pipe.vae.encode(
                    _vae_input,
                    device=pipe.device,
                )
                input_latents.append(_input_latents)
            input_latents = torch.cat(input_latents, dim=0)

            control_input_latents = []
            for _input_control in input_control:
                _vae_input = pipe.preprocess_video(_input_control)
                _vae_input = _vae_input.to(
                    dtype=pipe.torch_dtype, device=pipe.device)
                # print(f"_vae_input shape: {_vae_input.shape}")
                _input_latents = pipe.vae.encode(
                    _vae_input,
                    device=pipe.device,
                )
                control_input_latents.append(_input_latents)
            control_input_latents = torch.cat(control_input_latents, dim=0)
            # print(
            #     f"input_latents shape: {input_latents.shape}, control_input_latents shape: {control_input_latents.shape}"
            # )

        y = torch.zeros_like(latents).to(pipe.device)
        control_y = torch.zeros_like(latents).to(pipe.device)
        if not t2v:
            y[:, :, :1] = input_latents
            control_y[:, :, :1] = control_input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        control_y = control_y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # For debug  TODO
        # assert torch.allclose(control_y, y)
        return {
            "control_camera_latents_input": control_camera_latents_input,
            "y": y,
            "control_y": control_y,
        }


class WanVideoUnit_UnifiedSequenceParallel(InferenceStage):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoCameraPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join(
                [i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
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
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + 1) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros(
            (B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros(
            (1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    control_latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    control_y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    return_control_latents=False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
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
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    x = latents

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + \
            motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        control_latents = torch.cat([control_latents, control_y], dim=1)

        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    control_latents, (f_c, h_c, w_c) = dit.patchify(
        control_latents, control_camera_latents_input
    )
    assert (
        f == f_c and h == h_c and w == w_c
    ), f"Patchified latent shape mismatch: {x.shape} vs {control_latents.shape}, f: {f}, h: {h}, w: {w}, f_c: {f_c}, h_c: {h_c}, w_c: {w_c}"
    # print(f"Patchified latent shape: {x.shape}, f: {f}, h: {h}, w: {w}")

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]
            control_latents = torch.chunk(
                control_latents, get_sequence_parallel_world_size(), dim=1
            )[get_sequence_parallel_rank()]

    if tea_cache_update:
        x = tea_cache.update(x)
        control_latents = tea_cache.update(control_latents)

    else:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # for debug TODO
        # assert torch.allclose(x, control_latents), f"Latents and control latents are not the same."
        for block_id, block in enumerate(dit.blocks):

            # Control part
            if hasattr(dit, "control_blocks") and block_id in dit.control_block_index:

                control_block_id = dit.control_block_index.index(block_id)

                if block_id in dit.rgb_inject_blocks:
                    rgb_zero_linear_idx = dit.rgb_inject_blocks.index(block_id)
                    if dit.use_gate_3d_linear:
                        control_latents = control_latents + dit.rgb_zero_inits[
                            rgb_zero_linear_idx
                        ](x, control_latents, f, h, w)
                    else:
                        control_latents = control_latents + dit.rgb_zero_inits[
                            rgb_zero_linear_idx
                        ](x)

                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        control_latents = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(
                                dit.control_blocks[control_block_id]),
                            control_latents,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    control_latents = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(
                            dit.control_blocks[control_block_id]),
                        control_latents,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
                else:
                    control_latents = dit.control_blocks[control_block_id](
                        control_latents, context, t_mod, freqs
                    )
                if block_id in dit.depth_inject_blocks:
                    control_zero_linear_idx = dit.depth_inject_blocks.index(
                        block_id)
                    if dit.use_gate_3d_linear:
                        x = x + dit.control_zero_inits[control_zero_linear_idx](
                            x, control_latents, f, h, w
                        )
                    else:
                        x = x + dit.control_zero_inits[control_zero_linear_idx](
                            control_latents
                        )

            # Main block part
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)

            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if (
                    use_unified_sequence_parallel
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                ):
                    current_vace_hint = torch.chunk(
                        current_vace_hint, get_sequence_parallel_world_size(), dim=1
                    )[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
                control_latents = control_latents + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            tea_cache.store(control_latents)

    x = dit.head(x, t)
    control_latents = dit.head(control_latents, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            control_latents = get_sp_group().all_gather(control_latents, dim=1)

    x = dit.unpatchify(x, (f, h, w))
    control_latents = dit.unpatchify(control_latents, (f, h, w))
    if return_control_latents:
        return x, control_latents
    else:
        return x


