import gc
import json
import os
import types

import numpy as np
import torch
from safetensors.torch import load_file

from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl

from .utils import FlowMatchScheduler, SchedulerInterface
from .wan.modules.clip import CLIPModel
from .wan.modules.tokenizers import HuggingfaceTokenizer
from .wan.modules.vae import _video_vae


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

        self.text_encoder = (
            umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32, device=torch.device("cpu"))
            .eval()
            .requires_grad_(False)
        )

        # Dynamically construct path based on model_name
        t5_encoder_path = os.path.join(model_name, "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_encoder.load_state_dict(torch.load(t5_encoder_path, map_location="cpu", weights_only=True))

        # Dynamically construct tokenizer path
        tokenizer_path = os.path.join(model_name, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")

    @property
    def device(self):
        # Return actual device of text_encoder parameters
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: list[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens, strict=False):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name="ckpts/Wan-AI--Wan2.1-T2V-1.3B"):
        super().__init__()
        self.model_name = model_name
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path=f"{self.model_name}/Wan2.1_VAE.pth",
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

        self.dtype = torch.bfloat16

        self.vae_stride = (4, 8, 8)
        self.target_video_length = 81

    @torch.no_grad()
    def encode(self, pixel):
        device, dtype = pixel[0].device, self.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            output = [self.model.encode(u.to(self.dtype).unsqueeze(0), scale).squeeze(0) for u in pixel]
        return output

    def run_vae_encoder(self, img, new_target_video_length=None, add_first_to_kvcache=False):
        if add_first_to_kvcache == None:  # noqa: E711 - Keeps `==` for operands that may compare elementwise.
            add_first_to_kvcache = False

        if new_target_video_length is not None:
            self.target_video_length = new_target_video_length
        # Convert once to the target device and dtype to avoid repeated CPU/GPU copies.
        img = img.to(device="cuda", dtype=torch.bfloat16)
        h, w = img.shape[1:]
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        msk = torch.ones(
            1,
            self.target_video_length,
            lat_h,
            lat_w,
            device=img.device,
            dtype=torch.bfloat16,
        )

        #! Only plain I2V keeps the first frame as a given condition.
        msk[:, 1:] = 0

        #! Otherwise the first frame is denoised too, even when it is added to the KV cache.
        if add_first_to_kvcache:
            msk[:, 0:] = 0

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )  # Plain I2V: 1 for the first frame, 0 for the remaining 20.
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        #! Plain I2V
        if not add_first_to_kvcache:
            vae_encode_out = self.encode(
                [
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(
                                img[None],
                                size=(h, w),
                                mode="bicubic",
                            ).transpose(0, 1),  # [3, 1, H, W]
                            torch.zeros(
                                3,
                                self.target_video_length - 1,
                                h,
                                w,
                                device=img.device,
                                dtype=torch.bfloat16,
                            ),  # [3, F-1, H, W]
                        ],
                        dim=1,  # [3, F, H, W]
                    )
                ],
            )[0]  # [first frame image, zeros]

        else:
            vae_encode_out = self.encode(
                [
                    torch.concat(
                        [
                            torch.zeros(
                                3,
                                self.target_video_length,
                                h,
                                w,
                                device=img.device,
                                dtype=torch.bfloat16,
                            ),
                        ],
                        dim=1,
                    )
                ],
            )[0]

        # msk: 4 channels; plain I2V has 1 on the first frame and 0 elsewhere.

        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return vae_encode_out  # [20, 21, H, W]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            output = [self.model.encode(u.unsqueeze(0), scale).squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        decode_function = self.model.cached_decode if use_cache else self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanCLIPEncoder(torch.nn.Module):
    def __init__(self, model_name="ckpts/Wan-AI--Wan2.1-T2V-1.3B"):
        super().__init__()
        self.model_name = model_name
        self.image_encoder = CLIPModel(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(
                f"{self.model_name}/",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        )

    @property
    def device(self):
        # Return actual device of image_encoder parameters
        try:
            return next(self.image_encoder.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, img):
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        target_device = self.device
        if img.ndim == 3:
            img = img[:, None, :, :].to(target_device)
        elif img.ndim == 4:
            img = img.transpose(0, 1).to(target_device)

        clip_encoder_out = self.image_encoder.visual([img]).squeeze(0)
        return clip_encoder_out


class WanWrapperBase(torch.nn.Module):
    """
    Wan Wrapper Base Class - simplified model loading:

    Stage 1: Random initialization of base model + adapters/controllers
    Stage 2: Load official Wan pretrained weights (wan_model_name) - backbone only
    Stage 3: Load generator_backbone_ckpt_path as backbone weights (only if use_lora=False)

    NOTE:
    - LoRA is handled entirely in Task layer (using generator_lora_ckpt_path)
    - Submodule weights (control_adapter, sp_blocks) loaded in Task layer
    """

    def __init__(self, role="generator", config=None, **kwargs):
        super().__init__()
        self.config = config
        self.role = role

        # Extract parameters from config
        wan_model_name = getattr(config, "wan_model_name", None)
        generator_backbone_ckpt_path = getattr(config, "generator_backbone_ckpt_path", None)
        _generator_lora_ckpt_path = getattr(config, "generator_lora_ckpt_path", None)
        timestep_shift = getattr(config, "timestep_shift", 8.0)
        num_latent_frames = getattr(config, "num_latent_frames", config.image_or_video_shape[1])
        use_lora = getattr(config, "use_lora", False)
        # Default: if we already have a backbone checkpoint (non-LoRA), skip the official weights to avoid double loading a 14B model.
        default_load_official_backbone = use_lora or generator_backbone_ckpt_path is None
        load_official_backbone = getattr(config, "load_official_backbone", default_load_official_backbone)
        print(
            f"[Wrapper] load_official_backbone={load_official_backbone}, use_lora={use_lora}, generator_backbone_ckpt_path_set={generator_backbone_ckpt_path is not None}"
        )

        # ====== Stage 1: Random initialization ======
        self._init_base_model(**config)

        # Initialize scheduler
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

        self.seq_len = num_latent_frames * np.prod(self.config.image_or_video_shape[-2:]).item() // 4
        self.local_attn_size = num_latent_frames

        # ====== Stage 2: Load official Wan pretrained weights ======
        if wan_model_name is not None and load_official_backbone:
            self._load_official_weights(wan_model_name)
        elif wan_model_name is not None:
            print("[Wrapper] Skipping official Wan weights load (load_official_backbone=False)")

        # ====== Stage 3: Load backbone checkpoint ======
        if role == "generator" and generator_backbone_ckpt_path:
            self._load_backbone_checkpoint(generator_backbone_ckpt_path)

        # Freeze all parameters by default
        self.requires_grad_(False)
        torch.cuda.empty_cache()

    def _init_base_model(self, **kwargs):
        """
        Stage 1: Initialize base model with random weights.
        All sub-modules (adapters, controllers) are also randomly initialized here.
        Subclasses should override this to call load_caus_or_bir_basemodel.
        """
        self.load_caus_or_bir_basemodel(**kwargs)

    def _load_official_weights(self, wan_model_name: str):
        """
        Stage 2: Load official Wan pretrained weights.
        Only loads weights for the base model backbone.
        Sub-modules (adapters, controllers) remain randomly initialized.
        """
        print(f"[Wrapper] Loading official Wan weights from: {wan_model_name}")
        wan_state_dict = self._load_state_dict_from_pretrained(wan_model_name)

        # Filter out adapter/controller weights to keep them randomly initialized
        backbone_state_dict = {k: v for k, v in wan_state_dict.items() if not self._is_submodule_key(k)}

        missing, unexpected = self.model.load_state_dict(backbone_state_dict, strict=False)

        # Log missing keys (expected for adapters/controllers)
        if missing:
            submodule_missing = [k for k in missing if self._is_submodule_key(k)]
            backbone_missing = [k for k in missing if not self._is_submodule_key(k)]
            if backbone_missing:
                print(f"[Wrapper] WARNING: Missing backbone keys: {backbone_missing[:5]}...")
            if submodule_missing:
                print(f"[Wrapper] INFO: Submodule keys not loaded (expected): {len(submodule_missing)} keys")

        assert len(unexpected) == 0, f"Official weights load failed! Unexpected keys: {unexpected}"

        del wan_state_dict, backbone_state_dict
        gc.collect()
        torch.cuda.empty_cache()
        print("[Wrapper] Official weights loaded successfully")

    def _is_submodule_key(self, key: str) -> bool:
        """
        Check if a key belongs to a submodule (adapter/controller).
        These should be loaded separately, not from official weights.
        """
        submodule_prefixes = [
            "control_adapter",
            "sp_blocks",
            "sp_patch_embedding",
        ]
        return any(prefix in key for prefix in submodule_prefixes)

    def _load_backbone_checkpoint(self, generator_ckpt_path: str):
        """
        Load full backbone weights from checkpoint (non-LoRA case).
        """
        print(f"[Wrapper] Loading backbone checkpoint from: {generator_ckpt_path}")

        state_dict = self._load_checkpoint_file(generator_ckpt_path)

        # Load full model weights (excluding submodule keys)
        backbone_state_dict = {k: v for k, v in state_dict.items() if not self._is_submodule_key(k)}
        missing, unexpected = self.model.load_state_dict(backbone_state_dict, strict=False)

        if unexpected:
            # Filter out LoRA keys which are expected to be missing when not using LoRA
            non_lora_unexpected = [k for k in unexpected if "lora_" not in k]
            assert len(non_lora_unexpected) == 0, (
                f"Backbone checkpoint load failed! Unexpected keys: {non_lora_unexpected}"
            )

        print("[Wrapper] Backbone weights loaded successfully")
        del state_dict, backbone_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def _load_checkpoint_file(self, ckpt_path: str) -> dict:
        """
        Load checkpoint file and handle different formats.

        Respects config.load_ema setting:
        - load_ema=True (default): prefer generator_ema if available
        - load_ema=False: always use generator, ignore generator_ema
        """
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            # Use mmap for large checkpoint files to avoid OOM during loading
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

        # Check load_ema config (default True for backward compatibility)
        load_ema = getattr(self.config, "load_ema", True)

        # Handle different checkpoint formats
        if "generator" in state_dict or "generator_ema" in state_dict:
            if load_ema and "generator_ema" in state_dict:
                # Prefer EMA weights
                print("[Wrapper] Loading generator_ema weights (load_ema=True)")
                state_dict = state_dict["generator_ema"]
            elif "generator" in state_dict:
                # Use non-EMA weights
                print(f"[Wrapper] Loading generator weights (load_ema={load_ema})")
                state_dict = state_dict["generator"]
            else:
                # Only EMA available but load_ema=False, use EMA anyway with warning
                print("[Wrapper] WARNING: load_ema=False but only generator_ema found, using it anyway")
                state_dict = state_dict["generator_ema"]

            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        return state_dict

    def load_caus_or_bir_basemodel(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _load_state_dict_from_pretrained(model_path: str) -> dict:
        """
        Load state_dict directly from pretrained model path without instantiating the model class.
        This avoids memory issues when the pretrained model uses a different class (e.g., WanTransformer3DModel vs WanModel).

        Supports both single safetensors file and sharded safetensors files.
        """
        # Check for single safetensors file
        single_file = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(single_file):
            print(f"Loading state_dict from single file: {single_file}")
            return load_file(single_file, device="cpu")

        # Check for sharded safetensors files
        index_file = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_file):
            print(f"Loading state_dict from sharded files: {index_file}")
            with open(index_file) as f:
                index = json.load(f)

            # Get unique shard files
            shard_files = set(index["weight_map"].values())

            # Load and merge all shards
            state_dict = {}
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(model_path, shard_file)
                print(f"  Loading shard: {shard_file}")
                shard_dict = load_file(shard_path, device="cpu")
                state_dict.update(shard_dict)

            return state_dict

        # Fallback to .bin format
        bin_file = os.path.join(model_path, "diffusion_pytorch_model.bin")
        if os.path.exists(bin_file):
            print(f"Loading state_dict from bin file: {bin_file}")
            return torch.load(bin_file, map_location="cpu", weights_only=True)

        raise FileNotFoundError(
            f"No model weights found in {model_path}. "
            f"Expected one of: diffusion_pytorch_model.safetensors, "
            f"diffusion_pytorch_model.safetensors.index.json, or diffusion_pytorch_model.bin"
        )

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = (
            x.double().to(flow_pred.device) for x in [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def forward(
        self,
        noisy_image_or_video,
        timestep: torch.Tensor,
        context,
        clip_fea=None,
        y=None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        kv_start: int | None = None,
        kv_end: int | None = None,
        rope_start: int | None = None,
        y_camera=None,
        **kwargs,
    ) -> torch.Tensor:
        # [B, F] -> [B]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            # dzc: For training Self-Forcing; for infering DMD, Self-Forcing
            flow_pred = self.model(
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=context,
                clip_fea=clip_fea,
                y=list(y) if y is not None else None,  # y is [B, C, F, H, W], model expects list of [C, F, H, W]
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                kv_start=kv_start,
                kv_end=kv_end,
                rope_start=rope_start,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=context,
                clip_fea=clip_fea,
                y=list(y) if y is not None else None,  # y is [B, C, F, H, W], model expects list of [C, F, H, W]
                seq_len=self.seq_len,
                y_camera=y_camera,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()


# --- Additional import for BidirectionalWanWrapperSP ---

import torch  # noqa: E402 - Follows module-level definitions, as upstream.

from .wan.modules.model_state_adapter import (  # noqa: E402 - Follows module-level definitions, as upstream.
    StateAdapterWanModel,
)


class BidirectionalWanWrapperSP(WanWrapperBase):
    """
    Wrapper for LiveWorld: Camera-controlled video generation with State Adapter.

    Uses StateAdapterWanModel which is optimized for:
    - Per-frame timestep (R+P frames get t=0, T frames get sampled t)
    - State Adapter processes P+T scene projections while main model has R+P+T frames

    Note: State Adapter weights are loaded in the Task layer (_load_sp_weights),
    not in the wrapper, following the same pattern as camera adapter loading.
    """

    def __init__(self, role="generator", config=None, **kwargs):
        super().__init__(role=role, config=config, **kwargs)
        # LiveWorld always uses per-frame timestep
        self.uniform_timestep = False

    def load_caus_or_bir_basemodel(self, **kwargs):
        wan_config_dict = kwargs["wan_config_dict"]
        self.model = StateAdapterWanModel(**wan_config_dict, **kwargs)
        self.model.eval()

    def forward(
        self,
        noisy_image_or_video,
        timestep: torch.Tensor,
        context,
        clip_fea=None,
        y=None,
        sp_context=None,
        sp_context_scale=1.0,
        sp_hint_offset=0,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        kv_start: int | None = None,
        kv_end: int | None = None,
        rope_start: int | None = None,
        y_camera=None,
        **kwargs,
    ) -> torch.Tensor:
        # [B, F] -> [B]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        # Calculate seq_len based on actual input shape (may include R+P+T frames for LiveWorld)
        # noisy_image_or_video: [B, F, C, H, W]
        actual_frames = noisy_image_or_video.shape[1]
        H, W = noisy_image_or_video.shape[-2:]
        actual_seq_len = actual_frames * (H // 2) * (W // 2)  # after patch embedding with stride (1,2,2)

        # sp_context is already [C, T, H, W] format from task layer
        if kv_cache is not None:
            # dzc: For training Self-Forcing; for infering DMD, Self-Forcing
            flow_pred = self.model(
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=context,
                clip_fea=clip_fea,
                y=list(y) if y is not None else None,  # y is [B, C, F, H, W], model expects list of [C, F, H, W]
                sp_context=sp_context,
                sp_context_scale=sp_context_scale,
                sp_hint_offset=sp_hint_offset,
                seq_len=actual_seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                kv_start=kv_start,
                kv_end=kv_end,
                rope_start=rope_start,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=context,
                clip_fea=clip_fea,
                y=list(y) if y is not None else None,  # y is [B, C, F, H, W], model expects list of [C, F, H, W]
                sp_context=sp_context,
                sp_context_scale=sp_context_scale,
                sp_hint_offset=sp_hint_offset,
                seq_len=actual_seq_len,
                y_camera=y_camera,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0
