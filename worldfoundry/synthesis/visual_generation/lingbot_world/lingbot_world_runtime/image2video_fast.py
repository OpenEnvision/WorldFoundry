import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from worldfoundry.core import autocast_context
from worldfoundry.core.distributed.sharding.block_fsdp import shard_model
from worldfoundry.core.attention.parallel.causal_rope_sequence_parallel import (
    sp_attn_forward_causal,
    sp_attn_forward_causal_chunked,
    sp_dit_forward_causal,
    sp_dit_forward_causal_chunked,
)
from worldfoundry.core.distributed.sequence_parallel.ops import get_world_size
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.fast import WanModelFast
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import T5EncoderModel
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming import Wan2_1_VAE

from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from .distributed import distributed_barrier
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


_LINGBOT_DBG = os.getenv("WORLDFOUNDRY_LINGBOT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _dbg_stats(name, t):
    """Temporary numerical-diagnostic logger (rank 0 only, env-guarded)."""

    if not _LINGBOT_DBG:
        return
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = 0
    if rank != 0:
        return
    try:
        tf = t.detach().float()
        print(
            f"[LINGBOT_DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={tf.mean().item():.4f} std={tf.std().item():.4f} "
            f"min={tf.min().item():.4f} max={tf.max().item():.4f} "
            f"nan={bool(torch.isnan(tf).any().item())} inf={bool(torch.isinf(tf).any().item())}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[LINGBOT_DBG] {name}: err {exc}", flush=True)


class WanI2VFast:

    def __init__(
        self,
        config,
        checkpoint_dir,
        fast_checkpoint_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype: torch.dtype = torch.bfloat16,
        control_type=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        inferred_control_type = control_type
        if inferred_control_type is None and 'cam' in checkpoint_dir:
            inferred_control_type = 'cam'
        elif inferred_control_type is None and 'act' in checkpoint_dir:
            inferred_control_type = 'act'
        if inferred_control_type not in {'cam', 'act'}:
            raise ValueError("LingBot control_type must be 'cam' or 'act'; pass it explicitly for custom checkpoint paths.")
        self.control_type = inferred_control_type

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        fast_checkpoint_dir = checkpoint_dir if fast_checkpoint_dir is None else fast_checkpoint_dir
        fast_subfolder = config.fast_noise_checkpoint if fast_checkpoint_dir == checkpoint_dir else None

        logging.info(f"Creating WanModelFast from {fast_checkpoint_dir}")
        self.model = WanModelFast.from_pretrained(
            fast_checkpoint_dir,
            subfolder=fast_subfolder,
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
        )
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            replicated_sequence = os.getenv(
                "WORLDFOUNDRY_LINGBOT_SP_LAYOUT",
                "chunked",
            ).strip().lower() == "replicated"
            attention_forward = (
                sp_attn_forward_causal
                if replicated_sequence
                else sp_attn_forward_causal_chunked
            )
            model_forward = (
                sp_dit_forward_causal
                if replicated_sequence
                else sp_dit_forward_causal_chunked
            )
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    attention_forward, block.self_attn)
            model.forward = types.MethodType(model_forward, model)

        distributed_barrier(self.device)

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]
    
        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
    
        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 179, 358, 679],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            if self.control_type == 'act':
                # In 'act' mode, use rotation of c2ws to control orientation and wasd_action to drive movement.
                wasd_action = np.load(os.path.join(action_path, "action.npy")) # wasd action
                wasd_action = wasd_action[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]
        if _LINGBOT_DBG:
            try:
                _sel_sig = self.scheduler.sigmas[timesteps_index]
                print(
                    f"[LINGBOT_DBG] shift={shift} timesteps_index={timesteps_index} "
                    f"timesteps={[round(float(t),2) for t in timesteps.tolist()]} "
                    f"sel_sigmas={[round(float(s),4) for s in _sel_sig.tolist()]}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[LINGBOT_DBG] timesteps log err {_e}", flush=True)

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        # cam preparation (only if action_path is provided)
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(Ks,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=h,
                                    width_resize=w,
                                    height_final=h,
                                    width_final=w)
            Ks = Ks[0]
            
            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            if self.control_type == 'act':
                wasd_action = torch.from_numpy(wasd_action[::4]).float().to(self.device)
            else:
                wasd_action = None
            only_rays_d = wasd_action is not None
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            if wasd_action is not None:
                wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
                wasd_action_tensor = rearrange(
                    wasd_action_tensor,
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                    c1=int(h // lat_h),
                    c2=int(w // lat_w),
                )
                wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
                wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)
        
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])
        _dbg_stats("y_cond", y)
        if action_path is not None:
            _dbg_stats("c2ws_plucker_emb", c2ws_plucker_emb)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)
        if next(self.model.parameters()).device.type == 'cpu':
            self.model.to(self.device)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                      shape=self_kv_shape,
                                                      dtype=transformer_dtype,
                                                      device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                         shape=cross_kv_shape,
                                                         dtype=transformer_dtype,
                                                         device=self.device)
        # evaluation mode
        with (
                autocast_context(self.device, dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }
    
                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]
    
                    timestep = torch.stack(current_timestep).to(self.device)
                    _dbg_stats(f"xt[c{chunk_id}s{timestep_idx} t={float(current_timestep[0]):.1f}]", current_latent)

                    noise_pred = self.model(
                        x=latent_model_input, t=timestep, **kwargs)[0]
                    _dbg_stats(f"noise_pred[c{chunk_id}s{timestep_idx}]", noise_pred)

                    if offload_model:
                        torch.cuda.empty_cache()
                    
                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )
                    _dbg_stats(f"x0[c{chunk_id}s{timestep_idx}]", x0)

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                self.model(x=[x0], t=timestep, **kwargs)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)
            _dbg_stats("pred_latent_chunks", pred_latent_chunks)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])
                _dbg_stats("decoded_video", videos[0])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        distributed_barrier(self.device)

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the CrossAttb
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False
            })

        return crossattn_cache
