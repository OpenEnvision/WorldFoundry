import contextlib
from pathlib import Path

import torch
import torch.distributed as dist

from worldfoundry.core.vram import get_cuda_free_memory_gb, gpu, move_model_to_device_with_memory_preservation
from worldfoundry.synthesis.visual_generation.echo_infinity.echo_infinity_runtime.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _model_root_from_kwargs(model_kwargs):
    model_root = _config_get(model_kwargs, "model_root")
    if model_root is not None:
        return Path(model_root).expanduser().resolve()
    wan_root = _config_get(model_kwargs, "wan_root")
    if wan_root is None:
        raise ValueError("Echo-Infinity model_kwargs must define wan_root or model_root.")
    model_name = _config_get(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
    return (Path(wan_root).expanduser() / model_name).resolve()


def _move_conditioning_to_device(conditional_dict, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in conditional_dict.items()}


class CausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        model_kwargs = getattr(args, "model_kwargs", {})
        model_root = _model_root_from_kwargs(model_kwargs)
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True) if generator is None else generator
        inner = self.generator.module.model if hasattr(self.generator, "module") else self.generator.model
        memory_kwargs = getattr(args, "memory_kwargs", None)
        _use_sink_memory = (
            memory_kwargs.get("use_sink_memory", False)
            if isinstance(memory_kwargs, dict)
            else getattr(memory_kwargs, "use_sink_memory", False)
            if memory_kwargs is not None
            else False
        )
        _mem_enabled = (
            memory_kwargs.get("enabled", False)
            if isinstance(memory_kwargs, dict)
            else getattr(memory_kwargs, "enabled", False)
            if memory_kwargs is not None
            else False
        )
        if _use_sink_memory and getattr(inner, "sink_memory", None) is None:
            inner.setup_sink_memory(memory_kwargs)
        elif _mem_enabled and getattr(inner, "query_memory_encoder", None) is None:
            inner.setup_memory_encoder(memory_kwargs)
        self.text_encoder = (
            WanTextEncoder(model_root=model_root, device=device) if text_encoder is None else text_encoder
        )
        self.vae = WanVAEWrapper(model_root=model_root) if vae is None else vae
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        conditional_dict = _move_conditioning_to_device(self.text_encoder(text_prompts=text_prompts), noise.device)
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )
        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width], device=output_device, dtype=noise.dtype
        )
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})"
        )
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device, kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)
        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()
        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()
            noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64)
                    * current_timestep
                )
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)
            current_start_frame += current_num_frames
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if output.device == noise.device:
            if getattr(self.args.model_kwargs, "use_infinite_attention", False):
                video = self.vae.decode_to_pixel_chunk(output, use_cache=False)
            else:
                video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        else:
            video = torch.zeros([batch_size, 1, 3, 480, 832], device=output.device, dtype=torch.float32)
        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(
                    f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)"
                )
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")
            self.last_profile_info = {
                "init_ms": float(init_time),
                "diffusion_ms": float(diffusion_time),
                "vae_ms": float(vae_time),
                "total_ms": float(total_time),
                "block_ms": [float(b) for b in block_times],
                "num_output_frames": int(num_output_frames),
                "num_frame_per_block": int(self.num_frame_per_block),
                "num_blocks": int(num_blocks),
            }
        if return_latents:
            return (video, output)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        kv_cache1 = []
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        elif self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"  # noqa: F841 - Upstream assignment kept.
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
        updated_modules = []
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = self.generator.model.max_attention_size
            except Exception:
                prev = None
            self.generator.model.max_attention_size = target_size
            updated_modules.append("<root_model>")
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                with contextlib.suppress(Exception):
                    prev = module.max_attention_size  # noqa: F841 - Upstream assignment kept.
                try:
                    module.max_attention_size = target_size
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
