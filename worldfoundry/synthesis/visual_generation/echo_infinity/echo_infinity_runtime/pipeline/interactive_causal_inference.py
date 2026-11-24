import torch
import torch.distributed as dist

from worldfoundry.core.vram import get_cuda_free_memory_gb, gpu, move_model_to_device_with_memory_preservation
from worldfoundry.synthesis.visual_generation.echo_infinity.echo_infinity_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
    _move_conditioning_to_device,
)
from worldfoundry.synthesis.visual_generation.echo_infinity.echo_infinity_runtime.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        self.apr_enabled = getattr(args, "apr_enabled", False)
        self.apr_alpha_max = float(max(0.0, min(0.8, getattr(args, "apr_alpha_max", 0.8))))
        self.apr_d_window = getattr(args, "apr_d_window", None)
        self.apr_blend_sink = getattr(args, "apr_blend_sink", False)

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        sink_tok_size = None
        sink_backup = None
        sink_norm_before = None
        if self.global_sink and current_start_frame > 0:
            inner_tmp = self.generator.model
            sink_tok_size = inner_tmp.blocks[0].self_attn.sink_size * self.frame_seq_length
            sink_backup = [
                {
                    "k": self.kv_cache1[i]["k"][:, :sink_tok_size].clone(),
                    "v": self.kv_cache1[i]["v"][:, :sink_tok_size].clone(),
                }
                for i in range(self.num_transformer_blocks)
            ]
            sink_norm_before = torch.norm(self.kv_cache1[0]["k"][:, :sink_tok_size]).float().item()
            print(
                f"[Recache-Inf] switch@frame={current_start_frame}, global_sink=True, sink_tok={sink_tok_size}, sink_norm(block0) BEFORE = {sink_norm_before:.4f}"
            )
        elif not self.global_sink:
            print(
                f"[Recache-Inf] switch@frame={current_start_frame}, global_sink=False (sink will be recached with new prompt)"
            )
        apr_old_cache = None
        if self.apr_enabled and current_start_frame > 0:
            apr_old_cache = [
                {"k": self.kv_cache1[i]["k"].clone(), "v": self.kv_cache1[i]["v"].clone()}
                for i in range(self.num_transformer_blocks)
            ]
        if not self.global_sink:
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        inner = self.generator.model
        if current_start_frame == 0:
            return
        num_recache_frames = (
            current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        )
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        if frames_to_recache.device.type == "cpu":
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(
            f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}"
        )
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        context_timestep = (
            torch.ones([batch_size, num_recache_frames], device=device, dtype=torch.int64) * self.args.context_noise
        )
        self.generator.model.block_mask = block_mask
        _has_memory = (
            getattr(inner, "query_memory_encoder", None) is not None or getattr(inner, "sink_memory", None) is not None
        )
        if _has_memory:
            frame_seqlen = self.frame_seq_length
            sink_frames = inner.blocks[0].self_attn.sink_size
            max_attn = inner.blocks[0].self_attn.max_attention_size
            recent_window_frames = (max_attn - sink_frames * frame_seqlen) // frame_seqlen
            inner._ei_prev_window_start = max(sink_frames, current_start_frame - recent_window_frames)
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        if sink_backup is not None:
            for i in range(self.num_transformer_blocks):
                self.kv_cache1[i]["k"][:, :sink_tok_size] = sink_backup[i]["k"]
                self.kv_cache1[i]["v"][:, :sink_tok_size] = sink_backup[i]["v"]
            sink_norm_after = torch.norm(self.kv_cache1[0]["k"][:, :sink_tok_size]).float().item()
            delta = abs(sink_norm_after - sink_norm_before)
            status = "PRESERVED ✓" if delta < 0.001 else f"MISMATCH ✗ (delta={delta:.4e})"
            print(f"[Recache-Inf] global_sink=True: sink_norm AFTER = {sink_norm_after:.4f} [{status}]")
            del sink_backup
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        if self.apr_enabled and apr_old_cache is not None:
            self._apply_apr_blend(
                old_cache=apr_old_cache,
                recache_start_frame=recache_start_frame,
                current_start_frame=current_start_frame,
                num_recache_frames=num_recache_frames,
            )
            del apr_old_cache
        if getattr(inner, "query_memory_encoder", None) is not None:
            enc = inner.query_memory_encoder
            if getattr(enc, "memory_recache", False):
                enc.reset(batch_size=batch_size, device=frames_to_recache.device, dtype=torch.bfloat16)
                last_blk = self.num_transformer_blocks - 1
                cache = self.kv_cache1[last_blk]
                frame_seqlen = self.frame_seq_length
                sink_tok = inner.blocks[0].self_attn.sink_size * frame_seqlen
                local_end = cache["local_end_index"].item()
                if local_end > sink_tok:
                    recache_k = cache["k"][:, sink_tok:local_end].clone()
                    recache_v = cache["v"][:, sink_tok:local_end].clone()
                    sink_k = cache["k"][:, :sink_tok].clone() if sink_tok > 0 else None
                    sink_v = cache["v"][:, :sink_tok].clone() if sink_tok > 0 else None
                    enc.update(recache_k, recache_v, sink_k, sink_v)
        if getattr(inner, "sink_memory", None) is not None:
            sm = inner.sink_memory
            sm.reset()
            sink_frames_count = inner.blocks[0].self_attn.sink_size
            sink_output = output[:, :sink_frames_count]
            if sink_output.device.type == "cpu":
                sink_output = sink_output.to(device)
            sink_ts = (
                torch.ones([batch_size, sink_frames_count], device=device, dtype=torch.int64) * self.args.context_noise
            )
            n_heads = inner.blocks[0].self_attn.num_heads
            head_dim = inner.blocks[0].self_attn.head_dim
            sink_cache_size = sink_frames_count * self.frame_seq_length
            temp_kv = [
                {
                    "k": torch.zeros(
                        [batch_size, sink_cache_size, n_heads, head_dim], device=device, dtype=torch.bfloat16
                    ),
                    "v": torch.zeros(
                        [batch_size, sink_cache_size, n_heads, head_dim], device=device, dtype=torch.bfloat16
                    ),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
                }
                for _ in range(self.num_transformer_blocks)
            ]
            temp_crossattn = [
                {
                    "k": torch.zeros_like(self.crossattn_cache[0]["k"]),
                    "v": torch.zeros_like(self.crossattn_cache[0]["v"]),
                    "is_init": False,
                }
                for _ in range(self.num_transformer_blocks)
            ]
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=sink_output,
                    conditional_dict=new_conditional_dict,
                    timestep=sink_ts,
                    kv_cache=temp_kv,
                    crossattn_cache=temp_crossattn,
                    current_start=0,
                )
            inner._ei_prev_window_start = max(sink_frames, current_start_frame - recent_window_frames)
            print(f"[SinkMem] reset + re-captured on prompt switch at frame {current_start_frame}")

    def _apply_apr_blend(self, *, old_cache, recache_start_frame, current_start_frame, num_recache_frames):
        inner = self.generator.model
        frame_seqlen = self.frame_seq_length
        sink_size = inner.blocks[0].self_attn.sink_size
        sink_tok = sink_size * frame_seqlen
        D = self.apr_d_window if self.apr_d_window is not None else num_recache_frames
        D = max(1, int(D))
        alpha_max = self.apr_alpha_max
        n_frames = num_recache_frames
        with torch.no_grad():
            for blk_idx in range(self.num_transformer_blocks):
                new_k = self.kv_cache1[blk_idx]["k"]
                new_v = self.kv_cache1[blk_idx]["v"]
                old_k = old_cache[blk_idx]["k"]
                old_v = old_cache[blk_idx]["v"]
                if self.apr_blend_sink and (not self.global_sink):
                    for f in range(sink_size):
                        d_t = current_start_frame - f
                        alpha = min(alpha_max, max(0.0, 1.0 - d_t / D))
                        s0, s1 = (f * frame_seqlen, (f + 1) * frame_seqlen)
                        new_k[:, s0:s1] = (1 - alpha) * old_k[:, s0:s1] + alpha * new_k[:, s0:s1]
                        new_v[:, s0:s1] = (1 - alpha) * old_v[:, s0:s1] + alpha * new_v[:, s0:s1]
                base = sink_tok
                for f in range(n_frames):
                    absolute_frame = recache_start_frame + f
                    d_t = current_start_frame - absolute_frame
                    alpha = min(alpha_max, max(0.0, 1.0 - d_t / D))
                    s0, s1 = (base + f * frame_seqlen, base + (f + 1) * frame_seqlen)
                    new_k[:, s0:s1] = (1 - alpha) * old_k[:, s0:s1] + alpha * new_k[:, s0:s1]
                    new_v[:, s0:s1] = (1 - alpha) * old_v[:, s0:s1] + alpha * new_v[:, s0:s1]
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[APR] switch@{current_start_frame}: blended {n_frames} frames (α_max={alpha_max}, D_window={D})")

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: list[list[str]],
        switch_frame_indices: list[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        print(text_prompts_list)
        cond_list = [
            _move_conditioning_to_device(self.text_encoder(text_prompts=p), noise.device) for p in text_prompts_list
        ]
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )
        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width], device=output_device, dtype=noise.dtype
        )
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
            batch_size, dtype=noise.dtype, device=noise.device, kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0
        next_switch_pos = switch_frame_indices[segment_idx] if segment_idx < len(switch_frame_indices) else None
        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                next_switch_pos = switch_frame_indices[segment_idx] if segment_idx < len(switch_frame_indices) else None
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]
            noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64)
                    * current_timestep
                )
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
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
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            current_start_frame += current_num_frames
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return (video, output)
        return video
