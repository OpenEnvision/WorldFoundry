import time

import torch

from worldfoundry.core.vram import (
    get_cuda_free_memory_gb,
    gpu,
    move_model_to_device_with_memory_preservation,
)

from ..utils.misc import randn_like_fp32_cast
from ..utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from .causal_base import CausalPipelineBase


def _cuda_ms(ev_start: torch.cuda.Event, ev_end: torch.cuda.Event) -> float:
    """Return elapsed milliseconds between two recorded CUDA events."""
    torch.cuda.synchronize()
    return ev_start.elapsed_time(ev_end)


def _make_event() -> torch.cuda.Event:
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


class CausalInferencePipeline(CausalPipelineBase, torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        torch.nn.Module.__init__(self)
        # Step 1: Initialize all models
        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        print(f"before warp_denoising_step, denoising_step_list: {self.denoising_step_list}")
        if args.warp_denoising_step:
            # add 0 to the end of timesteps and use scheduler's timesteps with denoising_step_list as indices
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        print(f"after warp_denoising_step, denoising_step_list: {self.denoising_step_list}")
        self.num_transformer_blocks = 30
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.skip_clean_rerun = getattr(args, "skip_clean_rerun", False)
        self.skip_clean_rerun_copy_x_last_c_full = getattr(args, "skip_clean_rerun_copy_x_last_c_full", False)
        self.skip_clean_rerun_copy_x_last_c_last = getattr(args, "skip_clean_rerun_copy_x_last_c_last", False)
        self.use_flow_matching_denoising = getattr(args, "use_flow_matching_denoising", False)

        # With height-concat conditioning (use_cond_latent=True) height is doubled, so each
        # frame produces 2× tokens (3120 instead of 1560 for the default 60×52 patch grid).
        # MMDiT mode keeps separate streams → frame_seq_length stays at 1560.
        self.use_cond_latent = getattr(args, "use_cond_latent", False)
        self.is_mmdit = getattr(args, "model_kwargs", {}).get("is_mmdit", False)
        # MemRoPE Phase 2 — read the parsed config that WanDiffusionWrapper stored on the
        # model (set_memrope). When present, the split gen/cond three-tier KV cache is used
        # instead of the single sliding-window cache. Requires height-concat conditioning.
        self.memrope_cfg = getattr(getattr(self.generator, "model", None), "memrope_cfg", None)
        if self.memrope_cfg is not None:
            assert self.use_cond_latent and not self.is_mmdit, (
                "MemRoPE requires use_cond_latent=True (height-concat) and non-MMDiT model."
            )
            print(
                f"[CausalInferencePipeline] MemRoPE enabled: gen={self.memrope_cfg['gen']} "
                f"cond={self.memrope_cfg['cond']}"
            )
        # TODO: make the frame_seq_length adaptive to the number of frames in the input video
        if self.is_mmdit:
            self.frame_seq_length = 1560
        else:
            self.frame_seq_length = 3120 if self.use_cond_latent else 1560

        print(
            f"KV inference with {self.num_frame_per_block} frames per block"
            f" and frame_seq_length={self.frame_seq_length} tokens/frame "
            f"(use_cond_latent={self.use_cond_latent}, is_mmdit={self.is_mmdit} "
            f"skip_clean_rerun_copy_x_last_c_full={self.skip_clean_rerun_copy_x_last_c_full}, "
            f"skip_clean_rerun_copy_x_last_c_last={self.skip_clean_rerun_copy_x_last_c_last})"
        )
        # max_attention_size is patched at inference() time (not here) so it can
        # use the actual num_output_frames from the noise tensor. This makes the
        # global-attention case (local_attn_size=-1) correct for arbitrary video
        # lengths instead of being baked to the default num_max_frames=21.
        if self.is_mmdit:
            print(f"[CausalInferencePipeline] is_mmdit=True: frame_seq_length={self.frame_seq_length}")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        initial_latent: torch.Tensor | None = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        cond_latent: torch.Tensor | None = None,
        conditional_dict: dict | None = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise latent (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            cond_latent (torch.Tensor, optional): Conditioning render latent of shape
                (batch_size, num_output_frames, num_channels, height, width).
                When provided, each temporal block's slice is height-concatenated with
                the noisy input inside WanDiffusionWrapper.forward(). Requires
                use_cond_latent=True in the pipeline config (doubles frame_seq_length
                and KV cache capacity). Latent dims (T, C, H, W) are extracted from
                this tensor so they need not be hardcoded.
            conditional_dict (dict, optional): Pre-computed text conditioning dict
                {'prompt_embeds': Tensor [B, 512, 4096]}. When provided, the text
                encoder is bypassed. Useful for scene-id mode where text embeddings
                are stored alongside the latent tensors.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape  # (1,21,16,60,104)
        num_blocks, num_input_frames, num_output_frames = self._compute_num_blocks(num_frames, initial_latent)
        print(
            f"inference: noise shape={noise.shape}, initial_latent shape={initial_latent.shape if initial_latent is not None else None}, num_blocks={num_blocks}, num_input_frames={num_input_frames}, num_output_frames={num_output_frames}"
        )

        # Patch self-attention max_attention_size with the actual num_output_frames.
        # For local attention (local_attn_size != -1) the patch returns
        # local_attn_size * frame_seq_length and ignores num_max_frames, so this
        # is a no-op there. For global attention (-1) it correctly sizes the
        # window to the full output length.
        if not self.is_mmdit:
            self._patch_attn_max_size(self.generator, self.frame_seq_length, num_max_frames=num_output_frames)
        if conditional_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width], device=noise.device, dtype=noise.dtype
        )  # TODO: maybe empty is more effcient

        # Always-on timing
        timing_t0 = _make_event()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_output_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            self._reset_crossattn_cache(self.crossattn_cache)
            if self.memrope_cfg is not None:
                self._reset_memrope_kv_cache(self.kv_cache1)
            elif self.is_mmdit:
                self._reset_dual_kv_cache(self.kv_cache1, noise.device)
            else:
                self._reset_kv_cache(self.kv_cache1, noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            print("WARNING: i2v mode")
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cond_latent=cond_latent[:, :1] if cond_latent is not None else None,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :, current_start_frame : current_start_frame + self.num_frame_per_block
                ]
                output[:, current_start_frame : current_start_frame + self.num_frame_per_block] = current_ref_latents
                _cond_block = self._slice_cond(cond_latent, current_start_frame, self.num_frame_per_block)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                current_start_frame += self.num_frame_per_block

        init_end_ev = _make_event()

        # Step 3: Temporal denoising loop
        all_num_frames = self._build_all_num_frames(num_blocks, initial_latent)
        block_timing_stats = []  # list of {step_times_ms, ctx_ms, total_ms}
        for block_idx, current_num_frames in enumerate(all_num_frames):
            block_start_ev = _make_event()

            noisy_input = noise[
                :, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames
            ]

            print(
                f"Processing block {block_idx + 1}/{len(all_num_frames)} with num_input_frames={num_input_frames}, current_num_frames={current_num_frames}, noisy_input shape={noisy_input.shape}"
            )

            # Step 3.1: Spatial denoising loop
            step_times_ms = []
            for index, current_timestep in enumerate(self.denoising_step_list):
                step_start_ev = _make_event()
                # set current timestep
                timestep = (
                    torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64)
                    * current_timestep
                )

                _cond_block = self._slice_cond(cond_latent, current_start_frame, current_num_frames)
                print(
                    f"  Step {index + 1}/{len(self.denoising_step_list)}: timestep={current_timestep}, input shape={noisy_input.shape}, cond_latent block shape={_cond_block.shape if _cond_block is not None else None}"
                )
                if index < len(self.denoising_step_list) - 1:
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cond_latent=_cond_block,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    if self.use_flow_matching_denoising:
                        # Deterministic Euler ODE step: x_{t_next} = x_t + flow * (sigma_next - sigma_t)
                        noisy_input = self._flow_step(
                            flow_pred.flatten(0, 1),
                            noisy_input.flatten(0, 1),
                            timestep.flatten(0, 1),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # Stochastic consistency step: re-add fresh noise to pred_x0
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            randn_like_fp32_cast(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        # this has to be a new copy of the cond block to avoid in-place modifications, which will visually darken subsequent blocks when
                        cond_latent=self._slice_cond(cond_latent, current_start_frame, current_num_frames),
                    )
                step_end_ev = _make_event()
                step_times_ms.append(_cuda_ms(step_start_ev, step_end_ev))

            # Step 3.2: record the model's output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: update KV cache for use as context by the next block.
            # Two modes:
            #   skip_clean_rerun=False (default): rerun at t=context_noise (typically 0)
            #     with the clean prediction — one extra forward pass.
            #   skip_clean_rerun=True: skip the extra rerun.  If we exited the spatial
            #     denoising loop early (before the last step), continue running the
            #     remaining denoising steps (no grad) so that the KV cache ends in the
            #     same state as a clean re-run.
            ctx_start_ev = _make_event()
            if self.skip_clean_rerun_copy_x_last_c_full:
                # Skip the clean re-run forward pass and instead copy the
                # x-token KV entries from the last single-stream transformer
                # block to all layers (broadcasting).  C-token slots (height-
                # concat) and c-stream sub-buffers (MMDiT) are left unchanged.
                self._copy_x_last_to_all_layers(current_num_frames)
            elif self.skip_clean_rerun_copy_x_last_c_last:
                # Like _copy_x_last_to_all_layers but also copies c-tokens:
                # height-concat: full 3120 tokens (x+c) from last layer to all.
                # MMDiT: same x-broadcast as above, plus last double-stream
                #        block's c_k/c_v broadcast to all double-stream blocks.
                self._copy_all_last_to_all_layers(current_num_frames)
            elif self.skip_clean_rerun:
                pass
            else:
                context_timestep = torch.ones_like(timestep) * self.args.context_noise  # 0
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
            ctx_end_ev = _make_event()

            block_end_ev = _make_event()
            block_ms = _cuda_ms(block_start_ev, block_end_ev)
            ctx_ms = _cuda_ms(ctx_start_ev, ctx_end_ev)
            block_timing_stats.append(
                {
                    "step_times_ms": step_times_ms,
                    "ctx_ms": ctx_ms,
                    "total_ms": block_ms,
                }
            )
            step_summary = "  ".join(
                f"step{i}(t={int(self.denoising_step_list[i])})={t:.0f}ms" for i, t in enumerate(step_times_ms)
            )
            print(f"[Block {block_idx:2d}] total={block_ms:.0f}ms | {step_summary} | ctx_update={ctx_ms:.0f}ms")

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        vae_start_ev = _make_event()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        vae_end_ev = _make_event()

        # ── Timing summary ────────────────────────────────────────────────────
        init_ms = _cuda_ms(timing_t0, init_end_ev)
        diff_ms = sum(b["total_ms"] for b in block_timing_stats)
        vae_ms = _cuda_ms(vae_start_ev, vae_end_ev)
        total_ms = init_ms + diff_ms + vae_ms
        num_steps = len(self.denoising_step_list)
        print("\n=== Inference Timing Summary ===")
        print(f"  Init/cache : {init_ms:7.0f} ms  ({100 * init_ms / total_ms:.1f}%)")
        print(f"  Diffusion  : {diff_ms:7.0f} ms  ({100 * diff_ms / total_ms:.1f}%)")
        for b_idx, b in enumerate(block_timing_stats):
            pct = 100 * b["total_ms"] / diff_ms if diff_ms > 0 else 0.0
            print(f"    Block {b_idx:2d} : {b['total_ms']:6.0f} ms  ({pct:.1f}% of diffusion)")
        if block_timing_stats:
            print(f"  Per-step averages across {len(block_timing_stats)} blocks:")
            for i in range(num_steps):
                avg = sum(b["step_times_ms"][i] for b in block_timing_stats) / len(block_timing_stats)
                print(f"    step{i} (t={int(self.denoising_step_list[i])}): avg={avg:.0f}ms")
            avg_ctx = sum(b["ctx_ms"] for b in block_timing_stats) / len(block_timing_stats)
            print(f"    ctx_update            : avg={avg_ctx:.0f}ms")
        print(f"  VAE decode : {vae_ms:7.0f} ms  ({100 * vae_ms / total_ms:.1f}%)")
        print(f"  TOTAL      : {total_ms:7.0f} ms  ({total_ms / 1000:.2f}s)")
        print("=" * 35)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, num_output_frames=21):
        """Initialize the KV cache.

        Args:
            num_output_frames: Total output frames (used to size the global cache when
                local_attn_size == -1). With height-concat conditioning frame_seq_length is
                already doubled, so kv_cache_size = num_output_frames * frame_seq_length
                gives the correct doubled capacity automatically.
        """
        if self.memrope_cfg is not None:
            self.kv_cache1 = self._make_memrope_kv_cache(batch_size, dtype, device, self.memrope_cfg)
            return
        kv_cache_size = self._compute_kv_cache_size(self.local_attn_size, num_output_frames)
        if self.is_mmdit:
            self.kv_cache1 = self._make_dual_kv_cache(
                batch_size,
                kv_cache_size,
                dtype,
                device,
                self.generator.model.double_stream_layers,
            )
        else:
            self.kv_cache1 = self._make_kv_cache(batch_size, kv_cache_size, dtype, device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize the cross-attention cache."""
        self.crossattn_cache = self._make_crossattn_cache(batch_size, dtype, device)

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def _get_kv_cache_device(self) -> torch.device:
        """Extract device from the KV cache (handles standard, dual, and memrope layouts)."""
        entry = self.kv_cache1[0]
        if "k" in entry:
            return entry["k"].device
        elif "x_k" in entry:
            return entry["x_k"].device
        elif "gen" in entry:  # memrope split
            return entry["gen"]["sink_k"].device
        elif "sink_k" in entry:  # memrope fused
            return entry["sink_k"].device
        return torch.device("cuda")

    def _reset_all_kv_caches(self, device: torch.device) -> None:
        """Reset KV + cross-attn caches using the correct method for the current config."""
        if self.memrope_cfg is not None:
            self._reset_memrope_kv_cache(self.kv_cache1)
        elif self.is_mmdit:
            self._reset_dual_kv_cache(self.kv_cache1, device)
        else:
            self._reset_kv_cache(self.kv_cache1, device)
        self._reset_crossattn_cache(self.crossattn_cache)

    def initialize_streaming(
        self,
        batch_size: int = 1,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
        conditional_dict: dict | None = None,
        num_frames_for_cache: int = 21,
    ) -> None:
        """Initialize (or reset) a streaming generation session.

        Must be called before the first streaming_step(). Can be called again
        at any time to soft-reset the session (clear KV caches, restart from
        frame 0).

        Args:
            batch_size:           Batch size (usually 1).
            device:               Inference device; None = auto-detect from generator.
            dtype:                Tensor precision (default bfloat16).
            conditional_dict:     Pre-computed text conditioning dict
                                  {'prompt_embeds': Tensor [B, 512, 4096]}.
                                  Can be overridden per streaming_step().
            num_frames_for_cache: Maximum frames the KV cache can hold.
                                  For global attention (local_attn_size==-1) should
                                  match total generation length; for local attention
                                  equals the sliding window size.
        """
        if device is None:
            device = next(self.generator.parameters()).device

        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                num_output_frames=num_frames_for_cache,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )
        else:
            self._reset_all_kv_caches(device)

        self._stream_current_frame: int = 0
        self._stream_max_frames: int = num_frames_for_cache
        self._stream_conditional_dict: dict | None = conditional_dict

        # Clear VAE temporal decode cache to ensure each streaming session
        # starts from frame 0 (avoid stale feat_cache from previous sessions).
        if hasattr(self, "vae") and hasattr(self.vae, "model") and hasattr(self.vae.model, "clear_cache"):
            self.vae.model.clear_cache()

    def streaming_step(
        self,
        noise: torch.Tensor,  # [B, N, C, H, W]
        cond_latent: torch.Tensor,  # [B, N, C, H, W]
        conditional_dict: dict | None = None,
        _timing: dict | None = None,
    ) -> torch.Tensor:
        """Single streaming inference step: generate N frames without resetting KV cache.

        Each call generates N frames (N = noise.shape[1]). The KV cache persists
        between calls, enabling infinite-length causal autoregressive generation.

        Prerequisites:
            Must call initialize_streaming() first.

        Args:
            noise:            [B, N, C, H, W] initial noise for current N frames.
            cond_latent:      [B, N, C, H, W] rendered conditioning latent.
            conditional_dict: Text conditioning dict (None = reuse previous).
            _timing:          Optional empty dict; filled with per-stage timing (ms).

        Returns:
            pixel: [B, N, 3, H_px, W_px] decoded pixel frames in [0, 1].

        Raises:
            RuntimeError: If initialize_streaming() has not been called.
        """
        if not hasattr(self, "_stream_current_frame"):
            raise RuntimeError("Call initialize_streaming() before streaming_step().")

        if conditional_dict is not None:
            self._stream_conditional_dict = conditional_dict

        batch_size, num_frames = noise.shape[:2]
        current_start = self._stream_current_frame

        # ── KV cache overflow detection (global attention mode) ──────────
        if self.local_attn_size == -1 and current_start + num_frames > self._stream_max_frames:
            print(f"[streaming_step] KV cache full ({current_start}/{self._stream_max_frames} frames), auto-reset.")
            self._reset_all_kv_caches(noise.device)
            self._stream_current_frame = 0
            current_start = 0

        noisy_input = noise

        _do_timing = _timing is not None
        if _do_timing:
            torch.cuda.synchronize()
            _t_dit_start = time.perf_counter()

        # ── Multi-step denoising loop ────────────────────────────────────
        _num_steps = len(self.denoising_step_list)
        print(
            f"[streaming_step] frame={current_start}, {_num_steps} denoising steps, "
            f"skip_clean_rerun={self.skip_clean_rerun}"
        )
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = (
                torch.ones(
                    [batch_size, num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            is_last = index == len(self.denoising_step_list) - 1

            _cond = self._slice_cond(
                cond_latent if not is_last else cond_latent.clone(),
                0,
                num_frames,
            )

            _t0 = time.perf_counter()
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=self._stream_conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start * self.frame_seq_length,
                cond_latent=_cond,
            )
            torch.cuda.synchronize()
            print(
                f"[streaming_step]   denoise {index + 1}/{_num_steps} done "
                f"({(time.perf_counter() - _t0) * 1000:.0f} ms)"
            )

            if not is_last:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        # ── Clean rerun: update KV cache with denoised result ────────────
        if not self.skip_clean_rerun:
            _t0 = time.perf_counter()
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=self._stream_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start * self.frame_seq_length,
                cond_latent=self._slice_cond(cond_latent, 0, num_frames),
            )
            torch.cuda.synchronize()
            print(f"[streaming_step]   clean rerun done ({(time.perf_counter() - _t0) * 1000:.0f} ms)")

        # ── Update frame counter ─────────────────────────────────────────
        self._stream_current_frame += num_frames

        if _do_timing:
            torch.cuda.synchronize()
            _timing["dit_ms"] = (time.perf_counter() - _t_dit_start) * 1000.0
            _t_vae_start = time.perf_counter()

        # ── VAE decode → pixel frames ────────────────────────────────────
        pixel = self.vae.decode_to_pixel(denoised_pred, use_cache=True)
        pixel = (pixel * 0.5 + 0.5).clamp(0, 1)

        if _do_timing:
            torch.cuda.synchronize()
            _timing["vae_dec_ms"] = (time.perf_counter() - _t_vae_start) * 1000.0

        return pixel

    @property
    def streaming_frame_count(self) -> int:
        """Number of frames generated in the current streaming session."""
        return getattr(self, "_stream_current_frame", 0)

    def reset_streaming(self) -> None:
        """Manually reset streaming session (clear KV caches, zero frame counter)."""
        if self.kv_cache1 is not None:
            device = self._get_kv_cache_device()
            self._reset_all_kv_caches(device)
        self._stream_current_frame = 0
        # Clear VAE temporal decode cache as well
        if hasattr(self, "vae") and hasattr(self.vae, "model") and hasattr(self.vae.model, "clear_cache"):
            self.vae.model.clear_cache()
        print("[reset_streaming] KV caches reset, restarting from frame 0.")
