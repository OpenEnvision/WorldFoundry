import time
from tqdm import tqdm
from typing import List, Optional
import torch

from worldfoundry.base_models.diffusion_model.schedulers.flow_dpm import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler

from ..wan_utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        if need_vae:
            self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.prope_kv_cache_pos = None
        self.prope_kv_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Latency of producing the first chunk (set by inference()).
        self.last_chunk0_latency = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        return_video=True,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape

        # Start chunk0 latency timer
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        # Start chunk0 latency timer AFTER text encoder, BEFORE VAE decode
        # — matches HY15 latency definition (excludes both text encoder and decode).
        torch.cuda.synchronize()
        _chunk0_t0 = time.perf_counter()
        self.last_chunk0_latency = None

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            if viewmats is not None:
                self._initialize_prope_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            # reset prope kv cache
            if viewmats is not None and self.prope_kv_cache_pos is not None:
                for block_index in range(len(self.prope_kv_cache_pos)):
                    self.prope_kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                vm_slice = viewmats[:, current_start_frame:current_start_frame + 1] if viewmats is not None else None
                ks_slice = Ks[:, current_start_frame:current_start_frame + 1] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_slice,
                    Ks=ks_slice,
                    prope_kv_cache=self.prope_kv_cache_pos
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_slice,
                    Ks=ks_slice,
                    prope_kv_cache=self.prope_kv_cache_neg
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                vm_chunk = viewmats[:, current_start_frame:current_start_frame + self.num_frame_per_block] if viewmats is not None else None
                ks_chunk = Ks[:, current_start_frame:current_start_frame + self.num_frame_per_block] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_pos
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_neg
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Slice viewmats/Ks for current chunk
            vm_chunk = viewmats[:, current_start_frame:current_start_frame + current_num_frames] if viewmats is not None else None
            ks_chunk = Ks[:, current_start_frame:current_start_frame + current_num_frames] if Ks is not None else None

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_pos
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_neg
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Capture chunk0 latency: stop timer once the first chunk's denoised output is ready.
            if self.last_chunk0_latency is None:
                torch.cuda.synchronize()
                self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache_pos
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache_neg
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        if return_video:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            if return_latents:
                return video, output
            else:
                return video
        else:
            return output


    def inference_for_cd(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ):
        """
        Causal-forcing inference + record selected diffusion steps (per-chunk) for consistency distillation data.
        Record semantics: record xt BEFORE scheduler.step() at the specified progress_id (index in timesteps list).
        Also record the final latent of each chunk after the denoising loop.

        Returns:
            if return_video:
                (video, output_latents, cd_pack)
            else:
                (output_latents, cd_pack)

        cd_pack:
            {
            "record_step_indices": [...],
            "record_t_values": [t_i ...]  # same for all chunks
            "chunks": [
                {
                    "frame_start": int,
                    "frame_len": int,
                    "latents": Tensor [B, R, T, C, H, W]  (R = len(record_step_indices)+1, last one is final)
                }, ...
            ]
            }
        """
        self.sampling_steps = 48
        batch_size, num_frames, num_channels, height, width = noise.shape

        # ---- block counting (same logic as inference) ----
        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # ---- Step 1: init/reset caches (same as inference) ----
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # ---- validate record indices against scheduler length ----
        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        T = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if len(record_step_indices) == 0:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= T:
            raise ValueError(f"record_step_indices out of range: valid=[0,{T-1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        # ---- Step 2: cache context from initial_latent (same as inference) ----
        current_start_frame = start_frame_index
        cache_start_frame = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0

            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # ---- Step 3: causal-forcing denoising per chunk + record ----
        all_num_frames = [self.num_frame_per_block] * num_blocks

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            # noise slice for current window (same as inference)
            noisy_input = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # record list for this chunk
            chunk_records = []

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for progress_id, t in enumerate(tqdm(sample_scheduler.timesteps)):
                if progress_id in record_set:
                    print(f'{progress_id}: {t} saved')
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            # always append final latent of this chunk (like "-2")
            chunk_records.append(latents.detach().clone())
            chunk_records = torch.stack(chunk_records, dim=1)  # [B, R, T, C, H, W]

            full_chunk_record.append(chunk_records)
            # write output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # rerun at t=0 to update cache using clean context (same as inference)
            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )


            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames


        full_chunk_record = torch.cat(full_chunk_record, dim=2)
        # ---- Step 4: decode if needed ----

        return full_chunk_record


    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict = None,
        unconditional_dict = None,
        text_prompts = None,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize = 3
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            num_output_frames = num_frames + num_input_frames
        else:
            num_output_frames = num_frames

        if conditional_dict is None:
            assert text_prompts is not None
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_input.device,
            dtype=noisy_input.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)

        current_start_frame = 0
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noisy_input.device, dtype=torch.int64) * 0


        if initial_latent is not None:
            num_input_blocks = num_input_frames // chunksize
            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                output[:, cache_start_frame:cache_start_frame + chunksize] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize


        latents = noisy_input
        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        latent_model_input = latents
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_neg,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
            flow_pred_cond - flow_pred_uncond)

        latents = sample_scheduler.step(
            flow_pred,
            t,
            latents,
            return_dict=False)[0]

        return latents



    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Under SP, KV cache is stored in head-parallel domain (post all-to-all),
        so each rank only stores num_heads // sp_size heads.
        """
        num_heads = self._get_sp_num_heads(12)
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 31200

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    @staticmethod
    def _get_sp_num_heads(full_num_heads):
        """Return per-rank num_heads under SP (head-parallel domain)."""
        try:
            from worldfoundry.core.distributed.sequence_parallel.parallel_states import get_parallel_state
            ps = get_parallel_state()
            if ps.sp_enabled:
                return full_num_heads // ps.sp
        except (ImportError, AttributeError):
            pass
        return full_num_heads

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        num_heads = self._get_sp_num_heads(12)
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 31200
        entry = lambda: {
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        self.prope_kv_cache_pos = [entry() for _ in range(self.num_transformer_blocks)]
        self.prope_kv_cache_neg = [entry() for _ in range(self.num_transformer_blocks)]

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
