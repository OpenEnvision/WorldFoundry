# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1,
# inference/pipelines/causal_diffusion_inference.py.
"""Causal multi-step sampler: block-by-block UniPC denoising over the denoiser's KV caches."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import _invert_SE3
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler

from ...models.components import AstronexTextEncoder, AstronexVAE
from ...models.denoiser import AstronexDenoiser
from ...utils.cache_selection import frustum_overlap, select_history_frames
from .event_prompt import merge_event_into_captions


def anchor_viewmats_to_frame(viewmats: torch.Tensor, anchor_index: int) -> torch.Tensor:
    """Re-express camera<-world poses in the gauge of the camera at ``anchor_index``.

    A pure gauge change: relative geometry is preserved while transform magnitudes stay
    bounded around the anchor instead of growing with distance from the first frame.
    """
    if viewmats.shape[1] == 0:
        return viewmats
    return torch.matmul(viewmats, _invert_SE3(viewmats[:, anchor_index : anchor_index + 1]))


class CausalDiffusionInferencePipeline(torch.nn.Module):
    """Generates ``num_frame_per_block`` latent frames per block with classifier-free guidance."""

    _CACHES = (
        "kv_cache_pos",
        "kv_cache_neg",
        "crossattn_cache_pos",
        "crossattn_cache_neg",
        "prope_kv_cache_pos",
        "prope_kv_cache_neg",
    )

    def __init__(self, args):
        super().__init__()
        model_kwargs = dict(args.model_kwargs)
        self.generator = AstronexDenoiser(**model_kwargs, is_causal=True)
        self.text_encoder = AstronexTextEncoder(model_kwargs["model_root"])
        self.vae = AstronexVAE(model_kwargs["model_root"])
        self.vae_device = None

        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = int(args.get("sampling_steps", 50))
        self.shift = args.timestep_shift

        model = self.generator.model
        self.num_transformer_blocks = len(model.blocks)
        self.model_num_heads = model.num_heads
        self.model_head_dim = model.dim // model.num_heads
        self.text_len = model.text_len
        shape = args.image_or_video_shape
        self.frame_seq_length = (shape[3] // model.patch_size[1]) * (shape[4] // model.patch_size[2])
        self.local_attn_size = model.local_attn_size
        self.sink_size = model.sink_size
        self.prope_chunk_anchor = bool(model_kwargs.get("prope_chunk_anchor", False))

        for name in self._CACHES:
            setattr(self, name, None)
        self.args = args
        self.num_frame_per_block = int(args.get("num_frame_per_block", 1))
        self.independent_first_frame = bool(args.get("independent_first_frame", False))
        self.retrieval_window = bool(args.get("retrieval_window", False))
        self.retrieval_recent_frames = int(args.get("retrieval_recent_frames", 4))
        self.retrieval_min_overlap = float(args.get("retrieval_min_overlap", 0.3))
        self.retrieval_swap_per_block = int(args.get("retrieval_swap_per_block", 2))
        self.ref_sequence_conditioning = bool(args.get("ref_sequence_conditioning", False))
        self._prope_mirror = []
        self._prev_retrieved = []
        self._cache_keep = None

    def _release_denoiser_for_decode(self):
        """Free the KV caches and optionally host the denoiser while the VAE decodes.

        Returns the device to move the denoiser back to, or ``None`` when it stayed put.
        """
        released = False
        if bool(self.args.get("free_cache_before_decode", False)):
            for name in self._CACHES:
                setattr(self, name, None)
            released = True
        device = None
        if bool(self.args.get("offload_denoiser_for_decode", False)):
            device = next(self.generator.parameters()).device
            self.generator.to("cpu")
            released = True
        if released:
            torch.cuda.empty_cache()
        return device

    def _denoise(self, latents, conditional_dict, timestep, current_start_frame, cache, camera):
        """One denoiser call against the positive (``cache == "pos"``) or negative caches."""
        return self.generator(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=getattr(self, f"kv_cache_{cache}"),
            crossattn_cache=getattr(self, f"crossattn_cache_{cache}"),
            current_start=current_start_frame * self.frame_seq_length,
            prope_kv_cache=getattr(self, f"prope_kv_cache_{cache}"),
            **camera,
        )

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        initial_latent: torch.Tensor | None = None,
        return_latents: bool = False,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        event_prompts: str | None = None,
        event_start_frame: int = 0,
        actions: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ):
        """Sample ``noise`` (``[B, F, C, H, W]``) after an optional reference ``initial_latent``.

        Returns the decoded video in ``[0, 1]`` (and the latents when ``return_latents``).
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        self._prope_mirror = []
        self._prev_retrieved = []
        self._cache_keep = None
        sequence_ref = self.ref_sequence_conditioning and initial_latent is not None
        if sequence_ref:
            if self.independent_first_frame:
                raise ValueError("ref_sequence_conditioning requires independent_first_frame: false")
            if num_input_frames != 1:
                raise ValueError("ref_sequence_conditioning expects exactly one reference latent")
            if (num_frames + 1) % self.num_frame_per_block != 0:
                raise ValueError(
                    "reference + generated latent frames must form complete blocks: "
                    f"got 1 + {num_frames} with block size {self.num_frame_per_block}"
                )
            num_blocks = (num_frames + 1) // self.num_frame_per_block
        elif not self.independent_first_frame or initial_latent is not None:
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_output_frames = num_frames + num_input_frames

        # The event prompt reaches the model through the caption. With a start frame
        # the caption switches once the rollout reaches it, so earlier blocks match an
        # event-free run; both captions are encoded up front.
        event_in_caption = bool(self.args.get("event_in_caption", False))
        if event_prompts and not event_in_caption:
            raise ValueError("an event prompt requires event_in_caption: true")
        event_start_frame = int(event_start_frame or 0)
        conditioned_prompts = (
            merge_event_into_captions(text_prompts, event_prompts) if event_in_caption else text_prompts
        )
        scheduled_caption = bool(event_in_caption and event_prompts and event_start_frame)
        conditional_dict = self.text_encoder(text_prompts=(text_prompts if scheduled_caption else conditioned_prompts))
        event_caption_dict = self.text_encoder(text_prompts=conditioned_prompts) if scheduled_caption else None
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))
        if actions is not None:
            conditional_dict["actions"] = actions
            unconditional_dict["actions"] = actions
            if embodiment_id is not None:
                conditional_dict["embodiment_id"] = embodiment_id
                unconditional_dict["embodiment_id"] = embodiment_id
        event_caption_applied = False

        def apply_event(start_frame):
            """Switch to the event caption for the block beginning at ``start_frame``."""
            nonlocal event_caption_applied
            if event_caption_dict is None or event_caption_applied or start_frame < event_start_frame:
                return
            # Cross-attention K/V are cached from the first forward, so the positive cache restarts.
            conditional_dict.update(event_caption_dict)
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
            event_caption_applied = True
            print(f"[event] caption switched at latent frame {start_frame}", flush=True)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width], device=noise.device, dtype=noise.dtype
        )

        # Step 1: initialise or reset the caches.
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            if viewmats is not None:
                self._initialize_prope_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            caches = [*self.kv_cache_pos, *self.kv_cache_neg]
            if viewmats is not None and self.prope_kv_cache_pos is not None:
                caches += [*self.prope_kv_cache_pos, *self.prope_kv_cache_neg]
            for cache in caches:
                cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # Step 2: prefill the caches with the clean conditioning frames.
        current_start_frame = 0
        cache_start_frame = 0
        if initial_latent is not None and not sequence_ref:
            timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                camera = {
                    "viewmats": viewmats[:, :1] if viewmats is not None else None,
                    "Ks": Ks[:, :1] if Ks is not None else None,
                }
                for cache, condition in (("pos", conditional_dict), ("neg", unconditional_dict)):
                    self._denoise(initial_latent[:, :1], condition, timestep, current_start_frame, cache, camera)
                current_start_frame += 1
                cache_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                block = slice(cache_start_frame, cache_start_frame + self.num_frame_per_block)
                window = slice(current_start_frame, current_start_frame + self.num_frame_per_block)
                current_ref_latents = initial_latent[:, block]
                output[:, block] = current_ref_latents
                camera = {
                    "viewmats": viewmats[:, window] if viewmats is not None else None,
                    "Ks": Ks[:, window] if Ks is not None else None,
                }
                for cache, condition in (("pos", conditional_dict), ("neg", unconditional_dict)):
                    self._denoise(current_ref_latents, condition, timestep, current_start_frame, cache, camera)
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: temporal denoising loop.
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1, *all_num_frames]
        for current_num_frames in all_num_frames:
            apply_event(current_start_frame)
            is_reference_block = sequence_ref and cache_start_frame == 0
            if is_reference_block:
                # The external reference occupies output slot zero; the block's other
                # frames are the first block-1 generated noise frames.
                reference_latent = initial_latent[:, :1].to(noise.dtype)  # ty: ignore[not-subscriptable]
                latents = torch.cat([reference_latent, noise[:, : current_num_frames - 1]], dim=1)
            else:
                latents = noise[
                    :, cache_start_frame - num_input_frames : cache_start_frame + current_num_frames - num_input_frames
                ]
            camera = self._block_camera(viewmats, Ks, current_start_frame, current_num_frames)

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for t in tqdm(sample_scheduler.timesteps):
                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
                if is_reference_block:
                    timestep[:, :1] = 0
                flow_pred_cond = self._denoise(latents, conditional_dict, timestep, current_start_frame, "pos", camera)
                flow_pred_uncond = self._denoise(
                    latents, unconditional_dict, timestep, current_start_frame, "neg", camera
                )
                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]
                if is_reference_block:
                    latents[:, :1] = reference_latent

            output[:, cache_start_frame : cache_start_frame + current_num_frames] = latents

            # Rerun at timestep zero so the caches hold the clean block.
            for cache, condition in (("pos", conditional_dict), ("neg", unconditional_dict)):
                self._denoise(latents, condition, timestep * 0, current_start_frame, cache, camera)

            if is_reference_block:
                ref_scale = float(self.args.get("ref_cond_scale", 1.0))
                if ref_scale != 1.0:
                    # Scale the clean reference values in both the content and PRoPE caches so
                    # the two branches agree on the reference's weight.
                    span = self.frame_seq_length
                    caches = [*self.kv_cache_pos, *self.kv_cache_neg]
                    if self.prope_kv_cache_pos is not None:
                        caches += [*self.prope_kv_cache_pos, *self.prope_kv_cache_neg]
                    for layer_cache in caches:
                        layer_cache["v"][:, :span] *= ref_scale
                    print(f"[cond] reference values scaled by {ref_scale}", flush=True)

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: decode.
        output = self._stitch_blocks(output)
        offloaded = self._release_denoiser_for_decode()
        video = self.vae.decode_to_pixel(output.to(self.vae_device) if self.vae_device is not None else output)
        if offloaded is not None:
            self.generator.to(offloaded)
        video = self._stabilize_chroma((video * 0.5 + 0.5).clamp(0, 1))
        if return_latents:
            return video, output
        return video

    def _block_camera(self, viewmats, Ks, current_start_frame, current_num_frames):
        """Camera inputs for one block.

        With ``prope_chunk_anchor`` the PRoPE cache mirrors the content cache's resident
        frames (sink, kept history, this block) and the whole window is expressed in the
        pose of the frame before this block. With ``retrieval_window`` the kept history
        is chosen by frustum overlap with the block's last frame instead of by age.
        """
        if viewmats is None or self.prope_kv_cache_pos is None or not self.prope_chunk_anchor:
            window = slice(current_start_frame, current_start_frame + current_num_frames)
            return {
                "viewmats": viewmats[:, window] if viewmats is not None else None,
                "Ks": Ks[:, window] if Ks is not None else None,
            }
        cache_cap = max(1, int(self.local_attn_size))
        new_frames = list(range(current_start_frame, current_start_frame + current_num_frames))
        combined = list(self._prope_mirror) + new_frames
        self._cache_keep = None
        if len(combined) > cache_cap:
            evict = len(combined) - cache_cap
            sink = max(0, min(self.sink_size, len(combined)))
            if self.retrieval_window:
                body = combined[sink : len(combined) - len(new_frames)]
                keep_n = cache_cap - sink - len(new_frames)
                recent = min(keep_n, max(0, self.retrieval_recent_frames))
                candidates = body[:-recent] if recent else body
                room = keep_n - recent
                scores = frustum_overlap(viewmats, Ks, new_frames[-1], candidates) if room and candidates else {}
                body, self._prev_retrieved = select_history_frames(
                    body,
                    keep_n,
                    recent,
                    scores,
                    self.retrieval_min_overlap,
                    self._prev_retrieved,
                    self.retrieval_swap_per_block,
                )
                combined = combined[:sink] + body + new_frames
                self._cache_keep = list(body)
            else:
                combined = combined[:sink] + combined[sink + evict :]
        self._prope_mirror = list(combined)
        anchor_frame = max(0, current_start_frame - 1)
        if anchor_frame not in self._prope_mirror:
            anchor_frame = self._prope_mirror[0]
        vm_window = anchor_viewmats_to_frame(viewmats[:, self._prope_mirror], self._prope_mirror.index(anchor_frame))
        ks_window = Ks[:, self._prope_mirror] if Ks is not None else None
        return {
            "viewmats": vm_window[:, -current_num_frames:],
            "Ks": ks_window[:, -current_num_frames:] if ks_window is not None else None,
            "prope_window_viewmats": vm_window,
            "prope_window_Ks": ks_window,
            "cache_keep_frames": self._cache_keep,
        }

    def _stitch_blocks(self, output):
        """Blend each block's first latent toward the previous block's motion trend (``latent_stitch_strength``)."""
        stitch = float(self.args.get("latent_stitch_strength", 0.0))
        if stitch <= 0 or self.num_frame_per_block <= 1:
            return output
        output = output.clone()
        block = self.num_frame_per_block
        for start in range(block, output.shape[1], block):
            if start < 2:
                continue
            velocity = output[:, start - 1] - output[:, start - 2]
            expected = output[:, start - 1] + 0.5 * velocity
            offset = expected - output[:, start]
            # Clamp relative to the latent scale so a bad block cannot drag the next one.
            limit = 0.35 * output[:, start].float().std(dim=(1, 2, 3), keepdim=True).to(output.dtype)
            offset = offset.clamp(min=-limit, max=limit)
            if bool(self.args.get("latent_stitch_preserve_mean", False)):
                offset = offset - offset.mean(dim=(-2, -1), keepdim=True)
            end = min(start + block, output.shape[1])
            count = end - start
            weights = torch.linspace(1.0, 0.0, count + 1, device=output.device, dtype=output.dtype)[:-1]
            output[:, start:end] += stitch * weights.view(1, count, 1, 1, 1) * offset[:, None]
        return output

    def _stabilize_chroma(self, video):
        """Smooth the temporal trajectory of per-frame RGB means (``rgb_chroma_stabilize_strength``).

        Every frame receives one constant per-channel offset; no pixels are mixed.
        """
        strength = float(self.args.get("rgb_chroma_stabilize_strength", 0.0))
        if strength <= 0 or video.shape[1] <= 8:
            return video
        source = video.clone()
        means = source.float().mean(dim=(-2, -1))
        radius = min(max(1, int(self.args.get("rgb_chroma_stabilize_fade", 4))), video.shape[1] - 1)
        rising = torch.arange(1, radius + 2, device=video.device, dtype=torch.float32)
        kernel = torch.cat([rising, rising[:-1].flip(0)])
        kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(3, 1, 1)
        padded = F.pad(means.transpose(1, 2), (radius, radius), mode="reflect")
        smooth = F.conv1d(padded, kernel, groups=3).transpose(1, 2)
        correction = (smooth - means).clamp(-0.06, 0.06)
        print(f"[rgb-chroma] smoothed RGB means radius={radius} max_correction={correction.abs().max().item():.6f}")
        return (source + strength * correction.to(video.dtype)[..., None, None]).clamp(0, 1)

    def _kv_cache(self, batch_size, dtype, device):
        """Zeroed per-block sink + sliding-window cache (``local_attn_size`` frames, else 20)."""
        frames = self.local_attn_size if self.local_attn_size != -1 else 20
        size = frames * self.frame_seq_length
        return [
            {
                "k": torch.zeros(
                    [batch_size, size, self.model_num_heads, self.model_head_dim], dtype=dtype, device=device
                ),
                "v": torch.zeros(
                    [batch_size, size, self.model_num_heads, self.model_head_dim], dtype=dtype, device=device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache_pos = self._kv_cache(batch_size, dtype, device)
        self.kv_cache_neg = self._kv_cache(batch_size, dtype, device)

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        self.prope_kv_cache_pos = self._kv_cache(batch_size, dtype, device)
        self.prope_kv_cache_neg = self._kv_cache(batch_size, dtype, device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        shape = [batch_size, self.text_len, self.model_num_heads, self.model_head_dim]

        def entries():
            return [
                {
                    "k": torch.zeros(shape, dtype=dtype, device=device),
                    "v": torch.zeros(shape, dtype=dtype, device=device),
                    "is_init": False,
                }
                for _ in range(self.num_transformer_blocks)
            ]

        self.crossattn_cache_pos = entries()
        self.crossattn_cache_neg = entries()

    def _initialize_sample_scheduler(self, noise):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(self.sampling_steps, device=noise.device, shift=self.shift)
        return scheduler
