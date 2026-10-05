# Ported from SandAI MAGI-2-preview (Apache-2.0): inference/pipeline/sampler.py
# (Magi2PreviewSampler CFG driver). Distributed coupling (cp/ep/dp, psm,
# broadcast) stripped; the model is an injected single-GPU callable. All CFG /
# flow-matching math is preserved verbatim.

"""Single-GPU CFG sampling driver for MAGI-2-preview audio-video generation.

The upstream ``Magi2PreviewSampler`` wraps a distributed model + data-proxy and
loops over the flow-matching timesteps, at each step running a batch-of-2
cond/uncond forward, combining the two halves with classifier-free guidance
(separate video / audio guidance scales, an optional per-frame "cfg trick",
dynamic-CFG clamp, skimmed-CFG-linear, and CFG-rescale), then advancing the
video and audio latents with their own :class:`FlowUniPCMultistepScheduler`.

This port keeps that exact math but takes the model as an injected callable so
it runs single-GPU with no context/expert/data parallelism. The caller supplies
a ``model_forward`` with the contract documented on :class:`Magi2PreviewSampler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

# The model callable receives the batched (cond, uncond) inputs and the current
# (already /1000-normalized) timestep, and returns a (video, audio) velocity
# tuple, each a batch-of-2 tensor ordered [cond, uncond].
ModelForward = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


DEFAULT_NEGATIVE_PROMPT = (
    'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, '
    'static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra '
    'fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused '
    'fingers, still picture, messy background, three legs, many people in the background, walking '
    'backwards, low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, '
    'crackle, static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy,'
    ' unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, '
    'shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt '
    'cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, '
    'crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound, emotionless, flat '
    'delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, '
    'boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, '
    'exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak '
    'voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, '
    'vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible '
    'inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, speaking '
    'too slow, dragged out, unnatural pauses, awkward silence, choppy, disconnected, multiple speakers, '
    'two voices, background talking, out of tune, off-key, autotune artifacts'
)


@dataclass
class CFGConfig:
    """Classifier-free-guidance configuration (ported from preview_data_proxy)."""

    use_cfg_trick: bool = False
    cfg_trick_start_frame: int = 0
    cfg_trick_value: float = 0.0
    use_dynamic_cfg: bool = False
    dynamic_cfg_start_t: int = 0
    dynamic_cfg_cutoff_value: float = 0.0
    video_txt_guidance_scale: float = 0.0
    audio_txt_guidance_scale: float = 0.0
    use_ref_for_uncond: bool = False
    use_skimmed_cfg_linear: bool = False
    skimmed_cfg_scale: float = 5.0
    cfg_rescale: float = 0.0


@dataclass
class Magi2SamplingConfig:
    """MAGI-2-preview sampling defaults (from upstream ``EvaluationConfig``).

    Bakes the released magi2_preview.json defaults for the preview stage and the refiner
    renoise. Build a :class:`CFGConfig` from this via :meth:`to_cfg_config`.
    """

    num_inference_steps: int = 100
    shift: float = 7.0
    z_dim: int = 48
    vae_stride: Tuple[int, int, int] = (8, 16, 16)
    video_txt_guidance_scale: float = 5.0
    audio_txt_guidance_scale: float = 7.0
    use_cfg_trick: bool = False
    cfg_trick_start_frame: int = 13
    cfg_trick_value: float = 2.0
    use_dynamic_cfg: bool = False
    dynamic_cfg_start_t: int = 500
    dynamic_cfg_cutoff_value: float = 2.0
    use_ref_for_uncond: bool = True
    use_skimmed_cfg_linear: bool = False
    skimmed_cfg_scale: float = 3.0
    cfg_rescale: float = 0.0
    # Refiner stage.
    magi2_refiner_num_inference_steps: int = 5
    magi2_refiner_vae_stride: Tuple[int, int, int] = (4, 16, 16)
    magi2_refiner_noise_value: int = 220
    magi2_refiner_video_txt_guidance_scale: float = 2.0

    def to_cfg_config(self) -> CFGConfig:
        """Project the preview-stage CFG knobs into a :class:`CFGConfig`."""

        return CFGConfig(
            use_cfg_trick=self.use_cfg_trick,
            cfg_trick_start_frame=self.cfg_trick_start_frame,
            cfg_trick_value=self.cfg_trick_value,
            use_dynamic_cfg=self.use_dynamic_cfg,
            dynamic_cfg_start_t=self.dynamic_cfg_start_t,
            dynamic_cfg_cutoff_value=self.dynamic_cfg_cutoff_value,
            video_txt_guidance_scale=self.video_txt_guidance_scale,
            audio_txt_guidance_scale=self.audio_txt_guidance_scale,
            use_ref_for_uncond=self.use_ref_for_uncond,
            use_skimmed_cfg_linear=self.use_skimmed_cfg_linear,
            skimmed_cfg_scale=self.skimmed_cfg_scale,
            cfg_rescale=self.cfg_rescale,
        )


def _pad_or_trim(tensor, target_size: int, dim: int, pad_value: float = 0.0):
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding_tuple), 'constant', pad_value), current_size
    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], current_size


class Magi2PreviewSampler:
    """CFG sampling driver for the MAGI-2-preview audio-video model.

    Injected-model-callable contract
    --------------------------------
    ``model_forward`` is called once per denoising step as::

        video_out, audio_out = model_forward(
            x_t=<video latent, cat([cond, uncond], dim=0)>,      # (2, C, T, H, W)
            audio_x_t=<audio latent, cat([cond, uncond], dim=0)>,# (2, L, C)
            t=<timestep tensor / 1000.0>,                        # (2,)
            per_token_video_t=<(2, 1, T, H, W)>,
            per_token_audio_t=<(2, L, 1)>,
            context=<dict of the remaining conditioning tensors>,
        )

    and must return a ``(video_velocity, audio_velocity)`` tuple whose two
    tensors are each batched ``[cond, uncond]`` along dim 0 (matching the input
    ordering). Either element may be an empty tensor (``numel()==0``) to skip
    that modality. No context/expert/data-parallel ops are performed here.
    """

    def __init__(
        self,
        model_forward: ModelForward,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model_forward = model_forward
        self.device = device
        self.dtype = dtype

    def forward(self, model_input: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """One injected-model call; returns batched ``(video, audio)`` velocities."""

        return self.model_forward(**model_input)

    def sample(
        self,
        *,
        video_t_list: list[torch.Tensor],
        latent: torch.Tensor,          # B, C, T, H, W
        audio_latent: torch.Tensor,    # B, L, C
        txt_feat: torch.Tensor,        # B, L, D
        null_txt_feat: torch.Tensor,   # B, L, D
        video_scheduler,
        audio_scheduler,
        cfg_config: CFGConfig,
        ref_audio_feat: Optional[torch.Tensor] = None,
        ref_video_feat: Optional[torch.Tensor] = None,
        ref_image_feat: Optional[torch.Tensor] = None,
        ref_image_feat_len: Optional[torch.Tensor] = None,
        ref_image_special_token_embedding: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the preview flow-matching loop with per-step CFG.

        ``video_t_list`` is the scheduler's discrete timesteps.  Each step
        packs cond/uncond latents, calls ``model_forward``, mixes the two
        halves with :meth:`cfg_velocity`, then advances the video and audio
        UniPC schedulers independently.
        """

        video_cfgs, audio_cfgs = self.precalculate_cfg(
            video_t_list, latent.shape[2], cfg_config
        )

        latent = latent.clone()
        audio_latent = audio_latent.clone()

        iterator = zip(video_t_list, video_cfgs, audio_cfgs)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(video_t_list))
            except Exception:
                pass

        for t, video_cfg, audio_cfg in iterator:
            model_input = self.prepare_model_input(
                latent=latent,
                audio_latent=audio_latent,
                txt_feat=txt_feat,
                null_txt_feat=null_txt_feat,
                ref_audio_feat=ref_audio_feat,
                ref_video_feat=ref_video_feat,
                ref_image_feat=ref_image_feat,
                ref_image_feat_len=ref_image_feat_len,
                ref_image_special_token_embedding=ref_image_special_token_embedding,
                t=t,
                cfg_config=cfg_config,
            )
            model_pred = self.forward(model_input)

            latent, audio_latent, _, _ = self.step(
                model_pred,
                latent,
                audio_latent,
                video_cfg,
                audio_cfg,
                video_scheduler,
                audio_scheduler,
                t,
                cfg_config=cfg_config,
            )

        return latent, audio_latent

    def prepare_model_input(
        self,
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        txt_feat: torch.Tensor,
        null_txt_feat: torch.Tensor,
        ref_audio_feat: Optional[torch.Tensor] = None,
        ref_video_feat: Optional[torch.Tensor] = None,
        ref_image_feat: Optional[torch.Tensor] = None,
        ref_image_feat_len: Optional[torch.Tensor] = None,
        ref_image_special_token_embedding: Optional[torch.Tensor] = None,
        t: torch.Tensor | None = None,
        cfg_config: Optional[CFGConfig] = None,
    ) -> dict:
        """Build the batch-of-2 (cond, uncond) kwargs for ``model_forward``.

        Text is padded to a shared length and concatenated as ``[cond, null]``.
        Reference image tokens are either zeroed on the uncond half (default)
        or kept when ``use_ref_for_uncond`` is set.  Timesteps are divided by
        1000 to match the DiT's Fourier time scale.
        """

        audio_latent_len = audio_latent.shape[1]
        txt_feat_len = txt_feat.shape[1]
        null_txt_feat_len = null_txt_feat.shape[1]
        target_length = max(txt_feat_len, null_txt_feat_len)

        txt_feat, _ = _pad_or_trim(txt_feat, target_length, 1)
        null_txt_feat, _ = _pad_or_trim(null_txt_feat, target_length, 1)

        if ref_audio_feat is None:
            ref_audio_feat = torch.zeros(
                latent.shape[0], 0, audio_latent.shape[2],
                device=latent.device, dtype=audio_latent.dtype,
            )
        ref_audio_feat_len = ref_audio_feat.shape[1]

        if ref_video_feat is None:
            ref_video_feat = torch.empty_like(latent)
            ref_video_feat_len = 0
        else:
            ref_video_feat_len = (
                ref_video_feat.shape[3] * ref_video_feat.shape[4]
            ) // 4
        ref_video_feat = torch.cat(
            [ref_video_feat, torch.zeros_like(ref_video_feat)], dim=0
        )
        ref_video_feat_len = torch.tensor(
            [ref_video_feat_len, ref_video_feat_len], device=latent.device
        )

        (
            ref_image_feat_cfg,
            ref_image_feat_len_cfg,
            ref_image_special_tokens_cfg,
        ) = self._prepare_ref_image_cfg(
            ref_image_feat,
            ref_image_feat_len,
            ref_image_special_token_embedding,
            cfg_config,
        )

        if t is None:
            t_value = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)
        elif isinstance(t, torch.Tensor):
            t_value = t.reshape(-1)[0].to(device=latent.device, dtype=latent.dtype)
        else:
            t_value = torch.tensor(float(t), device=latent.device, dtype=latent.dtype)
        t_batch = torch.stack([t_value, t_value])
        t_normalized = t_batch / 1000.0

        batch_cfg = 2
        _, _, video_t, video_h, video_w = latent.shape
        audio_t = audio_latent.shape[1]
        per_token_video_t = (
            t_normalized.view(-1, 1, 1, 1, 1)
            .expand(batch_cfg, 1, video_t, video_h, video_w)
            .clone()
        )
        per_token_audio_t = (
            t_normalized.view(-1, 1, 1).expand(batch_cfg, audio_t, 1).clone()
        )

        context = {
            "audio_feat_len": torch.tensor(
                [audio_latent_len, audio_latent_len], device=latent.device
            ),
            "txt_feat": torch.cat([txt_feat, null_txt_feat], dim=0),
            "txt_feat_len": torch.tensor(
                [txt_feat_len, null_txt_feat_len], device=latent.device
            ),
            "ref_audio_feat": torch.cat(
                [ref_audio_feat, torch.zeros_like(ref_audio_feat)], dim=0
            ),
            "ref_audio_feat_len": torch.tensor(
                [ref_audio_feat_len, ref_audio_feat_len], device=latent.device
            ),
            "ref_video_feat": ref_video_feat,
            "ref_video_feat_len": ref_video_feat_len,
            "ref_image_feat": ref_image_feat_cfg,
            "ref_image_feat_len": ref_image_feat_len_cfg,
            "ref_image_special_token_embedding": ref_image_special_tokens_cfg,
        }

        return {
            "x_t": torch.cat([latent, latent], dim=0),
            "audio_x_t": torch.cat([audio_latent, audio_latent], dim=0),
            "t": t_normalized,
            "per_token_video_t": per_token_video_t,
            "per_token_audio_t": per_token_audio_t,
            "context": context,
        }

    @staticmethod
    def _prepare_ref_image_cfg(
        ref_image_feat: Optional[torch.Tensor],
        ref_image_feat_len: Optional[torch.Tensor],
        ref_image_special_token_embedding: Optional[torch.Tensor],
        cfg_config: Optional[CFGConfig],
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Duplicate the image conditioning along the cond/uncond batch axis.

        By default the unconditional half sees a zeroed image so CFG steers both
        the text and the image. With ``use_ref_for_uncond`` the image is kept on
        both halves, which leaves CFG steering the text alone.
        """
        if ref_image_feat is None:
            return None, None, None

        keep_ref = bool(cfg_config is not None and cfg_config.use_ref_for_uncond)
        uncond_feat = ref_image_feat if keep_ref else torch.zeros_like(ref_image_feat)
        ref_image_feat_cfg = torch.cat([ref_image_feat, uncond_feat], dim=0)
        ref_image_feat_len_cfg = (
            torch.cat([ref_image_feat_len, ref_image_feat_len], dim=0)
            if ref_image_feat_len is not None
            else None
        )
        ref_image_special_tokens_cfg = (
            torch.cat(
                [ref_image_special_token_embedding, ref_image_special_token_embedding],
                dim=0,
            )
            if ref_image_special_token_embedding is not None
            else None
        )
        return ref_image_feat_cfg, ref_image_feat_len_cfg, ref_image_special_tokens_cfg

    def precalculate_cfg(
        self, t_list: list[torch.Tensor], latent_length: int, cfg_config: CFGConfig
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Precompute per-timestep video/audio CFG scales.

        ``use_cfg_trick`` lowers the scale on the first ``cfg_trick_start_frame``
        latent frames.  ``use_dynamic_cfg`` clamps the scale before
        ``dynamic_cfg_start_t``.  Audio uses a scalar scale for every step.
        """

        all_video_cfgs = []
        all_audio_cfgs = []

        if cfg_config.use_cfg_trick:
            video_cfg_base = (
                torch.tensor(cfg_config.video_txt_guidance_scale, device=self.device)
                .expand(1, 1, latent_length, 1, 1)
                .clone()
            )
            video_cfg_base[:, :, : cfg_config.cfg_trick_start_frame] = min(
                cfg_config.cfg_trick_value, cfg_config.video_txt_guidance_scale
            )
        else:
            video_cfg_base = torch.tensor(
                [cfg_config.video_txt_guidance_scale], device=self.device
            )

        for t in t_list:
            video_cfg = video_cfg_base.clone()
            if cfg_config.use_dynamic_cfg and t < cfg_config.dynamic_cfg_start_t:
                video_cfg[video_cfg > cfg_config.dynamic_cfg_cutoff_value] = (
                    cfg_config.dynamic_cfg_cutoff_value
                )
            all_video_cfgs.append(video_cfg)
            all_audio_cfgs.append(cfg_config.audio_txt_guidance_scale)

        return all_video_cfgs, all_audio_cfgs

    @staticmethod
    def _cfg_scale_tensor(
        guidance_scale: torch.Tensor | float, like: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(guidance_scale, torch.Tensor):
            return guidance_scale.to(device=like.device, dtype=like.dtype)
        return torch.tensor(float(guidance_scale), device=like.device, dtype=like.dtype)

    @classmethod
    def _get_skimming_mask(
        cls,
        x_orig: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        x_orig = x_orig.to(device=cond.device, dtype=cond.dtype)
        denoised = x_orig - (
            (x_orig - uncond) + guidance_scale * ((x_orig - cond) - (x_orig - uncond))
        )
        matching_pred_signs = (cond - uncond).sign() == cond.sign()
        matching_diff_after = (
            cond.sign()
            == (cond * guidance_scale - uncond * (guidance_scale - 1)).sign()
        )
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        return matching_pred_signs & matching_diff_after & deviation_influence

    @classmethod
    def _apply_skimmed_cfg_linear(
        cls,
        x_orig: Optional[torch.Tensor],
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
        cfg_config: Optional[CFGConfig],
    ) -> torch.Tensor:
        if (
            cfg_config is None
            or not cfg_config.use_skimmed_cfg_linear
            or x_orig is None
            or not torch.any(uncond).item()
        ):
            return uncond

        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        valid_scale = guidance_scale > 1
        if not torch.any(valid_scale).item():
            return uncond

        scale_delta = torch.where(
            valid_scale, guidance_scale - 1, torch.ones_like(guidance_scale)
        )
        fallback_weight = (cfg_config.skimmed_cfg_scale - 1) / scale_delta

        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, cond, uncond, guidance_scale)
        uncond = torch.where(skim_mask & valid_scale, target_uncond, uncond)

        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, uncond, cond, guidance_scale)
        return torch.where(skim_mask & valid_scale, target_uncond, uncond)

    @staticmethod
    def _cfg_rescale_dims(tensor: torch.Tensor) -> tuple[int, ...]:
        if tensor.ndim <= 1:
            return ()
        if tensor.ndim == 2:
            return (1,)
        if tensor.ndim == 3:
            return (1,)
        return tuple(range(2, tensor.ndim))

    @classmethod
    def _apply_cfg_rescale(
        cls, pos: torch.Tensor, cfg: torch.Tensor, cfg_config: Optional[CFGConfig]
    ) -> torch.Tensor:
        if cfg_config is None or cfg_config.cfg_rescale <= 0 or cfg.numel() == 0:
            return cfg

        spatial_dims = cls._cfg_rescale_dims(cfg)
        if not spatial_dims:
            return cfg

        pos_std = pos.float().std(dim=spatial_dims, keepdim=True, unbiased=False)
        cfg_std = (
            cfg.float()
            .std(dim=spatial_dims, keepdim=True, unbiased=False)
            .clamp_min(1e-6)
        )
        factor = cfg_config.cfg_rescale * (pos_std / cfg_std) + (
            1 - cfg_config.cfg_rescale
        )
        return cfg * factor.to(device=cfg.device, dtype=cfg.dtype)

    def cfg_velocity(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        video_txt_guidance_scale: torch.Tensor,
        audio_txt_guidance_scale: torch.Tensor,
        cfg_config: Optional[CFGConfig] = None,
        latent: Optional[torch.Tensor] = None,
        audio_latent: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Combine cond/uncond velocities with optional skim + CFG-rescale."""

        cond_uncond_video = model_output[0]
        cond_uncond_audio = model_output[1]

        if cond_uncond_video.numel() > 0:
            cond_video, uncond_video = cond_uncond_video[0:1], cond_uncond_video[1:2]
            uncond_video = self._apply_skimmed_cfg_linear(
                latent, cond_video, uncond_video, video_txt_guidance_scale, cfg_config
            )
            cfg_video = uncond_video + video_txt_guidance_scale * (
                cond_video - uncond_video
            )
            cfg_video = self._apply_cfg_rescale(cond_video, cfg_video, cfg_config)
        else:
            cfg_video = None

        if cond_uncond_audio.numel() > 0:
            cond_audio, uncond_audio = cond_uncond_audio[0:1], cond_uncond_audio[1:2]
            uncond_audio = self._apply_skimmed_cfg_linear(
                audio_latent,
                cond_audio,
                uncond_audio,
                audio_txt_guidance_scale,
                cfg_config,
            )
            cfg_audio = uncond_audio + audio_txt_guidance_scale * (
                cond_audio - uncond_audio
            )
            cfg_audio = self._apply_cfg_rescale(cond_audio, cfg_audio, cfg_config)
        else:
            cfg_audio = None

        return cfg_video, cfg_audio

    def step(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        video_txt_guidance_scale: torch.Tensor,
        audio_txt_guidance_scale: torch.Tensor,
        video_scheduler,
        audio_scheduler,
        t: torch.Tensor,
        cfg_config: Optional[CFGConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply CFG and take one UniPC step on video and audio latents."""

        cfg_video, cfg_audio = self.cfg_velocity(
            model_output,
            video_txt_guidance_scale,
            audio_txt_guidance_scale,
            cfg_config,
            latent,
            audio_latent,
        )
        if cfg_video is not None:
            latent = video_scheduler.step(cfg_video, t, latent, return_dict=False)[0]
        if cfg_audio is not None:
            audio_latent = audio_scheduler.step(
                cfg_audio, t, audio_latent, return_dict=False
            )[0]
        return latent, audio_latent, cfg_video, cfg_audio


__all__ = [
    "CFGConfig",
    "Magi2PreviewSampler",
    "Magi2SamplingConfig",
    "ModelForward",
]
