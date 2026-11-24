"""Autoregressive text/image-to-video rollout for Echo-WM Flash."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import (
    decode_audio as vae_decode_audio,
)

from ..ltx_causal import (
    DEFAULT_CAUSAL_TIMESTEPS,
    CausalCacheConfig,
    CausalModelWrapper,
    causal_audio_frames,
    causal_rollout,
    causal_video_blocks,
)
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.model.video_vae.tiling import TilingConfig
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_core.tools import AudioLatentTools
from ..ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from .utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    encode_prompts,
    get_device,
)
from .utils.args import ImageConditioningInput
from .utils.helpers import create_noised_state, noise_video_state
from .utils.types import PipelineComponents

device = get_device()


DEFAULT_CACHE_CONFIG = CausalCacheConfig()


class CausalTI2VidPipeline:
    """Inference-only 4-step autoregressive I2V pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
        action_config=None,
        cache_config: CausalCacheConfig = DEFAULT_CACHE_CONFIG,
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device
        self.action_config = action_config
        cache_config.validate()
        self.cache_config = cache_config
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

    @torch.inference_mode()
    def __call__(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        action_cond: dict[str, torch.Tensor],
        timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
        video_tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)
        latent_frames = (num_frames - 1) // 8 + 1
        if num_frames != (latent_frames - 1) * 8 + 1:
            raise ValueError("causal --num-frames must be 1 + 8*n output frames")
        causal_video_blocks(latent_frames, self.cache_config.video_chunk_size)
        # The causal student is trained on one positive conditioning branch.
        (encoded_prompt,) = encode_prompts([prompt], self.model_ledger)
        if encoded_prompt.audio_encoding is None:
            raise ValueError("the causal AV checkpoint must provide audio text embeddings")

        output_shape = VideoPixelShape(1, num_frames, height, width, frame_rate)
        video_encoder = self.model_ledger.video_encoder()
        conditionings = combined_image_conditionings(images, height, width, video_encoder, self.dtype, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        del video_encoder
        cleanup_memory()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator)
        video_state, video_tools = noise_video_state(
            output_shape,
            noiser,
            conditionings,
            self.pipeline_components,
            self.dtype,
            self.device,
        )
        audio_frames = causal_audio_frames(latent_frames, self.cache_config.video_chunk_size)
        audio_shape = AudioLatentShape(batch=1, channels=8, frames=audio_frames, mel_bins=16)
        audio_tools = AudioLatentTools(self.pipeline_components.audio_patchifier, audio_shape)
        audio_state = create_noised_state(audio_tools, [], noiser, self.dtype, self.device)

        x0_model = self.model_ledger.transformer(action_config=self.action_config)
        wrapper = CausalModelWrapper(
            x0_model.velocity_model,
            patches_per_frame=(height // 32) * (width // 32),
            cache=self.cache_config,
        )
        generated_video, generated_audio = causal_rollout(
            wrapper=wrapper,
            clean_video=video_state.clean_latent,
            clean_audio=audio_state.clean_latent,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
            video_context=encoded_prompt.video_encoding,
            audio_context=encoded_prompt.audio_encoding,
            context_mask=encoded_prompt.attention_mask,
            action_cond=action_cond,
            seed=seed,
            timesteps=timesteps,
        )
        del wrapper, x0_model
        cleanup_memory()

        video_state = video_tools.unpatchify(
            video_tools.clear_conditioning(
                video_state.__class__(
                    latent=generated_video,
                    denoise_mask=video_state.denoise_mask,
                    positions=video_state.positions,
                    clean_latent=video_state.clean_latent,
                    attention_mask=None,
                )
            )
        )
        audio_state = audio_tools.unpatchify(
            audio_tools.clear_conditioning(
                audio_state.__class__(
                    latent=generated_audio,
                    denoise_mask=audio_state.denoise_mask,
                    positions=audio_state.positions,
                    clean_latent=audio_state.clean_latent,
                    attention_mask=None,
                )
            )
        )
        decoded_video = vae_decode_video(
            video_state.latent,
            self.model_ledger.video_decoder(),
            tiling_config=video_tiling_config,
            generator=generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.model_ledger.audio_decoder(),
            self.model_ledger.vocoder(),
        )
        return decoded_video, decoded_audio
