"""
VAE Wrappers for visualization and validation during DMD distillation.
"""

from collections.abc import Iterator

import torch
import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import encode_audio

from ...ltx_core.loader.registry import Registry
from ...ltx_core.model.video_vae.tiling import TilingConfig
from ...ltx_core.types import Audio


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """
    Infer target device/dtype from module parameters or buffers.
    """
    for tensor in module.parameters():
        return tensor.device, tensor.dtype
    for tensor in module.buffers():
        return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


class VideoVAEWrapper(nn.Module):
    """
    Wrapper for Video VAE encoder and decoder.

    Used for:
    - Encoding videos to latent space (for visualization)
    - Decoding latents to pixel space (for validation)
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            encoder: VideoEncoder instance (optional)
            decoder: VideoDecoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.

        Args:
            video: Pixel video [B, C, F, H, W] in range [-1, 1]

        Returns:
            Latent [B, F', C_latent, H', W']
        """
        if self.encoder is None:
            raise ValueError("Encoder not initialized")

        return self.encoder(video)

    @torch.no_grad()
    def _prepare_decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Decoder expects [B, C, F, H, W].
        # Our DMD code stores video as [B, F, C, H, W] where C=128.
        # Detect this by checking if dim 2 (not dim 1) equals 128.
        if latent.dim() == 5 and latent.shape[2] == 128:
            # Input is [B, F, C, H, W], need to permute to [B, C, F, H, W]
            latent = latent.permute(0, 2, 1, 3, 4)

        # Keep latent dtype/device consistent with decoder weights.
        dec_device, dec_dtype = _module_device_dtype(self.decoder)
        return latent.to(device=dec_device, dtype=dec_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel space.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video [B, C, F_out, H_out, W_out] in range [-1, 1]
        """
        latent = self._prepare_decode_latent(latent)

        return self.decoder(latent)

    @torch.no_grad()
    def decode_to_uint8_chunks(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
    ) -> Iterator[torch.Tensor]:
        """Yield decoded ``[F, H, W, C]`` chunks in CPU uint8 memory.

        Each tiled result is quantized and transferred immediately. This is
        important on Windows/WDDM: collecting full-resolution floating-point
        chunks on the GPU would recreate the memory pressure that tiling is
        intended to avoid.
        """
        latent = self._prepare_decode_latent(latent)
        for video in self.decoder.tiled_decode(latent, tiling_config):
            video_uint8 = (((video + 1) / 2).clamp(0, 1) * 255).to(torch.uint8)
            video_uint8 = video_uint8[0].permute(1, 2, 3, 0)
            yield video_uint8.cpu().contiguous()

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel video for visualization.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video frames suitable for logging (normalized to [0, 1])
        """
        video = self.decode(latent)
        # Normalize from [-1, 1] to [0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        return video


class AudioVAEWrapper(nn.Module):
    """
    Wrapper for Audio VAE decoder and vocoder.

    Used for:
    - Decoding audio latents to mel spectrogram
    - Converting mel to waveform via vocoder
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        vocoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            encoder: AudioEncoder instance (optional)
            decoder: AudioDecoder instance
            vocoder: Vocoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocoder = vocoder
        self.device = device
        self.dtype = dtype

    def get_output_sample_rate(self) -> int | None:
        """
        Return the vocoder waveform sample rate across 2.2/2.3 vocoder variants.
        """
        if self.vocoder is None:
            return None

        for attr in ("output_sample_rate", "output_sampling_rate"):
            value = getattr(self.vocoder, attr, None)
            if value is not None:
                return int(value)

        return None

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Encode waveform to transformer-format audio latents.

        Args:
            waveform: Audio waveform [B, C, samples] or [C, samples]
            sampling_rate: Input waveform sample rate

        Returns:
            Audio latent [B, T, C_latent * mel_bins]
        """
        if self.encoder is None:
            raise ValueError("Audio encoder not initialized")

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform [B, C, samples] or [C, samples], got {tuple(waveform.shape)}")

        enc_device, _ = _module_device_dtype(self.encoder)
        waveform = waveform.to(device=enc_device, dtype=torch.float32)
        latent = encode_audio(
            audio=Audio(waveform=waveform, sampling_rate=int(sampling_rate)),
            audio_encoder=self.encoder,
        )
        return latent.permute(0, 2, 1, 3).flatten(start_dim=2).contiguous()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to mel spectrogram.

        The DMD pipeline produces audio latents in the transformer's sequence
        format ``[B, T, C*F]`` (3D), but the ``AudioDecoder`` expects the VAE
        spatial format ``[B, C, T, F]`` (4D).  This method handles the
        conversion automatically using the decoder's ``z_channels`` and
        ``mel_bins`` attributes (set during checkpoint loading).

        Args:
            latent: Audio latent, either ``[B, T, C*F]`` (transformer) or
                    ``[B, C, T, F]`` (VAE).

        Returns:
            Mel spectrogram ``[B, out_ch, time, freq]``.
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Reshape 3D transformer latent → 4D VAE latent when necessary.
        # The transformer stores audio as [B, T, C*F] where C=z_channels and
        # F=latent_mel_bins.  The AudioDecoder expects [B, C, T, F].
        # Note: decoder.mel_bins is the *output* spectrogram size (e.g. 64),
        # NOT the latent mel dimension.  The latent mel dim = CF // z_channels.
        if latent.dim() == 3:
            B, T, CF = latent.shape
            z_channels = getattr(self.decoder, "z_channels", None)

            if z_channels is not None:
                latent_mel = CF // z_channels  # e.g. 128 // 8 = 16
                # "b t (c f) -> b c t f"
                latent = latent.reshape(B, T, z_channels, latent_mel).permute(0, 2, 1, 3)
            else:
                raise ValueError(
                    f"Cannot reshape 3D audio latent {latent.shape} to 4D: decoder is missing z_channels attribute."
                )

        # Keep latent dtype/device consistent with decoder weights.
        dec_device, dec_dtype = _module_device_dtype(self.decoder)
        latent = latent.to(device=dec_device, dtype=dec_dtype)

        return self.decoder(latent)

    @torch.no_grad()
    def decode_to_waveform(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to waveform.

        Args:
            latent: Audio latent [B, F, C]

        Returns:
            Waveform [B, 1, samples]
        """
        mel = self.decode(latent)

        if self.vocoder is None:
            raise ValueError("Vocoder not initialized")

        return self.vocoder(mel)


def create_vae_wrappers(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    with_video_encoder: bool = False,
    with_audio_encoder: bool = False,
    with_decoders: bool = True,
    decoder_device: torch.device | None = None,
    registry: Registry | None = None,
) -> tuple[VideoVAEWrapper, AudioVAEWrapper]:
    """
    Factory function to create VAE wrappers from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        device: Target device
        dtype: Model dtype
        with_decoders: Load the video/audio decoders and vocoder. Conditioning-only
            jobs can disable them to avoid loading unused weights.
        decoder_device: Device for video/audio decoders and vocoder. Defaults to
            ``device``; pass ``cpu`` during training init to avoid holding decode
            modules on every rank when only encoders are needed.

    Returns:
        Tuple of (VideoVAEWrapper, AudioVAEWrapper)
    """
    from ...ltx_pipelines.utils.model_ledger import ModelLedger

    if decoder_device is None:
        decoder_device = device

    # Load to CPU first to avoid safetensors device issues
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        registry=registry,
    )

    video_encoder = ledger.video_encoder() if with_video_encoder else None
    video_decoder = ledger.video_decoder() if with_decoders else None
    audio_encoder = ledger.audio_encoder() if with_audio_encoder else None
    audio_decoder = ledger.audio_decoder() if with_decoders else None
    vocoder = ledger.vocoder() if with_decoders else None

    # Move to target device
    if video_encoder is not None:
        video_encoder = video_encoder.to(device=device, dtype=dtype)
    if video_decoder is not None:
        video_decoder = video_decoder.to(device=decoder_device, dtype=dtype)
    if audio_encoder is not None:
        audio_encoder = audio_encoder.to(device=device, dtype=torch.float32)
    if audio_decoder is not None:
        audio_decoder = audio_decoder.to(device=decoder_device, dtype=dtype)
    if vocoder is not None:
        vocoder = vocoder.to(device=decoder_device, dtype=dtype)

    video_vae = VideoVAEWrapper(
        encoder=video_encoder,
        decoder=video_decoder,
        device=device,
        dtype=dtype,
    )

    audio_vae = AudioVAEWrapper(
        encoder=audio_encoder,
        decoder=audio_decoder,
        vocoder=vocoder,
        device=device,
        dtype=dtype,
    )

    return video_vae, audio_vae
