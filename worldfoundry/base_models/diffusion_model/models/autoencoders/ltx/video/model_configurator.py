"""Instantiate LTX video encoders and decoders from checkpoint JSON.

:class:`VideoEncoderConfigurator` and
:class:`VideoDecoderConfigurator` rebuild
:class:`~.video_vae.VideoEncoder` / :class:`VideoDecoder`
from metadata shipped with LTX weights.  LTX-2.5 ``CausalDiffusionVAE``
checkpoints nest the encoder under ``vae.encoder``; their neighborhood-attention
decoder is built by
:class:`~.diffusion_video_decoder.DiffusionVideoDecoderConfigurator`.

They do not run encode/decode or tiling; they only construct
modules.  :mod:`~..component` calls them at load time.

Sibling of the audio :mod:`~..audio.model_configurator`.
"""

from typing import Any

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import (
    LogVarianceType,
    NormLayerType,
    PaddingModeType,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoDecoder, VideoEncoder


class VideoEncoderConfigurator:
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> VideoEncoder:
        """Build the encoder from split-checkpoint safetensors metadata."""
        return cls.from_config(metadata.get("config", {}))

    @classmethod
    def from_config(cls: type[VideoEncoder], config: dict) -> VideoEncoder:
        config = config.get("vae", {})
        if "encoder" in config:
            return VideoEncoder(**_nested_encoder_kwargs(config))
        convolution_dimensions = config.get("dims", 3)
        in_channels = config.get("in_channels", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "zeros"))
        encoder_blocks = config.get("encoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        latent_log_var_str = config.get("latent_log_var", "uniform")

        return VideoEncoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=latent_channels,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            latent_log_var=LogVarianceType(latent_log_var_str),
            encoder_spatial_padding_mode=spatial_padding_mode,
        )


class VideoDecoderConfigurator:
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_config(cls: type[VideoDecoder], config: dict) -> VideoDecoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "reflect"))
        out_channels = config.get("out_channels", 3)
        decoder_blocks = config.get("decoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        causal = config.get("causal_decoder", False)
        timestep_conditioning = config.get("timestep_conditioning", True)
        base_channels = config.get("decoder_base_channels", 128)

        return VideoDecoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=latent_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=spatial_padding_mode,
            base_channels=base_channels,
        )


def _nested_encoder_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Read the ``vae.encoder`` section of an LTX-2.5 ``CausalDiffusionVAE`` config.

    Its ``out_channels`` is the latent width and ``blocks`` the encoder ladder.
    """
    encoder = config["encoder"]
    return {
        "convolution_dimensions": encoder.get("dims", config.get("dims", 3)),
        "in_channels": encoder.get("in_channels", 3),
        "out_channels": encoder.get("out_channels", config.get("latent_channels", 128)),
        "encoder_blocks": encoder.get("blocks", encoder.get("encoder_blocks", [])),
        "patch_size": encoder.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(
            encoder.get("spatial_padding_mode", config.get("encoder_spatial_padding_mode", "zeros"))
        ),
    }
