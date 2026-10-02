"""The RAEv2 temporal-downsampling video codec: encode/decode plus checkpoint loading.

:class:`VideoCodec` wraps the frozen-DINOv3 :class:`RAEEncoder` and the :class:`ViTVideoDecoder`.
It is video-only (no audio). The frozen DINOv3 backbone is loaded via ``torch.hub`` (see
:mod:`mira.codec.dino`); :meth:`VideoCodec.load_from_checkpoint` restores the backbone
weights from the checkpoint, so a deployment environment does not need a separate DINO weights path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

from mira.codec.config import VideoCodecConfig
from mira.codec.rae_encoder import RAEEncoder, RAEEncoderOutputs
from mira.codec.vit_decoder import ViTVideoDecoder
from mira.data.batch import VideoActionBatch
from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets

logger = logging.getLogger(__name__)

# Fields dropped from the codec schema but still written, at their no-op value, by older training
# runs; tolerated and stripped on load so released checkpoints keep validating (see
# `drop_removed_fields`). A non-no-op value raises rather than loading silently wrong.
REMOVED_CONFIG_FIELDS = {"is_audio_model": False}


@dataclass  # dataclass doesn't cause graph breaks when torch.compile()'ing, unlike Pydantic
class VideoCodecOutputs:
    input_video: Tensor
    output_video: Tensor
    z: Tensor
    dino_features: tuple[Tensor, ...] | None = None


class VideoCodec(nn.Module):
    def __init__(self, config: VideoCodecConfig, require_dino_weights: bool = True) -> None:
        super().__init__()

        self.config = config

        self.encoder = RAEEncoder(config.encoder, require_dino_weights=require_dino_weights)
        self.decoder = ViTVideoDecoder(config.decoder)

        self.temporal_downsampling, self.spatial_downsampling = self.encoder.get_downsampling_factors()
        decoder_patch_size_t = config.decoder.patch_size_t
        if decoder_patch_size_t != self.temporal_downsampling:
            raise ValueError(
                f"Codec encoder/decoder temporal mismatch: encoder downsamples by "
                f"{self.temporal_downsampling}x in time, decoder expands by "
                f"{decoder_patch_size_t}x (decoder.patch_size_t). Set them equal."
            )
        if config.decoder.latent_dim != config.encoder.latent_dim:
            raise ValueError(
                f"Codec encoder/decoder latent_dim mismatch: encoder={config.encoder.latent_dim}, "
                f"decoder={config.decoder.latent_dim}. Set them equal."
            )
        decoder_spatial = config.decoder.bottleneck.stride * config.decoder.patch_size
        if decoder_spatial != self.spatial_downsampling:
            raise ValueError(
                f"Codec encoder/decoder spatial mismatch: encoder downsamples by "
                f"{self.spatial_downsampling}x in space, decoder expands by {decoder_spatial}x "
                f"(decoder.bottleneck.stride * decoder.patch_size). Set them equal."
            )
        self.latent_dim = config.encoder.latent_dim

        # Non-weights checkpoint entries (everything saved alongside ``state_dict``), populated by
        # :meth:`load_from_checkpoint`. This is freeform metadata; the world model reads the codec
        # latent normalization from its ``latent_mean_std`` key when present.
        self.info_from_checkpoint: dict | None = None

    def preprocess_batch(self, batch: VideoActionBatch) -> None:
        """Preprocess a batch in-place.

        Watch out: the batch should be on the GPU, because resizing on CPU is so slow that it can
        bottleneck world model training.
        """
        video = batch.video / 255.0

        batch_size, n_frames, _, image_h, image_w = video.shape
        target_h, target_w = self.config.encoder.video.height, self.config.encoder.video.width

        # Pad with black to match the target aspect ratio, either on the right or bottom
        if image_w * target_h < target_w * image_h:
            right_pad = round(target_w / target_h * image_h) - image_w
            video = torch.nn.functional.pad(video, (0, right_pad))
        elif image_w * target_h > target_w * image_h:
            bottom_pad = round(target_h / target_w * image_w) - image_h
            video = torch.nn.functional.pad(video, (0, 0, 0, bottom_pad))

        if video.shape[-2:] != (target_h, target_w):
            video = torch.nn.functional.interpolate(
                rearrange(video, "b t c h w -> (b t) c h w"),
                size=(target_h, target_w),
                mode="bilinear",
                antialias=True,
            )
            video = rearrange(video, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames)

        batch.video = video

    def normalize_video(self, x: Tensor, trim_video: bool = True) -> Tensor:
        if trim_video:
            x = x[:, : self.config.encoder.video.timesteps]

        x = (x - 0.5) / 0.5  # [-1, 1] range

        return x

    def forward(self, batch: VideoActionBatch, trim_video: bool = True) -> VideoCodecOutputs:
        self.preprocess_batch(batch)

        input_video, encoder_output = self.encode(batch.video, trim_video=trim_video)
        output_video = self.decode(encoder_output.z)

        return VideoCodecOutputs(
            input_video=input_video,
            output_video=output_video,
            z=encoder_output.z,
            dino_features=encoder_output.dino_features,
        )

    def encode(self, video: Tensor, trim_video: bool = True) -> tuple[Tensor, RAEEncoderOutputs]:
        input_video = self.normalize_video(video, trim_video=trim_video)
        return input_video, self.encoder(input_video)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    CONFIG_FILENAME = "codec_config.yaml"

    def save_checkpoint(self, checkpoint_path: str | Path, extra_data: dict | None = None) -> None:
        """Save the model checkpoint to a file.

        Doesn't save the config; save it separately alongside the checkpoint.
        """
        if _find_codec_config(checkpoint_path) is None:
            raise FileNotFoundError(
                f"Could not find '{VideoCodec.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        checkpoint = {"state_dict": self.state_dict()}

        if extra_data is not None:
            for key in extra_data.keys():
                if key in checkpoint:
                    raise ValueError(f"Key {key} already exists in checkpoint, cannot overwrite")

            checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> VideoCodec:
        # `device` defaults to "cpu" rather than None: torch.load(..., map_location=None)
        # restores tensors to the device they were saved from (typically cuda:0), which
        # causes every DDP rank to materialise the codec on cuda:0 and silently leaks
        # several GiB onto rank 0's GPU. Callers that want GPU loading pass device explicitly.
        from omegaconf import OmegaConf  # noqa: PLC0415 -- optional dep, used only here

        config_path = _find_codec_config(checkpoint_path)

        if config_path is None:
            raise FileNotFoundError(
                f"Could not find '{VideoCodec.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        # Load via OmegaConf (already a dependency) so the config's ${..} interpolations resolve.
        config_raw = OmegaConf.load(config_path)

        # _target_ keys may exist at any nesting if saved from hydra instantiation; strip them all,
        # then drop fields removed from the schema since the checkpoint was written.
        raw_dict = strip_hydra_targets(
            OmegaConf.to_container(config_raw.model.architecture.config, resolve=True)
        )
        raw_dict = drop_removed_fields(raw_dict, REMOVED_CONFIG_FIELDS)
        assert isinstance(raw_dict, dict), "expected dict at architecture.config"
        config = VideoCodecConfig.model_validate(raw_dict)

        # We don't need pre-trained dino weights because the codec checkpoint includes them
        model = VideoCodec(config, require_dino_weights=False)
        model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])

        checkpoint.pop("state_dict")
        model.info_from_checkpoint = checkpoint

        return model


def _find_codec_config(checkpoint_path: str | Path) -> Path | None:
    for parent in Path(checkpoint_path).parents:
        candidate = parent / VideoCodec.CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None
