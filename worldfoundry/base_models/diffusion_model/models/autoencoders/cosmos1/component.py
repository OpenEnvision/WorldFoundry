"""Cosmos Predict1 Tokenize1 :class:`~...contracts.LatentEncoder` / decoder.

Wraps official TorchScript encoder/decoder artifacts.  Encode applies
per-chunk mean/std then ``latent_scale`` (σ_data = 0.5).  Decode inverts
that contract.  Only 1-frame stills or 121-pixel / 16-latent video chunks
are supported.  ``return_latent`` skips decode.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import NativeCheckpointResolver


def _first_tensor(value: object) -> torch.Tensor:
    while isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Cosmos Tokenize1 returned an empty sequence")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Cosmos Tokenize1 returned {type(value).__name__}, expected Tensor")
    return value


class Cosmos1VideoCodec:
    """One Tokenize1 artifact serving shared latent encode/decode roles."""

    latent_channels = 16
    spatial_compression = 8
    temporal_compression = 8
    pixel_chunk_duration = 121
    latent_chunk_duration = 16

    def __init__(
        self,
        encoder: torch.jit.ScriptModule,
        decoder: torch.jit.ScriptModule,
        *,
        image_mean: torch.Tensor,
        image_std: torch.Tensor,
        video_mean: torch.Tensor,
        video_std: torch.Tensor,
        dtype: torch.dtype,
        latent_scale: float = 0.5,
    ) -> None:
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.image_mean = image_mean.reshape(1, self.latent_channels, 1, 1, 1)
        self.image_std = image_std.reshape(1, self.latent_channels, 1, 1, 1)
        self.video_mean = video_mean.reshape(self.latent_channels, -1)[:, : self.latent_chunk_duration].reshape(
            1, self.latent_channels, self.latent_chunk_duration, 1, 1
        )
        self.video_std = video_std.reshape(self.latent_channels, -1)[:, : self.latent_chunk_duration].reshape(
            1, self.latent_channels, self.latent_chunk_duration, 1, 1
        )
        self.dtype = dtype
        self.latent_scale = float(latent_scale)
        if self.latent_scale <= 0:
            raise ValueError("Cosmos Tokenize1 latent_scale must be positive")

    def _stats(self, latent_frames: int, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_frames == 1:
            mean, std = self.image_mean, self.image_std
        elif latent_frames == self.latent_chunk_duration:
            mean, std = self.video_mean, self.video_std
        else:
            raise ValueError(
                "Cosmos Tokenize1 supports one image latent or one 121-frame/16-latent video chunk"
            )
        return mean.to(device=value.device, dtype=value.dtype), std.to(device=value.device, dtype=value.dtype)

    @torch.no_grad()
    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Run the TorchScript encoder without mean/std / σ_data scaling."""
        if images.ndim == 4:
            images = images.unsqueeze(2)
        if images.ndim != 5:
            raise ValueError("Cosmos Tokenize1 encoder expects BCHW or BCTHW pixels")
        if images.shape[2] not in {1, self.pixel_chunk_duration}:
            raise ValueError("Cosmos Tokenize1 input must contain 1 or 121 pixel frames")
        return _first_tensor(self.encoder(images.to(self.dtype))).to(images.dtype)

    @torch.no_grad()
    def decode_raw(self, latents: torch.Tensor) -> torch.Tensor:
        """Run the TorchScript decoder on already-unnormalized latents."""
        return _first_tensor(self.decoder(latents.to(self.dtype))).to(latents.dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixels and apply Tokenize1 mean/std × ``latent_scale``."""
        latent = self.encode_raw(images)
        mean, std = self._stats(latent.shape[2], latent)
        # Cosmos Predict1 trains the diffusion model on tokenizer-normalized
        # latents scaled by sigma_data (0.5).  Tokenize1's serialized mean/std
        # only performs the first half of that contract; omitting sigma_data
        # doubles the latent magnitude seen by the denoiser and collapses
        # GEN3C generations during decode.
        return ((latent - mean) / std) * self.latent_scale

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Invert σ_data / mean-std and decode; trim to ``request.num_frames``."""
        if bool(request.inputs.get("return_latent", False)):
            return latents
        if not bool(torch.isfinite(latents).all()):
            raise FloatingPointError("Cosmos Tokenize1 received non-finite latents for decode")
        input_dtype = latents.dtype
        mean, std = self._stats(latents.shape[2], latents)
        normalized = latents / self.latent_scale
        pixels = self.decode_raw((normalized * std + mean).to(input_dtype))
        if not bool(torch.isfinite(pixels).all()):
            raise FloatingPointError("Cosmos Tokenize1 decoder produced non-finite pixels")
        return pixels[:, :, : request.num_frames]


def load_cosmos1_video_codec(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Cosmos1VideoCodec:
    """Load Tokenize1 once for recipe and representation consumers."""

    root = Path(checkpoint_path).expanduser().resolve()
    target = torch.device(device)
    required = ("encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cosmos Tokenize1 checkpoint is missing files: {missing}")
    encoder = torch.jit.load(str(root / "encoder.jit"), map_location=target).to(dtype=dtype)
    decoder = torch.jit.load(str(root / "decoder.jit"), map_location=target).to(dtype=dtype)
    video_mean, video_std = torch.load(root / "mean_std.pt", map_location=target, weights_only=True)
    image_mean, image_std = torch.load(root / "image_mean_std.pt", map_location=target, weights_only=True)
    return Cosmos1VideoCodec(
        encoder,
        decoder,
        image_mean=image_mean,
        image_std=image_std,
        video_mean=video_mean,
        video_std=video_std,
        dtype=dtype,
    )


def build_cosmos1_video_codec(context: ComponentBuildContext) -> Cosmos1VideoCodec:
    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
    return load_cosmos1_video_codec(
        checkpoint.root,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )


__all__ = ["Cosmos1VideoCodec", "build_cosmos1_video_codec", "load_cosmos1_video_codec"]
