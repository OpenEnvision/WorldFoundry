"""Official DiffVAE and Gemma4 owners for LTX inference/preencoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...errors import BackendContractError
from .checkpoint import StrictCodecLoadReceipt
from .geometry import STABLE_GEOMETRY

# The LTX packages are vendored under the runtime root; class identities keep their upstream module names.
_RUNTIME_PREFIX = __name__.split(".solarwm.backends.")[0] + "."


def _qualified_class(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__.removeprefix(_RUNTIME_PREFIX)}.{cls.__qualname__}"


class OfficialDiffVAEDecoder:
    """Chunked-eager owner for the diffusion video VAE."""

    def __init__(self, checkpoint_path: str | Path, *, device: torch.device) -> None:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import (
            DiffVAEMode,
        )

        from ....ltx_core.model.video_vae import is_diffusion_video_vae
        from ....ltx_pipelines.utils.blocks import VideoDecoder

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        if not is_diffusion_video_vae(self.checkpoint_path):
            raise BackendContractError("LTX requires the official diffusion Video VAE")
        self.decoder = VideoDecoder(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            device=device,
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        )

    @property
    def implementation_class(self) -> str:
        return _qualified_class(self.decoder)

    def decode(self, latent: torch.Tensor, *, seed: int) -> torch.Tensor:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import (
            DiffVAEMode,
        )
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import VideoPixelShape

        from ....ltx_core.model.video_vae import AUTO_TILING
        from ....ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae

        if tuple(latent.shape) != (1, 128, 20, 16, 24):
            raise BackendContractError("DiffVAE input must be [1,128,20,16,24]")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        scale_factors = tiling_scale_factors_for_vae(self.checkpoint_path)
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=STABLE_GEOMETRY.pixel_frames,
                height=STABLE_GEOMETRY.height,
                width=STABLE_GEOMETRY.width,
                fps=24.0,
            ),
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
            device=self.device,
        )
        with torch.inference_mode():
            chunks = tuple(
                self.decoder(
                    latent,
                    tiling,
                    generator=generator,
                    dtype=torch.bfloat16,
                )
            )
        if not chunks or any(
            not isinstance(chunk, torch.Tensor) or chunk.ndim != 4 or chunk.shape[-1] != 3 for chunk in chunks
        ):
            raise BackendContractError("official DiffVAE returned invalid FHWC chunks")
        frames = torch.cat(chunks, dim=0)
        if tuple(frames.shape) != (153, 512, 768, 3):
            raise BackendContractError("official DiffVAE output geometry differs")
        if not bool(torch.isfinite(frames).all()) or not bool(((frames >= 0) & (frames <= 1)).all()):
            raise BackendContractError("official DiffVAE output is not finite [0,1]")
        return frames.permute(3, 0, 1, 2).unsqueeze(0).float().contiguous()

    def close(self) -> None:
        del self.decoder
        torch.cuda.empty_cache()


def codec_receipt(
    *,
    provider_identity: str,
    video_vae_class: str,
    gemma_feature_extractor_class: str = "",
    video_vae_operation: str = "diffvae_decode",
    video_vae_encoder_class: str = "",
) -> StrictCodecLoadReceipt:
    return StrictCodecLoadReceipt(
        provider_identity=provider_identity,
        video_vae_class=video_vae_class,
        diffvae_mode=("direct" if video_vae_operation == "direct_encode" else "chunked_eager"),
        gemma_feature_extractor_class=gemma_feature_extractor_class,
        caption_cache_stage=("gemma4_feature_extractor_preconnector" if gemma_feature_extractor_class else ""),
        video_vae_operation=video_vae_operation,
        video_vae_encoder_class=video_vae_encoder_class,
    )


__all__ = [
    "OfficialDiffVAEDecoder",
    "codec_receipt",
]
