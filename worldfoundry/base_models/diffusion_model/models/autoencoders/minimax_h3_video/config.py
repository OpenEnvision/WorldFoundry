# SPDX-License-Identifier: Apache-2.0
"""Config dataclasses for the MiniMax H3 video VAE.

:class:`MiniMaxH3VideoVAEArchConfig` describes the CNN/ViT
widths, downsample schedule, and 24-channel latent size.
:class:`MiniMaxH3VideoVAEConfig` adds tiling / parallel-decode
runtime flags.

No tensor math lives here.  :class:`~.model.MiniMaxH3VideoVAE`
passes these into :class:`~.klvae.AutoencoderKLLegacy`.

Sibling of :mod:`~..minimax_h3_audio.config`.
"""

from dataclasses import dataclass, field

from ..minimax_h3_common import validate_minimax_h3_vae_latent_stats


@dataclass
class MiniMaxH3VideoVAEArchConfig:
    """Architectural widths and downsample schedule for the H3 video VAE."""
    latent_channels: int = 24
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16
    vae_clip_length: int = 17
    vae_token_drop: int = 3
    vae_encoder_tiling: int = 1
    vae_decoder_tiling: int = 1
    vae_parallel_tiling: int = 1
    vae_tile_size: int = 256
    vae_tile_overlap_min: int = 64
    vae_chunk_dim: int = -1
    # sglang VAEArchConfig fields kept for parity / stat validation
    scaling_factor: float = 1.0


@dataclass
class MiniMaxH3VideoVAEConfig:
    """Runtime config (tiling, parallel decode) plus architecture."""
    arch_config: MiniMaxH3VideoVAEArchConfig = field(
        default_factory=MiniMaxH3VideoVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True
    use_tiling: bool = True
    use_parallel_tiling: bool = True
    # The released checkpoint's quality contract uses overlapping latent
    # tiles. Parallel tiling distributes whole tiles without changing that
    # recipe. Spatial-shard decode is rejected because validation found output
    # mismatches on H3.
    parallel_decode_mode: str = "tiled"
    # Single-GPU port: distributed decode is disabled. SGLang's VAEConfig
    # default was True; here we keep the field for API parity but force False
    # so the wrapper always takes the local tiled path.
    use_parallel_decode: bool = False

    def resolved_parallel_decode_mode(self) -> str:
        if self.parallel_decode_mode == "auto":
            return "tiled"
        if self.parallel_decode_mode in ("spatial", "spatial_shard"):
            raise ValueError(
                "MiniMax H3 rejects spatial-shard VAE decode because it failed "
                "the released quality contract; use tiled"
            )
        if self.parallel_decode_mode == "tiled":
            return "tiled"
        if self.parallel_decode_mode == "patch":
            raise ValueError("MiniMax H3 does not support patch VAE decode; use tiled")
        raise ValueError(
            f"unsupported MiniMax H3 VAE parallel decode mode "
            f"{self.parallel_decode_mode!r}"
        )

    def post_init(self) -> None:
        self.resolved_parallel_decode_mode()
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )


__all__ = ["MiniMaxH3VideoVAEArchConfig", "MiniMaxH3VideoVAEConfig"]
