# SPDX-License-Identifier: Apache-2.0
"""Top-level MiniMax H3 video VAE wrapper (CNN encode + ViT decode).

:class:`MiniMaxH3VideoVAE` subclasses
:class:`~.klvae.AutoencoderKLLegacy` with the published H3
architecture: causal 3-D CNN encoder, ViT3D decoder,
24-channel latents, BCTHW RGB pixels.

Tiling / parallel decode flags come from
:class:`~.config.MiniMaxH3VideoVAEConfig`.  Latent stats are
validated via :mod:`~..minimax_h3_common`.

Package entry re-exports this type from :mod:`..minimax_h3_video`.
"""

from .config import MiniMaxH3VideoVAEConfig
from .klvae import AutoencoderKLLegacy


class MiniMaxH3VideoVAE(AutoencoderKLLegacy):
    """Encode/decode MiniMax H3 video (CNN encoder, ViT decoder)."""
    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        arch = config.arch_config
        parallel_decode_mode = config.resolved_parallel_decode_mode()
        use_tiled_decode = config.use_tiling and parallel_decode_mode == "tiled"
        super().__init__(
            in_channels=3,
            out_ch=3,
            ch=128,
            embed_dim=24,
            z_channels=24,
            use_3d_conv=True,
            zq_ch_encoder=None,
            zq_ch_decoder=None,
            num_res_blocks=2,
            num_res_blocks_decoder=None,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            space_up=[1, 2, 2, 2, 2, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            time_up=None,
            padding_mode="reflect",
            padding_mode_t=None,
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_kwargs={
                "dim_head": 64,
                "ffn_activation_fn": "silu",
                "ffn_use_gated": True,
                "heads": 32,
                "norm_affine": True,
                "norm_type": "rms_norm",
                "num_layers": 36,
                "qk_norm_affine": False,
                "qk_norm_type": "rms_norm",
                "rope_dim_ratio": 0.75,
                "rope_theta": 100.0,
            },
            shift_factor=0.0,
            scaling_factor=1.0,
            pixel_norm_type="imagenet",
            clip_length=arch.vae_clip_length,
            token_drop=arch.vae_token_drop,
            encoder_tiling=bool(arch.vae_encoder_tiling),
            decoder_tiling=use_tiled_decode,
            # Single-GPU port: parallel tiling always disabled (no decode group).
            parallel_tiling=False,
            tile_size=int(arch.vae_tile_size),
            tile_overlap_min=int(arch.vae_tile_overlap_min),
            encoder_parallel=False,
            decoder_parallel=False,
            chunk_dim=int(arch.vae_chunk_dim),
        )
        self.sglang_config = config
        self.use_parallel_decode = config.use_parallel_decode
        self.parallel_decode_mode = parallel_decode_mode

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        return self.decoder.prepare_autocast_linear_weights(dtype)


__all__ = ["MiniMaxH3VideoVAE"]
