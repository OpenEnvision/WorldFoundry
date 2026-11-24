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

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .config import MiniMaxH3VideoVAEConfig
from .klvae import AutoencoderKLLegacy


def _resolve_autocast_dtype(dtype: torch.dtype | str | bool | None) -> torch.dtype | None:
    if dtype is None or dtype is False:
        return None
    if dtype is True:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"MiniMax H3 video VAE autocast dtype must be bf16 or fp16, got {dtype}")
        return dtype
    normalized = str(dtype).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported MiniMax H3 video VAE autocast dtype {dtype!r}")
    return mapping[normalized]


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
        self._decode_autocast_dtype: torch.dtype | None = None
        self._channels_last_3d: bool = False
        self._acceleration_report: dict[str, Any] = {}

    def prepare_decoder_autocast_weights(self, dtype: torch.dtype) -> int:
        return self.decoder.prepare_autocast_linear_weights(dtype)

    def configure_acceleration(
        self,
        *,
        autocast_dtype: torch.dtype | str | bool | None = torch.bfloat16,
        channels_last_3d: bool = True,
        stack_tiling: bool = True,
        tile_batch_size: int | None = None,
        fuse_projections: bool = True,
        compile_blocks: bool = False,
        compile_mode: str = "default",
    ) -> dict[str, Any]:
        """Configure inference acceleration (BF16 autocast, channels_last_3d, batched tiling).

        Applies explicit, auditable optimizations while keeping ``x_embedder``
        and ``proj_out`` in FP32 for high-PSNR (>50 dB) video reconstruction.
        """

        receipt: dict[str, Any] = {}
        if fuse_projections:
            from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.acceleration import (
                fuse_minimax_h3_projections,
            )

            receipt["projection_fusion"] = fuse_minimax_h3_projections(self.decoder)

        if channels_last_3d:
            from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

            conv_report = convert_convolution_weight_layouts(self, conv3d=True)
            self._channels_last_3d = True
            receipt["channels_last_3d"] = {
                "enabled": True,
                "conv3d_total": conv_report.conv3d_total,
                "conv3d_converted": conv_report.conv3d_converted,
            }
        else:
            self._channels_last_3d = False
            receipt["channels_last_3d"] = {"enabled": False}

        resolved_dtype = _resolve_autocast_dtype(autocast_dtype)
        self._decode_autocast_dtype = resolved_dtype
        if resolved_dtype is not None:
            converted = self.prepare_decoder_autocast_weights(resolved_dtype)
            receipt["decode_autocast"] = {
                "enabled": True,
                "dtype": str(resolved_dtype),
                "converted_linears": converted,
            }
        else:
            restored = 0
            if getattr(self.decoder, "_autocast_linear_dtype", None) is not None:
                base_dtype = getattr(self.decoder.x_embedder.weight, "dtype", torch.float32)
                restored = self.prepare_decoder_autocast_weights(base_dtype)
            receipt["decode_autocast"] = {"enabled": False, "restored_linears": restored}

        self.stack_tiling = bool(stack_tiling)
        self.tile_batch_size = None if tile_batch_size is None else int(tile_batch_size)
        receipt["stack_tiling"] = {
            "enabled": self.stack_tiling,
            "tile_batch_size": self.tile_batch_size,
        }

        if compile_blocks:
            compiled = 0
            for idx, block in enumerate(self.decoder.transformer_blocks):
                raw_block = getattr(block, "_orig_mod", block)
                self.decoder.transformer_blocks[idx] = torch.compile(raw_block, mode=compile_mode)  # type: ignore[assignment]
                compiled += 1
            receipt["compile_blocks"] = {"enabled": True, "blocks": compiled, "mode": compile_mode}
        else:
            for idx, block in enumerate(self.decoder.transformer_blocks):
                if hasattr(block, "_orig_mod"):
                    self.decoder.transformer_blocks[idx] = block._orig_mod
            receipt["compile_blocks"] = {"enabled": False}

        self._acceleration_report = receipt
        return receipt

    def acceleration_report(self) -> dict[str, Any]:
        """Return the active Video VAE acceleration configuration receipt."""

        return dict(self._acceleration_report)

    def _decode_autocast_context(self, tensor: torch.Tensor):
        if self._decode_autocast_dtype is None:
            return nullcontext()
        if tensor.is_cuda:
            return torch.autocast("cuda", dtype=self._decode_autocast_dtype)
        if tensor.device.type == "cpu" and self._decode_autocast_dtype == torch.bfloat16:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._channels_last_3d and x.ndim == 5 and x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return super().encode(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._channels_last_3d and z.ndim == 5 and z.is_cuda:
            z = z.contiguous(memory_format=torch.channels_last_3d)
        with self._decode_autocast_context(z):
            return super().decode(z)

    def decode_base(self, z: torch.Tensor, frame_num=None, process_image=False) -> torch.Tensor:
        with self._decode_autocast_context(z):
            return super().decode_base(z, frame_num=frame_num, process_image=process_image)


__all__ = ["MiniMaxH3VideoVAE"]
