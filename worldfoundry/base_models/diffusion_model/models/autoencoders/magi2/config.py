# Ported from SandAI MAGI-2-preview (Apache-2.0): plain dataclass configs + latent stats.
"""Architecture constants and latent statistics for the MAGI-2 VAEs.

The 48-dim video ``latents_mean`` / ``latents_std`` below are the normalization
statistics shipped inline in the upstream ``Wan2_2_VAE`` and ``TurboVAED``
classes (identical arrays in both). They are exposed here as the single source
of truth; the model modules import them from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
#  Wan2.2 (z_dim=48) video latent normalization statistics
#  Source: <SG>/model/vae2_2.py (Wan2_2_VAE.mean/std) and turbo_vaed.py.
# ---------------------------------------------------------------------------

WAN22_LATENTS_MEAN: List[float] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

WAN22_LATENTS_STD: List[float] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


@dataclass
class VideoVAEConfig:
    """Arch constants for the Wan2.2 causal-3D video encode VAE (``Wan2_2_VAE``).

    encode(video [B,3,T,H,W]) -> latent [B,48,1+(T-1)/4,H/16,W/16].
    Spatial /16 (patchify 2 + 3 spatial downsamples), temporal /4, deterministic.
    """

    z_dim: int = 48
    c_dim: int = 160  # encoder base channel width (`dim`)
    dim_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: Tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: List[float] = field(default_factory=lambda: list(WAN22_LATENTS_MEAN))
    latents_std: List[float] = field(default_factory=lambda: list(WAN22_LATENTS_STD))


@dataclass
class TurboDecoderConfig:
    """Arch constants for the decode-only student ``TurboVAED``.

    decode(z [B,48,T,H/16,W/16]) -> [B,3,~4T,H,W] via sliding-window temporal
    decode (first_chunk_size=3, step_size=5, overlap = 1 latent * temporal ratio).
    """

    out_channels: int = 3
    # NOTE: upstream's __init__ signature defaults are placeholders (latent_channels=128,
    # patch_size=4); the shipped config uses the values below (48-dim Wan2.2 latent and
    # the patch_size==2 that TurboVAEDDecoder3d asserts). Kept here as the working values.
    latent_channels: int = 48
    decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    decoder_layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4)
    decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False)
    patch_size: int = 2
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-6
    scaling_factor: float = 1.0
    decoder_causal: bool = False
    decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False)
    decoder_dw_kernel_size: int = 3
    decoder_spatio_only: Tuple[bool, ...] = (False, False, False, False)
    first_chunk_size: int = 3
    step_size: int = 5
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4
    use_unpatchify: bool = False
    latents_mean: List[float] = field(default_factory=lambda: list(WAN22_LATENTS_MEAN))
    latents_std: List[float] = field(default_factory=lambda: list(WAN22_LATENTS_STD))


@dataclass
class AudioVAEConfig:
    """Arch constants for the Stable-Audio-Open-1.0 Oobleck audio VAE.

    encode(audio [B,2,N]) -> [B,64,N/2048]; decode(latents) -> [B,2,N].
    Downsampling 2048 = prod(strides). io_channels=2 (stereo), 44100 Hz.
    The encoder emits 2*latent_dim (mean+scale) which the VAE bottleneck
    reparameterizes down to ``latent_dim`` channels.
    """

    latent_dim: int = 64
    downsampling_ratio: int = 2048
    sample_rate: int = 44100
    io_channels: int = 2
    channels: int = 128
    c_mults: Tuple[int, ...] = (1, 2, 4, 8, 16)
    strides: Tuple[int, ...] = (2, 4, 4, 8, 8)
    use_snake: bool = True
    final_tanh: bool = False
