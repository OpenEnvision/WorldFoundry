"""
3D posemb_sincos, Adapte from Qwen-2.5VL M-RoPE

We assum the raw image patches are 16x16.

1. For the spatial index, we use the index to indicate the grid point (not the patch center)

Here is the spatial index for one row (1x16) image patch:

0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
.--.--.--.--.--.--.--.--.--.--.--.--.--.--.--.--.
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
.--.--.--.--.--.--.--.--.--.--.--.--.--.--.--.--.

Therefore, for 8x8 patches (pooling from 16x16 to 8x8), the spatial index for one row (1x8) image patch should become:
   1     3     5     7     9     11    13    15
.-----.-----.-----.-----.-----.-----.-----.-----.
|     |     |     |     |     |     |     |     |
.-----.-----.-----.-----.-----.-----.-----.-----.


4x4 patches (pooling from 16x16 to 4x4), the spatial index for one row (1x4) image patch should become:
      2           6           10         14
.-----------.-----------.-----------.-----------.
|           |           |           |           |
.-----------.-----------.-----------.-----------.


2x2 patches (pooling from 16x16 to 2x2), the spatial index for one row (1x2) image patch should become:

            4                       12
.----------------------.------------------------.
|                       |                       |
.-----------------------.-----------------------.


The column index calculation is similar to the row index calculation.


2. For the temporal index, we use temporal interval

for the input videos, we first extract frames with temporal stride m.

0         |    0      | ... |    0      |     m      |     m     | ...
img1_tok1 | img1_tok2 | ... | img1_tokN |  img2_tok1 | img2_tok2 | ...

"""

import einops
import jax.numpy as jnp


class PosEmb3D:
    def __init__(self, dim: int, temporal_base: int = 10_000, spatial_base: int = 1_000):
        self.dim = dim
        if dim % 6:
            raise ValueError("Position embedding dimension must be divisible by 6")
        width = dim // 6
        # 2 for temporal (t sin, t cos), 4 for spatial (h sin, h cos, w sin, w cos)

        omega = jnp.arange(width) / (width - 1)
        self.temporal_omega = 1.0 / (temporal_base**omega)
        self.spatial_omega = 1.0 / (spatial_base**omega)

        # Pre-compute spatial embeddings for different pool sizes
        self.spatial_pe8x8 = self.compute_spatial_pe8x8()
        self.spatial_pe4x4 = self.compute_spatial_pe4x4()
        self.spatial_pe2x2 = self.compute_spatial_pe2x2()

    def compute_spatial_pe8x8(self):
        y, x = jnp.mgrid[:8, :8]
        y = 2 * y + 1
        x = 2 * x + 1
        y = jnp.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = jnp.einsum("m,d->md", x.flatten(), self.spatial_omega)
        spatial_pe = jnp.concatenate([jnp.sin(y), jnp.cos(y), jnp.sin(x), jnp.cos(x)], axis=-1)
        return spatial_pe

    def compute_spatial_pe4x4(self):
        y, x = jnp.mgrid[:4, :4]
        y = 4 * y + 2
        x = 4 * x + 2
        y = jnp.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = jnp.einsum("m,d->md", x.flatten(), self.spatial_omega)
        spatial_pe = jnp.concatenate([jnp.sin(y), jnp.cos(y), jnp.sin(x), jnp.cos(x)], axis=-1)
        return spatial_pe

    def compute_spatial_pe2x2(self):
        y, x = jnp.mgrid[:2, :2]
        y = 8 * y + 4
        x = 8 * x + 4
        y = jnp.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = jnp.einsum("m,d->md", x.flatten(), self.spatial_omega)
        spatial_pe = jnp.concatenate([jnp.sin(y), jnp.cos(y), jnp.sin(x), jnp.cos(x)], axis=-1)
        return spatial_pe

    def compute_temporal_pe(self, pos):
        sinusoid_input = jnp.einsum("m,d->md", pos, self.temporal_omega)
        temporal_pe = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
        return temporal_pe

    def __call__(self, pos, spatial_size: int):
        """
        Generate 3D positional embeddings.

        Args:
            pos: 1D array of temporal positions (frame indices)
            spatial_size: Spatial resolution (8, 4, or 2)

        Returns:
            3D positional embeddings of shape (T, H*W, D)
        """
        if spatial_size not in {8, 4, 2}:
            raise ValueError("spatial_size must be 8, 4, or 2")

        if spatial_size == 8:
            spatial_pe = self.spatial_pe8x8
        elif spatial_size == 4:
            spatial_pe = self.spatial_pe4x4
        else:
            spatial_pe = self.spatial_pe2x2

        temporal_pe = self.compute_temporal_pe(jnp.asarray(pos, dtype=jnp.float32))
        temporal_pe_repeat = einops.repeat(temporal_pe, "t d -> t hw d", hw=spatial_size * spatial_size)
        spatial_pe_repeat = einops.repeat(spatial_pe, "hw d -> t hw d", t=pos.shape[0])
        return jnp.concatenate([temporal_pe_repeat, spatial_pe_repeat], axis=-1)
