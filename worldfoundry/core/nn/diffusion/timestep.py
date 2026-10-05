"""Sinusoidal timestep embeddings and their MLP projections.

Diffusion models treat continuous ``t`` as a position: the same DDPM
sinusoid used by the scheduler, then a small MLP so AdaLN can consume it.
Keeping the sinusoid here (not inside each DiT) avoids flip-sin/cos and
``max_period`` drift across checkpoints.

- :func:`get_timestep_embedding` is the raw [N, dim] table (fractional t OK).
- :class:`Timesteps` is that table as a module (PixArt / diffusers layout).
- :class:`TimestepEmbedding` is the two-Linear + SiLU projection; optional
  ``cond_proj`` adds a class/guidance vector before the first Linear.
- :class:`ProjectedTimestepEmbedding` composes sinusoid + MLP and can run
  the sinusoid on a dedicated ``computation_device`` (CPU) to keep Graph
  capture free of host syncs.

Not this module:
    DiT-side :class:`~worldfoundry.core.nn.transformer.dit.SinusoidalTimestepEmbedder`
    is a checkpoint-named MLP over :func:`~worldfoundry.core.nn.transformer.shape.sinusoidal_embedding_1d`.
    Schedules that *produce* ``t`` live in
    :mod:`worldfoundry.core.nn.diffusion.schedulers`.

Public surface:

- :func:`get_timestep_embedding`
- :class:`Timesteps` / :class:`TimestepEmbedding` /
  :class:`ProjectedTimestepEmbedding`
- :class:`PixArtAlphaCombinedTimestepSizeEmbeddings`
"""

import math

import torch


# ──────────────────────────────────────────────────────────────────────────
# DDPM sinusoid — fractional t, optional cos-first, odd-dim zero pad
# ──────────────────────────────────────────────────────────────────────────


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# ──────────────────────────────────────────────────────────────────────────
# MLP projections — checkpoint-visible Linear names; optional cond_proj
# ──────────────────────────────────────────────────────────────────────────


class TimestepEmbedding(torch.nn.Module):
    """Two-layer SiLU projection of sinusoidal (or precomputed) timestep features."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        """Build ``linear_1`` / ``linear_2``; optional ``cond_proj`` has no bias.

        ``post_act_fn`` is accepted for diffusers-shaped constructors. Only
        ``None`` is wired today — a non-None value would leave ``post_act``
        unset and fail in :meth:`forward`.
        """

        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim

        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    @staticmethod
    def _linear_dtype(module: torch.nn.Module) -> torch.dtype:
        """Prefer ``computation_dtype`` (quantized / Graph wrappers) over ``weight.dtype``.

        Raises:
            TypeError: the module exposes neither a computation dtype nor a weight.
        """

        computation_dtype = getattr(module, "computation_dtype", None)
        if isinstance(computation_dtype, torch.dtype):
            return computation_dtype
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.dtype
        raise TypeError(f"{type(module).__name__} does not expose a linear computation dtype")

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Project ``[N, in_channels]``; add ``cond_proj(condition)`` before the first Linear.

        Casts follow each Linear's computation dtype so FP8/Graph wrappers
        do not see a silent promotion.
        """

        if condition is not None:
            condition = condition.to(dtype=self._linear_dtype(self.cond_proj))
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample.to(dtype=self._linear_dtype(self.linear_1)))

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample.to(dtype=self._linear_dtype(self.linear_2)))

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class ProjectedTimestepEmbedding(torch.nn.Module):
    """Sinusoidal timestep features followed by a checkpoint-visible MLP."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        computation_device: torch.device | str | None = None,
    ) -> None:
        """Optionally run the sinusoid off-device so CUDA Graph capture stays sync-free."""

        super().__init__()
        self.dim_in = int(dim_in)
        self.computation_device = computation_device
        self.timestep_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out),
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``[N]`` timesteps → ``[N, dim_out]`` in ``dtype`` after the sinusoid+MLP.

        Uses flip-sin-to-cos and ``downscale_freq_shift=0`` to match the
        PixArt / SD3 table this wrapper was introduced for.
        """

        source = timestep.to(self.computation_device) if self.computation_device is not None else timestep
        embedding = get_timestep_embedding(
            source,
            self.dim_in,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        if source.device != timestep.device:
            embedding = embedding.to(timestep.device)
        return self.timestep_embedder(embedding.to(dtype))


class Timesteps(torch.nn.Module):
    """Module wrapper around :func:`get_timestep_embedding` for diffusers-style graphs."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        """Store table knobs as attributes so they appear in the module graph."""

        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[N]`` → ``[N, num_channels]`` sinusoid; see :func:`get_timestep_embedding`."""

        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


# ──────────────────────────────────────────────────────────────────────────
# PixArt-Alpha — 256-wide sinusoid then the shared two-layer projection
# ──────────────────────────────────────────────────────────────────────────


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        """``size_emb_dim`` is stored as ``outdim`` for checkpoint-shaped callers.

        The time table is fixed at 256 channels (PixArt release); only the
        MLP output width follows ``embedding_dim``.
        """

        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        """``[N]`` → ``[N, embedding_dim]`` in ``hidden_dtype`` (matches AdaLN)."""

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb
