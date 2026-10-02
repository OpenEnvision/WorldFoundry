"""Reusable vision-language layers owned by WorldFoundry core.

This is the shared ``nn.Module`` kit (MLP, SwiGLU, DropPath, PatchEmbed,
LayerScale) used by ViT, SAM, and DiT blocks. Keep checkpoint-visible names
stable: ``Mlp`` is the timm two-layer FFN; ``SamHeadMLP`` / ``SamMLPBlock``
are SAM-specific so a mask head never aliases a transformer FFN.

Tuple helpers (``to_2tuple``, ``val2list``) exist so config scalars and
pairs share one constructor path. :func:`zero_module` detaches and zeros
parameters — used for residual adapters that must start as identity.
``XFORMERS_AVAILABLE`` is a probe, not an auto-enable: fused attention
still goes through :mod:`worldfoundry.core.attention`.

Not this module:
    DiT AdaLN / modulation lives in
    :mod:`worldfoundry.core.nn.transformer.dit`. Conv FFNs that
    take ``HW`` live in :mod:`worldfoundry.core.nn.blocks.convolutional_mlp`.
    2D patchify without a conv lives in :mod:`worldfoundry.core.nn.blocks.patching`.

Public surface:
    Tuple helpers, :class:`DropPath` / :class:`LayerNorm2d` /
    :class:`LayerScale`, SAM and timm MLPs, :class:`VisionAttention`,
    :class:`DomainAwareLinear`, :class:`PositionEmbeddingRandom`,
    :class:`PatchEmbed` / :class:`PatchEmbed_Mlp`, :class:`SwiGLUFFN` /
    :class:`SwiGLUFFNFused`.
"""

from __future__ import annotations

import collections.abc
import math
from itertools import repeat
from typing import Callable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ──────────────────────────────────────────────────────────────────────────
# Tuple / list helpers — config scalars and pairs share one constructor path
# ──────────────────────────────────────────────────────────────────────────


def to_2tuple(value: int | Sequence[int] | bool | Sequence[bool] | float | Sequence[float]):
    """Repeat a scalar into a 2-tuple; pass a sequence through as a pair."""
    return _ntuple(2)(value)


def to_3tuple(value: int | Sequence[int] | bool | Sequence[bool] | float | Sequence[float]):
    """Repeat a scalar into a 3-tuple; pass a sequence through as a triple."""
    return _ntuple(3)(value)


def make_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    """Require an int or a length-2 tuple; refuse longer sequences.

    Unlike :func:`to_2tuple`, a 3-tuple is an error so patch sizes cannot
    silently truncate.

    Raises:
        ValueError: ``value`` is a tuple whose length is not 2.
        TypeError: ``value`` is neither an int nor a tuple.
    """

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expected a two-item tuple.")
        return value
    if not isinstance(value, int):
        raise TypeError("expected an int or a two-item tuple.")
    return (value, value)


def val2tuple(value, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """Normalize a scalar or sequence and repeat one item to a minimum length."""

    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if values:
        values[idx_repeat:idx_repeat] = [values[idx_repeat]] * (min_len - len(values))
    return tuple(values)


def val2list(value, repeat_time: int = 1) -> list:
    """Normalize a scalar or sequence to a mutable list."""

    return list(value) if isinstance(value, (list, tuple)) else [value] * repeat_time


def list_sum(values: list):
    """Add a non-empty list of tensors or other additive values."""

    if not values:
        raise ValueError("list_sum requires at least one value")
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    """Return symmetric padding for odd scalar or n-D kernels."""

    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(size) for size in kernel_size)
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel size {kernel_size} must be odd")
    return kernel_size // 2


def drop_path(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> Tensor:
    """Apply per-sample stochastic depth."""

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def zero_module(module: nn.Module) -> nn.Module:
    """Detach and zero all parameters in a module."""

    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


# ──────────────────────────────────────────────────────────────────────────
# Stochastic depth and channel-first normalization
# ──────────────────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Drop residual paths per sample."""

    def __init__(self, drop_prob: float | None = 0.0, scale_by_keep: bool = True) -> None:
        """``drop_prob=None`` is treated as 0 so config YAML nulls are safe."""

        super().__init__()
        self.drop_prob = 0.0 if drop_prob is None else drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        """No-op in eval; in train, scale surviving samples when requested."""

        return drop_path(x, float(self.drop_prob), self.training, self.scale_by_keep)


class LayerNorm2d(nn.Module):
    """2D layer normalization over the channel dimension (dim=1)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Allocate channel-wise γ/β broadcast over spatial axes (SAM layout)."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize over dim=1 (C) for ``[B, C, H, W]``; last-dim LN would be wrong."""

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerScale(nn.Module):
    """Learnable per-channel residual scaling."""

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        """Create a learnable residual multiplier.

        Args:
            dim: Channel width of the final tensor dimension.
            init_values: Scalar or per-channel initialization for ``gamma``.
            inplace: Multiply the input in place during ``forward``.
            device: Optional parameter device.
        """
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Copy a tensor init or fill a scalar; used after meta-device materialize."""

        if isinstance(self.init_values, Tensor):
            with torch.no_grad():
                self.gamma.copy_(self.init_values)
        else:
            nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        """Scale the final dimension of ``x`` by the learned ``gamma``."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def extra_repr(self) -> str:
        """Include init and inplace flags so module printouts distinguish LayerScales."""

        return f"dim={self.dim}, init_values={self.init_values}, inplace={self.inplace}"


# ──────────────────────────────────────────────────────────────────────────
# MLP blocks — SAM names stay distinct from the timm ViT ``Mlp``
# ──────────────────────────────────────────────────────────────────────────


class SamMLPBlock(nn.Module):
    """SAM two-layer FFN block: lin1 → act → lin2 (used in SAM transformers)."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """Build the ``lin1`` / ``lin2`` names expected by SAM checkpoints."""

        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        """Token-wise FFN; shape ``[..., embedding_dim]`` is unchanged."""

        return self.lin2(self.act(self.lin1(x)))


class SamHeadMLP(nn.Module):
    """SAM-style N-layer MLP for mask / IoU prediction heads (not ViT ``Mlp``)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        """Stack ``num_layers`` linears with activation on every layer except the last."""

        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        """Optional sigmoid is for IoU / mask logits, not the transformer FFN."""

        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Mlp(nn.Module):
    """timm-style 2-layer ViT FFN (fc1 → act → fc2); distinct from ``SamHeadMLP``."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float | tuple[float, float] = 0.0,
        bias: bool | tuple[bool, bool] = True,
        device=None,
    ) -> None:
        """Construct a two-projection feed-forward block.

        Args:
            in_features: Input width.
            hidden_features: Intermediate width; defaults to ``in_features``.
            out_features: Output width; defaults to ``in_features``.
            act_layer: Activation module factory between projections.
            drop: One probability for both dropout sites or a pair for the
                first and second sites.
            bias: One bias flag for both projections or a pair.
            device: Optional projection parameter device.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias_pair = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_pair[0], device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_pair[1], device=device)
        if drop_probs[0] == drop_probs[1]:
            self.drop = nn.Dropout(drop_probs[0])
            self.drop1 = self.drop
            self.drop2 = self.drop
        else:
            self.drop1 = nn.Dropout(drop_probs[0])
            self.drop2 = nn.Dropout(drop_probs[1])
            self.drop = self.drop1

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``fc1 → activation → dropout → fc2 → dropout``."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ──────────────────────────────────────────────────────────────────────────
# Attention, domain linear, and SAM prompt positional encodings
# ──────────────────────────────────────────────────────────────────────────


class VisionAttention(nn.Module):
    """Checkpoint-compatible ViT self-attention used by native model graphs.

    This intentionally follows the small, stable ``timm`` attention parameter
    layout (``qkv`` and ``proj``) so model implementations do not need to pull
    in a second neural-network framework for two generic layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        """Fail fast when ``dim`` is not divisible by ``num_heads`` (timm layout)."""

        super().__init__()
        if dim % num_heads:
            raise ValueError(f"attention dim {dim} must be divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N, C]`` self-attention with optional QK-norm; output matches input shape."""

        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class DomainAwareLinear(nn.Module):
    """Per-sample linear projection selected by an integer domain id.

    The layer stores one flattened weight matrix and bias per domain.  It is
    useful for cross-embodiment policies whose tensor shapes are shared while
    input/output projections remain embodiment-specific.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        """Xavier on flattened weights, zero bias — per-domain start is identity-ish."""

        super().__init__()
        if input_size <= 0 or output_size <= 0 or num_domains <= 0:
            raise ValueError("input_size, output_size, and num_domains must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_domains = int(num_domains)
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size)
        self.bias = nn.Embedding(self.num_domains, self.output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: Tensor, domain_id: Tensor) -> Tensor:
        """Apply the domain's weight; accept ``[B, D]`` or ``[B, S, D]``.

        Raises:
            ValueError: ``domain_id`` rank/range is invalid, or ``x`` width mismatches.
        """

        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        if domain_id.ndim != 1:
            raise ValueError(f"domain_id must have shape [batch], got {tuple(domain_id.shape)}")
        domain_id = domain_id.to(device=x.device, dtype=torch.long)
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(f"domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}")
        batch = int(domain_id.shape[0])
        squeeze_sequence = x.ndim == 2
        if squeeze_sequence:
            x = x.unsqueeze(1)
        if x.ndim != 3 or int(x.shape[0]) != batch or int(x.shape[-1]) != self.input_size:
            raise ValueError(
                "DomainAwareLinear expects [batch, input] or [batch, sequence, input], "
                f"got x={tuple(x.shape)}, domain_id={tuple(domain_id.shape)}"
            )
        weight = self.fc(domain_id).view(batch, self.input_size, self.output_size)
        bias = self.bias(domain_id).view(batch, 1, self.output_size)
        output = torch.matmul(x, weight) + bias
        return output.squeeze(1) if squeeze_sequence else output


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """Register a non-learned Gaussian frequency matrix (SAM prompt encoder)."""

        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Map coords in [0, 1] to ``[-1, 1]`` then a 2π Fourier feature."""

        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Dense ``[C, H, W]`` encoding for a regular grid of size ``(H, W)``."""

        h, w = size
        device: torch.device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """Encode sparse ``[B, N, 2]`` xy coords normalized by ``image_size`` (W, H)."""

        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


# ──────────────────────────────────────────────────────────────────────────
# Patch embedding — conv proj, or unshuffle + MLP for some released ViTs
# ──────────────────────────────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """2D image patch embedding: ``(B, C, H, W) -> (B, N, D)``."""

    def __init__(
        self,
        img_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """Configure convolutional image-to-token projection.

        Args:
            img_size: Nominal image height/width used to report patch count.
            patch_size: Convolution kernel/stride height and width.
            in_chans: Input channel count.
            embed_dim: Output token width.
            norm_layer: Optional normalization factory applied per token.
            flatten_embedding: Return ``(B, N, D)`` when true, otherwise
                ``(B, grid_h, grid_w, D)``.
        """
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def reset_parameters(self) -> None:
        """Uniform init scaled by fan-in of the patch convolution (checkpoint match)."""

        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x: Tensor) -> Tensor:
        """Project a divisible NCHW image batch into patch embeddings.

        Raises:
            ValueError: Runtime height or width is not divisible by patch size.
        """
        _, _, height, width = x.shape
        patch_height, patch_width = self.patch_size
        if height % patch_height:
            raise ValueError(f"Input image height {height} is not a multiple of patch height {patch_height}.")
        if width % patch_width:
            raise ValueError(f"Input image width {width} is not a multiple of patch width {patch_width}.")

        x = self.proj(x)
        height, width = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, height, width, self.embed_dim)
        return x

    def flops(self) -> float:
        """Report conv + optional norm FLOPs for the *nominal* ``img_size`` grid."""

        ho, wo = self.patches_resolution
        flops = ho * wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += ho * wo * self.embed_dim
        return flops


class PatchEmbed_Mlp(PatchEmbed):
    """Patch embedding implemented with pixel unshuffle and MLP projection."""

    def __init__(
        self,
        img_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """Replace the conv ``proj`` with unshuffle + MLP; requires a square patch.

        Raises:
            ValueError: ``patch_size`` is not square.
        """

        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten_embedding)
        patch_hw = make_2tuple(patch_size)
        if patch_hw[0] != patch_hw[1]:
            raise ValueError("PatchEmbed_Mlp requires square patch_size.")
        patch = patch_hw[0]
        self.proj = nn.Sequential(
            PixelUnshuffle(patch),
            Permute((0, 2, 3, 1)),
            Mlp(in_chans * patch**2, 4 * embed_dim, embed_dim),
            Permute((0, 3, 1, 2)),
        )


class PixelUnshuffle(nn.Module):
    """Module wrapper for ``torch.nn.functional.pixel_unshuffle``."""

    def __init__(self, downscale_factor: int) -> None:
        """Keep ``downscale_factor`` as a buffer-free attribute so meta init works."""

        super().__init__()
        self.downscale_factor = int(downscale_factor)

    def forward(self, value: Tensor) -> Tensor:
        """Unshuffle ``[B, C, H, W]``; empty tensors use ``view`` because F. fails.

        Raises:
            ValueError: empty input spatial dims are 0 or not divisible by the factor.
        """

        if value.numel() == 0:
            channels, height, width = value.shape[-3:]
            factor = self.downscale_factor
            if not height or not width or height % factor or width % factor:
                raise ValueError("empty pixel_unshuffle input must have divisible spatial dimensions.")
            return value.view(*value.shape[:-3], channels * factor**2, height // factor, width // factor)
        return F.pixel_unshuffle(value, self.downscale_factor)


class Permute(nn.Module):
    """Module wrapper around ``Tensor.permute``."""

    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        """Record axes as a tuple so the module is picklable and checkpoint-stable."""

        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self) -> str:
        """Compact ``Permute(0, 2, 3, 1)`` form used in PatchEmbed_Mlp graphs."""

        return f"Permute{self.dims}"

    def forward(self, value: Tensor) -> Tensor:
        """Apply the stored axis permutation; no shape validation (caller contract)."""

        return value.permute(*self.dims)


# ──────────────────────────────────────────────────────────────────────────
# SwiGLU feed-forward — w12/w3 names; fused variant rounds for tensor cores
# ──────────────────────────────────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """``act_layer`` / ``drop`` are accepted for MLP-factory compatibility and ignored."""

        super().__init__()
        del act_layer, drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Last-dim SwiGLU matching the ``w12`` / ``w3`` checkpoint split."""

        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


SwiGLU = SwiGLUFFN
XFORMERS_AVAILABLE = False
XFORMERS_ENABLED = False


class SwiGLUFFNFused(SwiGLU):
    """SwiGLU FFN with hidden width rounded for tensor-core-friendly matmuls."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Construct a tensor-core-aligned SwiGLU feed-forward block.

        Args:
            in_features: Input width.
            hidden_features: Requested hidden width before the SwiGLU 2/3
                adjustment and upward rounding to a multiple of eight.
            out_features: Output width; defaults to ``in_features``.
            act_layer: Accepted for MLP-constructor compatibility; unused.
            drop: Accepted for MLP-constructor compatibility; unused.
            bias: Enable biases in the fused input and output projections.
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # SwiGLU paper 2/3 width, then round up to a multiple of 8 for tensor cores.
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
            bias=bias,
        )


def _ntuple(n: int):
    """Build a parser that repeats scalars to length ``n`` and tuples sequences."""

    def parse(value):
        """Return an n-tuple; strings are treated as scalars, not iterables."""

        if isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
            return tuple(value)
        return tuple(repeat(value, n))

    return parse


__all__ = [
    "DropPath",
    "LayerNorm2d",
    "LayerScale",
    "SamHeadMLP",
    "SamMLPBlock",
    "Mlp",
    "VisionAttention",
    "PatchEmbed",
    "PatchEmbed_Mlp",
    "Permute",
    "PixelUnshuffle",
    "PositionEmbeddingRandom",
    "SwiGLU",
    "SwiGLUFFN",
    "SwiGLUFFNFused",
    "XFORMERS_AVAILABLE",
    "XFORMERS_ENABLED",
    "drop_path",
    "make_2tuple",
    "to_2tuple",
    "to_3tuple",
    "val2tuple",
    "get_same_padding",
    "list_sum",
    "val2list",
    "zero_module",
]
