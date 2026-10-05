"""MAGI-2-preview DiT building blocks (single-GPU PyTorch port).

Ported faithfully from SandAI's Apache-2.0 ``model/magi2_preview.py``. This
module holds the leaf pieces used by :mod:`preview_dit`:

* per-modality grouped linears (``CustomGroupedLinear`` / ``create_linear``),
* the multi-modality RMSNorm (``MultiModalityRMSNorm``),
* the element-wise Fourier RoPE coord embedder + ``get_coords``,
* rotary application (``apply_rotary_emb_torch``),
* the ``ModalityDispatcher`` (permute / dispatch by modality),
* the MHC hyper-connection helper (``MHCHandler``) with the pure-torch
  einsum + Sinkhorn-Knopp fallbacks, and
* the simplified ``VarlenHandler``.

All upstream Triton / FA3 fast paths are dropped: the pure-torch fallbacks are
the numerical reference. The 8-GPU cp/ep/dp machinery is collapsed to
world_size == 1 (no all-to-all, no Ulysses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .ops import swiglu7_interleaved

_FP32 = torch.float32
_BF16 = torch.bfloat16


# ============================================================
# activation
# ============================================================


class MLPActivationType(Enum):
    """Supported dense-MLP activations (preview only uses SWIGLU7)."""

    SWIGLU7 = "swiglu7"


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    # TIME is an input-only marker; it is remapped to TEXT before the dispatcher
    # so parameter shapes / num_modality stay at 3.
    TIME = 3


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    """Return the dense-MLP activation; preview/refiner only ship SwiGLU7."""

    if activation_type is MLPActivationType.SWIGLU7:
        # Dense / interleaved SwiGLU7 (gate on ``[::2]``, linear on ``[1::2]``).
        return swiglu7_interleaved
    raise ValueError(f"Unknown activation type: {activation_type}")


# ============================================================
# grouped (per-modality) linear
# ============================================================


def _maybe_gather(input: torch.Tensor, gather_ids: Optional[torch.Tensor]) -> torch.Tensor:
    if gather_ids is None:
        return input
    gather_ids = gather_ids.to(input.device)
    if gather_ids.dim() == 1:
        return input.index_select(0, gather_ids)
    return torch.gather(input, 0, gather_ids)


def _m_splits_from(
    cu_seqlens: Optional[torch.Tensor], m_splits: Optional[Union[list[int], torch.Tensor]]
) -> list[int]:
    if isinstance(m_splits, torch.Tensor):
        return [int(v) for v in m_splits.detach().cpu().tolist()]
    if m_splits is not None:
        return [int(v) for v in m_splits]
    if cu_seqlens is None:
        raise ValueError("m_splits or cu_seqlens is required for grouped linear")
    return [int(v) for v in torch.diff(cu_seqlens).detach().cpu().tolist()]


def _torch_grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    m_splits: list[int],
    gather_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ordered_input = _maybe_gather(input, gather_ids)
    chunks = list(torch.split(ordered_input, m_splits, dim=0))
    outputs = [
        F.linear(chunk, weight[i], None if bias is None else bias[i])
        for i, chunk in enumerate(chunks)
    ]
    return (
        torch.cat(outputs, dim=0)
        if outputs
        else ordered_input.new_empty((0, weight.size(1)))
    )


class GroupedLinearBase(nn.Module):
    """A bank of ``num_experts`` linears sharing one ``[E*out, in]`` weight.

    ``num_experts == 1`` is a plain ``F.linear``; otherwise the input rows are
    split by the per-modality ``m_splits`` (rows must already be modality-sorted)
    and each group goes through its own expert weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 1,
        num_patterns: int = 1,
        bias: bool = False,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.num_patterns = num_patterns
        self.weight = nn.Parameter(
            torch.empty(num_experts * out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_experts * out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        m_splits: Optional[list[int]] = None,
        gather_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = self.weight.view(self.num_experts, self.out_features, self.in_features)
        bias = (
            self.bias.view(self.num_experts, self.out_features)
            if self.bias is not None
            else None
        )
        if self.num_experts == 1:
            return F.linear(input, weight[0], None if bias is None else bias[0])
        m_splits = _m_splits_from(cu_seqlens, m_splits)
        return _torch_grouped_linear(input, weight, bias, m_splits, gather_ids)


class CustomGroupedLinear(GroupedLinearBase):
    """Grouped linear that accepts a :class:`ModalityDispatcher` in ``forward``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        num_patterns: int,
        num_layers_for_initialization: int,
        bias: bool = False,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            num_patterns=num_patterns,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input: torch.Tensor,
        modality_dispatcher: Optional[Union["ModalityDispatcher", list[int]]] = None,
        gather_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(modality_dispatcher, ModalityDispatcher):
            m_splits = modality_dispatcher.group_size_cpu
            cu_seqlens = modality_dispatcher.cu_group_sizes
        elif isinstance(modality_dispatcher, torch.Tensor):
            m_splits = modality_dispatcher.tolist()
            cu_seqlens = None
        elif isinstance(modality_dispatcher, list):
            m_splits = modality_dispatcher
            cu_seqlens = None
        else:  # base linear, no split info needed
            m_splits = None
            cu_seqlens = None
        return super().forward(
            input=input, cu_seqlens=cu_seqlens, m_splits=m_splits, gather_ids=gather_ids
        )


def create_linear(
    in_features: int,
    out_features: int,
    num_layers: int = 1,
    num_experts: int = 1,
    num_patterns: int = 1,
    bias: bool = True,
    device=None,
    dtype: torch.dtype | None = None,
) -> CustomGroupedLinear:
    """Factory for checkpoint-shaped grouped (or plain) linears."""

    return CustomGroupedLinear(
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        num_patterns=num_patterns,
        num_layers_for_initialization=num_layers,
        bias=bias,
        device=device,
        dtype=dtype,
    )


# ============================================================
# multi-modality RMSNorm (pure-torch backend)
# ============================================================


class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with an optional per-modality (grouped) weight bank.

    fp32 accumulation; weight is stored as ``(weight - 1)`` (``weight_bias=1``),
    matching the upstream ``(weight.view(P, D) + 1)`` scaling. When
    ``num_modality > 1`` the token rows (already modality-sorted) are dispatched
    per modality and each group uses its own weight chunk.
    """

    __constants__ = ["dim", "eps", "num_modality"]

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        num_modality: int = 1,
        num_patterns: int = 1,
        out_dtype: torch.dtype | None = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.num_patterns = num_patterns
        self.out_dtype = out_dtype
        self.compute_dtype = _FP32
        self.weight_bias = 1.0
        self.weight = nn.Parameter(
            torch.zeros(num_patterns * dim * num_modality, device=device, dtype=_FP32)
        )

    def forward(
        self, x: torch.Tensor, modality_dispatcher: Optional["ModalityDispatcher"] = None
    ) -> torch.Tensor:
        if self.num_modality > 1:
            assert modality_dispatcher is not None, (
                "modality_dispatcher is required for multi-modality RMSNorm"
            )
            return self._forward_multi_experts(x, modality_dispatcher, self.out_dtype)
        return self._forward_single_expert(x, self.out_dtype)

    def _forward_multi_experts(
        self,
        x: torch.Tensor,
        modality_dispatcher: "ModalityDispatcher",
        target_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)

        weight_chunked = self.weight.chunk(self.num_modality, dim=-1)
        t_list = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (
                weight_chunked[i].view(self.num_patterns, self.dim) + self.weight_bias
            )
        t = modality_dispatcher.undispatch(*t_list)

        out_dtype = target_dtype if target_dtype is not None else original_dtype
        return t.to(out_dtype)

    def _forward_single_expert(
        self, x: torch.Tensor, target_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        out_dtype = target_dtype if target_dtype is not None else original_dtype
        return (t * (self.weight.view(self.num_patterns, self.dim) + self.weight_bias)).to(
            out_dtype
        )


# ============================================================
# element-wise Fourier RoPE embedding
# ============================================================

Coords: TypeAlias = Tuple[int, int, int]


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = (
        torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(_FP32)
        / num_bands
    )
    return 1.0 / (temperature**exp)


class ElementWiseFourierEmbed(nn.Module):
    """Element-wise Fourier coordinate embedding (RoPE frequencies).

    ``coords`` is ``[L, 9]`` with columns (t, h, w, T, H, W, ref_T, ref_H, ref_W);
    returns ``[L, dim]`` where ``dim`` is the per-head rotary dim.
    """

    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = _FP32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_xyz = coords[:, :3]  # (t, h, w)
        sizes = coords[:, 3:6]  # (T, H, W)
        refs = coords[:, 6:9]  # (ref_T, ref_H, ref_W)

        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        centers = (sizes - 1) / 2
        centers[:, 0] = 0  # do not center the time dim
        coords_xyz = coords_xyz - centers

        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        sin_proj = proj.sin()
        cos_proj = proj.cos()
        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        return freq_bands(
            self.dim // 8, temperature=self.temperature, step=1, device=self.device
        ).to(self.dtype)


def get_coords(
    shape: Coords,
    ref_feat_shape: Coords,
    offset_thw: Coords = (0, 0, 0),
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = _FP32,
    time_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build ``[T*H*W, 9]`` grid coords + (orig size, ref size) metadata."""
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw
    if time_positions is None:
        time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    else:
        time_rng = time_positions.to(device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    time_grid, height_grid, width_grid = torch.meshgrid(
        time_rng, height_rng, width_rng, indexing="ij"
    )
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)

    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype
    )
    meta_expanded = meta.expand(coords_flat.size(0), -1)
    return torch.cat([coords_flat, meta_expanded], dim=-1)


# ============================================================
# rotary application
# ============================================================


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """x: (b, s, nheads, headdim); cos/sin: (s, rotary_dim/2)."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


# ============================================================
# modality dispatcher
# ============================================================


def seqlens2cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(torch.cumsum(seqlens, dim=0), (1, 0))


class ModalityDispatcher:
    """Permute / dispatch packed tokens by modality (argsort of the tags)."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int) -> None:
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        self.permuted_modality_mapping = modality_mapping.index_select(
            0, self.permute_mapping
        )

        self.group_size = torch.bincount(
            self.permuted_modality_mapping, minlength=num_modalities
        ).to(torch.int32)
        self._group_size_cpu = [int(x) for x in self.group_size.to("cpu").tolist()]

        self.cu_group_sizes = seqlens2cu_seqlens(self.group_size)

        end_indices = torch.cumsum(self.group_size, dim=0)
        start_indices = torch.cat(
            [
                torch.zeros(1, dtype=end_indices.dtype, device=end_indices.device),
                end_indices[:-1],
            ],
            dim=0,
        )
        self.expert_ranges = (
            torch.stack([start_indices, end_indices], dim=1).to(torch.int32).cpu()
        )

    @property
    def group_size_cpu(self) -> list[int]:
        return self._group_size_cpu

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *processed_groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    def _permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.permute_mapping)

    def _inv_permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.inv_permute_mapping)


# ============================================================
# MHC hyper-connections (pure-torch backend)
# ============================================================

MHCTensorTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _sigmoid_affine(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    sigmoid_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return (
        sigmoid_scale * torch.sigmoid(alpha * matmul_scale * x + bias.unsqueeze(0))
    ).to(out_dtype)


def _sinkhorn_knopp(h: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


def _sinkhorn_knopp_affine(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    num_sk_iters: int,
    sk_eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    h = alpha * matmul_scale * x.to(_FP32) + bias.unsqueeze(0).to(_FP32)
    return _sinkhorn_knopp(h, num_sk_iters, sk_eps).to(out_dtype)


def _apply_hpre(h_pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tn,tnc->tc", h_pre, x)


def _hyper_connect(
    res: torch.Tensor,
    out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    out_mstream = torch.einsum("tn,tc->tnc", h_post, out)
    mixed_res = torch.einsum("tij,tjc->tic", h_res, res)
    return mixed_res + out_mstream


class MHCHandler:
    """4-stream MHC helper: Sinkhorn-normalized mix of residual streams.

    Each preview layer stores ``phi_fused`` / ``alpha`` / ``bias`` tensors that
    this helper turns into pre, post, and residual mix matrices.  Upstream uses
    a Triton kernel; the einsum + Sinkhorn-Knopp path here is the numerical
    reference at ``cp = 1``.
    """

    def __init__(
        self,
        n_stream: int,
        hidden_size: int,
        num_sk_iters: int = 20,
        sk_eps: float = 1e-12,
        dtype: torch.dtype = _FP32,
    ) -> None:
        self.n = n_stream
        self.hidden_size = hidden_size
        self.num_sk_iters = num_sk_iters
        self.sk_eps = sk_eps
        self.dtype = dtype
        self.matmul_scale = 1.0 / math.sqrt(float(n_stream * hidden_size))

    def flatten_mstream(self, x: torch.Tensor) -> torch.Tensor:
        self._check_mstream_shape(x)
        return x.view(x.size(0), -1)

    def apply_norm_and_compute_h_(
        self,
        x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor],
        phi_fused: torch.Tensor,
    ) -> MHCTensorTuple:
        self._check_flatten_mstream_shape(x)
        h_fused = torch.matmul(norm_fn(x).to(self.dtype), phi_fused)
        h_pre, h_post, h_res = torch.split(
            h_fused, [self.n, self.n, self.n * self.n], dim=-1
        )
        return h_pre, h_post, h_res.view(-1, self.n, self.n)

    def compute_and_apply_hpre(
        self,
        x: torch.Tensor,
        abh_pre: MHCTensorTuple,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self._check_mstream_shape(x)
        alpha, bias, h_pre_ = abh_pre
        dtype = out_dtype or x.dtype
        h_pre = _sigmoid_affine(
            h_pre_, alpha, bias, self.matmul_scale, sigmoid_scale=1.0, out_dtype=dtype
        )
        return _apply_hpre(h_pre, x)

    def compute_hpost_and_hres(
        self,
        abh_post: MHCTensorTuple,
        abh_res: MHCTensorTuple,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_post, bias_post, h_post_ = abh_post
        alpha_res, bias_res, h_res_ = abh_res
        dtype = out_dtype or h_post_.dtype
        h_post = _sigmoid_affine(
            h_post_, alpha_post, bias_post, self.matmul_scale, sigmoid_scale=2.0, out_dtype=dtype
        )
        h_res = _sinkhorn_knopp_affine(
            h_res_, alpha_res, bias_res, self.matmul_scale, self.num_sk_iters, self.sk_eps, dtype
        )
        return h_post, h_res

    def apply_hpost_and_hres(
        self,
        res: torch.Tensor,
        out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        self._check_mstream_shape(res)
        self._check_sstream_shape(out)
        return _hyper_connect(res, out, h_post, h_res)

    def _check_sstream_shape(self, x: torch.Tensor) -> None:
        if x.dim() != 2 or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected single-stream shape (T, {self.hidden_size}), got {tuple(x.shape)}"
            )

    def _check_mstream_shape(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(1) != self.n or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected multi-stream shape (T, {self.n}, {self.hidden_size}), got {tuple(x.shape)}"
            )

    def _check_flatten_mstream_shape(self, x: torch.Tensor) -> None:
        expected = self.n * self.hidden_size
        if x.dim() != 2 or x.size(-1) != expected:
            raise ValueError(
                f"Expected flattened multi-stream shape (T, {expected}), got {tuple(x.shape)}"
            )


# ============================================================
# varlen handler (single-GPU: just carries cu_seqlens)
# ============================================================


@dataclass
class VarlenHandler:
    """Carries packed-doc boundaries for ``ops.attention_with_sink``.

    On single GPU only ``cu_seqlens_q`` is consumed (q/k share boundaries for
    non-causal full attention); the other fields are kept for call-site parity.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


__all__ = [
    "Coords",
    "CustomGroupedLinear",
    "ElementWiseFourierEmbed",
    "GroupedLinearBase",
    "MHCHandler",
    "MLPActivationType",
    "Modality",
    "ModalityDispatcher",
    "MultiModalityRMSNorm",
    "VarlenHandler",
    "apply_rotary_emb_torch",
    "create_activation_func",
    "create_linear",
    "freq_bands",
    "get_coords",
    "rotate_half",
    "seqlens2cu_seqlens",
]
