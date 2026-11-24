# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law or agreed to in writing, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Frame-wise modulation and fused Triton AdaLN / residual-gate kernels for Hunyuan camera inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from worldfoundry.core.nn import DiTFinalLayer
from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="hy15-camera-triton")

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _hy_layernorm_modulate_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        out_ptr,
        rows,
        seq_len,
        token_len,
        latent_len,
        shift_row_stride,
        scale_row_stride,
        eps,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Fused affine-free FP32 LayerNorm + per-frame ``norm * (1 + scale) + shift``."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)

        mean = tl.sum(x_f, axis=0) / hidden_size
        diff = tl.where(mask, x_f - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / hidden_size
        normed = (diff * _libdevice.rsqrt(var + eps)).to(x_in.dtype).to(tl.float32)

        b_idx = row // seq_len
        s_idx = row % seq_len
        t_idx = s_idx // token_len
        mod_row = b_idx * latent_len + t_idx
        sc = (
            tl.load(scale_ptr + mod_row.to(tl.int64) * scale_row_stride + cols, mask=mask, other=0.0)
            .to(x_in.dtype)
            .to(tl.float32)
        )
        sh = (
            tl.load(shift_ptr + mod_row.to(tl.int64) * shift_row_stride + cols, mask=mask, other=0.0)
            .to(x_in.dtype)
            .to(tl.float32)
        )
        out = normed * (1.0 + sc) + sh
        tl.store(out_ptr + off, out.to(x_in.dtype), mask=mask)

    @triton.jit
    def _hy_residual_gate_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        out_ptr,
        rows,
        seq_len,
        token_len,
        latent_len,
        gate_row_stride,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        USE_TANH: tl.constexpr,
    ):
        """Fused ``x + y * gate`` with per-frame ``[B*T, C]`` gate broadcast."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        y_in = tl.load(y_ptr + off, mask=mask, other=0.0)

        b_idx = row // seq_len
        s_idx = row % seq_len
        t_idx = s_idx // token_len
        mod_row = b_idx * latent_len + t_idx
        g_raw = tl.load(gate_ptr + mod_row.to(tl.int64) * gate_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        if USE_TANH:
            g_raw = _libdevice.tanh(g_raw)
        g = g_raw.to(x_in.dtype).to(tl.float32)
        out = x_in.to(tl.float32) + (y_in.to(tl.float32) * g).to(x_in.dtype).to(tl.float32)
        tl.store(out_ptr + off, out.to(x_in.dtype), mask=mask)

    @triton.jit
    def _hy_residual_dual_gate_kernel(
        x_ptr,
        y1_ptr,
        y2_ptr,
        gate_ptr,
        out_ptr,
        rows,
        seq_len,
        token_len,
        latent_len,
        gate_row_stride,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Fused ``x + (y1 + y2) * gate`` for dual RoPE + PRoPE output projections."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        y1_in = tl.load(y1_ptr + off, mask=mask, other=0.0)
        y2_in = tl.load(y2_ptr + off, mask=mask, other=0.0)

        b_idx = row // seq_len
        s_idx = row % seq_len
        t_idx = s_idx // token_len
        mod_row = b_idx * latent_len + t_idx
        g = (
            tl.load(gate_ptr + mod_row.to(tl.int64) * gate_row_stride + cols, mask=mask, other=0.0)
            .to(x_in.dtype)
            .to(tl.float32)
        )
        y_sum = (y1_in.to(tl.float32) + y2_in.to(tl.float32)).to(x_in.dtype).to(tl.float32)
        out = x_in.to(tl.float32) + (y_sum * g).to(x_in.dtype).to(tl.float32)
        tl.store(out_ptr + off, out.to(x_in.dtype), mask=mask)

    _TRITON_MOD_AVAILABLE = True
except Exception:
    _TRITON_MOD_AVAILABLE = False


def modulate(x, shift=None, scale=None):
    """Modulate ``[B, S, C]`` tokens with per-frame ``[B*T, C]`` shift/scale via zero-copy broadcast."""
    if scale is None and shift is None:
        return x
    batch, seq_len, hidden = x.shape
    ref = scale if scale is not None else shift
    if ref.shape[0] == 1 and batch > 1:
        scale_v = None if scale is None else scale.reshape(1, 1, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(1, 1, hidden).type_as(x)
        if shift_v is None:
            return x * (1 + scale_v)
        if scale_v is None:
            return x + shift_v
        return x * (1 + scale_v) + shift_v
    if ref.shape[0] % batch != 0:
        raise ValueError(f"Modulation batch rows {ref.shape[0]} are not divisible by batch size {batch}")
    latent_length = ref.shape[0] // batch
    token_length = seq_len // latent_length
    if latent_length == seq_len:
        scale_v = None if scale is None else scale.reshape(batch, seq_len, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, seq_len, hidden).type_as(x)
        if shift_v is None:
            return x * (1 + scale_v)
        if scale_v is None:
            return x + shift_v
        return x * (1 + scale_v) + shift_v
    if latent_length == 1:
        scale_v = None if scale is None else scale.reshape(batch, 1, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, 1, hidden).type_as(x)
        if shift_v is None:
            return x * (1 + scale_v)
        if scale_v is None:
            return x + shift_v
        return x * (1 + scale_v) + shift_v
    if latent_length * token_length == seq_len:
        x_v = x.reshape(batch, latent_length, token_length, hidden)
        scale_v = None if scale is None else scale.reshape(batch, latent_length, 1, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, latent_length, 1, hidden).type_as(x)
        if shift_v is None:
            out = x_v * (1 + scale_v)
        elif scale_v is None:
            out = x_v + shift_v
        else:
            out = x_v * (1 + scale_v) + shift_v
        return out.reshape(batch, seq_len, hidden)
    raise ValueError(f"Sequence length {seq_len} is not divisible by modulation length {latent_length}")


def apply_gate(x, gate=None, tanh=False):
    """Apply a per-frame ``[B*T, C]`` residual gate to ``[B, S, C]`` tokens without repeat_interleave."""
    if gate is None:
        return x
    batch, seq_len, hidden = x.shape
    g = gate.tanh() if tanh else gate
    if gate.shape[0] == 1 and batch > 1:
        return x * g.reshape(1, 1, hidden).type_as(x)
    if gate.shape[0] % batch != 0:
        raise ValueError(f"Gate batch rows {gate.shape[0]} are not divisible by batch size {batch}")
    latent_length = gate.shape[0] // batch
    token_length = seq_len // latent_length
    if latent_length == seq_len:
        return x * g.reshape(batch, seq_len, hidden).type_as(x)
    if latent_length == 1:
        return x * g.reshape(batch, 1, hidden).type_as(x)
    if latent_length * token_length == seq_len:
        return (
            x.reshape(batch, latent_length, token_length, hidden)
            * g.reshape(batch, latent_length, 1, hidden).type_as(x)
        ).reshape(batch, seq_len, hidden)
    raise ValueError(f"Sequence length {seq_len} is not divisible by modulation length {latent_length}")


def fused_layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused affine-free ``LayerNorm(x, eps) * (1 + scale) + shift`` on CUDA inference tensors."""
    hidden = int(x.shape[-1])
    if (
        _TRITON_MOD_AVAILABLE
        and x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and shift is not None
        and scale is not None
        and shift.is_cuda
        and scale.is_cuda
        and shift.device == x.device
        and scale.device == x.device
        and shift.ndim == 2
        and scale.ndim == 2
        and shift.shape == scale.shape
        and scale.shape[1] == hidden
        and shift.stride(1) == 1
        and scale.stride(1) == 1
        and x.is_contiguous()
        and hidden <= 4096
        and int(x.shape[1]) > 0
    ):
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        if int(scale.shape[0]) % batch == 0:
            latent_len = int(scale.shape[0]) // batch
            if latent_len > 0 and seq_len % latent_len == 0:
                token_len = seq_len // latent_len
                rows = batch * seq_len
                out = torch.empty_like(x)
                block = triton.next_power_of_2(hidden)
                _hy_layernorm_modulate_kernel[(rows,)](
                    x,
                    shift,
                    scale,
                    out,
                    rows,
                    seq_len,
                    token_len,
                    latent_len,
                    int(shift.stride(0)),
                    int(scale.stride(0)),
                    float(eps),
                    hidden_size=hidden,
                    block=block,
                    num_warps=8 if block >= 1024 else 4,
                )
                return out
    return modulate(F.layer_norm(x, (hidden,), weight=None, bias=None, eps=eps), shift=shift, scale=scale)


def fused_residual_gate(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor | None,
    *,
    tanh: bool = False,
) -> torch.Tensor:
    """Fused ``x + apply_gate(y, gate, tanh=tanh)`` on CUDA inference tensors."""
    if gate is None:
        return x + y
    hidden = int(x.shape[-1])
    if (
        _TRITON_MOD_AVAILABLE
        and x.is_cuda
        and y.is_cuda
        and gate.is_cuda
        and y.device == x.device
        and gate.device == x.device
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and y.shape == x.shape
        and y.dtype == x.dtype
        and gate.ndim == 2
        and gate.shape[1] == hidden
        and gate.stride(1) == 1
        and x.is_contiguous()
        and y.is_contiguous()
        and hidden <= 4096
        and int(x.shape[1]) > 0
    ):
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        if int(gate.shape[0]) % batch == 0:
            latent_len = int(gate.shape[0]) // batch
            if latent_len > 0 and seq_len % latent_len == 0:
                token_len = seq_len // latent_len
                rows = batch * seq_len
                out = torch.empty_like(x)
                block = triton.next_power_of_2(hidden)
                _hy_residual_gate_kernel[(rows,)](
                    x,
                    y,
                    gate,
                    out,
                    rows,
                    seq_len,
                    token_len,
                    latent_len,
                    int(gate.stride(0)),
                    hidden_size=hidden,
                    block=block,
                    USE_TANH=tanh,
                    num_warps=8 if block >= 1024 else 4,
                )
                return out
    return x + apply_gate(y, gate=gate, tanh=tanh)


def fused_residual_dual_gate(
    x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Fused ``x + apply_gate(y1 + y2, gate)`` eliminating intermediate projection buffers."""
    if gate is None:
        return x + y1 + y2
    hidden = int(x.shape[-1])
    if (
        _TRITON_MOD_AVAILABLE
        and x.is_cuda
        and y1.is_cuda
        and y2.is_cuda
        and gate.is_cuda
        and y1.device == x.device
        and y2.device == x.device
        and gate.device == x.device
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and y1.shape == x.shape
        and y2.shape == x.shape
        and y1.dtype == x.dtype
        and y2.dtype == x.dtype
        and gate.ndim == 2
        and gate.shape[1] == hidden
        and gate.stride(1) == 1
        and x.is_contiguous()
        and y1.is_contiguous()
        and y2.is_contiguous()
        and hidden <= 4096
        and int(x.shape[1]) > 0
    ):
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        if int(gate.shape[0]) % batch == 0:
            latent_len = int(gate.shape[0]) // batch
            if latent_len > 0 and seq_len % latent_len == 0:
                token_len = seq_len // latent_len
                rows = batch * seq_len
                out = torch.empty_like(x)
                block = triton.next_power_of_2(hidden)
                _hy_residual_dual_gate_kernel[(rows,)](
                    x,
                    y1,
                    y2,
                    gate,
                    out,
                    rows,
                    seq_len,
                    token_len,
                    latent_len,
                    int(gate.stride(0)),
                    hidden_size=hidden,
                    block=block,
                    num_warps=8 if block >= 1024 else 4,
                )
                return out
    return x + apply_gate(y1 + y2, gate=gate)


class FinalLayer(DiTFinalLayer):
    def forward(self, value, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(fused_layernorm_modulate(value, shift=shift, scale=scale, eps=float(self.norm_final.eps)))
