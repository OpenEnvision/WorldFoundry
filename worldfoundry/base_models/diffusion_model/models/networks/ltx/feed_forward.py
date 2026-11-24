"""LTX DiT feed-forward: GELU-tanh expand then linear contract.

Operates on packed ``B T D`` tokens.  The Identity slot preserves the
checkpoint's three-module ``net`` key layout.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.ltx.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    """GELU-approx MLP used after self/cross-attn in each LTX block."""

    def __init__(self, dim: int, dim_out: int, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim, bias=bias)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out, bias=bias))
        self._fuse_fp8_ffn: bool = False
        self.fused_fp8_calls: int = 0
        self.fallback_fp8_calls: int = 0

    def enable_fused_fp8(self, enabled: bool = True) -> bool:
        """Enable fused GELU-tanh + rowwise FP8 quantization when ``net[2]`` is rowwise ``Float8Linear``."""
        if not enabled:
            self._fuse_fp8_ffn = False
            return False
        from worldfoundry.core.acceleration.quantization.linear import Float8Linear

        up = self.net[0]
        down = self.net[2]
        if (
            isinstance(up, GELUApprox)
            and isinstance(self.net[1], torch.nn.Identity)
            and isinstance(down, Float8Linear)
            and down.scaling == "rowwise"
        ):
            self._fuse_fp8_ffn = True
            return True
        self._fuse_fp8_ffn = False
        return False

    def _forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply GELU-tanh (+ fused rowwise FP8 quant when enabled) and down-projection."""
        from worldfoundry.core.acceleration.quantization.linear import (
            Float8Linear,
            _fp8_linear_eligible,
            _is_compiling,
        )

        down = cast(Float8Linear, self.net[2])
        eligible = (
            hidden.numel() > 0
            and down.low_precision_enabled
            and down.scaling == "rowwise"
            and down.input_scale is None
            and _fp8_linear_eligible(
                hidden,
                down.out_features,
                down._hardware_eligible,
                min_gemm_work=0.0 if down.weight is None else None,
            )
        )
        if not eligible:
            if not _is_compiling():
                self.fallback_fp8_calls += 1
            return down(F.gelu(hidden, approximate="tanh"))
        try:
            from .fused_ops import fused_ltx_gelu_tanh_rowwise_fp8
        except ImportError:
            if not _is_compiling():
                self.fallback_fp8_calls += 1
            return down(F.gelu(hidden, approximate="tanh"))

        codes, scales = fused_ltx_gelu_tanh_rowwise_fp8(
            hidden.reshape(-1, down.in_features),
            down.fp8_dtype,
        )
        result = cast(Any, down)._forward_quantized(codes, scales, hidden.shape, hidden.dtype)
        if not _is_compiling():
            self.fused_fp8_calls += 1
        return result

    def forward_from_adaln(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse AdaLN-zero ``rms_norm(x)*(1+scale)+shift`` with up-proj FP8 quantization when eligible."""
        up = self.net[0]
        if isinstance(up, GELUApprox) and isinstance(self.net[1], torch.nn.Identity):
            from worldfoundry.core.acceleration.quantization.linear import Float8Linear, _fp8_linear_eligible

            up_proj = up.proj
            if (
                isinstance(up_proj, Float8Linear)
                and up_proj.low_precision_enabled
                and up_proj.scaling == "rowwise"
                and up_proj.input_scale is None
            ):
                src = x
                scale_in = scale
                shift_in = shift
                if src.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
                    ac_dtype = torch.get_autocast_dtype("cuda")
                    if ac_dtype in (torch.float16, torch.bfloat16):
                        if src.dtype == torch.float32:
                            src = src.to(dtype=ac_dtype)
                        if scale_in.dtype == torch.float32:
                            scale_in = scale_in.to(dtype=ac_dtype)
                        if shift_in.dtype == torch.float32:
                            shift_in = shift_in.to(dtype=ac_dtype)
                if (
                    scale_in.dtype == src.dtype
                    and shift_in.dtype == src.dtype
                    and _fp8_linear_eligible(
                        src,
                        up_proj.out_features,
                        up_proj._hardware_eligible,
                        min_gemm_work=0.0 if up_proj.weight is None else None,
                    )
                ):
                    from .fused_ops import fused_ltx_rmsnorm_scale_shift_fp8

                    quant = fused_ltx_rmsnorm_scale_shift_fp8(
                        src,
                        scale_in,
                        shift_in,
                        eps=eps,
                        fp8_dtype=up_proj.fp8_dtype,
                    )
                    if quant is not None:
                        codes, scales = quant
                        hidden = cast(Any, up_proj)._forward_quantized(codes, scales, src.shape, src.dtype)
                        if self._fuse_fp8_ffn:
                            return self._forward_from_hidden(hidden)
                        return self.net[2](F.gelu(hidden, approximate="tanh"))
        from worldfoundry.core.kernels import rms_norm_scale_shift

        return self.forward(rms_norm_scale_shift(x, scale, shift, eps=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., dim]`` tokens to ``[..., dim_out]``."""
        if not self._fuse_fp8_ffn:
            return self.net(x)

        up = cast(GELUApprox, self.net[0])
        hidden = up.proj(x)
        return self._forward_from_hidden(hidden)
