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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., dim]`` tokens to ``[..., dim_out]``."""
        if not self._fuse_fp8_ffn:
            return self.net(x)

        from worldfoundry.core.acceleration.quantization.linear import (
            Float8Linear,
            _fp8_linear_eligible,
            _is_compiling,
        )

        up = cast(GELUApprox, self.net[0])
        down = cast(Float8Linear, self.net[2])
        hidden = up.proj(x)
        eligible = (
            down.low_precision_enabled
            and down.scaling == "rowwise"
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
            from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
        except ImportError:
            if not _is_compiling():
                self.fallback_fp8_calls += 1
            return down(F.gelu(hidden, approximate="tanh"))

        codes, scales = quantize_rowwise_fp8_triton(
            hidden.reshape(-1, down.in_features),
            down.fp8_dtype,
            activation="gelu_tanh",
        )
        result = cast(Any, down)._forward_quantized(codes, scales, hidden.shape, hidden.dtype)
        if not _is_compiling():
            self.fused_fp8_calls += 1
        return result
