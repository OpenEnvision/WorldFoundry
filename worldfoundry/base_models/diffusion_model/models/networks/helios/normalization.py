# SPDX-FileCopyrightText: 2025 The Helios Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

"""FP32 RMS normalization without optional accelerator-kernel imports."""

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from diffusers.utils import is_torch_npu_available, is_torch_version


def replace_rmsnorm_with_fp32(model: nn.Module) -> nn.Module:
    """Bind FP32 forwards in place without replacing modules or parameters."""
    patched_count = 0
    for _name, module in model.named_modules():
        if isinstance(module, (torch.nn.RMSNorm, RMSNorm)):

            def new_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                return FP32RMSNorm.forward(self, x)

            module.forward = new_forward.__get__(module, module.__class__)
            patched_count += 1
    print(f"Patched {patched_count} FP32_RMSNorm modules\n")
    return model


class FP32RMSNorm(RMSNorm):
    """Normalize and apply affine parameters in FP32, then restore the input dtype."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_torch_npu_available():
            raise ValueError("FP32RMSNorm is not available on NPU")

        if not is_torch_version(">=", "2.4"):
            raise ValueError("FP32RMSNorm is only available in PyTorch 2.4 or higher")

        original_dtype = hidden_states.dtype
        hidden_states = nn.functional.rms_norm(
            hidden_states.float(),
            normalized_shape=(hidden_states.shape[-1],),
            weight=self.weight.float(),
            eps=self.eps,
        )

        bias = getattr(self, "bias", None)
        if bias is not None:
            hidden_states = hidden_states + bias.float()

        return hidden_states.to(original_dtype)
