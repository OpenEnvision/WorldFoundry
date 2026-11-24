# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Canonical RMS arithmetic preserves model-specific affine and conditioning cast boundaries."""

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.lingbot.modeling import AdaRMSNorm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_lingbot_adarmsnorm_keeps_float32_modulation_until_output(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    model = AdaRMSNorm(8, 4).eval()
    assert set(model.state_dict()) == {"weight", "gamma.weight", "gamma.bias", "beta.weight", "beta.bias"}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(2, 3, 8).to(dtype)
    cond = torch.randn(2, 4)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + model.variance_epsilon)
    expected = (1 + model.gamma(cond).unsqueeze(1)) * (model.weight * normalized) + model.beta(cond).unsqueeze(1)
    actual = model(x, cond)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)


def test_qwen3_text_norms_keep_their_distinct_affine_cast_order() -> None:
    from worldfoundry.synthesis.action_generation.internvla_a1.qwen3_vl import Qwen3VLTextRMSNorm as A1Norm
    from worldfoundry.synthesis.action_generation.xiaomi_robotics_0.modeling_mibot import (
        Qwen3VLTextRMSNorm as XiaomiNorm,
    )

    torch.manual_seed(42)
    a1, xiaomi = A1Norm(8), XiaomiNorm(8)
    weights = {"weight": torch.randn(8)}
    a1.load_state_dict(weights, strict=True)
    xiaomi.load_state_dict(weights, strict=True)
    x = torch.randn(2, 3, 8).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    assert a1(x).dtype == torch.bfloat16
    assert xiaomi(x).dtype == torch.float32
    torch.testing.assert_close(a1(x), (weights["weight"] * normalized).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(xiaomi(x), weights["weight"] * normalized.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_lingbot_fixed_condition_norm_casts_linear_inputs(autocast: bool) -> None:
    from worldfoundry.synthesis.action_generation.lingbot_vla_v2.modeling.base import FixAdaRMSNorm

    model = FixAdaRMSNorm(8, 4).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    reference = AdaRMSNorm(8, 4).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    x, cond = torch.randn(2, 3, 8).bfloat16(), torch.randn(2, 4).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        torch.testing.assert_close(model(x, cond), reference(x, cond.float()), atol=0, rtol=0)
