# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA action-attention adapters retain FA3 packed batch and token axes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import action_attention_21

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")]


@pytest.mark.parametrize("return_tuple", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fa3_tensor_and_tuple_returns_preserve_every_batch_and_token(
    monkeypatch: pytest.MonkeyPatch, return_tuple: bool, dtype: torch.dtype
) -> None:
    query = torch.arange(2 * 7 * 3 * 8, device="cuda", dtype=torch.float32).reshape(2, 7, 3, 8).to(dtype)
    key = torch.ones(2, 9, 3, 8, device="cuda", dtype=dtype)
    value = torch.ones_like(key)
    packed_output = (query.flatten(0, 1).float() / 128).bfloat16()
    calls = []

    def provider(**kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        assert kwargs["q"].shape == (14, 3, 8)
        assert kwargs["k"].shape == kwargs["v"].shape == (18, 3, 8)
        assert kwargs["q"].dtype == kwargs["k"].dtype == kwargs["v"].dtype == torch.bfloat16
        torch.testing.assert_close(
            kwargs["cu_seqlens_q"], torch.tensor([0, 7, 14], device="cuda", dtype=torch.int32), rtol=0, atol=0
        )
        torch.testing.assert_close(
            kwargs["cu_seqlens_k"], torch.tensor([0, 9, 18], device="cuda", dtype=torch.int32), rtol=0, atol=0
        )
        return (packed_output, torch.zeros(3, 14, device="cuda")) if return_tuple else packed_output

    monkeypatch.setattr(action_attention_21, "FLASH_ATTN_3_AVAILABLE", True)
    monkeypatch.setattr(action_attention_21, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(
        action_attention_21, "flash_attn_interface", SimpleNamespace(flash_attn_varlen_func=provider), raising=False
    )

    actual = action_attention_21.flash_attention(query, key, value, version=3)

    assert len(calls) == 1
    assert actual.shape == query.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, packed_output.reshape(2, 7, 3, 8).to(dtype), rtol=0, atol=0)
