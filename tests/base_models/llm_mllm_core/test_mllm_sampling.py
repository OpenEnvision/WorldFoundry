# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared logits filtering retains the boundary token and mutates only filtered logits."""

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.mllm.sampling import top_k_top_p_filtering

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "top_k,top_p,minimum,kept", [(0, 1.0, 1, 5), (2, 1.0, 1, 2), (0, 0.6, 1, 1), (3, 0.75, 1, 2), (0, 0.1, 2, 3)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_top_k_top_p_filtering_preserves_inplace_boundary_selection(
    top_k: int, top_p: float, minimum: int, kept: int, dtype: torch.dtype
) -> None:
    logits = torch.tensor([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=dtype)
    expected = logits.clone()
    expected[0, kept:] = -torch.inf
    expected[1, : 5 - kept] = -torch.inf
    actual = top_k_top_p_filtering(logits, top_k, top_p, min_tokens_to_keep=minimum)
    assert actual is logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_open_magvit2_uses_canonical_logits_filtering() -> None:
    from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules.transformer import (
        gpt,
    )

    assert gpt.top_k_top_p_filtering is top_k_top_p_filtering
