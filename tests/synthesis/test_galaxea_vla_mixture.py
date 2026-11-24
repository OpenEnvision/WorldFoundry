"""Galaxea mixture attention's canonical KV-repetition binding."""

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import repeat_kv
from worldfoundry.synthesis.action_generation.galaxea_vla import mixture


def test_mixture_imports_canonical_repeat_kv() -> None:
    assert mixture.repeat_kv is repeat_kv


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("repetitions", [1, 3])
def test_mixture_repeat_kv_preserves_head_groups(dtype: torch.dtype, repetitions: int) -> None:
    states = torch.arange(80, dtype=torch.float32).reshape(2, 5, 2, 4).transpose(1, 2).to(dtype)
    repeated = mixture.repeat_kv(states, repetitions)
    expected = states.repeat_interleave(repetitions, dim=1)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    if repetitions == 1:
        assert repeated is states
