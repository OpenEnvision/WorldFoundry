# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.gamma_world import multiagent_rope
from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.models import wan_model

pytestmark = pytest.mark.unit


def test_recam_uses_gamma_complex_rotary_helpers() -> None:
    assert wan_model.precompute_freqs_cis is multiagent_rope.precompute_freqs_cis
    assert wan_model.rope_apply is multiagent_rope.rope_apply
    empty = multiagent_rope.precompute_freqs_cis(0, 4)
    assert empty.shape == (4, 0)
    assert empty.dtype == torch.complex128


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_gamma_rotary_multiplies_in_complex128_and_restores_dtype(dtype: torch.dtype) -> None:
    x = torch.randn(2, 5, 32, dtype=dtype)
    frequencies = multiagent_rope.precompute_freqs_cis(16, 5)[:, None]
    complex_x = torch.view_as_complex(x.reshape(2, 5, 2, 8, 2).double())
    expected = torch.view_as_real(complex_x * frequencies).flatten(2).to(dtype)
    actual = multiagent_rope.rope_apply(x, frequencies, 2)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
