# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""RDT/Go1 keeps its fixed embedding dtype while reusing the SANA frequency graph."""

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import (
    TimestepEmbedder as SanaTimestepEmbedder,
)
from worldfoundry.base_models.llm_mllm_core.action.rdt import TimestepEmbedder

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("frequency_dim", [7, 8])
def test_rdt_timestep_embedder_preserves_fixed_dtype_and_checkpoint_keys(
    dtype: torch.dtype, frequency_dim: int
) -> None:
    model = TimestepEmbedder(16, frequency_dim, dtype=dtype).to(dtype).eval()
    assert model.dtype == dtype
    assert set(model.state_dict()) == {"mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"}
    timesteps = torch.tensor([0.0, 2.0, 3.25])
    expected_basis = SanaTimestepEmbedder.timestep_embedding(timesteps, frequency_dim).to(dtype)
    torch.testing.assert_close(model.timestep_embedding(timesteps, frequency_dim), expected_basis, atol=0, rtol=0)
    torch.testing.assert_close(model(timesteps), model.mlp(expected_basis), atol=0, rtol=0)


def test_rdt_runtime_imports_the_writable_dtype_owner() -> None:
    from worldfoundry.synthesis.action_generation.rdt_1b import modeling

    assert modeling.TimestepEmbedder is TimestepEmbedder


def test_rdt_timestep_embedder_accepts_go1_runtime_dtype_assignment() -> None:
    model = TimestepEmbedder(16, 8, dtype=torch.bfloat16).float().eval()
    model.dtype = torch.float32
    assert model.dtype == torch.float32
    timesteps = torch.tensor([0.0, 2.0, 3.25])
    expected_basis = SanaTimestepEmbedder.timestep_embedding(timesteps, 8)
    torch.testing.assert_close(model.timestep_embedding(timesteps, 8), expected_basis, atol=0, rtol=0)
    torch.testing.assert_close(model(timesteps), model.mlp(expected_basis), atol=0, rtol=0)
