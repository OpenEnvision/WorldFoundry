
import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")
import torch

from worldfoundry.base_models.diffusion_model.recipes.gamma_world import (
    gamma_world_causal_few_step_recipe,
)
from worldfoundry.base_models.diffusion_model.models.networks.gamma_world.dit import Attention
from worldfoundry.pipelines.gamma_world.pipeline_gamma_world import GammaWorldPipeline


def test_gamma_world_uses_native_autoregressive_recipe() -> None:
    recipe = gamma_world_causal_few_step_recipe()

    assert recipe.execution.strategy == "autoregressive-window"
    assert recipe.metadata["native_inference"] is True
    assert GammaWorldPipeline.plan()["backend"] == "worldfoundry-native-diffusion"


def test_gamma_native_attention_flattens_heads_for_output_projection() -> None:
    attention = Attention(query_dim=16, n_heads=2, head_dim=8, backend="math")

    result = attention(torch.randn(1, 4, 16))

    assert result.shape == (1, 4, 16)


def test_gamma_native_attention_accepts_context_parallel_disable_contract() -> None:
    attention = Attention(query_dim=16, n_heads=2, head_dim=8, backend="math")

    attention.set_context_parallel_group(None, None, None)

    assert attention.attn_op.device_mesh is None
