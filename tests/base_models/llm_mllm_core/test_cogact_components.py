# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""CogACT reuses inference-only conditioning and the canonical DDPM/DDIM sampler."""

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.cogact import LabelEmbedder
from worldfoundry.base_models.llm_mllm_core.action.cogact_diffusion import SpacedDiffusion, create_diffusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dropout_prob", [0.0, 0.1])
def test_cogact_label_embedder_preserves_cfg_checkpoint_keys(dropout_prob: float) -> None:
    model = LabelEmbedder(8, 16, dropout_prob=dropout_prob, conditions_shape=(1, 1, 8)).eval()
    keys = {"linear.weight", "linear.bias"}
    if dropout_prob:
        keys.add("uncondition")
        assert model.uncondition.shape == (1, 8)
    assert set(model.state_dict()) == keys
    assert not hasattr(model, "token_drop")
    conditions = torch.randn(2, 1, 8)
    torch.testing.assert_close(model(conditions), model.linear(conditions), atol=0, rtol=0)


def test_cogact_and_db_cogact_share_the_sampler_factory() -> None:
    from worldfoundry.synthesis.action_generation.cogact.modeling.action import create_diffusion as cogact_factory
    from worldfoundry.synthesis.action_generation.db_cogact.modeling.action import create_diffusion as db_cogact_factory

    assert cogact_factory is db_cogact_factory is create_diffusion
    sampler = create_diffusion(
        "ddim5", noise_schedule="squaredcos_cap_v2", diffusion_steps=100, learn_sigma=False, sigma_small=True
    )
    assert type(sampler) is SpacedDiffusion
    assert sampler.timestep_map == [0, 20, 40, 60, 80]
    single_step = create_diffusion(
        "ddim1", noise_schedule="squaredcos_cap_v2", diffusion_steps=100, learn_sigma=False, sigma_small=True
    )
    assert single_step.timestep_map == [50]
