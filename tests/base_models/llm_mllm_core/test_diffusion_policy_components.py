# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Canonical action convolutions load through the serialized Diffusion Policy targets."""

import hydra
import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.diffusion_policy import Conv1dBlock, DictOfTensorMixin
from worldfoundry.synthesis.action_generation.diffusion_policy.runtime import (
    _rewrite_checkpoint_targets,
    _validate_safe_config,
)

pytestmark = pytest.mark.unit


def test_diffusion_policy_hydra_target_resolves_canonical_convolution() -> None:
    config = _rewrite_checkpoint_targets(
        {
            "_target_": "diffusion_policy.model.diffusion.conv1d_components.Conv1dBlock",
            "inp_channels": 4,
            "out_channels": 8,
            "kernel_size": 3,
            "n_groups": 2,
        }
    )
    _validate_safe_config(config)
    block = hydra.utils.instantiate(config)
    assert type(block) is Conv1dBlock
    assert block(torch.randn(2, 4, 9)).shape == (2, 8, 9)
    assert set(block.state_dict()) == {"block.0.weight", "block.0.bias", "block.1.weight", "block.1.bias"}


def test_diffusion_policy_normalizer_rebuilds_nested_parameter_dict() -> None:
    model = DictOfTensorMixin()
    state = {
        "params_dict.action.scale": torch.tensor([2.0, 3.0]),
        "params_dict.action.offset": torch.tensor([0.2, 0.3]),
        "params_dict.observation.input_stats.min": torch.tensor([-1.0]),
    }
    model.load_state_dict(state, strict=True)
    assert set(model.state_dict()) == set(state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    assert all(not parameter.requires_grad for parameter in model.parameters())
