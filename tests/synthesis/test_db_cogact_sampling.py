from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("cfg_scale", [1.0, 1.5])
def test_cached_action_model_uses_each_requests_ddim_steps(cfg_scale: float) -> None:
    from worldfoundry.synthesis.action_generation.db_cogact.modeling.action import ActionModel
    from worldfoundry.synthesis.action_generation.db_cogact.modeling.architecture import CogACTForCausalLM
    from worldfoundry.synthesis.action_generation.db_cogact.modeling.base import ActionOutputForCausalLM

    class Predictor(torch.nn.Module):
        inference_action = CogACTForCausalLM.inference_action
        _denorm = ActionOutputForCausalLM._denorm

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(chunk_size=2, action_dim=7)
            self.model = SimpleNamespace(
                action_head=ActionModel(
                    token_size=8,
                    model_type="DiT-S",
                    in_channels=7,
                    future_action_window_size=1,
                    past_action_window_size=0,
                ).eval()
            )

        def forward(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.ones(1, 1, 8))

    cached = Predictor()
    fresh_template = deepcopy(cached)
    previous_sampler = None
    for steps in (10, 5, 5, 10):
        options = {
            "num_ddim_steps": steps,
            "cfg_scale": cfg_scale,
            "seed": 42,
            "action_norms": {"min": -1, "max": 1},
        }
        actual = cached.inference_action(None, None, options)
        sampler = cached.model.action_head.ddim_diffusion
        assert sampler.num_timesteps == steps
        if previous_sampler is not None:
            assert (sampler is previous_sampler) == (previous_sampler.num_timesteps == steps)
        reference = deepcopy(fresh_template).inference_action(None, None, options)
        np.testing.assert_array_equal(actual, reference)
        previous_sampler = sampler
