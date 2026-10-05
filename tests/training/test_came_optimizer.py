from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from worldfoundry.training.optimizers import CAME
from worldfoundry.training.post_training.shared.building import build_post_training_optimizer
from worldfoundry.training.recipes import OptimizerSpec


class _Parameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float64))
        self.frozen = nn.Parameter(torch.ones(3, dtype=torch.float64), requires_grad=False)


def _build(module: nn.Module) -> CAME:
    optimizer = build_post_training_optimizer(
        OptimizerSpec(
            type="came",
            learning_rate=2.0e-3,
            weight_decay=1.0e-2,
            betas=(0.9, 0.999, 0.9999),
            epsilon=(1.0e-30, 1.0e-16),
            update_clip_threshold=1.0,
        ),
        module,
        fused=False,
        role="test",
    )
    assert isinstance(optimizer, CAME)
    return optimizer


def _set_first_gradients(module: _Parameters) -> None:
    module.matrix.grad = torch.tensor([[0.1, -0.2], [0.3, -0.4]], dtype=torch.float64)
    module.bias.grad = torch.tensor([0.25, -0.75], dtype=torch.float64)


def test_came_dispatch_owns_exact_trainable_inventory_and_official_hyperparameters() -> None:
    module = _Parameters()
    optimizer = _build(module)

    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert tuple(group["params"]) == (module.matrix, module.bias)
    assert group["lr"] == 2.0e-3
    assert group["weight_decay"] == 1.0e-2
    assert group["betas"] == (0.9, 0.999, 0.9999)
    assert group["eps"] == (1.0e-30, 1.0e-16)
    assert group["clip_threshold"] == 1.0


def test_came_one_step_matches_official_sana_update_and_state_layout() -> None:
    module = _Parameters()
    optimizer = _build(module)
    _set_first_gradients(module)

    optimizer.step()

    torch.testing.assert_close(
        module.matrix,
        torch.tensor(
            [[0.980582854201, 2.024152491287], [2.975494706746, 4.020245758239]],
            dtype=torch.float64,
        ),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        module.bias,
        torch.tensor([0.49979, -0.49979], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    matrix_state = optimizer.state[module.matrix]
    assert set(matrix_state) == {
        "step",
        "exp_avg",
        "exp_avg_sq_row",
        "exp_avg_sq_col",
        "exp_avg_res_row",
        "exp_avg_res_col",
        "RMS",
    }
    assert matrix_state["step"] == 1
    torch.testing.assert_close(
        matrix_state["exp_avg_sq_row"],
        torch.tensor([2.5e-5, 1.25e-4], dtype=torch.float64),
    )
    bias_state = optimizer.state[module.bias]
    assert set(bias_state) == {"step", "exp_avg", "exp_avg_sq", "RMS"}


def test_came_state_dict_resume_produces_the_same_next_update() -> None:
    source = _Parameters()
    source_optimizer = _build(source)
    _set_first_gradients(source)
    source_optimizer.step()
    saved = deepcopy(source_optimizer.state_dict())

    resumed = _Parameters()
    resumed.load_state_dict(source.state_dict())
    resumed_optimizer = _build(resumed)
    resumed_optimizer.load_state_dict(saved)

    source.matrix.grad = torch.tensor([[-0.4, 0.1], [0.2, -0.3]], dtype=torch.float64)
    source.bias.grad = torch.tensor([-0.5, 0.125], dtype=torch.float64)
    resumed.matrix.grad = source.matrix.grad.clone()
    resumed.bias.grad = source.bias.grad.clone()
    source_optimizer.step()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed.matrix, source.matrix, rtol=0.0, atol=0.0)
    torch.testing.assert_close(resumed.bias, source.bias, rtol=0.0, atol=0.0)
    source_state = source_optimizer.state_dict()
    resumed_state = resumed_optimizer.state_dict()
    assert source_state["param_groups"] == resumed_state["param_groups"]
    for parameter_id, state in source_state["state"].items():
        other = resumed_state["state"][parameter_id]
        assert state.keys() == other.keys()
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other[name])
            else:
                assert value == other[name]
