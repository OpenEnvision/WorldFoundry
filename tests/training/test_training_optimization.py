from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.engine import single_device  # noqa: E402
from worldfoundry.training.optimization import (  # noqa: E402
    audit_optimizer_parameters,
    build_adamw,
    trainable_parameters,
)


def test_single_device_keeps_canonical_optimizer_api_compatibility() -> None:
    assert single_device.build_adamw is build_adamw
    assert single_device.trainable_parameters is trainable_parameters
    assert single_device.audit_optimizer_parameters is audit_optimizer_parameters


def test_trainable_parameter_inventory_is_filtered_immutable_and_auditable() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Linear(3, 1),
    )
    module[1].requires_grad_(False)

    parameters = trainable_parameters(module)
    optimizer = build_adamw(parameters, learning_rate=1.0e-3, fused="auto")

    assert isinstance(parameters, tuple)
    assert parameters == tuple(module[0].parameters())
    assert audit_optimizer_parameters(optimizer, parameters, role="test") == parameters
    assert optimizer.defaults["fused"] is False


def test_optimizer_contract_rejects_empty_frozen_and_duplicate_inventories() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    frozen = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    with pytest.raises(ValueError, match="non-empty"):
        build_adamw((), learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="require gradients"):
        build_adamw((frozen,), learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="duplicates"):
        build_adamw((parameter, parameter), learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="fused"):
        build_adamw((parameter,), learning_rate=1.0e-3, fused="yes")  # type: ignore[arg-type]


def test_trainable_parameter_inventory_rejects_invalid_or_fully_frozen_modules() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        trainable_parameters(object())  # type: ignore[arg-type]

    module = torch.nn.Linear(2, 2)
    module.requires_grad_(False)
    with pytest.raises(ValueError, match="no parameters"):
        trainable_parameters(module)
