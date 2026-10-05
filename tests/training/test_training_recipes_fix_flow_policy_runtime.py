"""Regression tests for exact-type flow-policy runtime resolution (review TR-16)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from worldfoundry.training.post_training.rl.algorithms.flow_policy.runtime import (
    _FLOW_POLICY_RUNTIMES,
    resolve_flow_policy_algorithm_runtime,
)
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.algorithms import FlowGRPOAlgorithmSpec

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs/post_training"

_RECIPE_FILES = (
    "wan_1p3b_flow_grpo.yaml",
    "wan_1p3b_dance_grpo.yaml",
    "wan_1p3b_mix_grpo.yaml",
    "wan_1p3b_flow_dppo.yaml",
    "wan_1p3b_grpo_guard.yaml",
    "wan_1p3b_bagel_flow_unigrpo.yaml",
)


def _algorithm_spec(filename: str):
    pytest.importorskip("yaml")
    return PostTrainingRecipe.from_file(_CONFIG_ROOT / filename).algorithm


@pytest.mark.parametrize("filename", _RECIPE_FILES)
def test_every_registered_spec_resolves_to_its_own_runtime(filename: str) -> None:
    algorithm = _algorithm_spec(filename)
    runtime = resolve_flow_policy_algorithm_runtime(algorithm)
    assert runtime.algorithm_type is type(algorithm)


def test_subclass_specs_resolve_independently_of_registry_order() -> None:
    # DanceGRPO/MixGRPO derive from FlowGRPO; resolution must not depend on
    # the tuple ordering of _FLOW_POLICY_RUNTIMES.
    dance = _algorithm_spec("wan_1p3b_dance_grpo.yaml")
    mix = _algorithm_spec("wan_1p3b_mix_grpo.yaml")
    flow = _algorithm_spec("wan_1p3b_flow_grpo.yaml")
    assert isinstance(dance, FlowGRPOAlgorithmSpec)
    assert isinstance(mix, FlowGRPOAlgorithmSpec)
    resolved = {
        resolve_flow_policy_algorithm_runtime(spec).display_name for spec in (dance, mix, flow)
    }
    assert resolved == {"DANCE", "MixGRPO", "Flow-GRPO"}
    registered = {runtime.algorithm_type for runtime in _FLOW_POLICY_RUNTIMES}
    assert len(registered) == len(_FLOW_POLICY_RUNTIMES)


def test_unregistered_subclass_is_rejected_instead_of_running_parent_engine() -> None:
    flow = _algorithm_spec("wan_1p3b_flow_grpo.yaml")

    @dataclasses.dataclass(frozen=True, slots=True)
    class UnregisteredFlowGRPOSpec(FlowGRPOAlgorithmSpec):
        pass

    clone = UnregisteredFlowGRPOSpec(
        **{field.name: getattr(flow, field.name) for field in dataclasses.fields(flow)}
    )
    with pytest.raises(TypeError, match="unsupported native flow-policy algorithm"):
        resolve_flow_policy_algorithm_runtime(clone)
