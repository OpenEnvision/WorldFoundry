from __future__ import annotations

import ast
from pathlib import Path

import worldfoundry.training as training
import worldfoundry.training.post_training as post_training
from worldfoundry.training.post_training.distillation.dmd import builder as dmd_builder
from worldfoundry.training.post_training.distillation.dmd import engine as dmd_engine
from worldfoundry.training.post_training.distillation.dmd import session as dmd_session
from worldfoundry.training.post_training.rl.algorithms.flow_dppo import (
    engine as dppo_engine,
)
from worldfoundry.training.post_training.rl.algorithms.flow_dppo import (
    session as dppo_session,
)
from worldfoundry.training.post_training.rl.algorithms.flow_grpo import (
    engine as flow_engine,
)
from worldfoundry.training.post_training.rl.algorithms.flow_grpo import (
    session as flow_session,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy import (
    builder as flow_builder,
)

_ROOT = Path(__file__).resolve().parents[2]
_REMOVED_FLAT_MODULES = (
    "advantages.py",
    "batching.py",
    "builders.py",
    "checkpoints.py",
    "contracts.py",
    "distributed.py",
    "engines.py",
    "model_adapter.py",
    "policy_losses.py",
    "sde.py",
    "session.py",
    "trajectory.py",
    "videoalign.py",
)
_CANONICAL_RUNTIME = (
    _ROOT / "worldfoundry/training/post_training/distillation/dmd/engine.py",
    _ROOT / "worldfoundry/training/post_training/distillation/dmd/session.py",
    _ROOT / "worldfoundry/training/post_training/distillation/dmd/builder.py",
    _ROOT / "worldfoundry/training/post_training/rl/algorithms/flow_grpo/engine.py",
    _ROOT / "worldfoundry/training/post_training/rl/algorithms/flow_grpo/session.py",
    _ROOT / "worldfoundry/training/post_training/rl/algorithms/flow_dppo/engine.py",
    _ROOT / "worldfoundry/training/post_training/rl/algorithms/flow_dppo/session.py",
    _ROOT / "worldfoundry/training/post_training/rl/algorithms/flow_policy/builder.py",
)
_REMOVED_RL_ALGORITHM_MODULES = (
    "builders/flow_grpo.py",
    "engines/flow_grpo.py",
    "objectives/flow_dppo.py",
    "objectives/flow_grpo.py",
    "sessions/policy_gradient.py",
    "algorithms/flow_grpo/builder.py",
)
_REMOVED_RL_RUNTIME_PACKAGES = ("builders", "engines", "sessions")


def test_flat_runtime_compatibility_modules_do_not_return() -> None:
    package = _ROOT / "worldfoundry/training/post_training"
    assert all(not (package / name).exists() for name in _REMOVED_FLAT_MODULES)


def test_rl_algorithm_modules_exist_only_under_their_algorithm_package() -> None:
    package = _ROOT / "worldfoundry/training/post_training/rl"
    assert all(not (package / name).exists() for name in _REMOVED_RL_ALGORITHM_MODULES)
    assert all(not (package / name).exists() for name in _REMOVED_RL_RUNTIME_PACKAGES)


def test_public_facade_exports_are_the_canonical_runtime_objects() -> None:
    expected = {
        "DMD_ENGINE_STATE_SCHEMA": dmd_engine.DMD_ENGINE_STATE_SCHEMA,
        "DMDTrainResult": dmd_engine.DMDTrainResult,
        "NativeDMDTrainEngine": dmd_engine.NativeDMDTrainEngine,
        "DMDRunSummary": dmd_session.DMDRunSummary,
        "NativeDMDTrainingSession": dmd_session.NativeDMDTrainingSession,
        "NativeDMDTrainingStack": dmd_builder.NativeDMDTrainingStack,
        "build_native_dmd_training_stack": dmd_builder.build_native_dmd_training_stack,
        "FLOW_GRPO_ENGINE_STATE_SCHEMA": flow_engine.FLOW_GRPO_ENGINE_STATE_SCHEMA,
        "FlowGRPOStepResult": flow_engine.FlowGRPOStepResult,
        "NativeFlowGRPOEngine": flow_engine.NativeFlowGRPOEngine,
        "FlowGRPOIterationResult": flow_session.FlowGRPOIterationResult,
        "NativeFlowGRPOTrainingSession": flow_session.NativeFlowGRPOTrainingSession,
        "FLOW_DPPO_ENGINE_STATE_SCHEMA": dppo_engine.FLOW_DPPO_ENGINE_STATE_SCHEMA,
        "FlowDPPOStepResult": dppo_engine.FlowDPPOStepResult,
        "NativeFlowDPPOEngine": dppo_engine.NativeFlowDPPOEngine,
        "FlowDPPOIterationResult": dppo_session.FlowDPPOIterationResult,
        "NativeFlowDPPOTrainingSession": dppo_session.NativeFlowDPPOTrainingSession,
        "NativeFlowPolicyTrainingStack": flow_builder.NativeFlowPolicyTrainingStack,
        "build_native_flow_policy_training_stack": flow_builder.build_native_flow_policy_training_stack,
    }
    for name, canonical in expected.items():
        assert getattr(post_training, name) is canonical
    assert training.NativeFlowDPPOEngine is dppo_engine.NativeFlowDPPOEngine


def test_canonical_runtime_does_not_import_facades_or_package_aggregators() -> None:
    forbidden_relative_aggregators = {
        "builders",
        "engines",
        "objectives",
        "rewards",
        "session",
        "shared",
        "transitions",
    }
    forbidden_absolute = {
        "worldfoundry.training.post_training",
        "worldfoundry.training.post_training.builders",
        "worldfoundry.training.post_training.engines",
        "worldfoundry.training.post_training.session",
    }
    violations: list[str] = []
    for path in _CANONICAL_RUNTIME:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module in forbidden_absolute or (node.level > 0 and node.module in forbidden_relative_aggregators):
                violations.append(f"{path.relative_to(_ROOT)}:{node.lineno}: {node.module}")
    assert violations == []


def test_checkpoint_schema_names_remain_stable_across_the_move() -> None:
    assert dmd_engine.DMD_ENGINE_STATE_SCHEMA == "worldfoundry-dmd-engine"
    assert flow_engine.FLOW_GRPO_ENGINE_STATE_SCHEMA == "worldfoundry-flow-grpo-engine"
    assert dppo_engine.FLOW_DPPO_ENGINE_STATE_SCHEMA == "worldfoundry-flow-dppo-engine"
