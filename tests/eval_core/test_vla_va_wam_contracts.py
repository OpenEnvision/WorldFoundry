from __future__ import annotations

import json
from dataclasses import is_dataclass

import pytest

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, WorldModelConfig
from worldfoundry.evaluation.tasks.embodied import (
    CAPABILITY_SESSION_CONTROL,
    CAPABILITY_VLA_ACTION_PREDICTION,
    CAPABILITY_WAM_BRANCH,
    CAPABILITY_WAM_RESET,
    CAPABILITY_WAM_WORLD_ACTION_MODELING,
    ActionSpaceKind,
    ActionSpaceSpec,
    RunnerCapabilities,
    EmbodiedGenerationSpec,
    EvaluationTrack,
    RequestKind,
    SessionControl,
    VlaVaWamRunnerContract,
)


def _round_trip(contract):
    restored = contract.__class__.from_json(contract.to_json())
    assert restored == contract
    assert restored.to_dict() == contract.to_dict()
    assert restored.stable_hash() == contract.stable_hash()
    assert hash(restored) == hash(contract)
    return restored


def test_vla_contract_requires_observations_and_declares_action_capability() -> None:
    action_space = ActionSpaceSpec(kind=ActionSpaceKind.DISCRETE, actions=("open", "close"))
    spec = EmbodiedGenerationSpec(
        track=EvaluationTrack.VLA,
        kind=RequestKind.ACTION,
        task_name="robot_pick_place",
        action_space=action_space,
        observation_keys=("rgb", "instruction"),
        output_keys=("action",),
    )

    assert is_dataclass(spec)
    assert spec.track == EvaluationTrack.VLA
    assert spec.kind == RequestKind.ACTION
    assert CAPABILITY_VLA_ACTION_PREDICTION in spec.required_capabilities
    assert spec.to_dict()["track"] == "vla"
    assert spec.to_dict()["action_space"]["kind"] == "discrete"

    request = spec.to_generation_request(sample_id="sample-001", inputs={"instruction": "open drawer"})
    assert isinstance(request, GenerationRequest)
    assert request.task_name == spec.task_name
    assert request.controls["track"] == "vla"
    assert request.output_schema == {"action": {}}


def test_vla_contract_requires_required_fields() -> None:
    action_space = ActionSpaceSpec(kind="discrete", actions=("move",))

    with pytest.raises(ValueError, match="requires task_name"):
        EmbodiedGenerationSpec(
            track="vla",
            kind="action",
            task_name="",
            action_space=action_space,
            observation_keys=("rgb",),
        )

    with pytest.raises(ValueError, match="requires observation_keys"):
        EmbodiedGenerationSpec(
            track="vla",
            kind="action",
            task_name="robot_pick_place",
            action_space=action_space,
        )


def test_wam_contract_carries_reset_branch_and_session_fields() -> None:
    session = SessionControl(
        reset_supported=True,
        branch_supported=True,
        session_id="session-a",
        reset_seed=123,
        branch_from_session_id="session-root",
        branch_from_step=7,
        max_session_steps=32,
        deterministic_reset=True,
        state_checkpoint_format="json-state-v1",
    )
    spec = EmbodiedGenerationSpec(
        track="wam",
        kind="branch",
        task_name="interactive_world_rollout",
        action_space={"kind": "continuous", "dimensions": 4, "bounds": {"low": -1, "high": 1}},
        observation_keys=("rgb", "state"),
        output_keys=("next_state", "video"),
        session_control=session,
        horizon_steps=16,
    )

    payload = spec.to_dict()
    assert payload["track"] == "wam"
    assert payload["kind"] == "branch"
    assert payload["session_control"]["reset_supported"] is True
    assert payload["session_control"]["branch_supported"] is True
    assert payload["session_control"]["session_id"] == "session-a"
    assert payload["session_control"]["branch_from_session_id"] == "session-root"
    assert payload["session_control"]["branch_from_step"] == 7
    assert CAPABILITY_WAM_WORLD_ACTION_MODELING in spec.required_capabilities
    assert CAPABILITY_WAM_RESET in spec.required_capabilities
    assert CAPABILITY_WAM_BRANCH in spec.required_capabilities
    assert CAPABILITY_SESSION_CONTROL in spec.required_capabilities


def test_json_serialization_is_stable_and_round_trips() -> None:
    capabilities = RunnerCapabilities(
        model_id="sample-vla-wam",
        tracks=("wam", "vla"),
        capabilities=(CAPABILITY_WAM_RESET, CAPABILITY_VLA_ACTION_PREDICTION, CAPABILITY_WAM_RESET),
        action_spaces=(
            ActionSpaceSpec(kind="discrete", actions=("left", "right")),
            {"kind": "pose", "dimensions": 6},
        ),
        metadata={"provider": "local"},
    )

    restored = _round_trip(capabilities)
    assert restored.capabilities == (CAPABILITY_VLA_ACTION_PREDICTION, CAPABILITY_WAM_RESET)
    assert json.loads(capabilities.to_json()) == capabilities.to_dict()
    assert capabilities.to_json() == restored.to_json()
    assert capabilities.to_json().startswith('{"action_spaces":')


def test_invalid_track_and_action_space_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported track"):
        EmbodiedGenerationSpec(
            track="not-a-track",
            kind="action",
            task_name="bad",
            action_space={"kind": "discrete", "actions": ("noop",)},
            observation_keys=("rgb",),
        )

    with pytest.raises(ValueError, match="Unsupported action_space.kind"):
        ActionSpaceSpec(kind="not-an-action-space")

    with pytest.raises(ValueError, match="requires at least one action"):
        ActionSpaceSpec(kind="discrete")


def test_benchmark_contract_accepts_minimal_vla_va_wam_runner() -> None:
    class LocalRunner:
        model_id = "sample-vla-wam"
        capabilities = {CAPABILITY_VLA_ACTION_PREDICTION}

        @classmethod
        def from_config(cls, config: WorldModelConfig) -> "LocalRunner":
            assert config.model_id == cls.model_id
            return cls()

        def describe_capabilities(self) -> RunnerCapabilities:
            return RunnerCapabilities(
                model_id=self.model_id,
                tracks=(EvaluationTrack.VLA,),
                capabilities=tuple(self.capabilities),
                action_spaces=(ActionSpaceSpec(kind="discrete", actions=("noop",)),),
            )

        def generate(self, requests):
            return [GenerationResult(sample_id=request.sample_id, model_id=self.model_id) for request in requests]

        def cleanup(self) -> None:
            self.cleaned = True

    runner = LocalRunner.from_config(WorldModelConfig(model_id="sample-vla-wam", runner="local"))
    request = GenerationRequest(sample_id="sample-001", task_name="robot_pick_place")

    assert isinstance(runner, VlaVaWamRunnerContract)
    assert runner.describe_capabilities().tracks == (EvaluationTrack.VLA,)
    assert runner.generate([request])[0].sample_id == request.sample_id
