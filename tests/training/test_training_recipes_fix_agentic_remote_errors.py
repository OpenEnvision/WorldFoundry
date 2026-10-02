"""Regression tests for Ray agentic sibling error surfacing (review TR-15)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.training.post_training.agentic import (
    AgenticAssistantTurn,
    AgenticRolloutRequest,
    AgenticSampleRequest,
    AgenticSampleTrajectory,
    AgenticTurn,
    AgentMessage,
    RayAgenticRolloutAdapter,
    RayAgenticSampleResult,
)
from worldfoundry.training.post_training.agentic.remote import _summarize_rollout_errors


def _request() -> AgenticRolloutRequest:
    return AgenticRolloutRequest(
        samples=tuple(
            AgenticSampleRequest(
                sample_id=f"sample-{suffix}",
                group_id="prompt",
                messages=(AgentMessage(role="user", content="add 2 and 3"),),
                conditioning={},
            )
            for suffix in ("a", "b", "c")
        ),
        policy_revision="policy-root",
        sampling_temperature=0.7,
        max_turns=1,
    )


def _trajectory(sample: AgenticSampleRequest) -> AgenticSampleTrajectory:
    return AgenticSampleTrajectory(
        request=sample,
        turns=(
            AgenticTurn(
                assistant=AgenticAssistantTurn(
                    message=AgentMessage(role="assistant", content="done"),
                    token_ids=torch.tensor([3]),
                    old_log_probs=torch.zeros(1),
                    finish_reason="stop",
                ),
            ),
        ),
        terminal_reason="stop",
    )


class _FakeRolloutGroup:
    def __init__(self, *, errors_by_position: dict[int, str]) -> None:
        self.errors_by_position = dict(errors_by_position)

    def broadcast(self, method: str, policy_revision: str, weight_revision: int):
        del method, policy_revision, weight_revision
        return ()

    def map(self, method: str, items):
        assert method == "rollout_sample"
        results = []
        for request in items:
            error = self.errors_by_position.get(request.position)
            if error is not None:
                results.append(
                    RayAgenticSampleResult(position=request.position, trajectory=None, error=error)
                )
            else:
                results.append(
                    RayAgenticSampleResult(
                        position=request.position,
                        trajectory=_trajectory(request.sample),
                    )
                )
        return tuple(results)


class _FakeRayRuntime:
    def __init__(self, *, errors_by_position: dict[int, str]) -> None:
        self.rollout_group = _FakeRolloutGroup(errors_by_position=errors_by_position)

    def sync_rollout_weights(self, module, *, revision, kind):
        del module, revision, kind
        return None


def test_partial_sibling_failures_are_recorded_on_the_adapter() -> None:
    runtime = _FakeRayRuntime(errors_by_position={2: "RuntimeError: env down"})
    proxy = RayAgenticRolloutAdapter(runtime, nn.Linear(1, 1))

    trajectory = proxy.rollout_agentic(_request())

    assert trajectory.failed_sample_ids == ("sample-c",)
    assert proxy.last_rollout_error_counts == {"RuntimeError: env down": 1}

    # A later fully-successful rollout clears the diagnostic state.
    runtime.rollout_group.errors_by_position = {}
    proxy.rollout_agentic(_request())
    assert proxy.last_rollout_error_counts == {}


def test_all_sibling_failures_raise_with_aggregated_errors() -> None:
    runtime = _FakeRayRuntime(
        errors_by_position={
            0: "RuntimeError: env down",
            1: "RuntimeError: env down",
            2: "ValueError: bad tool call",
        }
    )
    proxy = RayAgenticRolloutAdapter(runtime, nn.Linear(1, 1))

    with pytest.raises(RuntimeError) as excinfo:
        proxy.rollout_agentic(_request())
    message = str(excinfo.value)
    assert "no trainable sibling group" in message
    assert "2x RuntimeError: env down" in message
    assert "1x ValueError: bad tool call" in message
    assert proxy.last_rollout_error_counts == {
        "RuntimeError: env down": 2,
        "ValueError: bad tool call": 1,
    }


def test_error_summary_truncates_distinct_messages() -> None:
    summary = _summarize_rollout_errors({f"E{index}": 1 for index in range(8)}, limit=3)
    assert summary.endswith("... 5 more distinct errors")
