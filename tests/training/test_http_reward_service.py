from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.training.post_training.rewards import (
    RewardComponentScorer as ExportedRewardComponentScorer,
)
from worldfoundry.training.post_training.rewards.contracts import RewardRequest
from worldfoundry.training.post_training.rewards.http import (
    HTTPRewardEvaluator,
    NativeRewardService,
    RewardComponentOutput,
    RewardScorerRegistry,
    WorkerGroupRewardScorer,
    create_reward_service_app,
    decode_wire_value,
    encode_wire_value,
)
from worldfoundry.training.post_training.rewards.scorers import (
    AgenticCorrectnessConfig,
    AgenticCorrectnessScorer,
    AgenticToolSuccessConfig,
    AgenticToolSuccessScorer,
)
from worldfoundry.training.post_training.rl import (
    HTTPTerminalRewardAdapter as ExportedHTTPTerminalRewardAdapter,
)
from worldfoundry.training.post_training.rl import (
    WorkerGroupTerminalRewardAdapter as ExportedWorkerGroupTerminalRewardAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.contracts import (
    DiffusionNFTRewardAdapter,
    DiffusionNFTTerminalLatents,
)
from worldfoundry.training.post_training.rl.contracts import FlowTrajectory, TrajectoryRewardAdapter
from worldfoundry.training.post_training.rl.remote_rewards import (
    HTTPTerminalRewardAdapter,
    WorkerGroupTerminalRewardAdapter,
)


class _MeanTensorScorer:
    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardComponentOutput, ...]:
        return tuple(
            RewardComponentOutput(
                float(request.artifacts["latents"].float().mean()),
                diagnostics={"frames": int(request.artifacts["latents"].shape[1])},
            )
            for request in requests
        )


def test_http_reward_public_exports_resolve() -> None:
    assert ExportedHTTPTerminalRewardAdapter is HTTPTerminalRewardAdapter
    assert ExportedWorkerGroupTerminalRewardAdapter is WorkerGroupTerminalRewardAdapter
    assert ExportedRewardComponentScorer.__name__ == "RewardComponentScorer"


def _request(index: int) -> RewardRequest:
    return RewardRequest(
        request_id=f"request-{index}",
        rollout_id=f"rollout-{index}",
        prompt="a moving red cube",
        conditions={"seed": index},
        artifacts={"latents": torch.full((1, 2, 2), float(index + 1))},
        reward_ids=("mean", "length"),
        metadata={},
    )


def test_reward_http_codec_roundtrips_tensor_bytes_and_path() -> None:
    value = {
        "tensor": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "bytes": b"video",
        "path": Path("/shared/sample.mp4"),
    }
    decoded = decode_wire_value(encode_wire_value(value))
    assert torch.equal(decoded["tensor"], value["tensor"])
    assert decoded["bytes"] == b"video"
    assert decoded["path"] == value["path"]


@pytest.mark.parametrize(
    "value",
    (
        torch.tensor(3.25, dtype=torch.float32),
        torch.tensor(-7, dtype=torch.int64),
        torch.tensor(True, dtype=torch.bool),
    ),
)
def test_reward_http_codec_roundtrips_scalar_tensors(value: torch.Tensor) -> None:
    decoded = decode_wire_value(encode_wire_value(value))

    assert isinstance(decoded, torch.Tensor)
    assert decoded.shape == value.shape == torch.Size([])
    assert decoded.dtype is value.dtype
    assert torch.equal(decoded, value)


def test_native_reward_service_batches_each_component_once() -> None:
    registry = RewardScorerRegistry()
    registry.register("mean", _MeanTensorScorer())
    registry.register("length", lambda requests: [len(request.prompt) for request in requests])
    results = NativeRewardService(registry).evaluate((_request(0), _request(1)))
    assert [result.values["mean"] for result in results] == [1.0, 2.0]
    assert all(result.values["length"] == len("a moving red cube") for result in results)
    assert results[0].diagnostics["mean"]["frames"] == 2


class _BatchWorkerGroup:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def submit(self, method, value):
        return method, value

    def gather(self, refs):
        return tuple(refs)

    def map_batches(self, method, requests, *, batch_size):
        assert method == "score"
        values = tuple(requests)
        output = []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            self.batch_sizes.append(len(batch))
            output.extend(float(len(request.prompt)) for request in batch)
        return tuple(output)


def test_worker_group_reward_scorer_preserves_batched_order() -> None:
    group = _BatchWorkerGroup()
    scorer = WorkerGroupRewardScorer(group, batch_size=2)
    requests = tuple(_request(index) for index in range(5))
    assert scorer.score(requests) == tuple(float(len(request.prompt)) for request in requests)
    assert group.batch_sizes == [2, 2, 1]


def test_http_reward_evaluator_roundtrips_through_fastapi() -> None:
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    registry = RewardScorerRegistry()
    registry.register("mean", _MeanTensorScorer())
    registry.register("length", lambda requests: [len(request.prompt) for request in requests])
    app = create_reward_service_app(NativeRewardService(registry))

    with testclient.TestClient(app) as session:
        evaluator = HTTPRewardEvaluator("http://testserver", session=session)
        assert evaluator.health()["reward_ids"] == ["length", "mean"]
        results = evaluator.evaluate((_request(0), _request(1)))

    assert [result.values["mean"] for result in results] == [1.0, 2.0]
    assert all(result.valid == {"mean": True, "length": True} for result in results)


def _agentic_request(
    request_id: str,
    *,
    prediction: str,
    expected_answer: str | None = "5",
    tool_failed: bool = False,
    include_result: bool = True,
) -> RewardRequest:
    transcript = [
        {"role": "user", "content": "Use the calculator for 2+3."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": f"{request_id}-call",
                    "name": "calculator",
                    "arguments": {"expression": "2+3"},
                }
            ],
        },
    ]
    if include_result:
        transcript.append(
            {
                "role": "tool",
                "content": "5",
                "tool_call_id": f"{request_id}-call",
                "name": "calculator",
                "tool_failed": tool_failed,
            }
        )
    transcript.append({"role": "assistant", "content": prediction})
    conditions = {"required_tool": "calculator"}
    if expected_answer is not None:
        conditions["answer"] = expected_answer
    return RewardRequest(
        request_id=request_id,
        rollout_id="agentic-rollout-00000003",
        prompt="Use the calculator for 2+3.",
        conditions=conditions,
        artifacts={
            "prediction": prediction,
            "transcript": transcript,
            "terminal_reason": "stop",
        },
        reward_ids=("correctness", "tool-success"),
        metadata={"turn_count": 2},
    )


def test_agentic_reward_http_roundtrip_preserves_order_and_invalidity() -> None:
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    registry = RewardScorerRegistry()
    registry.register(
        "correctness",
        AgenticCorrectnessScorer(AgenticCorrectnessConfig(expected_answer_condition="answer")),
    )
    registry.register(
        "tool-success",
        AgenticToolSuccessScorer(AgenticToolSuccessConfig(required_tool_condition="required_tool")),
    )
    requests = (
        _agentic_request("request-c", prediction="<answer> 5 </answer>"),
        _agentic_request("request-a", prediction="4", tool_failed=True),
        _agentic_request(
            "request-b",
            prediction="5",
            expected_answer=None,
            include_result=False,
        ),
    )
    app = create_reward_service_app(NativeRewardService(registry))

    with testclient.TestClient(app) as session:
        evaluator = HTTPRewardEvaluator("http://testserver", session=session)
        assert evaluator.health()["reward_ids"] == ["correctness", "tool-success"]
        results = evaluator.evaluate(requests)

    assert tuple(result.request_id for result in results) == ("request-c", "request-a", "request-b")
    assert dict(results[0].values) == {"correctness": 1.0, "tool-success": 1.0}
    assert dict(results[0].valid) == {"correctness": True, "tool-success": True}
    assert dict(results[1].values) == {"correctness": 0.0, "tool-success": 0.0}
    assert dict(results[1].valid) == {"correctness": True, "tool-success": True}
    assert dict(results[2].valid) == {"correctness": False, "tool-success": False}
    assert "conditions.answer" in results[2].diagnostics["correctness"]["error"]
    assert "transcript" in results[2].diagnostics["tool-success"]["error"]


class _TerminalDecoder:
    def decode(self, latents, request):
        return latents[:, :, None, None, None].expand(
            -1,
            3,
            request.num_frames,
            request.height,
            request.width,
        )


def _terminal_metadata() -> dict[str, object]:
    return {
        "prompt_by_group": {"group": "a moving red cube"},
        "generation_by_group": {
            "group": {"height": 2, "width": 2, "num_frames": 1},
        },
    }


def _flow_trajectory() -> FlowTrajectory:
    return FlowTrajectory(
        sample_ids=("sample-a", "sample-b"),
        group_ids=("group", "group"),
        policy_revision="policy-7",
        latents=torch.tensor([[[0.0], [1.0]], [[0.0], [2.0]]]),
        sigmas=torch.tensor([1.0, 0.0]),
        step_indices=(0,),
        old_log_probs=torch.zeros(2, 1),
        transition_means=torch.zeros(2, 1, 1),
        transition_scales=torch.ones(2, 1, 1),
        metadata=_terminal_metadata(),
    )


def _diffusion_nft_terminal() -> DiffusionNFTTerminalLatents:
    return DiffusionNFTTerminalLatents(
        collection_id="collection-7",
        policy_revision="policy-7",
        sample_ids=("sample-a", "sample-b"),
        group_ids=("group", "group"),
        clean_latents=torch.tensor([[1.0], [2.0]]),
        transition_count=1,
        metadata=_terminal_metadata(),
    )


@pytest.mark.parametrize("terminal", [_flow_trajectory(), _diffusion_nft_terminal()])
def test_http_terminal_adapter_scores_flow_and_diffusion_nft(terminal) -> None:
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    registry = RewardScorerRegistry()
    registry.register(
        "quality",
        lambda requests: [float(request.artifacts["video"].mean()) for request in requests],
    )
    app = create_reward_service_app(NativeRewardService(registry))

    with testclient.TestClient(app) as session:
        evaluator = HTTPRewardEvaluator("http://testserver", session=session)
        adapter = HTTPTerminalRewardAdapter(
            _TerminalDecoder(),
            evaluator,
            reward_ids=("quality",),
        )
        components = adapter.score(terminal)

    torch.testing.assert_close(components["quality"], torch.tensor([1.0, 2.0]))
    assert isinstance(adapter, TrajectoryRewardAdapter)
    assert isinstance(adapter, DiffusionNFTRewardAdapter)
    assert [result.request_id for result in adapter.last_results] == ["sample-a", "sample-b"]


class _TerminalWorkerGroup(_BatchWorkerGroup):
    def map_batches(self, method, requests, *, batch_size):
        values = tuple(requests)
        output = []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            self.batch_sizes.append(len(batch))
            offset = 10.0 if method == "alignment" else 0.0
            output.extend(float(request.artifacts["video"].mean()) + offset for request in batch)
        return tuple(output)


def test_worker_group_terminal_adapter_batches_named_components() -> None:
    group = _TerminalWorkerGroup()
    adapter = WorkerGroupTerminalRewardAdapter(
        _TerminalDecoder(),
        {
            "quality": WorkerGroupRewardScorer(group, method="quality", batch_size=1),
            "alignment": WorkerGroupRewardScorer(group, method="alignment", batch_size=2),
        },
    )

    components = adapter.score(_diffusion_nft_terminal())

    torch.testing.assert_close(components["quality"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(components["alignment"], torch.tensor([11.0, 12.0]))
    assert group.batch_sizes == [1, 1, 2]
    assert isinstance(adapter, TrajectoryRewardAdapter)
    assert isinstance(adapter, DiffusionNFTRewardAdapter)
