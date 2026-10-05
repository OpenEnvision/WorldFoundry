from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.agentic import (
    AgenticAssistantTurn,
    AgenticPrompt,
    AgenticRewardComponent,
    AgenticRolloutRequest,
    AgenticSampleRequest,
    AgenticTrajectoryRewardAdapter,
    AgentMessage,
    AgentToolCall,
    CausalLMAgenticPolicyAdapter,
    CausalLMGenerationConfig,
    HTTPAgenticRewardAdapter,
    LocalAgentTool,
    LocalToolExecutor,
    NativeAgenticRolloutAdapter,
    NativeAgenticTrainingSession,
    RayAgenticRolloutAdapter,
    RayAgenticSampleRequest,
    RayAgenticSampleResult,
    agentic_trajectory_from_packed,
    materialize_agentic_training_run,
)
from worldfoundry.training.post_training.rewards.contracts import (
    RewardRequest,
    RewardResult,
)
from worldfoundry.training.post_training.rewards.scalarization import (
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.contracts import (
    TokenReplayResult,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.engine import (
    NativeTokenPolicyEngine,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.session import (
    NativeTokenPolicyTrainingSession,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.stages import (
    TokenGRPOStage,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe


class _TinyChatCodec:
    eos_token_ids = (4,)

    def encode_prompt(self, messages, *, conditioning, device):
        del messages, conditioning
        return {
            "input_ids": torch.tensor([[2]], device=device),
            "attention_mask": torch.ones((1, 1), dtype=torch.int64, device=device),
        }

    def decode_assistant(self, token_ids):
        assert token_ids.tolist() == [3, 4]
        return AgentMessage(role="assistant", content="done")


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.full((5, 5), -40.0))
        self.dropout = nn.Dropout(p=0.75)
        with torch.no_grad():
            self.transitions[2, 3] = 40.0
            self.transitions[3, 4] = 40.0

    def forward(self, *, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        return type(
            "Output",
            (),
            {"logits": self.dropout(self.transitions[input_ids])},
        )()


def _request(*, max_turns: int = 3) -> AgenticRolloutRequest:
    return AgenticRolloutRequest(
        samples=(
            AgenticSampleRequest(
                sample_id="sample-a",
                group_id="prompt",
                messages=(AgentMessage(role="user", content="add 2 and 3"),),
                conditioning={"answer_token": 7},
            ),
            AgenticSampleRequest(
                sample_id="sample-b",
                group_id="prompt",
                messages=(AgentMessage(role="user", content="add 2 and 3"),),
                conditioning={"answer_token": 9},
            ),
        ),
        policy_revision="policy-root",
        sampling_temperature=0.7,
        max_turns=max_turns,
    )


def _three_sample_request() -> AgenticRolloutRequest:
    return AgenticRolloutRequest(
        samples=tuple(
            AgenticSampleRequest(
                sample_id=f"sample-{suffix}",
                group_id="prompt",
                messages=(AgentMessage(role="user", content="add 2 and 3"),),
                conditioning={"answer_token": token},
            )
            for suffix, token in zip(("a", "b", "c"), (7, 9, 11), strict=True)
        ),
        policy_revision="policy-root",
        sampling_temperature=0.7,
        max_turns=1,
    )


class _TwoTurnModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_turn(
        self,
        *,
        sample_id: str,
        messages: tuple[AgentMessage, ...],
        policy_revision: str,
        sampling_temperature: float,
        rollout_index: int,
        turn_index: int,
        conditioning: Mapping[str, object],
        generator: torch.Generator | None,
    ) -> AgenticAssistantTurn:
        del generator
        self.calls.append(
            {
                "sample_id": sample_id,
                "messages": messages,
                "policy_revision": policy_revision,
                "temperature": sampling_temperature,
                "rollout_index": rollout_index,
                "turn_index": turn_index,
                "conditioning": conditioning,
            }
        )
        answer_token = int(conditioning["answer_token"])
        if turn_index == 0:
            return AgenticAssistantTurn(
                message=AgentMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        AgentToolCall(
                            call_id=f"{sample_id}-addition",
                            name="add",
                            arguments={"left": 2, "right": 3},
                        ),
                    ),
                ),
                token_ids=torch.tensor([answer_token, 1]),
                old_log_probs=torch.tensor([-0.2, -0.3]),
                finish_reason="tool_calls",
            )
        assert messages[-1].role == "tool"
        assert messages[-1].content == "5"
        return AgenticAssistantTurn(
            message=AgentMessage(role="assistant", content="The answer is 5."),
            token_ids=torch.tensor([answer_token]),
            old_log_probs=torch.tensor([-0.1]),
            finish_reason="stop",
        )


def _tools() -> LocalToolExecutor:
    return LocalToolExecutor(
        (
            LocalAgentTool(
                "add",
                lambda arguments: int(arguments["left"]) + int(arguments["right"]),
            ),
        )
    )


def test_multiturn_tool_rollout_packs_every_assistant_token_and_restores_cursor() -> None:
    model = _TwoTurnModel()
    rollout = NativeAgenticRolloutAdapter(model, _tools())

    agentic = rollout.rollout_agentic(_request())
    packed = agentic.to_packed_token_trajectory()

    assert packed.sample_ids == ("sample-a", "sample-b")
    assert packed.group_ids == ("prompt", "prompt")
    assert packed.lengths.tolist() == [3, 3]
    assert packed.tokens.tolist() == [7, 1, 7, 9, 1, 9]
    torch.testing.assert_close(
        packed.old_log_probs,
        torch.tensor([-0.2, -0.3, -0.1, -0.2, -0.3, -0.1]),
    )
    assert [message.role for message in agentic.samples[0].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agentic.samples[0].turns[0].tool_results[0].tool_failed is False
    assert {int(call["rollout_index"]) for call in model.calls} == {0}

    restored_model = _TwoTurnModel()
    restored = NativeAgenticRolloutAdapter(restored_model, _tools())
    restored.load_state_dict(rollout.state_dict())
    next_trajectory = restored.rollout_agentic(_request())

    assert next_trajectory.rollout_index == 1
    assert {int(call["rollout_index"]) for call in restored_model.calls} == {1}


def test_causal_lm_adapter_generates_and_replays_exact_sampled_tokens() -> None:
    model = _TinyCausalLM()
    model.train()
    adapter = CausalLMAgenticPolicyAdapter(
        model,
        _TinyChatCodec(),
        generation=CausalLMGenerationConfig(max_new_tokens=4),
    )
    rollout = NativeAgenticRolloutAdapter(adapter, _tools())
    trajectory = rollout.rollout_agentic(_request(max_turns=1))
    packed = trajectory.to_packed_token_trajectory()

    assert packed.lengths.tolist() == [2, 2]
    assert packed.tokens.tolist() == [3, 4, 3, 4]
    replay = adapter.replay(packed, training=True)
    torch.testing.assert_close(replay.log_probs, packed.old_log_probs, rtol=0.0, atol=1.0e-7)
    assert not model.training
    replay.log_probs.mean().backward()
    assert model.transitions.grad is not None


def test_local_tool_failures_are_observable_to_the_next_policy_turn() -> None:
    executor = _tools()
    unknown = executor.execute(AgentToolCall(call_id="missing", name="lookup"))
    broken = LocalToolExecutor((LocalAgentTool("broken", lambda arguments: 1 / int(arguments["zero"])),)).execute(
        AgentToolCall(call_id="broken", name="broken", arguments={"zero": 0})
    )

    assert unknown.role == "tool" and unknown.tool_failed
    assert unknown.content == "unknown tool: lookup"
    assert broken.tool_failed
    assert broken.content.startswith("ZeroDivisionError:")


def test_agentic_reward_components_reduce_turn_signals() -> None:
    packed = NativeAgenticRolloutAdapter(_TwoTurnModel(), _tools()).rollout(_request().to_token_request())
    rewards = AgenticTrajectoryRewardAdapter(
        (
            AgenticRewardComponent(
                "tool_success",
                lambda sample: (
                    0.0 if result.tool_failed else 1.0 for turn in sample.turns for result in turn.tool_results
                ),
                reduction="sum",
            ),
            AgenticRewardComponent(
                "answer",
                lambda sample: [0.0, float(sample.messages[-1].content.endswith("5."))],
                reduction="last",
            ),
        )
    ).score(packed)

    torch.testing.assert_close(rewards["tool_success"], torch.ones(2))
    torch.testing.assert_close(rewards["answer"], torch.ones(2))


class _OneTurnModel:
    def generate_turn(
        self,
        *,
        sample_id: str,
        messages: tuple[AgentMessage, ...],
        policy_revision: str,
        sampling_temperature: float,
        rollout_index: int,
        turn_index: int,
        conditioning: Mapping[str, object],
        generator: torch.Generator | None,
    ) -> AgenticAssistantTurn:
        del messages, policy_revision, sampling_temperature, rollout_index, turn_index, generator
        token = int(conditioning["answer_token"])
        return AgenticAssistantTurn(
            message=AgentMessage(role="assistant", content=f"answer for {sample_id}"),
            token_ids=torch.tensor([token]),
            old_log_probs=torch.zeros(1),
            finish_reason="stop",
        )


class _FailingOneTurnModel(_OneTurnModel):
    def generate_turn(self, **kwargs):
        if kwargs["sample_id"] == "sample-b":
            raise RuntimeError("sample failed")
        return super().generate_turn(**kwargs)


class _FakeRolloutGroup:
    def __init__(self, *, failed_positions: tuple[int, ...] = ()) -> None:
        self.failed_positions = set(failed_positions)
        self.broadcasts: list[tuple[str, str, int]] = []
        self.batches: list[tuple[RayAgenticSampleRequest, ...]] = []

    def broadcast(self, method: str, policy_revision: str, weight_revision: int):
        self.broadcasts.append((method, policy_revision, weight_revision))
        return ()

    def map(self, method: str, items):
        assert method == "rollout_sample"
        requests = tuple(items)
        self.batches.append(requests)
        rollout = NativeAgenticRolloutAdapter(_OneTurnModel(), _tools())
        results = []
        for request in requests:
            if request.position in self.failed_positions:
                results.append(
                    RayAgenticSampleResult(
                        position=request.position,
                        trajectory=None,
                        error="sample failed",
                    )
                )
                continue
            full_request = AgenticRolloutRequest(
                samples=(request.sample,),
                policy_revision=request.policy_revision,
                sampling_temperature=request.sampling_temperature,
                max_turns=request.max_turns,
            )
            results.append(
                RayAgenticSampleResult(
                    position=request.position,
                    trajectory=rollout.rollout_sample(
                        request.sample,
                        full_request,
                        rollout_index=request.rollout_index,
                    ),
                )
            )
        return tuple(reversed(results))


class _FakeRayRuntime:
    def __init__(self, *, failed_positions: tuple[int, ...] = ()) -> None:
        self.rollout_group = _FakeRolloutGroup(failed_positions=failed_positions)
        self.sync_revisions: list[int] = []

    def sync_rollout_weights(self, module, *, revision, kind):
        assert isinstance(module, nn.Module)
        del kind
        self.sync_revisions.append(revision)
        return None


class _FakeRewardEvaluator:
    def __init__(self, *, invalid_ids: tuple[str, ...] = ()) -> None:
        self.invalid_ids = set(invalid_ids)
        self.batches: list[tuple[RewardRequest, ...]] = []

    def evaluate(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardResult, ...]:
        self.batches.append(requests)
        results = tuple(
            RewardResult(
                request_id=request.request_id,
                rollout_id=request.rollout_id,
                values={
                    "correctness": (
                        1.0 if request.request_id.endswith("a") or request.request_id.endswith("sample-0000") else -1.0
                    )
                },
                valid={"correctness": request.request_id not in self.invalid_ids},
                diagnostics={},
                latency_ms=0.0,
            )
            for request in requests
        )
        return tuple(reversed(results))


class _Closeable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_ray_agentic_proxy_syncs_every_rollout_and_excludes_failed_siblings() -> None:
    runtime = _FakeRayRuntime(failed_positions=(1,))
    proxy = RayAgenticRolloutAdapter(runtime, nn.Linear(1, 1))
    request = _three_sample_request()

    first = proxy.rollout_agentic(request)
    second = proxy.rollout_agentic(request)

    assert runtime.sync_revisions == [0, 1]
    assert runtime.rollout_group.broadcasts == [
        ("activate_policy_revision", "policy-root", 0),
        ("activate_policy_revision", "policy-root", 1),
    ]
    assert [tuple(item.position for item in batch) for batch in runtime.rollout_group.batches] == [
        (0, 1, 2),
        (0, 1, 2),
    ]
    assert tuple(sample.request.sample_id for sample in first.samples) == (
        "sample-a",
        "sample-c",
    )
    assert first.failed_sample_ids == ("sample-b",)
    assert (first.rollout_index, second.rollout_index) == (0, 1)
    restored = RayAgenticRolloutAdapter(_FakeRayRuntime(), nn.Linear(1, 1))
    restored.load_state_dict(proxy.state_dict())
    assert restored.completed_rollouts == 2
    assert restored.weight_revision == 1
    assert restored.active_policy_revision is None


def test_local_agentic_rollout_excludes_failed_sibling_and_trains_remaining_group() -> None:
    rollout = NativeAgenticRolloutAdapter(_FailingOneTurnModel(), _tools())
    reward = AgenticTrajectoryRewardAdapter(
        (
            AgenticRewardComponent(
                "correctness",
                lambda sample: 1.0 if sample.request.sample_id == "sample-a" else -1.0,
            ),
        )
    )
    replay = _ReplayAdapter()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        old_log_prob_source="rollout",
        replay_microbatch_size=1,
    )
    token_session = NativeTokenPolicyTrainingSession(
        rollout_adapter=rollout,
        reward_adapter=reward,
        scalarizer=WeightedRewardScalarizer({"correctness": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        group_size=3,
        sampling_temperature=0.7,
    )

    result = NativeAgenticTrainingSession(rollout, token_session).train_iteration(_three_sample_request())

    assert result.token_policy.trajectory.sample_ids == ("sample-a", "sample-c")
    assert result.trajectory.failed_sample_ids == ("sample-b",)
    assert replay.agentic_replay_samples == [("sample-a",), ("sample-c",)]


def test_http_agentic_rewards_batch_transcripts_and_filter_invalid_samples() -> None:
    evaluator = _FakeRewardEvaluator(invalid_ids=("sample-b",))
    reward = HTTPAgenticRewardAdapter(evaluator, reward_ids=("correctness",))
    rollout = NativeAgenticRolloutAdapter(_OneTurnModel(), _tools())
    replay = _ReplayAdapter()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        old_log_prob_source="rollout",
        replay_microbatch_size=1,
    )
    token_session = NativeTokenPolicyTrainingSession(
        rollout_adapter=rollout,
        reward_adapter=reward,
        scalarizer=WeightedRewardScalarizer({"correctness": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        group_size=3,
        sampling_temperature=0.7,
    )

    result = NativeAgenticTrainingSession(rollout, token_session).train_iteration(_three_sample_request())

    assert result.token_policy.trajectory.sample_ids == ("sample-a", "sample-c")
    assert result.trajectory.failed_sample_ids == ("sample-b",)
    torch.testing.assert_close(
        result.token_policy.rewards.scalar_rewards,
        torch.tensor([1.0, -1.0]),
    )
    assert replay.agentic_replay_samples == [("sample-a",), ("sample-c",)]
    requests = evaluator.batches[0]
    assert tuple(request.request_id for request in requests) == (
        "sample-a",
        "sample-b",
        "sample-c",
    )
    assert requests[0].artifacts["question"] == "add 2 and 3"
    assert requests[0].artifacts["prediction"] == "answer for sample-a"
    assert requests[0].artifacts["terminal_reason"] == "stop"
    assert [message["role"] for message in requests[0].artifacts["transcript"]] == [
        "user",
        "assistant",
    ]


def test_agentic_materializer_composes_ray_rollout_and_http_rewards(tmp_path) -> None:
    model = _TinyCausalLM()
    runtime = _FakeRayRuntime()
    rollout = RayAgenticRolloutAdapter(runtime, model)
    evaluator = _FakeRewardEvaluator()
    reward = HTTPAgenticRewardAdapter(evaluator, reward_ids=("correctness",))
    closeable = _Closeable()
    prompt = AgenticPrompt(
        prompt_id="prompt-a",
        messages=(AgentMessage(role="user", content="finish the sequence"),),
        conditioning={"answer_token": 3},
    )

    training_run = materialize_agentic_training_run(
        _agentic_recipe(),
        policy_module=model,
        codec=_TinyChatCodec(),
        rollout_adapter=rollout,
        reward_adapter=reward,
        closeables=(closeable,),
        prompts=(prompt,),
        output_dir=tmp_path / "remote-http",
        fused_adamw=False,
    )
    summary = training_run.run(max_iterations=1)
    training_run.close()

    assert training_run.rollout_adapter is rollout
    assert summary.completed_rollouts == 1
    assert runtime.sync_revisions == [0]
    assert len(runtime.rollout_group.batches) == 1
    assert len(evaluator.batches) == 1
    assert closeable.closed


class _ReplayAdapter:
    def __init__(self) -> None:
        self.module = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.module.weight)
        self.agentic_replay_samples: list[tuple[str, ...]] = []

    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        del training
        agentic = agentic_trajectory_from_packed(trajectory)
        self.agentic_replay_samples.append(tuple(sample.request.sample_id for sample in agentic.samples))
        values = trajectory.tokens.to(dtype=self.module.weight.dtype).unsqueeze(1)
        log_probs = self.module(values).squeeze(1)
        return TokenReplayResult(
            log_probs=log_probs,
            sampling_temperature=trajectory.sampling_temperature,
        )


def test_agentic_session_drives_the_shared_packed_token_policy_engine() -> None:
    rollout = NativeAgenticRolloutAdapter(_OneTurnModel(), _tools())
    reward = AgenticTrajectoryRewardAdapter(
        (
            AgenticRewardComponent(
                "correctness",
                lambda sample: 1.0 if sample.request.sample_id == "sample-a" else -1.0,
            ),
        )
    )
    replay = _ReplayAdapter()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        old_log_prob_source="rollout",
        replay_microbatch_size=1,
    )
    progress = TrainingProgress()
    token_session = NativeTokenPolicyTrainingSession(
        rollout_adapter=rollout,
        reward_adapter=reward,
        scalarizer=WeightedRewardScalarizer({"correctness": 1.0}),
        engine=engine,
        progress=progress,
        sampling_temperature=0.7,
    )
    session = NativeAgenticTrainingSession(rollout, token_session)

    result = session.train_iteration(_request(max_turns=1))

    assert result.trajectory.rollout_index == 0
    assert result.token_policy.trajectory.lengths.tolist() == [1, 1]
    assert len(result.token_policy.updates) == 1
    assert result.token_policy.updates[0].optimizer_committed
    assert replay.agentic_replay_samples == [("sample-a",), ("sample-b",)]
    assert progress.optimizer_steps == 1
    assert progress.samples_seen == 2
    assert progress.latent_tokens_seen == 2
    assert not torch.equal(replay.module.weight.detach(), torch.zeros_like(replay.module.weight))


def _agentic_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "agentic-resume", "output_dir": "unused-output"},
            "model": {"recipe": "tiny-causal-lm", "checkpoint": "policy-root"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused-prompts.jsonl",
                "shuffle": False,
                "shuffle_seed": 91,
                "tail_policy": "uneven",
                "options": {
                    "groups_per_batch": 1,
                    "max_new_tokens": 4,
                    "max_turns": 1,
                },
            },
            "algorithm": {
                "type": "token-grpo",
                "reward_weights": {"correctness": 1.0},
                "group_size": 2,
                "clip_range": 0.2,
                "sampling_temperature": 0.7,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "checkpoint": {"save_every_steps": 1, "async": False},
            "export": {"format": "distributed-checkpoint"},
        }
    )


def _agentic_prompts() -> tuple[AgenticPrompt, ...]:
    return (
        AgenticPrompt(
            prompt_id="prompt-a",
            messages=(AgentMessage(role="user", content="finish the sequence"),),
        ),
    )


def _correctness_reward() -> AgenticRewardComponent:
    return AgenticRewardComponent(
        "correctness",
        lambda sample: 1.0 if sample.request.sample_id.endswith("sample-0000") else -1.0,
    )


def _assert_optimizer_states_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id, values in left_state["state"].items():
        assert values.keys() == right_state["state"][parameter_id].keys()
        for name, value in values.items():
            other = right_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0.0, atol=0.0)
            else:
                assert value == other


def test_agentic_materializer_split_resume_is_exact(tmp_path) -> None:
    recipe = _agentic_recipe()
    initial_model = _TinyCausalLM()
    initial_state = {name: tensor.detach().clone() for name, tensor in initial_model.state_dict().items()}

    continuous_model = _TinyCausalLM()
    continuous_model.load_state_dict(initial_state)
    continuous = materialize_agentic_training_run(
        recipe,
        policy_module=continuous_model,
        codec=_TinyChatCodec(),
        tool_executor=_tools(),
        reward_components=(_correctness_reward(),),
        prompts=_agentic_prompts(),
        output_dir=tmp_path / "continuous",
        fused_adamw=False,
    )
    continuous_summary = continuous.run(max_iterations=2)

    split_model = _TinyCausalLM()
    split_model.load_state_dict(initial_state)
    split_output = tmp_path / "split"
    first = materialize_agentic_training_run(
        recipe,
        policy_module=split_model,
        codec=_TinyChatCodec(),
        tool_executor=_tools(),
        reward_components=(_correctness_reward(),),
        prompts=_agentic_prompts(),
        output_dir=split_output,
        fused_adamw=False,
    )
    first_summary = first.run(max_iterations=1)
    checkpoint = split_output / "checkpoints" / "step-00000001"
    assert first_summary.completed_rollouts == 1
    assert first.dataloader.completed_batches == 1

    resumed_model = _TinyCausalLM()
    resumed = materialize_agentic_training_run(
        recipe,
        policy_module=resumed_model,
        codec=_TinyChatCodec(),
        tool_executor=_tools(),
        reward_components=(_correctness_reward(),),
        prompts=_agentic_prompts(),
        output_dir=split_output,
        resume_checkpoint=checkpoint,
        fused_adamw=False,
    )
    assert resumed.engine.global_step == 1
    assert resumed.rollout_adapter.completed_rollouts == 1
    assert resumed.dataloader.completed_batches == 1
    resumed_summary = resumed.run(max_iterations=1)

    assert continuous_summary.final_optimizer_step == 2
    assert resumed_summary.final_optimizer_step == 2
    assert resumed_summary.completed_rollouts == 2
    assert resumed.dataloader.state_dict() == continuous.dataloader.state_dict()
    assert resumed.rollout_adapter.state_dict() == continuous.rollout_adapter.state_dict()
    assert resumed.engine.state_dict() == continuous.engine.state_dict()
    torch.testing.assert_close(
        resumed.checkpoint_state.objective_generator.get_state(),
        continuous.checkpoint_state.objective_generator.get_state(),
        rtol=0.0,
        atol=0.0,
    )
    for name, parameter in continuous_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            parameter,
            rtol=0.0,
            atol=0.0,
        )
    _assert_optimizer_states_equal(
        continuous.checkpoint_state.optimizer,
        resumed.checkpoint_state.optimizer,
    )
    artifact = resumed.export_policy()
    assert artifact.global_step == 2
