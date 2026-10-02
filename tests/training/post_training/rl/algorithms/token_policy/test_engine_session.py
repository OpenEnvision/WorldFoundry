from __future__ import annotations

import copy
import importlib

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy import (  # noqa: E402
    SEQUENCE_MEAN_TOKEN_MEAN,
    SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    TOKEN_MEAN,
    NativeTokenPolicyEngine,
    NativeTokenPolicyTrainingSession,
    PackedTokenTrajectory,
    TokenCPPOStage,
    TokenDPPOStage,
    TokenDRPOStage,
    TokenGRPOStage,
    TokenGSPOStage,
    TokenPolicyIterationResult,
    TokenReplayResult,
    TokenRolloutRequest,
)


class _ToyTokenReplay:
    def __init__(self, weight: float = 0.25) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(weight)
        self.calls: list[tuple[bool, tuple[str, ...], int]] = []

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        feature = (
            tokens.to(
                device=self.module.weight.device,
                dtype=self.module.weight.dtype,
            )
            / 10.0
        )
        logits = feature * self.module.weight.reshape(())
        return torch.nn.functional.logsigmoid(logits)

    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        self.module.train(training)
        self.calls.append((training, trajectory.sample_ids, trajectory.token_count))
        return TokenReplayResult(
            self.log_probs(trajectory.tokens),
            sampling_temperature=trajectory.sampling_temperature,
        )


def _trajectory(
    replay: _ToyTokenReplay,
    revision: str,
    *,
    rollout_log_prob_offset: float = 0.0,
) -> PackedTokenTrajectory:
    tokens = torch.tensor([1, 2, 3, 4, 5, 6])
    with torch.no_grad():
        old_log_probs = replay.log_probs(tokens).detach() + rollout_log_prob_offset
    return PackedTokenTrajectory(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision=revision,
        tokens=tokens,
        lengths=torch.tensor([2, 0, 3, 1]),
        old_log_probs=old_log_probs,
        conditioning={"prompt_ids": torch.arange(4)},
    )


def _rewards() -> torch.Tensor:
    return torch.tensor([0.0, 2.0, -1.0, 3.0])


def _fully_nonempty_trajectory(
    replay: _ToyTokenReplay,
    revision: str,
) -> PackedTokenTrajectory:
    tokens = torch.tensor([1, 2, 3, 4, 5, 6])
    with torch.no_grad():
        old_log_probs = replay.log_probs(tokens).detach()
    return PackedTokenTrajectory(
        sample_ids=("a", "b", "c", "d", "e", "f"),
        group_ids=("first", "first", "second", "second", "third", "third"),
        policy_revision=revision,
        tokens=tokens,
        lengths=torch.ones(6, dtype=torch.long),
        old_log_probs=old_log_probs,
    )


@pytest.mark.parametrize("old_log_prob_source", ["rollout", "replay"])
def test_first_update_is_ratio_one_and_anchor_stays_frozen_across_updates(
    old_log_prob_source: str,
) -> None:
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        old_log_prob_source=old_log_prob_source,
        updates_per_trajectory=2,
        replay_microbatch_size=1,
    )
    trajectory = _trajectory(
        replay,
        "policy-root",
        rollout_log_prob_offset=(0.4 if old_log_prob_source == "replay" else 0.0),
    )
    anchor_id = engine.prepare_trajectory(trajectory, _rewards())

    first = engine.train_step(anchor_id=anchor_id)
    calls_after_first = tuple(replay.calls)
    second = engine.train_step(anchor_id=anchor_id)

    torch.testing.assert_close(first.metrics["ratio_mean"], torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(first.metrics["ratio_min"], torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(first.metrics["ratio_max"], torch.tensor(1.0), rtol=0, atol=0)
    assert not torch.isclose(second.metrics["ratio_mean"], torch.tensor(1.0))
    assert first.trajectory_complete is False
    assert second.trajectory_complete is True
    assert not engine.has_active_trajectory
    expected_anchor_calls = 3 if old_log_prob_source == "replay" else 0
    assert sum(not training for training, _, _ in calls_after_first) == expected_anchor_calls
    assert sum(not training for training, _, _ in replay.calls) == expected_anchor_calls
    training_calls = [sample_ids for training, sample_ids, _ in replay.calls if training]
    assert training_calls == [("a",), ("c",), ("d",)]
    assert (first.sample_count, first.token_count, first.replay_microbatches) == (2, 2, 1)
    assert (second.sample_count, second.token_count, second.replay_microbatches) == (2, 4, 2)


def test_updates_are_disjoint_balanced_partitions_not_full_trajectory_epochs() -> None:
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        updates_per_trajectory=3,
    )
    trajectory = _fully_nonempty_trajectory(replay, "policy-root")
    anchor_id = engine.prepare_trajectory(
        trajectory,
        torch.tensor([0.0, 1.0, 0.0, 2.0, -1.0, 3.0]),
    )

    results = tuple(engine.train_step(anchor_id=anchor_id) for _ in range(3))

    training_calls = [sample_ids for training, sample_ids, _ in replay.calls if training]
    assert training_calls == [("a", "b"), ("c", "d"), ("e", "f")]
    assert [result.sample_count for result in results] == [2, 2, 2]
    assert [result.token_count for result in results] == [2, 2, 2]
    assert [result.replay_microbatches for result in results] == [1, 1, 1]
    assert not results[0].trajectory_complete
    assert results[-1].trajectory_complete


def test_partition_plan_rejects_more_updates_than_samples_and_empty_token_partitions() -> None:
    replay = _ToyTokenReplay()
    too_many = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        updates_per_trajectory=5,
    )
    with pytest.raises(ValueError, match="cannot exceed trajectory batch_size"):
        too_many.prepare_trajectory(
            _trajectory(replay, "policy-root"),
            _rewards(),
        )

    empty_first_tokens = torch.tensor([3, 4, 5, 6])
    with torch.no_grad():
        empty_first_log_probs = replay.log_probs(empty_first_tokens).detach()
    empty_first = PackedTokenTrajectory(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        tokens=empty_first_tokens,
        lengths=torch.tensor([0, 0, 3, 1]),
        old_log_probs=empty_first_log_probs,
    )
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
    )
    with pytest.raises(ValueError, match="partition.*response token"):
        engine.prepare_trajectory(empty_first, _rewards())


@pytest.mark.parametrize(
    "stage",
    [
        pytest.param(
            TokenGRPOStage(reduction=TOKEN_MEAN, clip_range=0.2),
            id="grpo-token-mean",
        ),
        pytest.param(
            TokenGRPOStage(
                reduction=SEQUENCE_MEAN_TOKEN_MEAN,
                clip_range=0.2,
            ),
            id="grpo-sequence-token-mean",
        ),
        pytest.param(
            TokenGRPOStage(
                reduction=SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
                horizon=8,
                clip_range=0.2,
            ),
            id="grpo-sequence-token-sum",
        ),
        pytest.param(TokenGSPOStage(clip_range=0.2), id="gspo-sequence-mean"),
        pytest.param(
            TokenDPPOStage(delta=1.0e-6, reduction=TOKEN_MEAN),
            id="dppo-token-mean",
        ),
        pytest.param(
            TokenDPPOStage(
                delta=1.0e-6,
                reduction=SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
                horizon=8,
            ),
            id="dppo-sequence-token-sum",
        ),
        pytest.param(
            TokenDRPOStage(epsilon=0.5, reduction=TOKEN_MEAN),
            id="drpo-token-mean",
        ),
        pytest.param(
            TokenDRPOStage(
                epsilon=0.5,
                reduction=SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
                horizon=8,
            ),
            id="drpo-sequence-token-sum",
        ),
        pytest.param(
            TokenCPPOStage(delta=1.0e-6, reduction=TOKEN_MEAN),
            id="cppo-token-mean",
        ),
        pytest.param(
            TokenCPPOStage(
                delta=1.0e-6,
                reduction=SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
                horizon=8,
            ),
            id="cppo-sequence-token-sum",
        ),
    ],
)
def test_sequence_microbatching_matches_unsplit_objective_and_reduction(
    stage,
) -> None:
    full_replay = _ToyTokenReplay()
    split_replay = _ToyTokenReplay()
    full_engine = NativeTokenPolicyEngine(
        full_replay,
        torch.optim.SGD(full_replay.module.parameters(), lr=0.05),
        algorithm=stage,
        initial_policy_revision="policy-root",
        max_grad_norm=100.0,
        updates_per_trajectory=2,
    )
    split_engine = NativeTokenPolicyEngine(
        split_replay,
        torch.optim.SGD(split_replay.module.parameters(), lr=0.05),
        algorithm=stage,
        initial_policy_revision="policy-root",
        max_grad_norm=100.0,
        replay_microbatch_size=1,
        updates_per_trajectory=2,
    )

    full_anchor = full_engine.prepare_trajectory(
        _trajectory(full_replay, "policy-root"),
        _rewards(),
    )
    split_anchor = split_engine.prepare_trajectory(
        _trajectory(split_replay, "policy-root"),
        _rewards(),
    )
    full_first = full_engine.train_step(anchor_id=full_anchor)
    split_first = split_engine.train_step(anchor_id=split_anchor)
    full_second = full_engine.train_step(anchor_id=full_anchor)
    split_second = split_engine.train_step(anchor_id=split_anchor)

    torch.testing.assert_close(full_first.loss, split_first.loss)
    torch.testing.assert_close(full_second.loss, split_second.loss)
    torch.testing.assert_close(
        full_replay.module.weight,
        split_replay.module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert full_first.replay_microbatches == 1
    assert split_first.replay_microbatches == 1
    assert full_second.replay_microbatches == 1
    assert split_second.replay_microbatches == 2
    assert sum(result.sample_count for result in (split_first, split_second)) == 4
    assert sum(result.token_count for result in (split_first, split_second)) == 6


def test_first_update_preserves_the_official_rollout_replay_gap() -> None:
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        old_log_prob_source="rollout",
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(
            replay,
            "policy-root",
            rollout_log_prob_offset=0.1,
        ),
        _rewards(),
    )

    result = engine.train_step(anchor_id=anchor_id)

    assert result.optimizer_committed is True
    assert engine.global_step == 1
    assert not engine.is_poisoned


class _WrongTemperatureReplay(_ToyTokenReplay):
    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        result = super().replay(trajectory, training=training)
        return TokenReplayResult(
            result.log_probs,
            sampling_temperature=result.sampling_temperature * 2.0,
        )


def test_replay_rejects_sampling_temperature_drift() -> None:
    replay = _WrongTemperatureReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(replay, "policy-root"),
        _rewards(),
    )

    with pytest.raises(ValueError, match="different sampling temperature"):
        engine.train_step(anchor_id=anchor_id)

    assert engine.global_step == 0
    assert not engine.is_poisoned


class _TrainingOnlyDriftReplay(_ToyTokenReplay):
    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        result = super().replay(trajectory, training=training)
        if not training:
            return result
        return TokenReplayResult(
            result.log_probs + 0.01,
            sampling_temperature=result.sampling_temperature,
        )


def test_first_update_rejects_training_mode_only_anchor_drift() -> None:
    replay = _TrainingOnlyDriftReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        old_log_prob_source="replay",
        first_update_log_ratio_tolerance=1.0e-5,
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(replay, "policy-root"),
        _rewards(),
    )

    with pytest.raises(ValueError, match="first differentiable.*must match"):
        engine.train_step(anchor_id=anchor_id)

    assert engine.global_step == 0
    assert not engine.is_poisoned


def test_engine_state_restore_reproduces_next_update_exactly() -> None:
    replay = _ToyTokenReplay()
    optimizer = torch.optim.SGD(
        replay.module.parameters(),
        lr=0.05,
        momentum=0.9,
    )
    engine = NativeTokenPolicyEngine(
        replay,
        optimizer,
        algorithm=TokenGRPOStage(
            clip_range=0.2,
            clip_schedule="linear-decay",
            clip_schedule_steps=4,
        ),
        initial_policy_revision="policy-root",
    )
    engine.train_step(
        anchor_id=engine.prepare_trajectory(
            _trajectory(replay, "policy-root"),
            _rewards(),
        )
    )
    saved_model = copy.deepcopy(replay.module.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_engine = copy.deepcopy(engine.state_dict())

    expected = engine.train_step(
        anchor_id=engine.prepare_trajectory(
            _trajectory(replay, engine.current_policy_revision),
            _rewards(),
        )
    )
    expected_weight = replay.module.weight.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    restored_replay = _ToyTokenReplay(weight=-1.0)
    restored_optimizer = torch.optim.SGD(
        restored_replay.module.parameters(),
        lr=0.05,
        momentum=0.9,
    )
    restored = NativeTokenPolicyEngine(
        restored_replay,
        restored_optimizer,
        algorithm=TokenGRPOStage(
            clip_range=0.2,
            clip_schedule="linear-decay",
            clip_schedule_steps=4,
        ),
        initial_policy_revision="policy-root",
    )
    restored_replay.module.load_state_dict(saved_model)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored.load_state_dict(saved_engine)
    actual = restored.train_step(
        anchor_id=restored.prepare_trajectory(
            _trajectory(restored_replay, restored.current_policy_revision),
            _rewards(),
        )
    )

    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.metrics["clip_range"], torch.tensor(0.175))
    torch.testing.assert_close(actual.metrics["clip_range"], expected.metrics["clip_range"], rtol=0, atol=0)
    torch.testing.assert_close(
        restored_replay.module.weight,
        expected_weight,
        rtol=0,
        atol=0,
    )
    assert restored_optimizer.state_dict() == expected_optimizer
    assert restored.state_dict() == engine.state_dict()


def test_failed_engine_state_validation_does_not_mutate_global_step() -> None:
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
    )
    engine.train_step(
        anchor_id=engine.prepare_trajectory(
            _trajectory(replay, "policy-root"),
            _rewards(),
        )
    )
    invalid = dict(engine.state_dict())
    invalid["global_step"] = 7
    invalid["current_policy_revision"] = "not-the-candidate-revision"

    with pytest.raises(ValueError, match="logical token-policy revision"):
        engine.load_state_dict(invalid)

    assert engine.global_step == 1


class _RaiseAfterOptimizerStep(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("optimizer failed after mutating parameters")


def test_optimizer_step_exception_poisons_engine_and_blocks_checkpoint() -> None:
    replay = _ToyTokenReplay()
    optimizer = _RaiseAfterOptimizerStep(replay.module.parameters(), lr=0.05)
    engine = NativeTokenPolicyEngine(
        replay,
        optimizer,
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(replay, "policy-root"),
        _rewards(),
    )

    with pytest.raises(RuntimeError, match="after mutating parameters"):
        engine.train_step(anchor_id=anchor_id)

    assert engine.is_poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.train_step(anchor_id=anchor_id)
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.load_state_dict({})


def test_all_empty_trajectory_is_a_no_op_optimizer_boundary() -> None:
    replay = _ToyTokenReplay()
    optimizer = torch.optim.SGD(replay.module.parameters(), lr=0.05)
    engine = NativeTokenPolicyEngine(
        replay,
        optimizer,
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        updates_per_trajectory=1,
    )
    trajectory = PackedTokenTrajectory(
        sample_ids=("a", "b"),
        group_ids=("prompt", "prompt"),
        policy_revision="policy-root",
        tokens=torch.empty(0, dtype=torch.long),
        lengths=torch.tensor([0, 0]),
        old_log_probs=torch.empty(0),
    )
    before = replay.module.weight.detach().clone()

    result = engine.train_step(
        anchor_id=engine.prepare_trajectory(
            trajectory,
            torch.tensor([0.0, 1.0]),
        )
    )

    assert result.optimizer_committed is False
    assert result.trajectory_complete is True
    assert result.replay_microbatches == 0
    assert result.sample_count == 2
    assert result.token_count == 0
    assert engine.global_step == 0
    assert not engine.has_active_trajectory
    assert replay.calls == []
    torch.testing.assert_close(replay.module.weight, before, rtol=0, atol=0)


class _FixedRollout:
    def __init__(self, replay: _ToyTokenReplay) -> None:
        self.replay = replay

    def rollout(self, request: TokenRolloutRequest, *, generator=None):
        del generator
        return _trajectory(self.replay, request.policy_revision)


class _FixedReward:
    reward_ids = ("quality",)

    def score(self, trajectory: PackedTokenTrajectory):
        del trajectory
        return {"quality": _rewards()}


def test_session_composes_rollout_reward_multi_update_and_progress() -> None:
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(clip_range=0.2),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
        old_log_prob_source="replay",
    )
    progress = TrainingProgress()
    events = []
    session = NativeTokenPolicyTrainingSession(
        rollout_adapter=_FixedRollout(replay),
        reward_adapter=_FixedReward(),
        scalarizer=WeightedRewardScalarizer({"quality": 1.0}),
        engine=engine,
        progress=progress,
        step_sink=events.append,
    )
    request = TokenRolloutRequest(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
    )

    result = session.train_iteration(request)

    assert isinstance(result, TokenPolicyIterationResult)
    assert len(result.updates) == 2
    assert len(events) == 2
    assert progress.optimizer_steps == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 4
    assert progress.latent_tokens_seen == 6
    assert [update.sample_count for update in result.updates] == [2, 2]
    assert [update.token_count for update in result.updates] == [2, 4]


class _RecordingParallelContext:
    rank = 0
    world_size = 3
    process_group = None

    def __init__(self) -> None:
        self.local_weights: list[int] = []

    def audit_synchronized_module(self, module, *, role: str) -> None:
        del module, role

    def audit_local_group_ownership(self, group_ids) -> None:
        del group_ids

    def scale_local_mean(self, local_mean, local_weight):
        self.local_weights.append(int(local_weight))
        return local_mean


def test_distributed_partition_rejects_mismatched_backward_call_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_module = importlib.import_module("worldfoundry.training.post_training.rl.algorithms.token_policy.engine")

    def mismatched_counts(gathered, local, *, group) -> None:
        del group
        for value in gathered:
            value.copy_(local)
        gathered[-1].add_(1)

    monkeypatch.setattr(engine_module.dist, "all_gather", mismatched_counts)
    replay = _ToyTokenReplay()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
        replay_microbatch_size=1,
        parallel_context=_RecordingParallelContext(),
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(replay, "policy-root"),
        _rewards(),
    )

    with pytest.raises(ValueError, match="same backward-call count"):
        engine.train_step(anchor_id=anchor_id)

    assert replay.calls == []
    assert engine.global_step == 0


def test_data_parallel_path_scales_by_exact_local_reduction_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_module = importlib.import_module("worldfoundry.training.post_training.rl.algorithms.token_policy.engine")

    def equal_microbatch_counts(gathered, local, *, group) -> None:
        del group
        for value in gathered:
            value.copy_(local)

    monkeypatch.setattr(engine_module.dist, "all_gather", equal_microbatch_counts)

    def maximum_anchor_error(value, *, op, group) -> None:
        del value, op, group

    monkeypatch.setattr(engine_module.dist, "all_reduce", maximum_anchor_error)
    replay = _ToyTokenReplay()
    parallel = _RecordingParallelContext()
    engine = NativeTokenPolicyEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        algorithm=TokenGRPOStage(reduction=TOKEN_MEAN),
        initial_policy_revision="policy-root",
        parallel_context=parallel,
    )

    engine.train_step(
        anchor_id=engine.prepare_trajectory(
            _trajectory(replay, "policy-root"),
            _rewards(),
        )
    )

    assert parallel.local_weights == [6]
    assert engine.state_dict()["data_parallel_size"] == 3
