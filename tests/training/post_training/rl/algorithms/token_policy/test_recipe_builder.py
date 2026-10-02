from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
import worldfoundry.training.post_training.rl as rl  # noqa: E402
import worldfoundry.training.post_training.rl.algorithms as algorithms  # noqa: E402
import worldfoundry.training.recipes as recipes  # noqa: E402
from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms import token_policy  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms.token_policy import (  # noqa: E402
    NativeTokenPolicyTrainingStack,
    PackedTokenTrajectory,
    TokenCPPOStage,
    TokenDPPOStage,
    TokenDRPOStage,
    TokenGRPOStage,
    TokenGSPOStage,
    TokenReplayResult,
    TokenRolloutRequest,
)
from worldfoundry.training.recipes import (  # noqa: E402
    PostTrainingRecipe,
    TokenCPPOAlgorithmSpec,
    TokenDPPOAlgorithmSpec,
    TokenDRPOAlgorithmSpec,
    TokenGRPOAlgorithmSpec,
    TokenGSPOAlgorithmSpec,
)

_SPEC_TYPES = {
    "token-grpo": TokenGRPOAlgorithmSpec,
    "token-gspo": TokenGSPOAlgorithmSpec,
    "token-dppo": TokenDPPOAlgorithmSpec,
    "token-drpo": TokenDRPOAlgorithmSpec,
    "token-cppo": TokenCPPOAlgorithmSpec,
}
_STAGE_TYPES = {
    "token-grpo": TokenGRPOStage,
    "token-gspo": TokenGSPOStage,
    "token-dppo": TokenDPPOStage,
    "token-drpo": TokenDRPOStage,
    "token-cppo": TokenCPPOStage,
}
_STAGE_FIELDS = {
    "token-grpo": (
        "clip_range",
        "clip_range_high",
        "clip_schedule",
        "clip_schedule_steps",
        "reduction",
        "horizon",
    ),
    "token-gspo": (
        "clip_range",
        "clip_range_high",
        "clip_schedule",
        "clip_schedule_steps",
    ),
    "token-dppo": ("delta", "reduction", "horizon"),
    "token-drpo": ("epsilon", "mu_weighted", "reduction", "horizon"),
    "token-cppo": ("delta", "w_min", "delta_b", "reduction", "horizon"),
}


def _algorithm_mapping(algorithm_type: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": algorithm_type,
        "updates_per_trajectory": 2,
        "group_size": 4,
        "old_log_prob_source": "replay",
        "advantage_epsilon": 1.0e-7,
        "advantage_clip_max": 3.0,
        "advantage_normalization": "group-mean-global-sample-std",
        "sampling_temperature": 0.7,
        "replay_microbatch_size": 1,
        "first_update_log_ratio_tolerance": 2.0e-5,
        "reward_weights": {"correctness": 1.0, "format": 0.25},
    }
    if algorithm_type == "token-grpo":
        payload["old_log_prob_source"] = "rollout"
        payload.update(
            {
                "clip_range": 0.2,
                "clip_range_high": 0.3,
                "clip_schedule": "linear_decay",
                "clip_schedule_steps": 20,
                "reduction": "seq-mean-token-mean",
                "horizon": 32,
            }
        )
    elif algorithm_type == "token-gspo":
        payload.update(
            {
                "clip_range": 0.01,
                "clip_range_high": 0.02,
                "clip_schedule": "cosine_decay",
                "clip_schedule_steps": 20,
            }
        )
    elif algorithm_type == "token-dppo":
        payload.update(
            {
                "delta": 0.1,
                "reduction": "seq-mean-token-sum-norm",
                "horizon": 32,
            }
        )
    elif algorithm_type == "token-drpo":
        payload.update(
            {
                "epsilon": 9.0,
                "mu_weighted": False,
                "reduction": "seq-mean-token-sum-norm",
                "horizon": 32,
            }
        )
    elif algorithm_type == "token-cppo":
        payload.update(
            {
                "delta": 0.18,
                "w_min": 0.75,
                "delta_b": 0.03,
                "reduction": "seq-mean-token-sum-norm",
                "horizon": 32,
            }
        )
    else:
        raise AssertionError(algorithm_type)
    return payload


def _recipe_mapping(algorithm_type: str) -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {
            "id": f"{algorithm_type}-test",
            "output_dir": f"runs/{algorithm_type}-test",
        },
        "model": {"recipe": "qwen3-4b", "checkpoint": "policy"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/reasoning.jsonl", "shuffle": False},
        "algorithm": _algorithm_mapping(algorithm_type),
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.02,
            "max_grad_norm": 0.75,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


class _ReplayAdapter:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(0.25)

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        feature = tokens.to(dtype=self.module.weight.dtype) / 10.0
        return torch.nn.functional.logsigmoid(feature * self.module.weight.reshape(()))

    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        self.module.train(training)
        return TokenReplayResult(
            self.log_probs(trajectory.tokens),
            sampling_temperature=trajectory.sampling_temperature,
        )


class _RolloutAdapter:
    def __init__(self, replay: _ReplayAdapter) -> None:
        self.replay = replay

    def rollout(self, request: TokenRolloutRequest, *, generator=None):
        del generator
        tokens = torch.tensor([1, 2, 3, 4, 5, 6])
        with torch.no_grad():
            old_log_probs = self.replay.log_probs(tokens).detach() + 0.25
        return PackedTokenTrajectory(
            sample_ids=request.sample_ids,
            group_ids=request.group_ids,
            policy_revision=request.policy_revision,
            tokens=tokens,
            lengths=torch.tensor([2, 0, 3, 1]),
            old_log_probs=old_log_probs,
            sampling_temperature=request.sampling_temperature,
            conditioning=request.conditioning,
        )


class _RewardAdapter:
    reward_ids = ("correctness", "format")

    def score(self, trajectory: PackedTokenTrajectory):
        del trajectory
        return {
            "correctness": torch.tensor([0.0, 2.0, -1.0, 3.0]),
            "format": torch.tensor([1.0, 0.0, 0.5, 1.0]),
        }


@pytest.mark.parametrize("algorithm_type", tuple(_SPEC_TYPES))
def test_token_policy_recipes_are_strict_and_round_trip(
    algorithm_type: str,
) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(algorithm_type))

    assert isinstance(recipe.algorithm, _SPEC_TYPES[algorithm_type])
    assert recipe.algorithm.updates_per_trajectory == 2
    assert recipe.algorithm.replay_microbatch_size == 1
    assert recipe.algorithm.first_update_log_ratio_tolerance == 2.0e-5
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe


def test_token_policy_recipe_rejects_unknown_and_unused_fields() -> None:
    unknown = _recipe_mapping("token-grpo")
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    invalid_reduction = _recipe_mapping("token-dppo")
    invalid_reduction["algorithm"]["reduction"] = "seq-mean-token-mean"
    with pytest.raises(ValueError, match="reduction must be one of"):
        PostTrainingRecipe.from_mapping(invalid_reduction)

    fake_score = _recipe_mapping("token-drpo")
    fake_score["fake_score_optimizer"] = {
        "type": "adamw",
        "learning_rate": 0.001,
    }
    with pytest.raises(ValueError, match="cannot configure fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(fake_score)

    replay_grpo = _recipe_mapping("token-grpo")
    replay_grpo["algorithm"]["old_log_prob_source"] = "replay"
    with pytest.raises(ValueError, match="requires rollout old log probabilities"):
        PostTrainingRecipe.from_mapping(replay_grpo)

    missing_horizon = _recipe_mapping("token-gspo")
    missing_horizon["algorithm"].pop("clip_schedule_steps")
    with pytest.raises(TypeError, match="requires integer clip_schedule_steps"):
        PostTrainingRecipe.from_mapping(missing_horizon)

    unused_horizon = _recipe_mapping("token-grpo")
    unused_horizon["algorithm"]["clip_schedule"] = "constant"
    with pytest.raises(ValueError, match="clip_schedule_steps is unused"):
        PostTrainingRecipe.from_mapping(unused_horizon)


@pytest.mark.parametrize("algorithm_type", tuple(_SPEC_TYPES))
def test_builder_consumes_each_algorithm_spec(
    algorithm_type: str,
) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(algorithm_type))
    replay = _ReplayAdapter()
    rollout = _RolloutAdapter(replay)
    reward = _RewardAdapter()

    stack = post_training.build_native_token_policy_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=reward,
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )

    assert isinstance(stack, NativeTokenPolicyTrainingStack)
    assert isinstance(stack.engine.algorithm, _STAGE_TYPES[algorithm_type])
    assert dict(stack.engine.algorithm.state_fields) == {
        name: getattr(recipe.algorithm, name) for name in _STAGE_FIELDS[algorithm_type]
    }
    assert stack.engine.old_log_prob_source == recipe.algorithm.old_log_prob_source
    assert stack.engine.updates_per_trajectory == 2
    assert stack.engine.replay_microbatch_size == 1
    assert stack.engine.first_update_log_ratio_tolerance == 2.0e-5
    assert stack.engine.max_grad_norm == 0.75
    assert stack.optimizer.param_groups[0]["lr"] == 0.001
    assert stack.group_size == 4
    assert stack.advantage_epsilon == 1.0e-7
    assert stack.advantage_clip_max == 3.0
    assert stack.advantage_normalization == "group-mean-global-sample-std"
    assert stack.sampling_temperature == 0.7
    assert dict(stack.scalarizer.weights) == {
        "correctness": 1.0,
        "format": 0.25,
    }
    assert stack.checkpoint_state_kwargs()["algorithm_state"] is stack.scalarizer


def test_recipe_bound_stack_runs_session_and_checkpoints_scalarizer() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping("token-grpo"))
    replay = _ReplayAdapter()
    stack = post_training.build_native_token_policy_training_stack(
        recipe,
        rollout_adapter=_RolloutAdapter(replay),
        replay_adapter=replay,
        reward_adapter=_RewardAdapter(),
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )
    progress = TrainingProgress()
    session = stack.build_session(progress)
    request = TokenRolloutRequest(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("group", "group", "group", "group"),
        policy_revision="policy-root",
        sampling_temperature=stack.sampling_temperature,
    )

    result = session.train_iteration(request)

    assert len(result.updates) == 2
    assert progress.optimizer_steps == 2
    assert stack.engine.global_step == 2
    scalarizer_state = stack.scalarizer.state_dict()
    stack.scalarizer.load_state_dict(scalarizer_state)


def test_builder_rejects_reward_contract_drift() -> None:
    class _WrongReward(_RewardAdapter):
        reward_ids = ("correctness",)

    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping("token-gspo"))
    replay = _ReplayAdapter()

    with pytest.raises(ValueError, match="ids differ from reward_weights"):
        post_training.build_native_token_policy_training_stack(
            recipe,
            rollout_adapter=_RolloutAdapter(replay),
            replay_adapter=replay,
            reward_adapter=_WrongReward(),
            initial_policy_revision="policy-root",
            fused_adamw=False,
        )


def test_recipe_bound_session_rejects_wrong_group_size_before_rollout() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping("token-gspo"))
    replay = _ReplayAdapter()
    rollout = _RolloutAdapter(replay)
    stack = post_training.build_native_token_policy_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=_RewardAdapter(),
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )
    session = stack.build_session(TrainingProgress())
    request = TokenRolloutRequest(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        sampling_temperature=stack.sampling_temperature,
    )

    with pytest.raises(ValueError, match="must each contain 4 samples"):
        session.train_iteration(request)


def test_token_policy_stack_fixtures_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    for algorithm_type, spec_type in _SPEC_TYPES.items():
        path = root / (f"tests/training/fixtures/recipes/qwen3_4b_{algorithm_type.replace('-', '_')}_stack.yaml")
        recipe = PostTrainingRecipe.from_file(path)
        assert isinstance(recipe.algorithm, spec_type)
        assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    assert recipes.TokenPolicyAlgorithmSpec.__module__.endswith(".algorithms.token_policy")
    for spec_type in _SPEC_TYPES.values():
        assert getattr(recipes, spec_type.__name__) is spec_type
    for name in token_policy.__all__:
        canonical = getattr(token_policy, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
