from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training import (  # noqa: E402
    DecodedTerminalRewardAdapter,
    DiffusionNFTRewardAdapter,
    DiffusionNFTTerminalLatents,
    FlowTrajectory,
    RewardRequest,
    RewardResult,
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rewards.contracts import (  # noqa: E402
    RewardRequest as CanonicalRewardRequest,
)
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer as CanonicalWeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.trajectory_rewards import (  # noqa: E402
    DecodedTerminalRewardAdapter as CanonicalDecodedTerminalRewardAdapter,
)


def test_reward_public_exports_resolve_to_the_canonical_layered_modules() -> None:
    assert RewardRequest is CanonicalRewardRequest
    assert DecodedTerminalRewardAdapter is CanonicalDecodedTerminalRewardAdapter
    assert WeightedRewardScalarizer is CanonicalWeightedRewardScalarizer


def test_reward_contract_preserves_vector_components() -> None:
    request = RewardRequest(
        request_id="request-a",
        rollout_id="rollout-a",
        prompt="a rolling red cube",
        conditions={"camera": "orbit"},
        artifacts={"video": {"path": "rollout-a.mp4"}},
        reward_ids=("alignment", "motion"),
    )
    result = RewardResult(
        request_id=request.request_id,
        rollout_id=request.rollout_id,
        values={"alignment": 0.8, "motion": 0.4},
        valid={"alignment": True, "motion": True},
        diagnostics={},
        latency_ms=17.5,
    )

    assert result.values["alignment"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="same non-empty keys"):
        RewardResult(
            request_id="request",
            rollout_id="rollout",
            values={"alignment": 1.0},
            valid={"motion": True},
            diagnostics={},
            latency_ms=1.0,
        )


def test_reward_scalarizer_uses_frozen_calibration() -> None:
    scalarizer = WeightedRewardScalarizer(
        {"alignment": 2.0, "motion": -0.5},
        calibration_mean={"alignment": 0.5, "motion": 1.0},
        calibration_std={"alignment": 0.25, "motion": 2.0},
        normalization_epsilon=1.0e-8,
    )
    result = scalarizer.scalarize(
        {
            "alignment": torch.tensor([0.5, 0.75]),
            "motion": torch.tensor([1.0, 3.0]),
        }
    )

    torch.testing.assert_close(result.normalized_components["alignment"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(result.normalized_components["motion"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(result.scalar_rewards, torch.tensor([0.0, 1.5]))


def test_reward_scalarizer_normalization_epsilon_is_explicit_and_checkpointed() -> None:
    scalarizer = WeightedRewardScalarizer(
        {"quality": 1.0},
        calibration_mean={"quality": 2.0},
        calibration_std={"quality": 4.0},
        normalization_epsilon=1.0,
    )

    result = scalarizer.scalarize({"quality": torch.tensor([7.0])})

    torch.testing.assert_close(result.scalar_rewards, torch.tensor([1.0]))
    assert scalarizer.state_dict()["normalization_epsilon"] == 1.0
    with pytest.raises(ValueError, match="normalization_epsilon"):
        WeightedRewardScalarizer({"quality": 1.0}, normalization_epsilon=-1.0)


def test_reward_scalarizer_rejects_invalid_by_default_and_zero_policy_is_explicit() -> None:
    values = {"quality": torch.tensor([1.0, float("nan")])}
    with pytest.raises(ValueError, match="1 invalid"):
        WeightedRewardScalarizer({"quality": 1.0}).scalarize(values)

    result = WeightedRewardScalarizer({"quality": 1.0}, invalid_policy="zero").scalarize(values)
    torch.testing.assert_close(result.scalar_rewards, torch.tensor([1.0, 0.0]))
    assert torch.equal(result.valid_mask, torch.tensor([True, False]))


class _Decoder:
    def decode(self, latents, request):
        assert tuple(request.prompt) == ("a red cube", "a red cube")
        assert latents.shape == (2, 1)
        return torch.zeros(2, 3, request.num_frames, request.height, request.width)


class _Evaluator:
    def __init__(self) -> None:
        self.requests = ()

    def evaluate(self, requests):
        self.requests = requests
        return tuple(
            RewardResult(
                request_id=request.request_id,
                rollout_id=request.rollout_id,
                values={"quality": float(index + 1), "alignment": 0.5},
                valid={"quality": True, "alignment": True},
                diagnostics={},
                latency_ms=1.0,
            )
            for index, request in enumerate(requests)
        )


def _reward_metadata() -> dict[str, object]:
    return {
        "prompt_by_group": {"group": "a red cube"},
        "generation_by_group": {"group": {"height": 16, "width": 16, "num_frames": 1}},
    }


def _flow_trajectory() -> FlowTrajectory:
    return FlowTrajectory(
        sample_ids=("sample-a", "sample-b"),
        group_ids=("group", "group"),
        policy_revision="policy",
        latents=torch.zeros(2, 2, 1),
        sigmas=torch.tensor([1.0, 0.0]),
        step_indices=(0,),
        old_log_probs=torch.zeros(2, 1),
        transition_means=torch.zeros(2, 1, 1),
        transition_scales=torch.ones(2, 1, 1),
        metadata=_reward_metadata(),
    )


def _diffusion_nft_terminal(
    *,
    metadata: dict[str, object] | None = None,
) -> DiffusionNFTTerminalLatents:
    return DiffusionNFTTerminalLatents(
        collection_id="collection",
        policy_revision="policy",
        sample_ids=("sample-a", "sample-b"),
        group_ids=("group", "group"),
        clean_latents=torch.zeros(2, 1),
        transition_count=1,
        metadata=_reward_metadata() if metadata is None else metadata,
    )


@pytest.mark.parametrize(
    ("terminal_state", "identity_key", "identity_value"),
    [
        (_flow_trajectory(), "step_indices", [0]),
        (_diffusion_nft_terminal(), "collection_id", "collection"),
    ],
)
def test_decoded_terminal_reward_adapter_scores_both_native_terminal_types(
    terminal_state,
    identity_key,
    identity_value,
) -> None:
    evaluator = _Evaluator()
    adapter = DecodedTerminalRewardAdapter(
        _Decoder(),
        evaluator,
        reward_ids=("quality", "alignment"),
        evaluator_identity={"model": "native-test"},
    )

    components = adapter.score(terminal_state)

    torch.testing.assert_close(components["quality"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(components["alignment"], torch.tensor([0.5, 0.5]))
    expected_device = (
        terminal_state.clean_latents.device
        if isinstance(terminal_state, DiffusionNFTTerminalLatents)
        else terminal_state.latents.device
    )
    assert components["quality"].device == expected_device
    assert [request.request_id for request in evaluator.requests] == ["sample-a", "sample-b"]
    assert all(request.metadata[identity_key] == identity_value for request in evaluator.requests)
    assert len(adapter.last_results) == 2
    assert adapter.identity == {
        "schema": "worldfoundry-decoded-terminal-reward",
        "reward_ids": ["quality", "alignment"],
        "evaluator": {"model": "native-test"},
    }
    assert isinstance(adapter, DiffusionNFTRewardAdapter)


def test_decoded_terminal_reward_adapter_rejects_unsupported_and_incomplete_inputs() -> None:
    adapter = DecodedTerminalRewardAdapter(
        _Decoder(),
        _Evaluator(),
        reward_ids=("quality", "alignment"),
        evaluator_identity={"model": "native-test"},
    )

    with pytest.raises(TypeError, match="FlowTrajectory or DiffusionNFTTerminalLatents"):
        adapter.score(object())
    with pytest.raises(ValueError, match="prompt_by_group and generation_by_group"):
        adapter.score(_diffusion_nft_terminal(metadata={}))


def test_decoded_terminal_reward_adapter_rejects_malformed_decode_and_result_order() -> None:
    class _WrongShapeDecoder:
        def decode(self, latents, request):
            del latents, request
            return torch.zeros(2, 3, 1, 8, 8)

    malformed_decode = DecodedTerminalRewardAdapter(
        _WrongShapeDecoder(),
        _Evaluator(),
        reward_ids=("quality", "alignment"),
        evaluator_identity={"model": "native-test"},
    )
    with pytest.raises(ValueError, match="must have shape"):
        malformed_decode.score(_diffusion_nft_terminal())

    class _NonFiniteDecoder:
        def decode(self, latents, request):
            del latents
            return torch.full(
                (2, 3, request.num_frames, request.height, request.width),
                float("nan"),
            )

    non_finite_decode = DecodedTerminalRewardAdapter(
        _NonFiniteDecoder(),
        _Evaluator(),
        reward_ids=("quality", "alignment"),
        evaluator_identity={"model": "native-test"},
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        non_finite_decode.score(_flow_trajectory())

    class _OutOfOrderEvaluator(_Evaluator):
        def evaluate(self, requests):
            return tuple(reversed(super().evaluate(requests)))

    out_of_order = DecodedTerminalRewardAdapter(
        _Decoder(),
        _OutOfOrderEvaluator(),
        reward_ids=("quality", "alignment"),
        evaluator_identity={"model": "native-test"},
    )
    with pytest.raises(ValueError, match="identity/components"):
        out_of_order.score(_flow_trajectory())
