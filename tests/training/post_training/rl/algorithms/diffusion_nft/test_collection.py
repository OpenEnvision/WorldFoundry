from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_nft import (  # noqa: E402
    NativeDiffusionNFTTerminalCollector,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch  # noqa: E402


class _RecordingPolicy:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(2, 2, bias=False)
        self.calls: list[tuple[tuple[str, ...], torch.Tensor, bool, bool]] = []

    def predict_velocity(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del branch
        self.calls.append(
            (
                sample_ids,
                conditioning["context"].detach().clone(),
                training,
                torch.is_grad_enabled(),
            )
        )
        assert sigmas.shape == (len(sample_ids),)
        return torch.full_like(noisy_latents, 2)

    def predict_clean(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning, training, branch
        return noisy_latents


def _batch(*, sigmas: torch.Tensor | None = None) -> FlowRolloutBatch:
    return FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="old-policy-3",
        initial_latents=torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ),
        sigmas=(torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32) if sigmas is None else sigmas),
        conditioning={"context": torch.arange(4, dtype=torch.float32).unsqueeze(1)},
        metadata={"prompt_by_group": {"first": "one", "second": "two"}},
    )


def test_terminal_collector_integrates_chunks_without_grad_and_keeps_only_terminal() -> None:
    policy = _RecordingPolicy()
    collector = NativeDiffusionNFTTerminalCollector(
        policy,
        sigmas=(1.0, 0.5, 0.0),
        group_size=2,
        latent_dtype=torch.float32,
        forward_batch_size=2,
    )

    result = collector.collect(_batch(), collection_id="collection-3")

    torch.testing.assert_close(
        result.clean_latents,
        _batch().initial_latents - 2,
        rtol=0,
        atol=0,
    )
    assert result.clean_latents.grad_fn is None
    assert result.transition_count == 2
    assert result.policy_revision == "old-policy-3"
    assert result.with_rewards(torch.ones(result.batch_size)).policy_revision == ("old-policy-3")
    assert [call[0] for call in policy.calls] == [
        ("a", "b"),
        ("c", "d"),
        ("a", "b"),
        ("c", "d"),
    ]
    assert all(training is False and grad_enabled is False for _, _, training, grad_enabled in policy.calls)
    torch.testing.assert_close(policy.calls[0][1], torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(policy.calls[1][1], torch.tensor([[2.0], [3.0]]))


def test_terminal_collector_rejects_recipe_schedule_and_group_mismatch() -> None:
    collector = NativeDiffusionNFTTerminalCollector(
        _RecordingPolicy(),
        sigmas=(1.0, 0.5, 0.0),
        group_size=4,
        latent_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="group sizes differ"):
        collector.collect(_batch(), collection_id="wrong-groups")

    matching_groups = NativeDiffusionNFTTerminalCollector(
        _RecordingPolicy(),
        sigmas=(1.0, 0.5, 0.0),
        group_size=2,
        latent_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="sigmas differ"):
        matching_groups.collect(
            _batch(sigmas=torch.tensor([1.0, 0.25, 0.0])),
            collection_id="wrong-schedule",
        )
