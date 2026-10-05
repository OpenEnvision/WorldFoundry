from __future__ import annotations

from dataclasses import dataclass

import pytest

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch, TrainStepResult


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def test_training_batch_enforces_canonical_visual_layout() -> None:
    batch = TrainingBatch(
        sample_ids=("a", "b"),
        prompts=("one", "two"),
        pixel_values=_Tensor((2, 3, 1, 32, 48)),
        valid_mask=_Tensor((2, 1, 32, 48)),
        sample_weights=_Tensor((2,)),
    )

    assert batch.batch_size == 2

    with pytest.raises(ValueError, match=r"\[B,C,T,H,W\]"):
        TrainingBatch(
            sample_ids=("a", "b"),
            prompts=("one", "two"),
            pixel_values=_Tensor((2, 3, 32, 48)),
        )


def test_training_batch_rejects_duplicate_sample_identity() -> None:
    with pytest.raises(ValueError, match="unique"):
        TrainingBatch(sample_ids=("same", "same"), prompts=("one", "two"))


def test_prepared_and_objective_batch_validate_tensor_tree_shapes() -> None:
    prepared = PreparedBatch(
        sample_ids=("a", "b"),
        clean_latents={"video": _Tensor((2, 4, 3, 8, 8))},
        loss_mask={"video": _Tensor((2, 3, 8, 8))},
    )

    assert prepared.batch_size == 2
    with pytest.raises(ValueError, match="do not match"):
        ObjectiveBatch(
            sample_ids=("a", "b"),
            model_input=_Tensor((2, 4, 3, 8, 8)),
            target=_Tensor((2, 4, 4, 8, 8)),
            sigmas=_Tensor((2,)),
            timesteps=_Tensor((2,)),
        )


def test_train_step_result_requires_scalar_loss_and_explicit_counts() -> None:
    result = TrainStepResult(
        loss=_Tensor(()),
        losses={"flow": _Tensor(())},
        metrics={"sigma": _Tensor(())},
        sample_count=2,
        latent_token_count=128,
    )

    assert result.sample_count == 2
    with pytest.raises(ValueError, match="scalar"):
        TrainStepResult(
            loss=_Tensor((2,)),
            losses={},
            metrics={},
            sample_count=2,
            latent_token_count=128,
        )
