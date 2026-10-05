from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training import (  # noqa: E402
    NativeClassifierFreeGuidance,
    NativeFlowPredictionAdapter,
)


class _NativeAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (torch.nn.Linear,)

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.trainable_module.weight.fill_(1.0)
        self.calls: list[object] = []

    def prepare_batch(self, batch):
        raise AssertionError("not used")

    def forward_train(self, batch):
        raise AssertionError("CFG must use the explicit forward_model seam")

    def forward_model(self, batch, *, training: bool, branch: str):
        self.calls.append((batch, training, branch))
        self.trainable_module.train(training)
        return self.trainable_module(batch.conditioning["context"])


class _MixedPrecisionAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (torch.nn.Linear,)

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            self.trainable_module.weight.fill_(2.0)
        self.seen_model_dtype = None
        self.seen_sigma_dtype = None

    def forward_model(self, batch, *, training: bool, branch: str):
        del training, branch
        self.seen_model_dtype = batch.model_input.dtype
        self.seen_sigma_dtype = batch.sigmas.dtype
        return self.trainable_module(batch.model_input)


def test_native_cfg_uses_one_unconditional_then_conditional_model_batch() -> None:
    adapter = _NativeAdapter()
    policy = NativeClassifierFreeGuidance(
        NativeFlowPredictionAdapter(adapter),
        guidance_scale=5.0,
    )
    noisy = torch.zeros(2, 1)
    output = policy.predict_velocity(
        noisy,
        torch.tensor([1.0, 0.7]),
        sample_ids=("first", "second"),
        conditioning={
            "context": torch.tensor([[3.0], [4.0]]),
            "negative_context": torch.tensor([[1.0], [2.0]]),
        },
        training=True,
    )

    torch.testing.assert_close(output, torch.tensor([[11.0], [12.0]]))
    assert len(adapter.calls) == 1
    batch, training, branch = adapter.calls[0]
    assert training is True
    assert branch == "positive"
    assert batch.sample_ids == (
        "unconditional::first",
        "unconditional::second",
        "conditional::first",
        "conditional::second",
    )
    torch.testing.assert_close(
        batch.conditioning["context"],
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    output.sum().backward()
    torch.testing.assert_close(
        adapter.trainable_module.weight.grad,
        torch.tensor([[23.0]]),
    )


def test_native_cfg_rejects_implicit_or_mismatched_conditioning() -> None:
    policy = NativeClassifierFreeGuidance(
        NativeFlowPredictionAdapter(_NativeAdapter()),
        guidance_scale=5.0,
    )

    with pytest.raises(ValueError, match="exactly context and negative_context"):
        policy.predict_velocity(
            torch.zeros(2, 1),
            1.0,
            sample_ids=("first", "second"),
            conditioning={"context": torch.zeros(2, 1)},
            training=False,
        )
    with pytest.raises(ValueError, match=r"share shape \[B,...\]"):
        policy.predict_velocity(
            torch.zeros(2, 1),
            1.0,
            sample_ids=("first", "second"),
            conditioning={
                "context": torch.zeros(2, 1),
                "negative_context": torch.zeros(1, 1),
            },
            training=False,
        )


def test_native_prediction_casts_only_model_compute_and_restores_trajectory_dtype() -> None:
    adapter = _MixedPrecisionAdapter()
    prediction = NativeFlowPredictionAdapter(
        adapter,
        autocast_dtype=torch.bfloat16,
    )
    trajectory = torch.tensor([[3.0], [4.0]], dtype=torch.float32, requires_grad=True)

    output = prediction.predict_velocity(
        trajectory,
        torch.tensor([0.8, 0.2], dtype=torch.float64),
        sample_ids=("first", "second"),
        conditioning={},
        training=True,
    )

    assert adapter.seen_model_dtype is torch.bfloat16
    assert adapter.seen_sigma_dtype is torch.float32
    assert output.dtype is torch.float32
    torch.testing.assert_close(output, torch.tensor([[6.0], [8.0]]))
    output.sum().backward()
    assert trajectory.grad is not None
    assert trajectory.grad.dtype is torch.float32
    torch.testing.assert_close(trajectory.grad, torch.full_like(trajectory, 2.0))
    assert adapter.trainable_module.weight.grad is not None
    assert adapter.trainable_module.weight.grad.dtype is torch.bfloat16


def test_native_prediction_binds_and_propagates_checkpoint_identity() -> None:
    prediction = NativeFlowPredictionAdapter(
        _NativeAdapter(),
        checkpoint_identity="teacher-checkpoint",
    )
    guided = NativeClassifierFreeGuidance(prediction, guidance_scale=5.0)

    assert prediction.checkpoint_identity == "teacher-checkpoint"
    assert guided.checkpoint_identity == "teacher-checkpoint"
    with pytest.raises(ValueError, match="checkpoint_identity"):
        NativeFlowPredictionAdapter(_NativeAdapter(), checkpoint_identity="  ")
