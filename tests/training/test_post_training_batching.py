from __future__ import annotations

from collections.abc import Mapping

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.data import (  # noqa: E402
    CacheTensorDescriptor,
    RolloutConditionedPrompt,
    RolloutPromptRecord,
    SharedConditioningArtifact,
    SharedConditioningIdentity,
    collate_rollout_conditioned_prompts,
)
from worldfoundry.training.post_training import (  # noqa: E402
    NativeDMDDataLoader,
    NativeFlowPolicyDataLoader,
    RolloutPrompt,
    dmd_batch_from_prepared,
    flow_rollout_batch_from_prompts,
)
from worldfoundry.training.safety import PromptSafetyAudit  # noqa: E402
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
)


def _prepared(batch_size: int = 2) -> PreparedBatch:
    return PreparedBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        clean_latents=torch.zeros(batch_size, 4, 2, 2),
        conditioning={"context": torch.arange(batch_size * 12, dtype=torch.float32).reshape(batch_size, 3, 4)},
        loss_mask=torch.ones(batch_size, 1, 2, 2),
        sample_weights=torch.ones(batch_size),
        metadata={"source": "unit-test"},
    )


def test_prepared_dmd_bridge_expands_one_verified_unconditional_branch() -> None:
    prepared = _prepared()
    shared = torch.full((3, 4), -2.0)
    batch = dmd_batch_from_prepared(
        prepared,
        shared_unconditional_conditioning={"context": shared},
    )

    assert batch.clean_latents is prepared.clean_latents
    assert tuple(batch.unconditional_conditioning["context"].shape) == (2, 3, 4)
    torch.testing.assert_close(batch.unconditional_conditioning["context"][0], shared)
    torch.testing.assert_close(batch.unconditional_conditioning["context"][1], shared)

    with pytest.raises(ValueError, match="keys must match"):
        dmd_batch_from_prepared(
            prepared,
            shared_unconditional_conditioning={"other": shared},
        )


class _Adapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (torch.nn.Linear,)

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        return _prepared(len(batch.sample_ids))

    def forward_train(self, batch):
        raise AssertionError("not used")


class _StatefulSource:
    def __init__(self) -> None:
        self.position = 0

    def __iter__(self):
        self.position += 1
        yield TrainingBatch(
            sample_ids=("raw-0", "raw-1"),
            prompts=("first", "second"),
        )

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.position = int(state_dict["position"])


def test_native_dmd_loader_forwards_source_checkpoint_state() -> None:
    source = _StatefulSource()
    loader = NativeDMDDataLoader(
        source,
        _Adapter(),
        shared_unconditional_conditioning={"context": torch.zeros(3, 4)},
    )

    batch = next(iter(loader))
    state = loader.state_dict()
    assert batch.batch_size == 2
    assert state["source"] == {"position": 1}

    source.position = 9
    loader.load_state_dict(state)
    assert source.position == 1


def test_flow_rollout_bridge_expands_complete_prompt_groups_and_noise() -> None:
    prompts = (
        RolloutPrompt(
            prompt_id="first",
            prompt="a red cube rolls",
            conditions={"context": torch.ones(1, 3, 4)},
        ),
        RolloutPrompt(
            prompt_id="second",
            prompt="a blue cube stops",
            conditions={"context": torch.full((3, 4), 2.0)},
        ),
    )
    generator = torch.Generator().manual_seed(17)
    batch = flow_rollout_batch_from_prompts(
        prompts,
        group_size=3,
        policy_revision="a" * 64,
        latent_shape=(4, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    )

    assert batch.group_ids == ("first", "first", "first", "second", "second", "second")
    assert tuple(batch.initial_latents.shape) == (6, 4, 2, 2)
    assert tuple(batch.conditioning["context"].shape) == (6, 3, 4)
    torch.testing.assert_close(batch.conditioning["context"][:3], torch.ones(3, 3, 4))
    torch.testing.assert_close(batch.conditioning["context"][3:], torch.full((3, 3, 4), 2.0))
    torch.testing.assert_close(batch.sigmas, torch.tensor([1.0, 0.5, 0.0]))
    assert batch.metadata["prompt_by_group"]["first"] == "a red cube rolls"
    assert set(batch.metadata) == {"prompt_by_group", "generation_by_group"}


def test_flow_rollout_bridge_repeats_initial_noise_per_group_and_expands_empty_prompt_context() -> None:
    prompts = (
        RolloutPrompt(
            prompt_id="first",
            prompt="first prompt",
            conditions={"context": torch.ones(3, 4)},
        ),
        RolloutPrompt(
            prompt_id="second",
            prompt="second prompt",
            conditions={"context": torch.full((3, 4), 2.0)},
        ),
    )
    batch = flow_rollout_batch_from_prompts(
        prompts,
        group_size=3,
        policy_revision="a" * 64,
        latent_shape=(2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(101),
        shared_negative_conditioning={"context": torch.full((3, 4), -3.0)},
        init_same_noise=True,
    )

    torch.testing.assert_close(
        batch.initial_latents[:3],
        batch.initial_latents[0].expand_as(batch.initial_latents[:3]),
    )
    torch.testing.assert_close(
        batch.initial_latents[3:],
        batch.initial_latents[3].expand_as(batch.initial_latents[3:]),
    )
    assert not torch.equal(batch.initial_latents[0], batch.initial_latents[3])
    torch.testing.assert_close(
        batch.conditioning["negative_context"],
        torch.full((6, 3, 4), -3.0),
    )


def test_flow_rollout_keeps_conditioning_dtype_independent_from_trajectory_dtype() -> None:
    prompts = (
        RolloutPrompt(
            prompt_id="first",
            prompt="first prompt",
            conditions={"context": torch.ones(3, 4, dtype=torch.bfloat16)},
        ),
    )

    batch = flow_rollout_batch_from_prompts(
        prompts,
        group_size=2,
        policy_revision="a" * 64,
        latent_shape=(4, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(103),
        shared_negative_conditioning={"context": torch.full((3, 4), -2.0)},
    )

    assert batch.initial_latents.dtype is torch.float32
    assert batch.sigmas.dtype is torch.float32
    assert batch.conditioning["context"].dtype is torch.bfloat16
    assert batch.conditioning["negative_context"].dtype is torch.bfloat16


def _conditioned(
    prompt_id: str,
    prompt: str,
    value: float,
    *,
    generation: Mapping[str, object] | None = None,
) -> RolloutConditionedPrompt:
    audit = PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={key: 0.01 for key in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )
    record = RolloutPromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        safety_audit=audit,
        generation=({"height": 32, "width": 32, "num_frames": 5} if generation is None else generation),
    )
    identity = SharedConditioningIdentity(
        branch=f"rollout-{prompt_id}",
        prompt=prompt,
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repository": "test/conditioner", "revision": "test"},
        tokenizer={"repository": "test/tokenizer", "revision": "test"},
        tensors={
            "context": CacheTensorDescriptor(
                dtype="float32",
                shape=(3, 4),
                layout="sequence-features",
            )
        },
    )
    artifact = SharedConditioningArtifact(
        identity=identity,
        object_size_bytes=100,
        object_path=f"shared-objects/{identity.branch}.safetensors",
    )
    return RolloutConditionedPrompt(
        record=record,
        conditioning={"context": torch.full((3, 4), value)},
        artifact=artifact,
    )


class _FlowSource:
    def __init__(self) -> None:
        self.position = 0

    def __iter__(self):
        self.position += 1
        yield (
            _conditioned("first", "first prompt", 1.0),
            _conditioned("second", "second prompt", 2.0),
        )

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.position = int(state_dict["position"])


class _ListFlowSource(_FlowSource):
    def __iter__(self):
        self.position += 1
        yield [
            _conditioned("first", "first prompt", 1.0),
            _conditioned("second", "second prompt", 2.0),
        ]


def test_native_flow_loader_uses_live_revision_and_forwards_source_state() -> None:
    source = _FlowSource()
    revision = ["1" * 64]
    loader = NativeFlowPolicyDataLoader(
        source,
        group_size=2,
        policy_revision=lambda: revision[0],
        latent_shape=(4, 2, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(19),
    )

    first = next(iter(loader))
    state = loader.state_dict()
    revision[0] = "2" * 64
    second = next(iter(loader))

    assert first.policy_revision == "1" * 64
    assert second.policy_revision == "2" * 64
    assert first.group_ids == ("first", "first", "second", "second")
    assert tuple(first.initial_latents.shape) == (4, 4, 2, 2, 2)
    assert state["source"] == {"position": 1}
    source.position = 9
    loader.load_state_dict(state)
    assert source.position == 1


def test_native_flow_loader_accepts_sequence_batches_rebuilt_as_lists() -> None:
    loader = NativeFlowPolicyDataLoader(
        _ListFlowSource(),
        group_size=2,
        policy_revision=lambda: "4" * 64,
        latent_shape=(4, 2, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(29),
    )

    batch = next(iter(loader))

    assert batch.group_ids == ("first", "first", "second", "second")


def test_native_flow_loader_accepts_stateful_pin_memory_batches() -> None:
    stateful_dataloader = pytest.importorskip("torchdata.stateful_dataloader")
    source = stateful_dataloader.StatefulDataLoader(
        [_conditioned("first", "first prompt", 1.0)],
        batch_size=1,
        collate_fn=collate_rollout_conditioned_prompts,
        pin_memory=True,
        num_workers=0,
    )
    loader = NativeFlowPolicyDataLoader(
        source,
        group_size=2,
        policy_revision=lambda: "5" * 64,
        latent_shape=(4, 2, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(31),
    )

    batch = next(iter(loader))

    assert batch.group_ids == ("first", "first")
    assert loader.state_dict()["source"]


class _DefaultGenerationFlowSource(_FlowSource):
    def __iter__(self):
        self.position += 1
        yield (
            _conditioned(
                "first",
                "first prompt",
                1.0,
                generation={},
            ),
        )


def test_native_flow_loader_merges_generation_defaults_and_namespaces_groups() -> None:
    loader = NativeFlowPolicyDataLoader(
        _DefaultGenerationFlowSource(),
        group_size=2,
        policy_revision=lambda: "3" * 64,
        latent_shape=(4, 2, 2, 2),
        sigmas=(1.0, 0.5, 0.0),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(23),
        generation_defaults={"height": 64, "width": 80, "num_frames": 9},
        group_namespace="rank-00000003",
    )

    batch = next(iter(loader))

    assert batch.group_ids == (
        "rank-00000003:first",
        "rank-00000003:first",
    )
    assert batch.metadata["generation_by_group"] == {
        "rank-00000003:first": {
            "height": 64,
            "width": 80,
            "num_frames": 9,
        }
    }
    assert set(batch.metadata) == {"prompt_by_group", "generation_by_group"}
