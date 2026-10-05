from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.training.distributed.weight_sync import (
    ModuleWeightReceiver,
    NativeWeightSynchronizer,
    WeightBucket,
    WeightKind,
    WeightUpdateHeader,
    build_weight_buckets,
    materialize_weight_tensors,
)


class _AdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2)
        self.lora_A = nn.Parameter(torch.zeros(2, 3))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class _IncompatibleAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2)
        self.lora_A = nn.Parameter(torch.zeros(1, 3))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


def test_full_weight_sync_stages_then_updates_rollout_module() -> None:
    source = _AdapterModel()
    target = _AdapterModel()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
        for parameter in target.parameters():
            parameter.zero_()

    report = NativeWeightSynchronizer(max_bucket_bytes=32).sync(
        source,
        [ModuleWeightReceiver(target)],
        revision=4,
        kind=WeightKind.FULL,
    )

    assert report.transmitted
    assert report.bucket_count > 1
    assert all(
        torch.equal(source_value, target.state_dict()[name]) for name, source_value in source.state_dict().items()
    )


def test_lora_weight_sync_leaves_rollout_base_unchanged() -> None:
    source = _AdapterModel()
    target = _AdapterModel()
    original_base = {name: value.clone() for name, value in target.base.state_dict().items()}
    with torch.no_grad():
        source.lora_A.fill_(3.0)
        source.lora_B.fill_(7.0)

    receiver = ModuleWeightReceiver(target)
    NativeWeightSynchronizer().sync(
        source,
        [receiver],
        revision=1,
        kind="lora",
    )

    assert torch.equal(target.lora_A, source.lora_A)
    assert torch.equal(target.lora_B, source.lora_B)
    assert all(torch.equal(value, target.base.state_dict()[name]) for name, value in original_base.items())
    assert receiver.last_revision == 1


def test_weight_materialization_and_bucketing_are_deterministic() -> None:
    model = _AdapterModel()
    tensors = materialize_weight_tensors(model, kind="lora")
    buckets = build_weight_buckets(tensors, revision=2, max_bucket_bytes=1)
    assert tuple(tensors) == tuple(sorted(tensors))
    assert tuple(bucket.index for bucket in buckets) == tuple(range(len(buckets)))
    assert {name for bucket in buckets for name in bucket.tensors} == set(tensors)


def test_receiver_rejects_incompatible_shapes_before_loading() -> None:
    model = _AdapterModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    receiver = ModuleWeightReceiver(model)
    receiver.begin_weight_update(
        WeightUpdateHeader(
            revision=1,
            kind=WeightKind.LORA,
            tensor_names=("lora_A",),
            bucket_count=1,
        )
    )
    receiver.write_weight_bucket(WeightBucket(revision=1, index=0, tensors={"lora_A": torch.ones(1)}))
    with pytest.raises(ValueError, match="shapes"):
        receiver.commit_weight_update(1)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_multi_receiver_validation_aborts_before_any_live_weight_commit() -> None:
    source = _AdapterModel()
    good_target = _AdapterModel()
    bad_target = _IncompatibleAdapterModel()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(4.0)
        for parameter in good_target.parameters():
            parameter.zero_()
    before = {name: value.clone() for name, value in good_target.state_dict().items()}
    good_receiver = ModuleWeightReceiver(good_target)
    bad_receiver = ModuleWeightReceiver(bad_target)

    with pytest.raises(ValueError, match="shapes"):
        NativeWeightSynchronizer(max_bucket_bytes=16).sync(
            source,
            [good_receiver, bad_receiver],
            revision=0,
            kind="full",
        )

    assert good_receiver.last_revision == -1
    assert bad_receiver.last_revision == -1
    assert all(torch.equal(before[name], value) for name, value in good_target.state_dict().items())


def test_committed_revision_cannot_be_replayed() -> None:
    source = _AdapterModel()
    target = _AdapterModel()
    receiver = ModuleWeightReceiver(target)
    synchronizer = NativeWeightSynchronizer()
    synchronizer.sync(source, [receiver], revision=2, kind="full")

    with pytest.raises(ValueError, match="not newer"):
        synchronizer.sync(source, [receiver], revision=2, kind="full")
    assert receiver.last_revision == 2
