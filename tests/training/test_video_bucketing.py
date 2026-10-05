from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdata")

from worldfoundry.training.data import (  # noqa: E402
    MediaReference,
    SamplerStateMismatchError,
    TrainingSample,
    build_stateful_dataloader,
)
from worldfoundry.training.data.latent_token_sampler import (  # noqa: E402
    LatentTokenBatchSampler,
)
from worldfoundry.training.data.video_bucketing import (  # noqa: E402
    VideoBucketKey,
    VideoBucketSelectionPolicy,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
)


class _BucketDataset:
    def __init__(self, keys: tuple[VideoBucketKey, ...]) -> None:
        self.keys = keys
        self.sample_ids = tuple(f"sample-{index}" for index in range(len(keys)))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> int:
        return index


def _training_sample(
    sample_id: str,
    *,
    task: str = "t2v",
    frames: int = 65,
    height: int = 576,
    width: int = 1024,
) -> TrainingSample:
    return TrainingSample(
        sample_id=sample_id,
        task=task,
        prompt="a moving subject",
        media=MediaReference(uri=f"{sample_id}.mp4"),
        width=width,
        height=height,
        num_frames=frames,
        fps=24.0,
        conditions={},
        split="train",
        safety={"accepted": True},
    )


def _sampler(
    dataset: _BucketDataset,
    *,
    rank: int = 0,
    world_size: int = 1,
    tail_policy: str = "drop",
) -> LatentTokenBatchSampler:
    return LatentTokenBatchSampler(
        dataset,
        bucket_keys=dataset.keys,
        max_latent_tokens=64,
        seed=19,
        shuffle=True,
        rank=rank,
        world_size=world_size,
        tail_policy=tail_policy,
    )


def _loader_sampler_state(state: object) -> dict[str, object]:
    matches: list[dict[str, object]] = []

    def visit(value: object) -> None:
        if not isinstance(value, dict):
            return
        candidate = value.get("_index_sampler_state")
        if isinstance(candidate, dict):
            matches.append(candidate)
        for item in value.values():
            visit(item)

    visit(state)
    if len(matches) != 1:
        raise AssertionError(f"expected one native sampler state, found {len(matches)}")
    return matches[0]


def test_video_latent_geometry_requires_model_explicit_temporal_alignment() -> None:
    first_frame = VideoLatentGeometry(32, 32, 8, "first-frame")
    assert first_frame.latent_shape(num_frames=17, height=256, width=512).to_dict() == {
        "frames": 3,
        "height": 8,
        "width": 16,
    }
    with pytest.raises(ValueError, match="num_frames - 1"):
        first_frame.latent_shape(num_frames=16, height=256, width=512)

    uniform = VideoLatentGeometry(8, 8, 4, "uniform")
    assert uniform.latent_shape(num_frames=16, height=256, width=512).frames == 4
    with pytest.raises(ValueError, match="num_frames divisible"):
        uniform.latent_shape(num_frames=17, height=256, width=512)


def test_bucket_assignment_is_shape_task_and_condition_layout_aware() -> None:
    sample = _training_sample("wide")
    geometry = VideoLatentGeometry(8, 8, 4, "first-frame")
    buckets = (
        VideoResolutionBucket(17, 512, 512, "text", tasks=("t2v",)),
        VideoResolutionBucket(17, 288, 512, "text", tasks=("t2v",)),
        VideoResolutionBucket(17, 288, 512, "image-text", tasks=("i2v",)),
    )
    assignment = assign_video_buckets(
        (sample,),
        buckets=buckets,
        geometry=geometry,
        conditioning_layout="text",
    )[0]

    assert (assignment.target_num_frames, assignment.target_height, assignment.target_width) == (17, 288, 512)
    assert assignment.bucket_key == VideoBucketKey("t2v", 5, 36, 64, "16:9", "text")
    assert assignment.latent_tokens == 5 * 36 * 64

    too_small = _training_sample("small", frames=9, height=128, width=128)
    with pytest.raises(ValueError, match="no eligible bucket"):
        assign_video_buckets(
            (too_small,),
            buckets=buckets,
            geometry=geometry,
            conditioning_layout="text",
        )
    padded = assign_video_buckets(
        (too_small,),
        buckets=(VideoResolutionBucket(17, 256, 256, "text"),),
        geometry=geometry,
        conditioning_layout="text",
        policy=VideoBucketSelectionPolicy(allow_spatial_upscale=True, allow_temporal_padding=True),
    )[0]
    assert padded.target_num_frames == 17


def test_token_budget_sampler_keeps_buckets_homogeneous_and_ranks_in_lockstep() -> None:
    small = VideoBucketKey("t2v", 1, 4, 4, "1:1", "text")
    large = VideoBucketKey("t2v", 1, 4, 6, "3:2", "text")
    dataset = _BucketDataset((small,) * 7 + (large,) * 5)

    drop_samplers = [_sampler(dataset, rank=rank, world_size=3, tail_policy="drop") for rank in range(3)]
    drop_batches = [list(sampler) for sampler in drop_samplers]
    assert [len(batches) for batches in drop_batches] == [1, 1, 1]
    flattened_drop = [index for batches in drop_batches for batch in batches for index in batch]
    assert len(flattened_drop) == len(set(flattened_drop))
    for sampler, batches in zip(drop_samplers, drop_batches):
        for batch in batches:
            stats = sampler.describe_batch(batch)
            assert stats.latent_tokens <= 64
            assert len({dataset.keys[index] for index in batch}) == 1

    pad_samplers = [_sampler(dataset, rank=rank, world_size=3, tail_policy="pad") for rank in range(3)]
    pad_batches = [list(sampler) for sampler in pad_samplers]
    assert [len(batches) for batches in pad_batches] == [2, 2, 2]
    assert set(index for batches in pad_batches for batch in batches for index in batch) == set(range(len(dataset)))
    assert pad_samplers[0].global_batch_count == 5
    assert pad_samplers[0].global_padded_batch_count == 1


def test_token_budget_sampler_state_replays_queues_and_next_batch_exactly() -> None:
    small = VideoBucketKey("t2v", 1, 4, 4, "1:1", "text")
    large = VideoBucketKey("t2v", 1, 4, 6, "3:2", "text")
    dataset = _BucketDataset((small,) * 9 + (large,) * 7)
    sampler = _sampler(dataset, rank=0, world_size=2, tail_policy="pad")
    iterator = iter(sampler)
    next(iterator)
    state = json.loads(json.dumps(sampler.state_dict()))
    expected_tail = list(iterator)

    restored = _sampler(dataset, rank=0, world_size=2, tail_policy="pad")
    restored.load_state_dict(state)
    assert list(restored) == expected_tail
    assert state["next_batch_sample_ids"] is not None

    active_state = restored.state_dict()
    tampered = json.loads(json.dumps(active_state))
    tampered["next_batch_sample_ids"] = ["forged"]
    with pytest.raises(SamplerStateMismatchError, match="deterministic replay"):
        restored.load_state_dict(tampered)
    assert restored.state_dict() == active_state

    with pytest.raises(SamplerStateMismatchError, match="invariants"):
        _sampler(dataset, rank=0, world_size=1, tail_policy="pad").load_state_dict(state)

    changed_content = _BucketDataset(dataset.keys)
    changed_content.sample_ids = tuple(f"other-{index}" for index in range(len(changed_content)))
    with pytest.raises(SamplerStateMismatchError, match="invariants"):
        _sampler(changed_content, rank=0, world_size=2, tail_policy="pad").load_state_dict(state)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_loader_resumes_a_custom_token_batch_sampler(num_workers: int) -> None:
    key = VideoBucketKey("t2v", 1, 4, 4, "1:1", "text")
    dataset = _BucketDataset((key,) * 13)
    loader = build_stateful_dataloader(
        dataset,
        batch_sampler=_sampler(dataset, tail_policy="pad"),
        worker_seed=23,
        num_workers=num_workers,
        multiprocessing_context=None if num_workers == 0 else "spawn",
        snapshot_every_n_steps=1,
    )
    iterator = iter(loader)
    first = next(iterator).tolist()
    state = loader.state_dict()
    expected_tail = [batch.tolist() for batch in iterator]

    restored_dataset = _BucketDataset((key,) * 13)
    restored = build_stateful_dataloader(
        restored_dataset,
        batch_sampler=_sampler(restored_dataset, tail_policy="pad"),
        worker_seed=23,
        num_workers=num_workers,
        multiprocessing_context=None if num_workers == 0 else "spawn",
        snapshot_every_n_steps=1,
    )
    restored.load_state_dict(state)
    assert len(first) == 4
    assert [batch.tolist() for batch in restored] == expected_tail

    with pytest.raises(ValueError, match="batch_size must be omitted"):
        build_stateful_dataloader(
            dataset,
            batch_size=2,
            batch_sampler=_sampler(dataset, tail_policy="pad"),
        )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_loader_resume_at_last_yield_matches_the_next_epoch(num_workers: int) -> None:
    key = VideoBucketKey("t2v", 1, 4, 4, "1:1", "text")
    dataset = _BucketDataset((key,))
    loader = build_stateful_dataloader(
        dataset,
        batch_sampler=_sampler(dataset, tail_policy="pad"),
        worker_seed=23,
        num_workers=num_workers,
        multiprocessing_context=None if num_workers == 0 else "spawn",
        snapshot_every_n_steps=1,
    )
    iterator = iter(loader)
    assert next(iterator).tolist() == [0]
    state_at_last_yield = loader.state_dict()

    # Uninterrupted session behavior observes StopIteration, creates the next
    # iterator, and advances the native token sampler to its next epoch.
    with pytest.raises(StopIteration):
        next(iterator)
    expected = next(iter(loader)).tolist()

    restored_dataset = _BucketDataset((key,))
    restored = build_stateful_dataloader(
        restored_dataset,
        batch_sampler=_sampler(restored_dataset, tail_policy="pad"),
        worker_seed=23,
        num_workers=num_workers,
        multiprocessing_context=None if num_workers == 0 else "spawn",
        snapshot_every_n_steps=1,
    )
    restored.load_state_dict(state_at_last_yield)
    actual = next(iter(restored)).tolist()

    assert actual == expected == [0]
    # TorchData's iterator-local yield counters may differ depending on
    # whether StopIteration was observed before checkpointing.  The native
    # sampler owns the logical epoch/queue/position and must match exactly.
    assert _loader_sampler_state(restored.state_dict()) == _loader_sampler_state(loader.state_dict())
