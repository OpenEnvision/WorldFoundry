import functools
import math

import jax
import torch

from .batch_sampler import EvalBatchSampler
from .dataset import collate_segments_to_batch


def build_data_loader(
    dataset,
    batch_size,
    num_workers,
    num_frames,
    seed_data,
    eval,
    eval_num_samples=None,
    eval_pseudo_process_index=None,
    eval_pseudo_process_count=None,
    seed_offset=None,
):
    """Build loader for torch Dataset."""

    if not eval:
        raise ValueError("The in-tree Solaris runtime only supports eval/inference dataloading.")

    local_batch_size = batch_size // jax.process_count()

    assert eval_num_samples == batch_size, "eval_num_samples should be equal to batch_size in eval mode"
    if eval_pseudo_process_index is not None or eval_pseudo_process_count is not None:
        assert not jax.distributed.is_initialized(), (
            "jax.distributed should be disabled with eval_pseudo_process_index and eval_pseudo_process_count!"
        )
        assert batch_size == local_batch_size, (
            "batch_size should be equal to local_batch_size when using eval_pseudo_process_index and eval_pseudo_process_count!"
        )
    rank = eval_pseudo_process_index if eval_pseudo_process_index is not None else jax.process_index()
    num_replicas = eval_pseudo_process_count if eval_pseudo_process_count is not None else jax.process_count()
    sampler = EvalBatchSampler(
        dataset,
        rank=rank,
        num_replicas=num_replicas,
        batch_size=local_batch_size,
        num_frames=num_frames,
        num_global_samples=eval_num_samples,
    )
    num_batches = len(sampler)
    pad_batch_to = calculate_last_batch_padding(eval_num_samples, batch_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_segments_to_batch, sampler.num_frames, pad_batch_to),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, num_batches


def calculate_last_batch_padding(num_global_samples, batch_size):
    last_batch_samples = num_global_samples % batch_size
    num_devices = jax.device_count()
    per_device = math.ceil(last_batch_samples / num_devices)
    return per_device * jax.local_device_count()
