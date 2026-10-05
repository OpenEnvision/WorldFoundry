import json
import logging
import math
import os

import numpy as np
import torch

from worldfoundry.core.io.paths import resolve_data_path

from .dataset import DatasetMultiplayer
from .segment import SegmentId, SegmentIdMultiplayer


def _init_torch_sampler_with_optional_dataset(sampler, dataset):
    try:
        super(type(sampler), sampler).__init__(dataset)
    except TypeError:
        super(type(sampler), sampler).__init__()


class EvalBatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        num_global_samples=None,  # global number of samples
        seed=[0],
    ):
        _init_torch_sampler_with_optional_dataset(self, dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        logging.info(f"Eval num_frames: {num_frames}")
        self._seed = seed
        self.reset_rng()

        # hardcoded full episodes for worldmem_demo
        if "worldmem_demo" in str(dataset.directory):
            self.ids = [
                [0, 0, 1024],
                [1, 0, 1024],
                [2, 0, 1024],
                [3, 0, 1024],
                [4, 0, 1024],
                [5, 0, 1024],
            ]
        else:
            eval_ids_path = resolve_data_path(
                "models",
                "runtime",
                "configs",
                "solaris",
                "eval_ids",
                f"eval_ids_{dataset.dataset_name}.json",
            )
            with open(eval_ids_path, "r") as f:
                self.ids = json.load(f)

        assert num_frames <= 1024, "num_frames must be at most 1024"
        num_global_samples = (
            min(num_global_samples, len(self.ids))
            if num_global_samples is not None
            else len(self.ids)
        )
        if isinstance(dataset, DatasetMultiplayer):
            self.examples = [
                SegmentIdMultiplayer(
                    episode_id,
                    bot1_start,
                    bot1_start + self.num_frames,
                    bot2_start,
                    bot2_start + self.num_frames,
                )
                for (episode_id, bot1_start, _, bot2_start, _) in self.ids
            ]
        else:
            # For other datasets, replace the endpoint with start + self.num_frames
            # Note this could potentially lead to an error later on when calling Dataset.__getitem__(...) if out of bounds
            self.examples = [
                SegmentId(episode_id, start, start + self.num_frames)
                for (episode_id, start, _) in self.ids
            ]
        self.examples = self.examples[:num_global_samples]
        self.examples = self.examples[self.rank :: self.world_size]
        self.num_batches = math.ceil(len(self.examples) / self.batch_size)

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.examples))
            yield self.examples[start:end]
