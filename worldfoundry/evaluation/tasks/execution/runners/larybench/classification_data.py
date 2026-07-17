import csv

import numpy as np
import torch
from .paths import resolve_la_path
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class LatentActionVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dataset: str,
        split: str,
        model: str,
        action_to_id: dict[str, int] | None = None,
    ) -> None:
        self.csv_path = csv_path

        with open(csv_path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if (row.get("la_path") or "").strip() and (row.get("action") or "").strip()]
        if not rows:
            raise ValueError(f"No usable latent-action rows in {csv_path}")

        if action_to_id is None:
            actions = sorted({str(row["action"]) for row in rows})
            action_to_id = {action: index for index, action in enumerate(actions)}
        unknown = sorted({str(row["action"]) for row in rows} - action_to_id.keys())
        if unknown:
            raise ValueError(f"Unknown validation actions in {csv_path}: {unknown}")
        self.action_to_id = dict(action_to_id)

        self.samples = [
            (
                resolve_la_path(str(row["la_path"]).strip(), dataset, split, model),
                self.action_to_id[str(row["action"])],
            )
            for row in rows
        ]
        counts = torch.bincount(
            torch.tensor([label for _, label in self.samples]),
            minlength=len(self.action_to_id),
        ).double()
        self.sample_weights = torch.tensor([1.0 / counts[label] for _, label in self.samples], dtype=torch.double)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with np.load(path, allow_pickle=False) as payload:
                tokens = np.asarray(payload["tokens"])
            if tokens.size == 0 or not np.issubdtype(tokens.dtype, np.number):
                raise ValueError(f"invalid token array: shape={tokens.shape}, dtype={tokens.dtype}")
            if not np.isfinite(tokens).all():
                raise ValueError("token array contains non-finite values")
            return label, torch.from_numpy(tokens.astype(np.float32, copy=False))
        except Exception as error:
            raise RuntimeError(f"Failed to load latent action {path}: {error}") from error


class DistributedWeightedSampler(DistributedSampler):
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.dataset.sample_weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])


def make_latent_action_dataloader(
    dataset: LatentActionVideoDataset,
    batch_size: int,
    world_size: int,
    rank: int,
    training: bool,
    num_workers: int,
    timeout: int = 0,
) -> tuple[DataLoader, DistributedSampler]:
    sampler: DistributedSampler
    if training:
        sampler = DistributedWeightedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        timeout=timeout,
    )
    return loader, sampler
