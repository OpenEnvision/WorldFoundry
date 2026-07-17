import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from worldfoundry.core.utils import set_random_seed

from .config import get_config
from .paths import resolve_data_path
from .registry import MODEL


def setup_seed(seed=42):
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoActionDataset(Dataset):
    def __init__(self, data, model: nn.Module, image_size=224, dataset=None, split=None):
        self.data = data.copy()
        self.model = model
        self.image_size = image_size
        if "video_path" in self.data.columns:
            self.data["video_path"] = self.data["video_path"].apply(
                lambda value: resolve_data_path(str(value), dataset, split) if pd.notna(value) else value
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.model.process_video(self.data, index, self.image_size)


class ImagePairDataset(Dataset):
    def __init__(self, data, model: nn.Module, image_size=224, dataset=None, split=None):
        self.data = data.copy()
        self.model = model
        self.image_size = image_size
        for column in ("src_img", "tgt_img", "src_state", "tgt_state", "action"):
            if column in self.data.columns:
                self.data[column] = self.data[column].apply(
                    lambda value: resolve_data_path(str(value), dataset, split) if pd.notna(value) else value
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        global_index = self.data.index[index]
        try:
            tensor = self.model.process_image(row["src_img"], row["tgt_img"], self.image_size)
        except Exception as error:
            print(f"Error loading images at index {global_index}: {error}")
            return None
        return global_index, tensor


def video_collate_fn(batch):
    batch = [value for value in batch if value is not None]
    if not batch:
        return None
    indices, tensors, relative_indices = zip(*batch)
    data = list(tensors) if isinstance(tensors[0], list) else torch.stack(tensors)
    return indices, data, torch.stack(relative_indices)


def image_collate_fn(batch):
    batch = [value for value in batch if value is not None]
    if not batch:
        return None
    indices, tensors = zip(*batch)
    data = list(tensors) if isinstance(tensors[0], list) else torch.stack(tensors)
    return indices, data


@dataclass
class ExtractionConfig:
    model: str
    dataset: str
    split: str = "all"
    batch_size: int = 16
    num_workers: int = 8
    partition: int = 0
    mode: str = "video"
    stride: int = 5


class LatentActionExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = MODEL.build(config.model)

    def extract(self, data, output_dir):
        if self.config.mode == "image":
            dataset = ImagePairDataset(
                data,
                self.model,
                dataset=self.config.dataset,
                split=self.config.split,
            )
            collate_fn = image_collate_fn
        else:
            dataset = VideoActionDataset(
                data,
                self.model,
                dataset=self.config.dataset,
                split=self.config.split,
            )
            collate_fn = video_collate_fn
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Partition {self.config.partition}"):
                if batch is None:
                    continue
                indices, batch_data = batch[:2]
                relative_indices = batch[2] if self.config.mode == "video" else None
                tokens, token_ids = self.model.get_latent_action(batch_data, relative_indices, self.config)
                for offset, global_index in enumerate(indices):
                    filename = f"latent_action_{global_index:08d}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, filename),
                        tokens=tokens[offset],
                        indices=token_ids[offset],
                    )
                    data.at[global_index, "la_path"] = os.path.join(
                        self.config.dataset,
                        self.config.split,
                        self.model.name,
                        filename,
                    )
        return data


def extract_latent_actions(
    model,
    dataset,
    input_file=None,
    output_dir=None,
    split="all",
    batch_size=16,
    num_workers=8,
    gpus=None,
    mode="video",
    stride=5,
    num_partitions=1,
    partition=0,
):
    config = get_config()
    if input_file is None:
        candidates = [
            config.data_dir / f"{dataset}_metadata_{split}.csv",
            config.data_dir / f"{dataset}_{split}.csv",
        ]
        input_file = next((str(path) for path in candidates if path.exists()), None)
        if input_file is None:
            expected = "\n".join(f"  {path}" for path in candidates)
            raise ValueError(
                f"Input file not found for dataset '{dataset}' / split '{split}'.\n"
                f"Expected one of:\n{expected}\n"
                "Use --input to specify the metadata CSV directly."
            )
    output_dir = output_dir or os.environ.get("LARY_LA_DIR")
    if output_dir is None:
        output_dir = str(config.log_dir / "latent_action")

    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[partition % len(gpus)])
    if not 0 <= partition < num_partitions:
        raise ValueError(f"partition must be in [0, {num_partitions}), received {partition}")
    setup_seed()

    save_dir = Path(output_dir) / dataset / split / model
    save_dir.mkdir(parents=True, exist_ok=True)
    full_data = pd.read_csv(input_file)
    if "la_path" not in full_data.columns:
        full_data["la_path"] = ""
    if num_partitions > 1:
        indices = np.array_split(np.arange(len(full_data)), num_partitions)[partition]
        current_data = full_data.iloc[indices].copy()
    else:
        current_data = full_data.copy()

    extractor = LatentActionExtractor(
        ExtractionConfig(
            model=model,
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            partition=partition,
            mode=mode,
            stride=stride,
        )
    )
    processed = extractor.extract(current_data, str(save_dir))

    config.data_dir.mkdir(parents=True, exist_ok=True)
    stride_suffix = f"_{stride}" if mode == "image" else ""
    partition_suffix = f"_{partition}" if num_partitions > 1 else ""
    output_csv = config.data_dir / (f"{split}_la_{dataset}{stride_suffix}_{model}{partition_suffix}.csv")
    processed.to_csv(output_csv, index=False)
    print(output_csv)
