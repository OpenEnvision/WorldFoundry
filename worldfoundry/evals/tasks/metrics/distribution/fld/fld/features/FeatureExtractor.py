from __future__ import annotations

import json
import os
from collections.abc import Callable, Sized
from typing import Any, cast

import torch
from tqdm import tqdm

NUM_WORKERS = 4
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper class for dataset to add preprocess transform"""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    # Retain the upstream idx keyword rather than Dataset's index parameter.
    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        img, labels = self.dataset[idx]
        img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.dataset)


class FeatureExtractor:
    name: str
    features_size: int | None
    preprocess: Callable[[Any], Any] | None

    def __init__(self, save_path: str | None):
        if save_path is None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                save_path = config["feature_cache_path"]
                save_path = os.path.join(save_path, self.name)
        else:
            save_path = os.path.join(save_path, self.name)

        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # TO BE IMPLEMENTED BY EACH MODULE
        self.features_size = None
        self.preprocess = None

    def get_feature_batch(self, img_batch: torch.Tensor):
        """TO BE IMPLEMENTED BY EACH MODULE"""

    def get_features(self, imgs: torch.utils.data.Dataset, name=None, recompute=False):
        """
        Gets the features from imgs (either a Dataset) or a tensor of images.
        - name: Unique name of set of images for caching purposes
        - recompute: Whether to recompute cached features
        """
        if self.save_path and name:
            file_path = os.path.join(self.save_path, f"{name}.pt")

            if not recompute and os.path.exists(file_path):
                return torch.load(file_path)

        if isinstance(imgs, torch.utils.data.Dataset):
            features = self.get_dataset_features(imgs)
        else:
            raise NotImplementedError(f"Cannot get features from '{type(imgs)}'. Expected torch.utils.data.Dataset")

        if self.save_path and name:
            torch.save(features, file_path)

        return features

    def get_dataset_features(self, dataset: torch.utils.data.Dataset):
        size = len(cast(Sized, dataset))
        features = torch.zeros(size, cast(int, self.features_size))

        # Add preprocessing to dataset transforms
        dataset = TransformedDataset(dataset, self.preprocess)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        start_idx = 0
        for img_batch, _ in tqdm(dataloader, leave=False):
            feature = self.get_feature_batch(img_batch.to(DEVICE))

            # If going to overflow, just get required amount and break
            if size and start_idx + feature.shape[0] > size:
                features[start_idx:] = feature[: size - start_idx]
                break

            features[start_idx : start_idx + feature.shape[0]] = feature
            start_idx = start_idx + feature.shape[0]

        return features
