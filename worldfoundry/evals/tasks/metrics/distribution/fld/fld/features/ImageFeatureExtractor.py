from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from ..datasets.ImageFilesDataset import ImageFilesDataset
    from ..datasets.ImageTensorDataset import ImageTensorDataset
    from .FeatureExtractor import FeatureExtractor
else:
    from fld.datasets.ImageFilesDataset import ImageFilesDataset
    from fld.datasets.ImageTensorDataset import ImageTensorDataset
    from fld.features.FeatureExtractor import FeatureExtractor
from tqdm import tqdm

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFeatureExtractor(FeatureExtractor):
    def get_tensor_features(self, img_tensor: torch.Tensor, name=None, recompute=False):
        img_tensor = img_tensor.cpu()  # Ensure the tensor is on CPU (required for ToPilImage)
        dataset = ImageTensorDataset(img_tensor)
        return self.get_features(dataset, name, recompute)

    def get_dir_features(self, dir: str, extension="png", name=None, recompute=False):
        dataset = ImageFilesDataset(dir, extension=extension)
        return self.get_features(dataset, name, recompute)

    def get_model_features(self, gen_fn: Callable[[], torch.Tensor], num_samples: int):
        img_tensor = gen_fn()
        batch_size = img_tensor.shape[0]

        num_batches = math.ceil(num_samples / batch_size)
        features = torch.zeros(num_batches * batch_size, cast(int, self.features_size))

        for i in tqdm(range(num_batches)):
            img_tensor = gen_fn()
            curr_features = self.get_tensor_features(img_tensor)
            features[i * batch_size : (i + 1) * batch_size] = curr_features

        return features[:num_samples]
