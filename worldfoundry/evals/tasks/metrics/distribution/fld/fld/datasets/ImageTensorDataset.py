import torch
import torchvision
from torch.utils.data import Dataset


class ImageTensorDataset(Dataset):
    """Creates Dataset from a tensor of images"""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    # Retain the upstream idx keyword rather than Dataset's index parameter.
    def __getitem__(self, idx: int):  # ty: ignore[invalid-method-override]
        img = self.tensor[idx]
        img = torchvision.transforms.ToPILImage()(img)

        return img, 0
