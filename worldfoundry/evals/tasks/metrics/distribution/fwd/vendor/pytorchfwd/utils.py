"""Utilities file."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch as th
from PIL import Image


def _parse_args():
    """Argument parser."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for wavelet packet transform.",
    )
    parser.add_argument("--num-processes", type=int, default=None, help="Number of multiprocess.")
    parser.add_argument("--save-packets", action="store_true", help="Save the packets as npz file.")
    parser.add_argument("--wavelet", type=str, default="Haar", help="Choice of wavelet.")
    parser.add_argument("--max_level", type=int, default=4, help="wavelet decomposition level")
    parser.add_argument("--log_scale", action="store_true", help="Use log scaling for wavelets.")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize the images to specified resolution.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Set PyTorch to deterministic mode, for perfect reproducability.",
    )
    parser.add_argument(
        "path",
        type=str,
        nargs=2,
        help="Path to the generated images or path to .npz statistics file.",
    )
    return parser.parse_args()


class ImagePathDataset(th.utils.data.Dataset):
    """Image dataset."""

    def __init__(self, files, transforms=None):
        """File initialization."""
        self.files = files
        self.transforms = transforms

    def __len__(self):
        """Length of dataset."""
        return len(self.files)

    def __getitem__(self, i):
        """Load the image."""
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img
