import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, imagefolder_path, img_size=256, seq_len=16, stride=1):
        self.imagefolder_path = imagefolder_path
        self.folder_image_list = os.listdir(imagefolder_path)
        self.img_size = img_size
        self.data = []
        for img_name in self.folder_image_list:
            # TODO: remove this line
            if "_dwpose" in img_name:
                continue
            image_path = os.path.join(self.imagefolder_path, img_name)
            files = os.listdir(image_path)
            files.sort()
            files = [os.path.join(image_path, f) for f in files if f.endswith((".jpg", ".png", ".jpeg"))]
            for i in range(0, len(files) - seq_len + 1, stride):
                self.data.append(files[i : i + seq_len])

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        files = self.data[idx]
        frames = []
        for f in files:
            image = Image.open(f)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = self.transform(image)
            frames.append(image)
        frames = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2)  # S,C,H,W
        return frames


class VideoDatasetNP(Dataset):
    def __init__(self, video_list, img_size=256, seq_len=16, stride=1):
        self.video_list = video_list
        self.img_size = img_size
        self.data = []
        for video in self.video_list:
            for i in range(0, len(video) - seq_len + 1, stride):
                self.data.append(video[i : i + seq_len])

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # S,C,H,W
        frames = self.transform(frames)
        return frames
