"""
mid.metric
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    """Generated MSRVTT and WebVid Video Captions.

    Attributes:
        prompt_file (string): The test prompt dataset
        media_size (int): Shorter one between height and width
    """

    def __init__(
        self,
        prompt_file: str | None = None,
        gen_path: str = ".",
        media_size: int = 256,
        max_num_frm: int = 64,
        preprocessor=None,
    ):
        """
        Args:
            gen_path (str, optional): Path containing generated video data
            prompt_file (str, optional): File containing test prompts
            media_size (int, optional): Shorter one between height and width
            max_text_len (int, optional): Max length of text
            max_num_frm: Max number of frames
        """
        super().__init__()
        self.prompt_file = prompt_file
        self.media_size = media_size
        self.max_num_frm = max_num_frm

        self.gen_path = gen_path
        self.load_captions()
        self.transforms = transforms.Compose(
            [
                transforms.Resize(media_size),
                transforms.CenterCrop(media_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.preprocessor = preprocessor

    def load_captions(self):
        self.captions, self.video_ids, self.sent_ids = {}, [], []
        with open(self.prompt_file) as fp:
            lines = fp.readlines()
        for sent_id, line in enumerate(lines):
            data = json.loads(line)
            video_id = str(data["video_id"])
            self.captions[sent_id] = data["prompt"]
            self.video_ids.append(video_id)
            self.sent_ids.append(sent_id)

    def get_video_frames(self, path: str):
        """
        Args:
            video_type: choose from {'real', 'fake'}
        """
        if path is None:
            return "none"

        def _get_image(path: str, filename: str) -> torch.Tensor:
            img = Image.open(os.path.join(path, filename)).convert("RGB")
            if self.preprocessor is not None:
                return self.preprocessor(img)
            else:
                return self.transforms(img)

        filenames = os.listdir(path)
        # Sort the files based on the frame index
        filenames = {int(re.findall("\d+", fn)[0]): fn for fn in filenames}
        filenames = sorted(filenames.items(), key=lambda x: x[0])
        try:
            filenames = list(zip(*filenames, strict=False))[1]
        except IndexError as error:
            print(path)
            print(os.listdir(path))
            print(filenames)
            raise ValueError(f"No video frames found in {path}") from error
        frames = []
        for filename in filenames:
            frames.append(_get_image(path, filename))
        frames = [f.unsqueeze(0) for f in frames]
        frames = torch.cat(frames, 0)

        if len(frames) > self.max_num_frm:
            frame_indices = np.arange(
                0, len(frames), len(frames) / self.max_num_frm, dtype=int
            )  # Uniformly sampling a subset of frames
            frames = frames[frame_indices]
        return frames

    def __len__(self):
        """The nubmer of samples.

        Returns:
            long: The number of samples
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """Implemetation of the `__getitem__` magic method.

        Args:
            idx (int): The index of samples

        Returns:
            Tuple[torch.Tensor, str, str, Dict]: A sample enveloped in dict
        """
        video_id = self.video_ids[idx]
        sent_id = self.sent_ids[idx]
        caption = self.captions[sent_id]
        video_path = os.path.join(self.gen_path, f"sent{sent_id}_frames")
        # The video should exist, except for real videos of unusual prompts
        if not (video_id == "None" and "real_video" in video_path):
            assert os.path.exists(video_path)
        if os.path.exists(video_path):
            video = self.get_video_frames(video_path)
            self.video_shape = video.shape
        else:
            video = torch.zeros(
                self.video_shape
            )  # for the unusual prompts, there is no reference videos, set the tensor to zero as a placeholder

        video_len = video.shape[0]
        if video_len > self.max_num_frm:
            frame_indices = np.arange(0, video_len, video_len / self.max_num_frm, dtype=int)
            video = video[frame_indices, :]

        return video, caption, video_id
