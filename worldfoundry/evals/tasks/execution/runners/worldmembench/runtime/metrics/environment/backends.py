# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Local-only full-frame DINOv2 and SegFormer inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_REJECT_LABELS = (
    "person",
    "car",
    "boat",
    "bus",
    "truck",
    "van",
    "ship",
    "minibike",
    "animal",
    "bicycle",
    "sky",
    "water",
    "sea",
    "river",
    "lake",
    "mirror",
)


class DinoVisibleFeatures:
    def __init__(self, checkpoint, device, batch_size=8, *, model):
        import torch
        from transformers import AutoImageProcessor

        if batch_size < 1:
            raise ValueError("DINO batch size must be positive")
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.torch, self.device, self.batch_size = torch, torch.device(device), batch_size
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = model

    def embed(self, images_rgb):
        from PIL import Image

        output = []
        for start in range(0, len(images_rgb), self.batch_size):
            images = [
                Image.fromarray(np.asarray(value, dtype=np.uint8))
                for value in images_rgb[start : start + self.batch_size]
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                features = self.model(**inputs).float()
                features = self.torch.nn.functional.normalize(features, dim=-1)
            output.append(features.cpu().numpy())
        return np.concatenate(output, axis=0)


class SegformerMasks:
    def __init__(self, checkpoint, device, batch_size=4, reject_labels=DEFAULT_REJECT_LABELS):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        if batch_size < 1:
            raise ValueError("semantic batch size must be positive")
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.torch, self.device, self.batch_size = torch, torch.device(device), batch_size
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = (
            AutoModelForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=True).to(self.device).eval()
        )
        rejected = {str(label).strip().casefold() for label in reject_labels}
        # Exact label matching: rejecting car must not reject carpet.
        self.rejected_ids = [
            int(index)
            for index, label in self.model.config.id2label.items()
            if str(label).strip().casefold() in rejected
        ]

    def keep_masks(self, images_rgb):
        from PIL import Image

        output = []
        for start in range(0, len(images_rgb), self.batch_size):
            chunk = images_rgb[start : start + self.batch_size]
            images = [Image.fromarray(np.asarray(value, dtype=np.uint8)) for value in chunk]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits
            segmentations = self.processor.post_process_semantic_segmentation(
                SimpleNamespace(logits=logits), target_sizes=[tuple(image.shape[:2]) for image in chunk]
            )
            output.extend(~np.isin(labels.cpu().numpy(), self.rejected_ids) for labels in segmentations)
        return output
