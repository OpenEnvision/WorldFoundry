# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""ADE20K scene labels and canonical RepViT-SAM region proposals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class SceneSegmenter:
    """Resident segmentation weights; each call owns its image embeddings."""

    def __init__(self, semantic_model: Any, processor: Any, mask_generator: Any, *, device: str) -> None:
        self.semantic_model = semantic_model
        self.processor = processor
        self.mask_generator = mask_generator
        self.device = device

    @classmethod
    def from_pretrained(
        cls, oneformer_path: str | Path, repvit_checkpoint: str | Path, *, device: str
    ) -> SceneSegmenter:
        """Load local OneFormer metadata and the existing RepViT-SAM network."""
        from transformers import (
            CLIPTokenizer,
            OneFormerForUniversalSegmentation,
            OneFormerImageProcessor,
            OneFormerProcessor,
        )

        from .repvit_sam import SamAutomaticMaskGenerator
        from .repvit_sam.build_sam import build_sam_repvit

        root = Path(oneformer_path).expanduser()
        if not (root / "ade20k_panoptic.json").is_file():
            raise FileNotFoundError(f"OneFormer ADE20K metadata is missing: {root / 'ade20k_panoptic.json'}")
        image_config = json.loads((root / "preprocessor_config.json").read_text())
        image_config["repo_path"] = str(root)
        processor = OneFormerProcessor(
            image_processor=OneFormerImageProcessor.from_dict(image_config),
            tokenizer=CLIPTokenizer.from_pretrained(root, local_files_only=True),
        )
        model = (
            OneFormerForUniversalSegmentation.from_pretrained(root, local_files_only=True, use_safetensors=True)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        sam = build_sam_repvit(checkpoint=str(repvit_checkpoint)).eval().requires_grad_(False).to(device)
        generator = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.9)
        return cls(model, processor, generator, device=device)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Return ADE20K labels and area-descending independent region masks."""
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.semantic_model(**inputs)
        labels = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])[
            0
        ]
        try:
            regions = self.mask_generator.generate(np.asarray(image))
        finally:
            self.mask_generator.predictor.reset_image()
        return labels, sorted(regions, key=lambda item: item["area"], reverse=True)
