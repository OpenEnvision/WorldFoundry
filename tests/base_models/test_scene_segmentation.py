# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Scene segmentation uses local metadata and canonical RepViT-SAM."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def test_oneformer_loads_local_class_metadata_and_repvit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transformers import OneFormerForUniversalSegmentation
    from transformers.models.oneformer import image_processing_oneformer

    from worldfoundry.base_models.perception_core.segment import repvit_sam
    from worldfoundry.base_models.perception_core.segment.scene_segmentation import SceneSegmenter

    (tmp_path / "ade20k_panoptic.json").write_text(json.dumps({"0": {"name": "wall", "isthing": False}}))
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "class_info_file": "ade20k_panoptic.json",
                "repo_path": "remote/repo",
                "size": {"shortest_edge": 32, "longest_edge": 64},
            }
        )
    )
    (tmp_path / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    (tmp_path / "tokenizer_config.json").write_text('{"tokenizer_class":"CLIPTokenizer","model_max_length":77}')

    def reject_download(*args: object, **kwargs: object) -> None:
        pytest.fail("OneFormer class metadata must load locally during construction")

    monkeypatch.setattr(image_processing_oneformer, "hf_hub_download", reject_download)
    monkeypatch.setattr(
        OneFormerForUniversalSegmentation, "from_pretrained", lambda *args, **kwargs: torch.nn.Identity()
    )
    builder = importlib.import_module("worldfoundry.base_models.perception_core.segment.repvit_sam.build_sam")
    monkeypatch.setattr(builder, "build_sam_repvit", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(repvit_sam, "SamAutomaticMaskGenerator", lambda *args, **kwargs: object())
    segmenter = SceneSegmenter.from_pretrained(tmp_path, tmp_path / "repvit.pt", device="cpu")
    assert segmenter.processor.image_processor.repo_path == str(tmp_path)
    assert segmenter.processor.image_processor.metadata["0"] == "wall"
    inputs = segmenter.processor(images=Image.new("RGB", (32, 32)), task_inputs=["semantic"], return_tensors="pt")
    assert inputs["pixel_values"].shape == (1, 3, 32, 32)
    assert inputs["task_inputs"].shape == (1, 77)
    assert isinstance(segmenter.semantic_model, torch.nn.Identity)
