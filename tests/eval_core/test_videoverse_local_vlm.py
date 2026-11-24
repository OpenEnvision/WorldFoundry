# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""VideoVerse forwards the canonical Qwen utility's image/video pair."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest

from worldfoundry.evals.tasks.execution.runners.videoverse.videoverse_local_vlm import local_vlm_generate

pytestmark = pytest.mark.unit


class _Inputs(dict[str, Any]):
    device: str

    @property
    def input_ids(self) -> list[list[int]]:
        return self["input_ids"]

    def to(self, device: str) -> _Inputs:
        self.device = device
        return self


def test_local_vlm_uses_canonical_two_value_vision_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    model_path = tmp_path / "checkpoint"
    video_inputs = ["sampled-video"]
    pixel_values = object()
    inputs = _Inputs(input_ids=[[10, 11]], pixel_values_videos=pixel_values)
    processor = Mock(return_value=inputs)
    processor.apply_chat_template.return_value = "formatted-prompt"
    processor.batch_decode.return_value = ["yes"]
    model = Mock(device="model-device")
    model.generate.return_value = [[10, 11, 20, 21]]
    model_loader = Mock(return_value=model)
    processor_loader = Mock(return_value=processor)
    bfloat16 = object()

    def process_vision_info(messages: list[dict[str, Any]]) -> tuple[None, list[str]]:
        return None, video_inputs

    vision = create_autospec(process_vision_info, side_effect=process_vision_info)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=bfloat16))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            Qwen2_5_VLForConditionalGeneration=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.base_models.llm_mllm_core.mllm.qwen.qwen_vl_utils",
        SimpleNamespace(process_vision_info=vision),
    )

    result = local_vlm_generate(prompt="Does the ball roll?", video_path=video_path, model_path=model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path), "max_pixels": 360 * 420, "fps": 8},
                {"type": "text", "text": "Does the ball roll?"},
            ],
        }
    ]
    assert result == "yes"
    vision.assert_called_once_with(messages)
    model_loader.assert_called_once_with(
        str(model_path),
        torch_dtype=bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=True,
    )
    processor_loader.assert_called_once_with(str(model_path), use_fast=True)
    model.eval.assert_called_once_with()
    processor.apply_chat_template.assert_called_once_with(messages, tokenize=False, add_generation_prompt=True)
    processor.assert_called_once_with(
        text=["formatted-prompt"], images=None, videos=video_inputs, padding=True, return_tensors="pt"
    )
    assert inputs.device == "model-device"
    model.generate.assert_called_once_with(**inputs, max_new_tokens=128, do_sample=False)
    processor.batch_decode.assert_called_once_with(
        [[20, 21]], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
