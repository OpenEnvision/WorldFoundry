# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""VQAScore entrypoints preserve answer probabilities without weights or API calls."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

_PACKAGE = "worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models"


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    return str(path)


@pytest.mark.parametrize(("answer", "token_id"), [("Yes", 1), ("No", 2)])
def test_llama32_forward_selects_answer_probability(image_path: str, answer: str, token_id: int) -> None:
    module = importlib.import_module(f"{_PACKAGE}.llama32_model")
    scorer = object.__new__(module.LLaMA32VisionModel)
    inputs = Mock()
    inputs.to.return_value = {"input_ids": torch.tensor([[0]])}
    processor = Mock(return_value=inputs)
    processor.apply_chat_template.return_value = "formatted question"
    processor.tokenizer.encode.return_value = [0, token_id]
    logits = [torch.tensor([[0.0, 1.0, 2.0, 3.0]]), torch.tensor([[3.0, 2.0, 1.0, 0.0]])]
    scorer.processor = processor
    scorer.model = SimpleNamespace(
        device="cpu", generate=Mock(side_effect=[SimpleNamespace(scores=[row]) for row in logits])
    )

    result = scorer.forward([image_path, image_path], ["a cat", "a dog"], answer_template=answer)

    expected = torch.tensor([math.exp(row[0, token_id].item()) / sum(map(math.exp, row[0])) for row in logits])
    torch.testing.assert_close(result, expected)
    assert result.shape == (2,)
    assert result.dtype == torch.get_default_dtype()
    assert result.device.type == "cpu"
    assert not result.requires_grad
    assert processor.tokenizer.encode.call_count == 2
    processor.tokenizer.encode.assert_called_with(answer)
    inputs.to.assert_called_with("cpu")
    for call in scorer.model.generate.call_args_list:
        assert call.kwargs["max_new_tokens"] == 1
        assert call.kwargs["output_scores"] is True
        assert call.kwargs["return_dict_in_generate"] is True
    question = processor.apply_chat_template.call_args.args[0][0]["content"][1]["text"]
    assert question == 'Does this image show "a dog"? Answer the question with Yes or No'


def test_molmo_forward_accepts_matching_images_and_texts(image_path: str) -> None:
    module = importlib.import_module(f"{_PACKAGE}.molmo_model")
    scorer: Any = object.__new__(module.MOLMOVisionModel)
    processor = Mock()
    processor.process.return_value = {"input_ids": torch.tensor([0, 1])}
    processor.tokenizer.encode.return_value = [2]
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    scorer.processor = processor
    scorer.model = SimpleNamespace(
        device="cpu", generate_from_batch=Mock(return_value=SimpleNamespace(scores=[logits]))
    )

    result = scorer.forward(images=[image_path], texts=["a cat"])

    torch.testing.assert_close(result, torch.tensor([math.exp(2) / sum(map(math.exp, range(4)))]))
    assert result.shape == (1,)
    assert result.dtype == torch.get_default_dtype()
    assert result.device.type == "cpu"
    assert not result.requires_grad
    processor.tokenizer.encode.assert_called_once_with(" Yes")
    processed = processor.process.call_args.kwargs
    assert processed["text"] == 'Does this image show "a cat"? Answer the question with Yes or No'
    assert processed["images"][0].mode == "RGB"
    batch, config = scorer.model.generate_from_batch.call_args.args
    assert torch.equal(batch["input_ids"], torch.tensor([[0, 1]]))
    assert config.max_new_tokens == 1
    assert config.output_scores is True
    assert config.return_dict_in_generate is True


@pytest.mark.parametrize(
    ("module_name", "class_name"), [("llama32_model", "LLaMA32VisionModel"), ("molmo_model", "MOLMOVisionModel")]
)
def test_vision_forward_rejects_mismatched_lengths_before_loading(module_name: str, class_name: str) -> None:
    module = importlib.import_module(f"{_PACKAGE}.{module_name}")
    scorer = object.__new__(getattr(module, class_name))
    scorer.load_images = Mock()

    with pytest.raises(AssertionError, match="^Number of paths and texts must match$"):
        scorer.forward(["not-loaded.png"], [])

    scorer.load_images.assert_not_called()


@pytest.fixture
def gemini_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Load the real module in isolation so optional SDK stubs cannot leak to other tests.
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    types = ModuleType("google.genai.types")
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setattr(types, "GenerateContentConfig", Mock(), raising=False)
    monkeypatch.setattr(types, "Part", SimpleNamespace(from_bytes=Mock(return_value="image part")), raising=False)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    package = importlib.import_module(_PACKAGE)
    source = Path(package.__file__).with_name("gemini_model.py")
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}._test_gemini_model", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_gemini_forward_single_preserves_probability_and_retry_limit(gemini_module: Any, failures: int) -> None:
    scorer = object.__new__(gemini_module.GeminiModel)
    scorer.model_name = "gemini-2.5-flash"
    candidates = [
        SimpleNamespace(token="No", log_probability=-0.01),
        SimpleNamespace(token=" yes ", log_probability=-0.8),
        SimpleNamespace(token="Yes", log_probability=-0.1),
    ]
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(logprobs_result=SimpleNamespace(top_candidates=[SimpleNamespace(candidates=candidates)]))
        ]
    )
    generate_content = Mock(side_effect=[RuntimeError("transient provider error")] * failures + [response])
    scorer.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    data = {"type": "image", "data": b"image", "mime_type": "image/png", "path": "image.png"}

    result = scorer.forward_single(data, "Does this image show a cat?", "Yes")

    assert result.shape == (1,)
    assert result.device.type == "cpu"
    assert result.item() == pytest.approx(0.0 if failures == 2 else math.exp(-0.1))
    assert generate_content.call_count == min(failures + 1, 2)
    for call in generate_content.call_args_list:
        assert call.kwargs["model"] == scorer.model_name
        assert call.kwargs["contents"] == ["Does this image show a cat?", "image part"]
    if failures:
        assert generate_content.call_args_list[0] == generate_content.call_args_list[1]
