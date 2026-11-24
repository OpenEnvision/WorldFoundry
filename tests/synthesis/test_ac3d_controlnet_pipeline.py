# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free prompt encoding contracts for the AC3D ControlNet pipeline."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def pipeline_module(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = importlib.import_module("worldfoundry.synthesis.visual_generation.ac3d.ac3d_runtime.controlnet_pipeline")
    logger = logging.getLogger(module.__name__)
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    return module


class T5TokenizerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.decoded_ids: torch.Tensor | None = None

    def __call__(
        self,
        prompt: list[str],
        *,
        padding: str,
        max_length: int | None = None,
        truncation: bool = False,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "padding": padding,
                "max_length": max_length,
                "truncation": truncation,
                "add_special_tokens": add_special_tokens,
                "return_tensors": return_tensors,
            }
        )
        rows = [[int(word) for word in text.split()] + [1] for text in prompt]
        if truncation:
            assert max_length is not None
            rows = [[*row[: max_length - 1], 1] if len(row) > max_length else row for row in rows]
        width = max_length if padding == "max_length" else max(map(len, rows))
        assert width is not None
        return SimpleNamespace(input_ids=torch.tensor([row + [0] * (width - len(row)) for row in rows]))

    def batch_decode(self, ids: torch.Tensor) -> list[str]:
        self.decoded_ids = ids.clone()
        return [" ".join(map(str, row.tolist())) for row in ids]


class TextEncoderStub:
    def __init__(self) -> None:
        self.input_ids: torch.Tensor | None = None

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor]:
        self.input_ids = ids.clone()
        return (ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 8),)


@pytest.mark.parametrize("words", [50, 225, 226, 300], ids=["short", "exact-limit", "one-over", "long"])
@pytest.mark.parametrize("videos", [1, 2])
def test_get_t5_prompt_embeds_preserves_tokens_and_warns_on_truncation(
    pipeline_module: ModuleType, caplog: pytest.LogCaptureFixture, words: int, videos: int
) -> None:
    pipeline = pipeline_module.ControlnetCogVideoXPipeline.__new__(pipeline_module.ControlnetCogVideoXPipeline)
    tokenizer = T5TokenizerStub()
    encoder = TextEncoderStub()
    object.__setattr__(pipeline, "tokenizer", tokenizer)
    object.__setattr__(pipeline, "text_encoder", encoder)
    prompt = " ".join(str(token) for token in range(2, words + 2))

    with caplog.at_level(logging.WARNING, logger=pipeline_module.__name__):
        embeddings = pipeline._get_t5_prompt_embeds(
            prompt, num_videos_per_prompt=videos, device=torch.device("cpu"), dtype=torch.float64
        )

    expected_ids = [*list(range(2, min(words, 225) + 2)), 1]
    expected_ids += [0] * (226 - len(expected_ids))
    assert encoder.input_ids is not None
    assert encoder.input_ids.tolist() == [expected_ids]
    assert encoder.input_ids.device.type == "cpu"
    assert tokenizer.calls == [
        {
            "padding": "max_length",
            "max_length": 226,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        },
        {
            "padding": "longest",
            "max_length": None,
            "truncation": False,
            "add_special_tokens": True,
            "return_tensors": "pt",
        },
    ]
    expected_embeddings = torch.tensor(expected_ids, dtype=torch.float64).view(1, 226, 1).repeat(videos, 1, 8)
    torch.testing.assert_close(embeddings, expected_embeddings, rtol=0, atol=0)
    assert embeddings.device.type == "cpu"
    records = [record for record in caplog.records if record.name == pipeline_module.__name__]
    if words > 225:
        removed_tokens = list(range(227, words + 2))
        assert tokenizer.decoded_ids is not None
        assert tokenizer.decoded_ids.tolist() == [removed_tokens]
        removed_text = [" ".join(map(str, removed_tokens))]
        assert [record.getMessage() for record in records] == [
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" 226 tokens: {removed_text}"
        ]
        assert records[0].levelno == logging.WARNING
    else:
        assert tokenizer.decoded_ids is None
        assert records == []
