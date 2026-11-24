# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""The T2VWorldBench evaluator imports with the canonical LLaVA helpers."""

from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava import constants, conversation, mm_utils

pytestmark = pytest.mark.unit

_MODULE = "worldfoundry.evals.tasks.execution.runners.t2vworldbench.runtime.t2vworldbench.eval"


def test_evaluator_binds_the_canonical_llava_helpers() -> None:
    evaluator = importlib.import_module(_MODULE)

    assert evaluator.IMAGE_TOKEN_INDEX == constants.IMAGE_TOKEN_INDEX
    assert evaluator.DEFAULT_IMAGE_TOKEN == constants.DEFAULT_IMAGE_TOKEN
    assert evaluator.conv_templates is conversation.conv_templates
    assert evaluator.process_images is mm_utils.process_images
    assert evaluator.tokenizer_image_token is mm_utils.tokenizer_image_token


@pytest.mark.parametrize("location", ["local", "http", "https"])
def test_images_load_from_files_and_urls_as_rgb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, location: str) -> None:
    evaluator = importlib.import_module(_MODULE)
    source = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
    buffer = BytesIO()
    source.save(buffer, format="PNG")
    get = Mock(return_value=SimpleNamespace(content=buffer.getvalue()))
    monkeypatch.setattr(evaluator.requests, "get", get)
    if location == "local":
        path = tmp_path / "frame.png"
        path.write_bytes(buffer.getvalue())
        image_file = str(path)
    else:
        image_file = f"{location}://example.test/frame.png"

    image = evaluator.load_image(image_file)

    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.tobytes() == source.convert("RGB").tobytes()
    if location == "local":
        get.assert_not_called()
    else:
        get.assert_called_once_with(image_file)
