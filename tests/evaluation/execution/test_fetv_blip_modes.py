# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""FETV BLIP scoring supports both enabled and disabled image-text matching."""

from __future__ import annotations

import ast
import datetime
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/fetv/runtime/fetv_eval/auto_eval.py"
)


def _score_function() -> Callable[..., Any]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"pre_caption", "compute_blipscore"}
    ]
    namespace: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "re": re,
        "time": time,
        "datetime": datetime,
        "tqdm": lambda iterable: iterable,
    }
    exec(compile(tree, str(_SOURCE), "exec"), namespace)
    return namespace["compute_blipscore"]


class _Loader(list[tuple[torch.Tensor, None, None]]):
    dataset: SimpleNamespace


@pytest.mark.parametrize("mode", ["default", "enabled", "disabled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_blip_modes_preserve_scores_and_optional_itm_result(mode: str, dtype: torch.dtype) -> None:
    text_features = torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    visual_features = torch.arange(1, 33, dtype=dtype).reshape(4, 2, 4).requires_grad_()
    logits = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=dtype, requires_grad=True)
    model = SimpleNamespace(
        eval=Mock(),
        text_encoder=Mock(return_value=SimpleNamespace(last_hidden_state=text_features)),
        text_proj=Mock(side_effect=lambda value: value),
        visual_encoder=Mock(return_value=visual_features),
        vision_proj=Mock(side_effect=lambda value: value),
        itm_head=Mock(return_value=logits),
    )
    tokens = SimpleNamespace(
        input_ids=torch.arange(1, 7, dtype=torch.long).reshape(2, 3),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )
    tokens.to = Mock(return_value=tokens)
    tokenizer = Mock(return_value=tokens)
    tokenizer.additional_special_tokens_ids = [99]
    video = torch.arange(48, dtype=dtype).reshape(2, 2, 3, 2, 2)
    original_video = video.clone()
    loader = _Loader([(video, None, None)])
    loader.dataset = SimpleNamespace(captions={0: "First caption!", 1: "Second caption."})
    kwargs = {} if mode == "default" else {"use_itm": mode == "enabled"}

    dot_scores, itm_scores = _score_function()(model, loader, tokenizer, "cpu", {"batch_size": 2}, None, **kwargs)

    expected_video = F.normalize(visual_features[:, 0, :].reshape(2, 2, 4).mean(dim=1), dim=-1)
    expected_text = F.normalize(text_features[:, 0, :], dim=-1)
    expected_dots = (expected_video * expected_text).sum(dim=-1)
    assert torch.equal(dot_scores, expected_dots)
    assert dot_scores.dtype == dtype
    assert dot_scores.device.type == "cpu"
    assert not dot_scores.requires_grad
    assert torch.equal(video, original_video)
    assert torch.equal(tokens.input_ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    model.eval.assert_called_once_with()
    tokenizer.assert_called_once_with(
        ["first caption", "second caption"],
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    )
    tokens.to.assert_called_once_with("cpu")
    model.visual_encoder.assert_called_once()
    assert torch.equal(model.visual_encoder.call_args.args[0], original_video.reshape(4, 3, 2, 2))
    assert model.text_encoder.call_args_list[0].kwargs["mode"] == "text"
    if mode == "disabled":
        assert itm_scores is None
        model.itm_head.assert_not_called()
        assert model.text_encoder.call_count == 1
    else:
        assert torch.equal(itm_scores, F.softmax(logits, dim=1)[:, 1])
        assert itm_scores.dtype == dtype
        assert itm_scores.device.type == "cpu"
        assert not itm_scores.requires_grad
        model.itm_head.assert_called_once()
        assert model.text_encoder.call_count == 2
        itm_call = model.text_encoder.call_args_list[1]
        assert torch.equal(itm_call.args[0], torch.tensor([[99, 2, 3], [99, 5, 6]]))
        context = itm_call.kwargs
        assert context["return_dict"] is True
        assert torch.equal(context["encoder_hidden_states"], visual_features.reshape(2, 4, 4))
        assert torch.equal(context["encoder_attention_mask"], torch.ones(2, 4, dtype=torch.long))
