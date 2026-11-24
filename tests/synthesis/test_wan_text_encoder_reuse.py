# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and conditioning contracts for canonical Wan UMT5 consumers."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl

pytestmark = pytest.mark.unit


def test_liveworld_imports_canonical_umt5() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = root / "worldfoundry/synthesis/visual_generation/liveworld/liveworld_runtime/liveworld"
    tree = ast.parse((runtime / "wrapper.py").read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "worldfoundry.base_models.diffusion_model.models.encoders.wan.reference"
        and any(alias.name == "umt5_xxl" for alias in node.names)
        for node in tree.body
    )
    assert not (runtime / "wan/modules/t5.py").exists()


def test_infinite_world_loads_canonical_umt5_and_preserves_full_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    runtime = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime.models.umt5"
    )
    assert runtime.umt5_xxl is umt5_xxl
    config = {
        "vocab_size": 32,
        "dim": 16,
        "dim_attn": 16,
        "dim_ffn": 24,
        "num_heads": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "num_buckets": 8,
        "shared_pos": False,
        "dropout": 0.0,
    }
    reference = umt5_xxl(encoder_only=True, **config).eval()
    path = tmp_path / "umt5.pth"
    torch.save(reference.state_dict(), path)
    ids = torch.tensor([[2, 3, 1, 1], [4, 5, 6, 1]])
    mask = ids.ne(1).long()

    class Tokenizer:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"name": "local-tokenizer", "seq_len": 4, "clean": "whitespace"}

        def __call__(self, texts: list[str], **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            assert texts == ["first", "second"]
            assert kwargs == {"return_mask": True, "add_special_tokens": True}
            return ids, mask

    def build(**kwargs: Any) -> torch.nn.Module:
        return umt5_xxl(**config, **kwargs)

    monkeypatch.setattr(runtime, "umt5_xxl", build)
    monkeypatch.setattr(runtime, "HuggingfaceTokenizer", Tokenizer)
    encoder = runtime.T5EncoderModel(
        model_max_length=4,
        dtype=torch.float32,
        device="cpu",
        checkpoint_path=path,
        tokenizer_path="local-tokenizer",
    )
    with torch.no_grad():
        result = encoder.encode(["first", "second"])
        expected = reference(ids, mask).float()[:, None]
    assert type(encoder.model) is type(reference)
    assert encoder.model.state_dict().keys() == reference.state_dict().keys()
    torch.testing.assert_close(result["y"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["y_mask"], mask, rtol=0, atol=0)
    assert result["y"].shape == (2, 1, 4, 16)
    assert encoder.output_dim == 16
    assert encoder.t5 is encoder
