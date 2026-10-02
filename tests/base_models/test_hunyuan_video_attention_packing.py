"""Regression tests for original HunyuanVideo prompt-aware attention packing."""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video import original
from worldfoundry.core.attention import get_cu_seqlens


class _AttentionRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Size, dict[str, object]]] = []

    def __call__(self, query, key, value, **kwargs):
        del key
        self.calls.append((query.shape, kwargs))
        return value.reshape(value.shape[0], value.shape[1], -1)


def _assert_dynamic_offsets(call, expected_offsets: torch.Tensor, *, combined_length: int) -> None:
    shape, kwargs = call
    assert shape[1] == combined_length
    assert torch.equal(kwargs["cu_seqlens_q"], expected_offsets)
    assert torch.equal(kwargs["cu_seqlens_k"], expected_offsets)
    assert kwargs["max_seqlen_q"] == combined_length
    assert kwargs["max_seqlen_k"] == combined_length


def test_double_stream_uses_prompt_mask_offsets(monkeypatch) -> None:
    recorder = _AttentionRecorder()
    monkeypatch.setattr(original, "attention", recorder)
    block = original.MMDoubleStreamBlock(hidden_size=8, heads_num=2, mlp_width_ratio=2)
    image = torch.randn(1, 3, 8)
    text = torch.randn(1, 4, 8)
    conditioning = torch.randn(1, 8)
    text_mask = torch.tensor([[1, 1, 0, 0]])
    offsets = get_cu_seqlens(text_mask, image.shape[1])

    image_out, text_out = block(
        image,
        text,
        conditioning,
        None,
        cu_seqlens_q=offsets,
        cu_seqlens_kv=offsets,
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )

    assert image_out.shape == image.shape
    assert text_out.shape == text.shape
    assert len(recorder.calls) == 1
    _assert_dynamic_offsets(recorder.calls[0], offsets, combined_length=7)


def test_single_stream_uses_prompt_mask_offsets(monkeypatch) -> None:
    recorder = _AttentionRecorder()
    monkeypatch.setattr(original, "attention", recorder)
    block = original.MMSingleStreamBlock(hidden_size=8, heads_num=2, mlp_width_ratio=2)
    hidden = torch.randn(1, 7, 8)
    conditioning = torch.randn(1, 8)
    text_mask = torch.tensor([[1, 0, 0, 0]])
    offsets = get_cu_seqlens(text_mask, img_len=3)

    output = block(
        hidden,
        conditioning,
        txt_len=4,
        cu_seqlens_q=offsets,
        cu_seqlens_kv=offsets,
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )

    assert output.shape == hidden.shape
    assert len(recorder.calls) == 1
    _assert_dynamic_offsets(recorder.calls[0], offsets, combined_length=7)
