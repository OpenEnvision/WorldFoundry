"""The dispatcher accepts a request-scoped provider without global mutation."""

from __future__ import annotations

import torch

from worldfoundry.core.attention.backends.probe import normalize_attention_backend
from worldfoundry.core.model_loading.policy import AttentionBackend


def test_attention_forward_normalizes_request_scoped_backend(monkeypatch) -> None:
    from worldfoundry.core.attention.backends import dispatch

    selected: list[str] = []

    def fake_select(preferred, *args, **kwargs):
        del args, kwargs
        selected.append(preferred)
        return ("torch",)

    sentinel = torch.ones(1, 2, 4, 8)
    monkeypatch.setattr(dispatch, "_select_attention_backends_cached", fake_select)
    monkeypatch.setattr(dispatch, "torch_sdpa", lambda *args, **kwargs: sentinel)
    q = torch.zeros_like(sentinel)

    output = dispatch.attention_forward(q, q, q, backend="flash2")

    assert output is sentinel
    assert selected == ["flash_attention_2"]
    assert dispatch.ATTENTION_IMPLEMENTATION == "auto"


def test_flash_attention_4_public_aliases_are_explicit() -> None:
    for alias in ("flash4", "flash-attn-4", "flash_attn4", "flash_attention_4"):
        assert normalize_attention_backend(alias) == "flash_attention_4"
        assert AttentionBackend(alias) is AttentionBackend.FLASH_ATTENTION_4


def test_attention_forward_routes_explicit_flash_attention_4(monkeypatch) -> None:
    from worldfoundry.core.attention.backends import dispatch

    selected: list[str] = []
    sentinel = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)

    def fake_select(preferred, *args, **kwargs):
        del args, kwargs
        selected.append(preferred)
        return ("flash_attention_4", "torch")

    monkeypatch.setattr(dispatch, "_select_attention_backends_cached", fake_select)
    monkeypatch.setattr(dispatch, "flash_attention_4", lambda *args, **kwargs: sentinel)
    q = torch.zeros_like(sentinel)

    output = dispatch.attention_forward(q, q, q, backend="flash4")

    assert output is sentinel
    assert selected == ["flash_attention_4"]
