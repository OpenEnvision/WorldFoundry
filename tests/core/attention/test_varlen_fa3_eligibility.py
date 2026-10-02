"""CPU contracts for explicit FlashAttention-3 varlen selection.

The shared dispatcher deliberately keeps external providers request-scoped;
an unversioned dense call stays on PyTorch SDPA.  An explicit FA3 request on an
ineligible runtime must report its fallback and preserve output shape/dtype.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.core.attention.sequence import varlen


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(2, 8, 2, 16)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_unversioned_dense_call_does_not_probe_external_fa3(monkeypatch) -> None:
    monkeypatch.setattr(
        varlen,
        "probe_attention_backends",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default dense attention probed an external provider")
        ),
    )
    query, key, value = _qkv()
    output = varlen.flash_attention(query, key, value, dtype=torch.bfloat16)
    assert output.shape == query.shape
    assert output.dtype == query.dtype


def test_explicit_fa3_request_on_cpu_reports_exact_fallback(monkeypatch) -> None:
    fake_fa3 = SimpleNamespace(
        flash_attn_varlen_func=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("ineligible FA3 kernel was invoked")
        )
    )
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fake_fa3)
    monkeypatch.setattr(
        varlen,
        "probe_attention_backends",
        lambda *_args, **_kwargs: {
            "flash_attention_3": SimpleNamespace(usable=False),
            "flash_attention_2": SimpleNamespace(usable=False),
        },
    )
    query, key, value = _qkv()
    with pytest.warns(UserWarning, match="FlashAttention 3 is unavailable"):
        output = varlen.flash_attention(
            query,
            key,
            value,
            dtype=torch.bfloat16,
            version=3,
        )
    assert output.shape == query.shape
    assert output.dtype == query.dtype
