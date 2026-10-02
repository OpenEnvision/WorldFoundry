"""Compatibility guards for selecting PyTorch SDPA kernels under Dynamo."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from worldfoundry.core.attention.backends import native


def test_sdpa_kernel_context_is_skipped_while_compiling(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_sdpa_kernel(*args, **kwargs):
        assert not args
        calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)

    with native._sdpa_kernel_context(backends=("math",)):
        pass

    assert calls == []


def test_native_sdpa_is_cpu_fullgraph_compilable() -> None:
    query = torch.randn(1, 2, 8, 16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with torch.no_grad():
        expected = native.scaled_dot_product_attention(
            query,
            key,
            value,
            backends=("math",),
        )
        compiled = torch.compile(
            native.scaled_dot_product_attention,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(query, key, value, backends=("math",))
    torch.testing.assert_close(actual, expected)
