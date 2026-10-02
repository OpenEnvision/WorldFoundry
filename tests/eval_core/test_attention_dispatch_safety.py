from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_fa3_is_not_marked_usable_on_ampere(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import backends

    def fake_package_capability(**kwargs):
        return backends.AttentionKernelCapability(
            name=kwargs["name"],
            package=kwargs["package"],
            available=True,
            usable=bool(kwargs["usable_if"]),
            reason="" if kwargs["usable_if"] else kwargs["unusable_reason"],
        )

    monkeypatch.setattr(backends, "_cuda_compute_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(backends, "_package_capability", fake_package_capability)
    backends.probe_attention_backends.cache_clear()
    try:
        capabilities = backends.probe_attention_backends()
        assert capabilities["flash_attention_2"].usable is True
        assert capabilities["flash_attention_3"].usable is False
        assert "Hopper" in capabilities["flash_attention_3"].reason
    finally:
        backends.probe_attention_backends.cache_clear()


def test_fa3_is_marked_usable_only_on_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import backends

    monkeypatch.setattr(backends, "_cuda_compute_capability", lambda device=None: (9, 0))
    assert backends.gpu_supports_flash_attention_3() is True
    monkeypatch.setattr(backends, "_cuda_compute_capability", lambda device=None: (10, 0))
    assert backends.gpu_supports_flash_attention_3() is False
    monkeypatch.setattr(backends, "_cuda_compute_capability", lambda device=None: (12, 0))
    assert backends.gpu_supports_flash_attention_3() is False


def test_torch_sdpa_propagates_oom_instead_of_allocating_dense_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.core.attention import dispatch

    def raise_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(dispatch, "_worldfoundry_scaled_dot_product_attention", raise_oom)
    query = torch.zeros(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match="out of memory"):
        dispatch.torch_sdpa(query, query, query)


def _configure_flash2(monkeypatch: pytest.MonkeyPatch, dispatch, error: RuntimeError) -> None:
    monkeypatch.setattr(
        dispatch,
        "resolve_attention_backend",
        lambda preferred=None, device=None: "flash_attention_2",
    )
    monkeypatch.setattr(
        dispatch,
        "attention_backend_capability",
        lambda name, device=None: SimpleNamespace(usable=True),
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(dispatch, "flash_attention_2", fail)


def test_attention_forward_propagates_backend_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import dispatch

    _configure_flash2(monkeypatch, dispatch, RuntimeError("CUDA out of memory"))
    query = torch.zeros(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match="out of memory"):
        dispatch.attention_forward(query, query, query)


def test_attention_forward_falls_back_only_for_explicit_unsupported_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.core.attention import dispatch

    _configure_flash2(monkeypatch, dispatch, RuntimeError("kernel only supports head dimension 128"))
    sentinel = torch.ones(1, 1, 2, 4)
    monkeypatch.setattr(dispatch, "torch_sdpa", lambda *args, **kwargs: sentinel)
    query = torch.zeros_like(sentinel)
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert dispatch.attention_forward(query, query, query) is sentinel


def test_attention_forward_propagates_unknown_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import dispatch

    _configure_flash2(monkeypatch, dispatch, RuntimeError("invalid tensor stride"))
    query = torch.zeros(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match="invalid tensor stride"):
        dispatch.attention_forward(query, query, query)


def test_attention_forward_propagates_backend_internal_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import dispatch

    monkeypatch.setattr(dispatch, "resolve_attention_backend", lambda preferred=None, device=None: "flash_attention_2")
    monkeypatch.setattr(
        dispatch,
        "attention_backend_capability",
        lambda name, device=None: SimpleNamespace(usable=True),
    )

    def fail(*args, **kwargs):
        raise OSError("failed reading an internal tuning database")

    monkeypatch.setattr(dispatch, "flash_attention_2", fail)
    query = torch.zeros(1, 1, 2, 4)
    with pytest.raises(OSError, match="tuning database"):
        dispatch.attention_forward(query, query, query)


def test_attention_forward_falls_back_for_missing_optional_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention import dispatch

    monkeypatch.setattr(dispatch, "resolve_attention_backend", lambda preferred=None, device=None: "flash_attention_2")
    monkeypatch.setattr(
        dispatch,
        "attention_backend_capability",
        lambda name, device=None: SimpleNamespace(usable=True),
    )

    def fail(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'flash_attn'", name="flash_attn")

    sentinel = torch.ones(1, 1, 2, 4)
    monkeypatch.setattr(dispatch, "flash_attention_2", fail)
    monkeypatch.setattr(dispatch, "torch_sdpa", lambda *args, **kwargs: sentinel)
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert dispatch.attention_forward(sentinel, sentinel, sentinel) is sentinel


@pytest.mark.parametrize(
    "relative_path",
    (
        "worldfoundry/core/attention/patch_xdit_context_parallel.py",
        "worldfoundry/core/attention/scope_xdit_context_parallel.py",
    ),
)
def test_xdit_attention_hot_path_does_not_empty_allocator_cache(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    assert "empty_cache(" not in source
