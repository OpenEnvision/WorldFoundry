"""Truthfulness guards for generic attention provider selection and execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.core.attention import backends, dispatch
from worldfoundry.core.attention.backends import ModelSpecificAttentionBackendError

_MODEL_SPECIFIC_BACKENDS = (
    "video_sparse_attention",
    "flex_block_attention",
    "vmoba_attention",
    "sla_attention",
    "sage_sla_attention",
)


@pytest.mark.parametrize("backend", _MODEL_SPECIFIC_BACKENDS)
def test_model_specific_backend_selection_fails_before_package_fallback(backend: str) -> None:
    with pytest.raises(
        ModelSpecificAttentionBackendError,
        match=r"model-specific.*generic Q/K/V dispatcher.*requires",
    ):
        backends.resolve_attention_backend(backend, device="cuda:0")


@pytest.mark.parametrize("backend", _MODEL_SPECIFIC_BACKENDS)
def test_generic_forward_never_silently_runs_sdpa_for_sparse_name(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_sdpa(*args, **kwargs):
        raise AssertionError("generic sparse request silently reached Torch SDPA")

    monkeypatch.setattr(dispatch, "torch_sdpa", forbidden_sdpa)
    query = torch.zeros(1, 1, 2, 4)
    with pytest.raises(ModelSpecificAttentionBackendError, match="model-specific"):
        dispatch.attention_forward(
            query,
            query,
            query,
            backend=backend,
            # These used to short-circuit before validating the requested
            # provider, which made a sparse request look successful.
            compatibility_mode=True,
        )


def test_installed_sparse_packages_are_not_reported_generic_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "PathFinder",
        SimpleNamespace(
            find_spec=lambda name, path=None: SimpleNamespace(submodule_search_locations=[])
        ),
    )
    backends.probe_attention_backends.cache_clear()
    try:
        capabilities = backends._probe_attention_backends_cached(
            (9, 0),
            False,
            True,
        )
        for name in _MODEL_SPECIFIC_BACKENDS:
            assert capabilities[name].available is True
            assert capabilities[name].usable is False
            assert "model" in capabilities[name].reason.lower()
    finally:
        backends.probe_attention_backends.cache_clear()


def test_model_specific_owner_can_resolve_installed_flexblock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = backends.AttentionKernelCapability(
        name="flex_block_attention",
        package="flex_block_attn",
        available=True,
        usable=False,
        reason="model-specific metadata required",
    )
    monkeypatch.setattr(
        backends,
        "probe_attention_backends",
        lambda device=None: {"flex_block_attention": capability},
    )
    monkeypatch.setattr(
        backends,
        "_cuda_compute_capability",
        lambda device=None: (9, 0),
    )

    assert (
        backends.resolve_attention_backend(
            "flex_block_attention",
            device="cuda:0",
            allow_model_specific=True,
        )
        == "flex_block_attention"
    )


def test_provider_runtime_counter_proves_successful_kernel_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = torch.ones(1, 1, 2, 8, dtype=torch.float16)
    query = torch.zeros_like(sentinel)
    dispatch.clear_attention_dispatch_cache()
    dispatch.reset_attention_provider_runtime()
    monkeypatch.setattr(
        dispatch,
        "_select_attention_backends_cached",
        lambda *args, **kwargs: ("flash_attention_2", "torch"),
    )
    monkeypatch.setattr(dispatch, "flash_attention_2", lambda *args, **kwargs: sentinel)

    assert dispatch.attention_forward(query, query, query) is sentinel
    runtime = dispatch.attention_provider_runtime_report()
    assert runtime["flash_attention_2"] == {
        "attempts": 1,
        "successes": 1,
        "fallbacks": 0,
        "errors": 0,
        "quarantined_skips": 0,
        "compiled_graph_traces": 0,
    }
    assert "torch" not in runtime


def test_provider_runtime_counter_distinguishes_fallback_from_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = torch.ones(1, 1, 2, 8, dtype=torch.float16)
    query = torch.zeros_like(sentinel)
    dispatch.clear_attention_dispatch_cache()
    dispatch.reset_attention_provider_runtime()
    monkeypatch.setattr(
        dispatch,
        "_select_attention_backends_cached",
        lambda *args, **kwargs: ("flash_attention_2", "torch"),
    )

    def unsupported(*args, **kwargs):
        raise RuntimeError("kernel only supports a different head dimension")

    monkeypatch.setattr(dispatch, "flash_attention_2", unsupported)
    monkeypatch.setattr(dispatch, "torch_sdpa", lambda *args, **kwargs: sentinel)

    with pytest.warns(RuntimeWarning, match="trying the next eligible backend"):
        assert dispatch.attention_forward(query, query, query) is sentinel
    runtime = dispatch.attention_dispatch_report()["provider_calls"]
    assert runtime["flash_attention_2"] == {
        "attempts": 1,
        "successes": 0,
        "fallbacks": 1,
        "errors": 1,
        "quarantined_skips": 0,
        "compiled_graph_traces": 0,
    }
    assert runtime["torch"]["attempts"] == 1
    assert runtime["torch"]["successes"] == 1


def test_compiled_provider_receipt_has_no_eager_call_counter() -> None:
    dispatch.clear_attention_dispatch_cache()
    dispatch.reset_attention_provider_runtime()
    query = torch.randn(1, 1, 4, 8)

    def compiled_attention(value: torch.Tensor) -> torch.Tensor:
        return dispatch.attention_forward(
            value,
            value,
            value,
            compatibility_mode=True,
        )

    compiled = torch.compile(compiled_attention, backend="eager", fullgraph=True)
    wrapper_receipts: dict[str, int] = {}
    with dispatch.attention_compile_receipt_scope(wrapper_receipts):
        output = compiled(query)
    assert output.shape == query.shape
    runtime = dispatch.attention_provider_runtime_report()["torch"]
    assert runtime["compiled_graph_traces"] > 0
    assert wrapper_receipts["torch"] > 0
    assert runtime["attempts"] == 0
    assert runtime["successes"] == 0
    assert runtime["fallbacks"] == 0
    assert runtime["errors"] == 0
    assert runtime["quarantined_skips"] == 0


def test_load_time_attention_policy_does_not_configure_unwired_sparse_backend() -> None:
    from worldfoundry.core.model_loading.optimize import apply_attention_policy

    class Seam(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.configured: list[str] = []

        def set_attention_backend(self, value: str) -> None:
            self.configured.append(value)

    model = Seam()
    with pytest.raises(ModelSpecificAttentionBackendError, match="checkpoint-compatible"):
        apply_attention_policy(model, "video_sparse_attention", device="cuda:0")
    assert model.configured == []
