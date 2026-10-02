"""Contract tests for the shared CUDA-graph denoiser mixin (CPU-only).

GPU capture/replay parity is covered by the H100 validation scripts; here we
lock the opt-in wiring that must hold on any device: disabled by default (no
runner, direct passthrough), correct passthrough of positional + keyword args,
and the report shape. These are the invariants each denoiser relies on when it
routes its inner ``self.model(...)`` call through ``_run_network``.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.graph_wrapped import (
    GraphWrappedDenoiserMixin,
    resolve_cuda_graph_option,
    validate_cuda_graph_options,
)


class _StubDenoiser(GraphWrappedDenoiserMixin):
    def __init__(self, model, *, enable_cuda_graph: bool = False) -> None:
        self.model = model
        self._init_graph_runner(model, enabled=enable_cuda_graph, extra_key="stub")

    def run(self, *args, **kwargs):
        return self._run_network(*args, **kwargs)


def _model(x: torch.Tensor, *, scale: float = 1.0, bias=None) -> torch.Tensor:
    y = x * scale
    if bias is not None:
        y = y + bias
    return y


def test_disabled_by_default_has_no_runner() -> None:
    d = _StubDenoiser(_model)
    assert d._graph_runner is None
    assert d.graph_report() is None


def test_passthrough_matches_direct_call_when_disabled() -> None:
    d = _StubDenoiser(_model)
    x = torch.randn(4, 8)
    bias = torch.randn(4, 8)
    out = d.run(x, scale=2.0, bias=bias)
    torch.testing.assert_close(out, _model(x, scale=2.0, bias=bias))


def test_enabled_falls_back_to_eager_on_cpu_but_stays_correct() -> None:
    # CPU tensors are not graph-eligible, so the runner runs eagerly; the
    # output must still match the direct call and the runner must exist.
    d = _StubDenoiser(_model, enable_cuda_graph=True)
    assert d._graph_runner is not None
    x = torch.randn(3, 5)
    out = d.run(x, scale=0.5)
    torch.testing.assert_close(out, _model(x, scale=0.5))
    report = d.graph_report()
    assert report is not None and report["eager"] >= 1 and report["capture"] == 0


def test_graph_mixin_starts_a_clean_request_window() -> None:
    d = _StubDenoiser(_model, enable_cuda_graph=True)
    x = torch.randn(3, 5)
    d.run(x)
    first = d.graph_report()
    assert first is not None and first["eager"] == 1

    d._begin_graph_request_window()

    clean = d.graph_report()
    assert clean is not None
    assert clean["window_id"] == 1
    assert clean["eager"] == 0
    assert clean["lifetime"]["eager"] == 1


class _Ctx:
    def __init__(self, component: dict, policy_options: dict) -> None:
        self.component_options = component
        self.policy = type("P", (), {"options": policy_options})()


def test_resolve_option_precedence() -> None:
    # Component-level flag overrides the run-wide policy flag.
    assert resolve_cuda_graph_option(_Ctx({"cuda_graph": True}, {"cuda_graph": False})) is True
    assert resolve_cuda_graph_option(_Ctx({}, {"cuda_graph": True})) is True
    assert resolve_cuda_graph_option(_Ctx({}, {})) is False
    assert resolve_cuda_graph_option(_Ctx({"cuda_graph": False}, {"cuda_graph": True})) is False


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("approximate_attention", {"kind": "vsa"}),
        ("dualblock", {"residual_diff_threshold": 0.1}),
        ("dynamicblock", {"residual_diff_threshold": 0.1}),
        ("firstblock", {"residual_diff_threshold": 0.1}),
        ("static_cross_kv", True),
        ("teacache", 0.15),
        ("teacache_thresh", 0.2),
    ),
)
def test_cuda_graph_rejects_stateful_runtime_options(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        validate_cuda_graph_options(_Ctx({}, {"cuda_graph": True, name: value}))


def test_cuda_graph_rejects_component_level_stateful_options() -> None:
    with pytest.raises(ValueError, match="static_cross_kv"):
        validate_cuda_graph_options(
            _Ctx(
                {"cuda_graph": True, "static_cross_kv": True},
                {},
            )
        )


def test_disabled_cuda_graph_does_not_reject_other_optimizations() -> None:
    validate_cuda_graph_options(
        _Ctx(
            {"cuda_graph": False},
            {
                "cuda_graph": True,
                "static_cross_kv": True,
                "teacache": 0.15,
                "approximate_attention": "vsa",
            },
        )
    )
