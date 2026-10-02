from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
    fused_rope_runtime_report,
    install_fused_rope_runtime,
    record_compiled_fused_rope_graph_trace,
)


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttention(), SelfAttention()])


def _dispatch(
    *,
    implementation: str,
    accelerated: bool,
    fallback: bool,
    failures: list[str] | None = None,
    quarantined: list[str] | None = None,
) -> dict[str, object]:
    return {
        "dispatches": [
            {
                "op": "hidden_qk_rmsnorm_rope_3d",
                "implementation": implementation,
                "backend": "triton" if accelerated else "torch",
                "accelerated": accelerated,
                "fallback": fallback,
                "cache_hit": False,
                "failures": [] if failures is None else failures,
                "quarantined": [] if quarantined is None else quarantined,
                "reason": None,
            }
        ]
    }


def test_fused_rope_state_requires_real_provider_execution() -> None:
    model = _TinyModel()
    state = install_fused_rope_runtime(model)
    assert state.installed_blocks == 2
    assert all(
        block._worldfoundry_fused_rope_runtime is state
        for block in model.blocks
    )

    state.record_eager_dispatch(
        _dispatch(
            implementation="triton_hidden_qk_rmsnorm_rope_3d",
            accelerated=True,
            fallback=False,
        )
    )
    report = fused_rope_runtime_report(state)
    assert report["effective"] == "accelerated-provider-executed"
    assert report["provider_calls"] == 1
    assert report["torch_fallback_calls"] == 0
    assert report["provider_paths"] == ["triton_hidden_qk_rmsnorm_rope_3d"]

    state.record_eager_dispatch(
        _dispatch(
            implementation="torch",
            accelerated=False,
            fallback=True,
        )
    )
    report = fused_rope_runtime_report(state)
    assert report["effective"] == "torch-or-failed-provider"
    assert report["torch_fallback_calls"] == 1


def test_fused_rope_request_reset_preserves_graph_receipt() -> None:
    state = install_fused_rope_runtime(_TinyModel())
    assert record_compiled_fused_rope_graph_trace(state) == 1
    state.record_eager_dispatch(
        _dispatch(
            implementation="triton_hidden_qk_rmsnorm_rope_3d",
            accelerated=True,
            fallback=False,
        )
    )

    state.reset_request_window()
    report = fused_rope_runtime_report(state)
    assert report["compiled_graph_traces"] == 1
    assert report["eager_calls"] == 0
    assert report["provider_calls"] == 0
    assert report["effective"] == "compiled-graph-traced"


def test_fused_rope_malformed_receipt_is_not_effective() -> None:
    state = install_fused_rope_runtime(_TinyModel())
    state.record_eager_dispatch({"dispatches": []})

    report = fused_rope_runtime_report(state)
    assert report["eager_calls"] == 1
    assert report["malformed_receipts"] == 1
    assert report["provider_calls"] == 0
    assert report["effective"] == "torch-or-failed-provider"


def test_fused_rope_install_is_idempotent() -> None:
    model = _TinyModel()
    first = install_fused_rope_runtime(model)
    second = install_fused_rope_runtime(model)

    assert second is first
    assert first.installed_blocks == 2
    assert isinstance(next(model.parameters()), torch.Tensor)
