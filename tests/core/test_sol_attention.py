"""Strict opt-in Sol-Attn contracts and optional real CUDA execution."""

import importlib.util

import pytest
import torch

from worldfoundry.core.attention.backends.dispatch import (
    attention_forward,
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.core.attention.backends.probe import probe_attention_backends, resolve_attention_backend


def test_sol_is_explicit_and_unavailable_cpu_fails():
    assert resolve_attention_backend("auto", "cpu") == "torch"
    assert not probe_attention_backends("cpu")["sol_attn"].usable
    with pytest.raises(RuntimeError, match="Explicit sol_attn"):
        resolve_attention_backend("sol", "cpu")


def test_sol_mask_does_not_silently_execute_dense_attention():
    value = torch.ones(1, 2, 4, 128)
    with pytest.raises(ValueError, match="masks"):
        attention_forward(value, value, value, backend="sol_attn", attn_mask=torch.ones(4, 4))


def test_options_cannot_be_silently_ignored_by_dense_backend():
    value = torch.ones(1, 2, 4, 128)
    with pytest.raises(ValueError, match="backend_options"):
        attention_forward(value, value, value, backend="torch", backend_options={"tau": 1.0})


@pytest.mark.parametrize(
    "options", [{"tau": True}, {"tau": float("nan")}, {"thresh_type": "unknown"}, {"kv_splits": 2}, {"kv_splits": 1.0}]
)
def test_invalid_provider_parameters_fail_before_execution(options):
    from worldfoundry.core.attention.backends.sol import validate_sol_options

    with pytest.raises(ValueError):
        validate_sol_options(**options)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("sol_attn") is None,
    reason="requires optional Sol-Attn CUDA environment",
)
def test_sol_cuda_executes_with_receipt_and_validates_shape():
    if not probe_attention_backends("cuda")["sol_attn"].usable:
        pytest.skip("Sol-Attn runtime version requirements are not satisfied")
    torch.manual_seed(21)
    value = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
    reset_attention_provider_runtime()
    with torch.inference_mode():
        expected = attention_forward(
            value,
            value,
            value,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            backend="torch",
        )
        actual = attention_forward(
            value,
            value,
            value,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            backend="sol_attn",
        )
        assert torch.isfinite(actual).all()
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.01
        with pytest.raises(ValueError, match="self-attention"):
            attention_forward(
                value[..., :64],
                value[..., :64],
                value[..., :64],
                backend="sol_attn",
                q_pattern="b s n d",
                k_pattern="b s n d",
                v_pattern="b s n d",
            )
    assert attention_provider_runtime_report()["sol_attn"]["successes"] == 1
    assert attention_provider_runtime_report()["sol_attn"]["fallbacks"] == 0
