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
    "options",
    [
        {"tau": True},
        {"tau": float("nan")},
        {"thresh_type": "unknown"},
        {"kv_splits": 2},
        {"kv_splits": 1.0},
        {"compress_kv": 0},
        {"compress_kv": 1},
        {"compress_kv": "false"},
        {"compress_kv": None},
    ],
)
def test_invalid_provider_parameters_fail_before_execution(options):
    from worldfoundry.core.attention.backends.sol import validate_sol_options

    with pytest.raises(ValueError):
        validate_sol_options(**options)


def test_compression_is_explicit_and_keeps_approximate_installation_receipt(monkeypatch):
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
    from worldfoundry.core.attention.backends import probe
    from worldfoundry.core.attention.backends.sol import validate_sol_options
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    assert validate_sol_options()["compress_kv"] is True
    assert validate_sol_options(compress_kv=False)["compress_kv"] is False
    monkeypatch.setattr(probe, "resolve_attention_backend", lambda *args: "sol_attn")
    model = torch.nn.Module()
    model.self_attn = SelfAttention(128, 1)
    session = install_diffusion_accelerations(
        model,
        {"attention_policy": {"self": {"backend": "sol_attn", "compress_kv": False}}},
        RuntimePolicy(dtype=torch.bfloat16),
    )
    receipt = session.report()["installed"][0]
    assert receipt["approximate"] is True
    assert receipt["scopes"]["self"]["options"]["compress_kv"] is False
    assert model.self_attn.attn._worldfoundry_attention_options["compress_kv"] is False
    session.uninstall()


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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("sol_attn") is None,
    reason="requires optional Sol-Attn CUDA environment",
)
def test_uncompressed_cuda_path_matches_upstream_sink_and_preserves_scale(monkeypatch):
    import sol_attn

    from worldfoundry.core.attention.backends.sol import sol_attention

    if not probe_attention_backends("cuda")["sol_attn"].usable:
        pytest.skip("Sol-Attn runtime version requirements are not satisfied")
    torch.manual_seed(91)
    q, k, v = [torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    original = sol_attn.sol_attn
    forwarded = []

    def capture(*args, **kwargs):
        forwarded.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sol_attn, "sol_attn", capture)
    with torch.inference_mode():
        for scale in (None, 0.07):
            actual = sol_attention(q, k, v, scale=scale, tau=0.0, compress_kv=False)
            assert forwarded[-1]["sink_tokens"] == q.shape[1] and forwarded[-1]["scale"] == scale
            expected = original(q, k, v, scale=scale, tau=0.0, sink_tokens=q.shape[1])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dense = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(),
                k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
                scale=scale,
            ).transpose(1, 2)
            assert torch.isfinite(actual).all()
            assert (actual.float() - dense).norm() / dense.norm() < 0.01
        compressed = sol_attention(q, k, v, tau=0.0)
        assert forwarded[-1]["sink_tokens"] == 0
        torch.testing.assert_close(compressed, original(q, k, v, tau=0.0), rtol=0, atol=0)
