"""CPU contracts for the explicit dense TMA provider."""

import builtins
import importlib.util
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("triton")

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.attention.backends import probe as backends
from worldfoundry.core.attention.backends import triton_tma as tma
from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend
from worldfoundry.core.attention.backends.triton_tma import is_triton_tma_supported, triton_tma_sdpa


def test_dense_tma_cpu_requests_are_rejected_without_sdpa_fallback():
    query = torch.ones(1, 37, 2, 64, dtype=torch.bfloat16)
    key = torch.ones(1, 53, 2, 64, dtype=torch.bfloat16)
    assert not is_triton_tma_supported(query, key, key)
    with patch.object(
        torch.nn.functional, "scaled_dot_product_attention", side_effect=AssertionError("dense fallback")
    ):
        with pytest.raises(RuntimeError, match="CUDA FP16/BF16"):
            triton_tma_sdpa(query, key, key)


def test_dense_tma_shape_and_empty_key_contracts_are_explicit():
    query = torch.ones(1, 37, 2, 64, dtype=torch.bfloat16)
    key = torch.ones(1, 53, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape"):
        triton_tma_sdpa(query.squeeze(0), key, key)
    with pytest.raises(ValueError, match="batch, head"):
        triton_tma_sdpa(query, key[:, :, :1], key[:, :, :1])
    with pytest.raises(ValueError, match="identical"):
        triton_tma_sdpa(query, key, key[:, :1])
    with pytest.raises(ValueError, match="positive"):
        triton_tma_sdpa(query, key[:, :0], key[:, :0])


def test_dense_tma_is_explicit_and_masked_requests_cannot_fall_back_to_torch():
    assert normalize_attention_backend("triton_tma") == "triton_tma"
    assert resolve_attention_backend("auto", "cpu") == "torch"
    with pytest.raises(RuntimeError, match="triton_tma.*unavailable"):
        resolve_attention_backend("triton_tma", "cpu")
    query = torch.ones(1, 2, 37, 64, dtype=torch.bfloat16)
    with patch.object(dispatch, "_invoke_torch_sdpa_audited", side_effect=AssertionError("dense fallback")):
        with pytest.raises(RuntimeError, match="CUDA FP16/BF16"):
            dispatch.attention_forward(query, query, query, backend="triton_tma")
        for options in ({"attn_mask": torch.ones(37, 37, dtype=torch.bool)}, {"compatibility_mode": True}):
            with pytest.raises(ValueError, match="does not support"):
                dispatch.attention_forward(query, query, query, backend="triton_tma", **options)


@pytest.mark.parametrize("installed", ["3.3.1", "3.4.0", "3.5.0.dev0", "invalid"])
def test_dense_tma_old_versions_fail_before_kernel_decoration_or_allocator_mutation(monkeypatch, installed):
    monkeypatch.setattr(tma.triton, "__version__", installed)
    query = torch.ones(1, 37, 2, 64, dtype=torch.bfloat16)
    with (
        patch.object(tma.triton, "autotune", side_effect=AssertionError("decorated old-runtime kernel")),
        patch.object(tma.triton, "set_allocator", side_effect=AssertionError("unsafe allocator fallback")),
        patch.object(dispatch, "_invoke_torch_sdpa_audited", side_effect=AssertionError("dense fallback")),
    ):
        spec = importlib.util.spec_from_file_location("worldfoundry_tma_version_rejection", tma.__file__)
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(RuntimeError, match="Triton >=3.5.0.*context-local allocator"):
            spec.loader.exec_module(module)
        assert not is_triton_tma_supported(query, query, query)
        with pytest.raises(RuntimeError, match="Triton >=3.5.0"):
            triton_tma_sdpa(query, query, query)
        with pytest.raises(RuntimeError, match="Triton >=3.5.0"):
            dispatch.attention_forward(
                query.transpose(1, 2), query.transpose(1, 2), query.transpose(1, 2), backend="triton_tma"
            )


@pytest.mark.parametrize("installed", ["3.3.1", "3.4.0", "3.5.0.dev0", "invalid"])
def test_dense_tma_light_probe_rejects_old_metadata_without_importing_triton(monkeypatch, installed):
    real_import = builtins.__import__

    def no_triton_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise AssertionError("light capability probe imported Triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(backends.metadata, "version", lambda package: installed)
    monkeypatch.setattr(backends, "_cuda_compute_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(backends, "_torch_cuda_accelerator_available", lambda device=None: True)
    backends.probe_attention_backends.cache_clear()
    try:
        with patch.object(builtins, "__import__", no_triton_import):
            capability = backends.probe_attention_backends("cuda:0")["triton_tma"]
            assert capability.available and not capability.usable
            assert "Triton >=3.5.0" in capability.reason
            assert "context-local allocator" in capability.reason
            with pytest.raises(RuntimeError, match="triton_tma.*unavailable.*Triton >=3.5.0"):
                resolve_attention_backend("triton_tma", "cuda:0")
            assert resolve_attention_backend("auto", "cpu") == "torch"
    finally:
        backends.probe_attention_backends.cache_clear()
