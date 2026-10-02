"""CPU contracts for the explicit dense TMA provider."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("triton")

from worldfoundry.core.attention.backends import dispatch
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
