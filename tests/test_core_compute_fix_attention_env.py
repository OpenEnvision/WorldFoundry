"""CPU-only tests for the attention backend-selection fixes.

Covers:
- CC-31: WORLDFOUNDRY_ATTENTION_BACKEND accepts both the SDPA policy
  vocabulary and the attention-dispatch vocabulary without crashing.
- CC-06: explicitly requested backends that are unusable degrade to torch
  with a warning log naming the request, the effective backend, and the reason.
- CC-05: auto resolution logs usable-but-not-enabled external backends.
- CC-07: importing the dispatch module must not initialize CUDA.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import types

import pytest
import torch

from worldfoundry.core.execution import inference as core_inference
from worldfoundry.core.attention.backends import probe as backends
from worldfoundry.core.attention.backends.probe import AttentionKernelCapability


def test_varlen_flash_attention_accepts_precomputed_max_lengths(monkeypatch):
    from worldfoundry.core.attention.sequence import varlen

    observed: dict[str, object] = {}

    def fake_flash_attn_varlen_func(**kwargs):
        observed.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    fake_flash_attn = types.ModuleType("flash_attn")
    fake_flash_attn.flash_attn_varlen_func = fake_flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash_attn)
    monkeypatch.setattr(
        varlen,
        "probe_attention_backends",
        lambda _device: {"flash_attention_2": types.SimpleNamespace(usable=True)},
    )

    query = torch.randn(2, 4, 2, 8)
    lengths = torch.tensor([3, 1], dtype=torch.int32)
    output = varlen.flash_attention(
        query,
        query,
        query,
        q_lens=lengths,
        k_lens=lengths,
        version=2,
        max_seqlen_q=3,
        max_seqlen_k=3,
    )

    assert output.shape == query.shape
    assert observed["max_seqlen_q"] == 3
    assert observed["max_seqlen_k"] == 3


def test_varlen_flash_attention_rejects_too_small_precomputed_max(monkeypatch):
    from worldfoundry.core.attention.sequence import varlen

    query = torch.randn(2, 4, 2, 8)
    lengths = torch.tensor([3, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="max_seqlen_q=2"):
        varlen.flash_attention(
            query,
            query,
            query,
            q_lens=lengths,
            k_lens=lengths,
            max_seqlen_q=2,
            max_seqlen_k=3,
        )


# ---------------------------------------------------------------------------
# CC-31: unified WORLDFOUNDRY_ATTENTION_BACKEND parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        ("flash", "flash"),
        ("cudnn", "cudnn"),
        ("efficient", "efficient"),
        ("math", "math"),
        ("sdpa", "auto"),
        ("default", "auto"),
        ("", "auto"),
        ("mem-efficient", "efficient"),
        ("memory_efficient", "efficient"),
        ("flash-attention", "flash"),
    ],
)
def test_sdpa_policy_vocabulary_is_preserved(value, expected):
    assert core_inference._normalize_attention_backend(value) == expected


@pytest.mark.parametrize(
    "value",
    ["flash_attention_2", "flash-attention-3", "sage_attention", "xformers", "torch", "torch_sdpa", "sage3"],
)
def test_dispatch_vocabulary_is_accepted_with_deprecation_warning(value):
    core_inference._DISPATCH_BACKEND_POLICY_WARNED.clear()
    with pytest.warns(DeprecationWarning, match="attention dispatch backend"):
        assert core_inference._normalize_attention_backend(value) == "auto"


def test_unknown_backend_value_raises_with_both_vocabularies():
    with pytest.raises(ValueError, match="SDPA kernel policy .* dispatch backend"):
        core_inference._normalize_attention_backend("definitely_not_a_backend")


def test_install_infra_does_not_crash_on_dispatch_vocabulary(monkeypatch):
    """The reported crash path: install with a dispatch-layer env value."""

    monkeypatch.setenv("WORLDFOUNDRY_ATTENTION_BACKEND", "flash_attention_2")
    core_inference._DISPATCH_BACKEND_POLICY_WARNED.clear()
    try:
        with pytest.warns(DeprecationWarning):
            state = core_inference.install_worldfoundry_inference_infra(
                enable_tf32=None,
                patch_sdpa=False,
            )
        assert state.attention_backend == "auto"
    finally:
        core_inference.uninstall_worldfoundry_inference_infra()


# ---------------------------------------------------------------------------
# CC-06: explicit request downgrade warning
# ---------------------------------------------------------------------------


def _fake_capabilities(**usable_flags):
    names = (
        "flash_attention_3",
        "flash_attention_2",
        "sage_attention",
        "sage_attention_3",
        "xformers",
        "video_sparse_attention",
        "flex_block_attention",
        "vmoba_attention",
        "sla_attention",
        "sage_sla_attention",
    )
    table = {}
    for name in names:
        usable = bool(usable_flags.get(name, False))
        table[name] = AttentionKernelCapability(
            name=name,
            package=name,
            available=usable,
            usable=usable,
            reason="" if usable else f"{name} is not installed (test)",
        )
    table["torch"] = AttentionKernelCapability(name="torch", package="torch", available=True, usable=True)
    return table


def test_explicit_unusable_backend_warns_and_degrades(monkeypatch, caplog):
    monkeypatch.setattr(backends, "probe_attention_backends", lambda device=None: _fake_capabilities())
    backends._EXPLICIT_FALLBACK_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.attention.backends.probe"):
        resolved = backends.resolve_attention_backend("flash_attention_2")
    assert resolved == "torch"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "flash_attention_2" in message and "'torch'" in message and "not installed (test)" in message
        for message in messages
    ), messages


def test_explicit_usable_backend_resolves_without_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        backends,
        "probe_attention_backends",
        lambda device=None: _fake_capabilities(flash_attention_2=True),
    )
    backends._EXPLICIT_FALLBACK_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.attention.backends.probe"):
        resolved = backends.resolve_attention_backend("flash_attention_2")
    assert resolved == "flash_attention_2"
    assert not caplog.records


def test_flash_auto_unusable_warns_with_reasons(monkeypatch, caplog):
    monkeypatch.setattr(backends, "probe_attention_backends", lambda device=None: _fake_capabilities())
    backends._EXPLICIT_FALLBACK_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.attention.backends.probe"):
        resolved = backends.resolve_attention_backend("flash")
    assert resolved == "torch"
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert "flash_attention_3" in combined and "flash_attention_2" in combined


# ---------------------------------------------------------------------------
# CC-05: auto keeps torch but reports usable faster backends once
# ---------------------------------------------------------------------------


def test_auto_logs_usable_external_backends(monkeypatch, caplog):
    monkeypatch.setattr(
        backends,
        "probe_attention_backends",
        lambda device=None: _fake_capabilities(flash_attention_2=True, xformers=True),
    )
    backends._AUTO_FASTER_BACKENDS_LOGGED.clear()
    with caplog.at_level(logging.INFO, logger="worldfoundry.core.attention.backends.probe"):
        resolved = backends.resolve_attention_backend("auto")
    assert resolved == "torch"
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert "flash_attention_2" in combined and "xformers" in combined
    assert "not enabled" in combined

    # The hint is emitted once per usable-backend set.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="worldfoundry.core.attention.backends.probe"):
        backends.resolve_attention_backend("auto")
    assert not caplog.records


# ---------------------------------------------------------------------------
# CC-07: dispatch import must not initialize CUDA
# ---------------------------------------------------------------------------


def test_dispatch_import_does_not_initialize_cuda():
    code = (
        "import torch\n"
        "import worldfoundry.core.attention.backends.dispatch as d\n"
        "assert not torch.cuda.is_initialized(), 'import initialized CUDA'\n"
        "assert isinstance(d.ATTENTION_IMPLEMENTATION, str)\n"
        "assert isinstance(d.FLASH_ATTN_2_AVAILABLE, bool)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
