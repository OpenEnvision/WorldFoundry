# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenAI CLIP visual scoring and local vocabulary contracts."""

from __future__ import annotations

import gzip
import hashlib
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_PACKAGE = "worldfoundry.base_models.perception_core.general_perception.openai_clip_runtime.clip"
_BPE = "bpe_simple_vocab_16e6.txt.gz"


def test_harnesseval_visual_scoring_imports_without_loading_bpe() -> None:
    code = """
import gzip
from unittest.mock import patch
import torch

original = gzip.open
def check_vocab(filename, *args, **kwargs):
    if str(filename).endswith('bpe_simple_vocab_16e6.txt.gz'):
        raise AssertionError('visual scoring loaded the text vocabulary')
    return original(filename, *args, **kwargs)

with patch.object(gzip, 'open', check_vocab):
    from worldfoundry.evals.tasks.execution.runners.harnesseval_w.runtime.metrics.background_consistency import (
        BackgroundConsistencyMetric,
    )
    from worldfoundry.base_models.perception_core.general_perception.openai_clip_runtime.clip.model import (
        ResidualAttentionBlock,
    )
    assert BackgroundConsistencyMetric.__name__ == 'BackgroundConsistencyMetric'
    block = ResidualAttentionBlock(8, 2).eval()
    x = torch.linspace(-2.0, 2.0, 80).reshape(5, 2, 8)
    with torch.inference_mode():
        attended = x + block.attention(block.ln_1(x))
        hidden = block.mlp.c_fc(block.ln_2(attended))
        expected = attended + block.mlp.c_proj(hidden * torch.sigmoid(1.702 * hidden))
        torch.testing.assert_close(block(x), expected, atol=0, rtol=0)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("layout", ["bundled", "installed", "wrong_digest"])
def test_openai_clip_tokenize_resolves_only_the_selected_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    tokenizer = import_module(_PACKAGE + ".simple_tokenizer")
    api = import_module(_PACKAGE + ".clip")
    bundled = tmp_path / "bundled" / _BPE
    installed = tmp_path / "site-packages" / "open_clip" / _BPE
    vocabulary = bundled if layout == "bundled" else installed
    vocabulary.parent.mkdir(parents=True)
    with gzip.open(vocabulary, "wt", encoding="utf-8") as stream:
        stream.write("#version: 0.2\na b</w>")
    locate = Mock(return_value=installed)
    package = Mock(return_value=SimpleNamespace(locate_file=locate))
    monkeypatch.setattr(tokenizer, "__file__", str(bundled.with_name("simple_tokenizer.py")))
    monkeypatch.setattr(tokenizer, "distribution", package)
    if layout != "wrong_digest":
        monkeypatch.setattr(tokenizer, "_OPENAI_BPE_SHA256", hashlib.sha256(vocabulary.read_bytes()).hexdigest())

    tokenizer.default_bpe.cache_clear()
    api._get_tokenizer.cache_clear()
    try:
        if layout == "wrong_digest":
            with pytest.raises(ValueError, match="differs from the official file"):
                api.tokenize("ab", context_length=5)
        else:
            assert tokenizer.default_bpe() == str(vocabulary)
            assert api.tokenize("ab", context_length=5).tolist() == [[513, 512, 514, 0, 0]]
        if layout == "bundled":
            package.assert_not_called()
        else:
            package.assert_called_once_with("open-clip-torch")
            locate.assert_called_once_with(f"open_clip/{_BPE}")
    finally:
        tokenizer.default_bpe.cache_clear()
        api._get_tokenizer.cache_clear()
