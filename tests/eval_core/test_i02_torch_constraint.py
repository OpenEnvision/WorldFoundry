"""Installer contract for retaining the CUDA-index torch stack."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "conda_install.sh"


def test_second_pip_pass_uses_exact_torch_constraint() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TORCH_CONSTRAINT_FILE=" in text
    assert '--constraint "$TORCH_CONSTRAINT_FILE"' in text
    assert 'version(name)' in text
    assert 'trap cleanup_torch_constraint EXIT' in text


def test_post_install_checks_selected_torch_cuda_tier() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "torch.version.cuda" in text
    assert '"cu121": "12.1"' in text
    assert '"cu124": "12.4"' in text
    assert '"cu128": "12.8"' in text
    assert "does not match selected tier" in text
