"""Tests for the CUDA-tier torch constraint drift checker."""

from __future__ import annotations

from pathlib import Path

from scripts.setup.check_cuda_torch_constraints import check_tier, main
from worldfoundry.runtime.cuda_tiers import TIER_TORCH_SPECS, torch_specs_for_tier

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_constraint_stubs_match_ssot() -> None:
    for tier, expected in TIER_TORCH_SPECS.items():
        path = REPO_ROOT / "requirements" / "cuda" / f"{tier}-torch.txt"
        assert check_tier(path, expected=expected) == []


def test_checker_main_accepts_repo_stubs() -> None:
    assert main([]) == 0


def test_torch_specs_for_tier_returns_tier_matrix_copy() -> None:
    assert torch_specs_for_tier("cu121")["torch"] == "torch>=2.4,<2.6"
    assert torch_specs_for_tier("cu124")["torch"] == "torch>=2.4,<2.7"
    cu128 = torch_specs_for_tier("cu128")
    assert cu128["torch"] == "torch>=2.7,<2.12.0"
    cu128["torch"] = "changed"
    assert TIER_TORCH_SPECS["cu128"]["torch"] == "torch>=2.7,<2.12.0"


def test_conda_installer_reads_defaults_from_resolved_tier_report() -> None:
    script = (REPO_ROOT / "scripts" / "setup" / "conda_install.sh").read_text(encoding="utf-8")
    assert 'json.load(sys.stdin)["torch_specs"]["torch"]' in script
    assert 'json.load(sys.stdin)["torch_specs"]["torchvision"]' in script
    assert 'json.load(sys.stdin)["torch_specs"]["torchaudio"]' in script
    assert 'TORCH_SPEC="${WORLDFOUNDRY_TORCH_SPEC:-}"' in script


def test_checker_rejects_duplicate_and_drift(tmp_path: Path) -> None:
    path = tmp_path / "tier.txt"
    path.write_text("torch>=2.4\ntorch>=2.5\n", encoding="utf-8")
    errors = check_tier(path, expected={"torch": "torch>=2.4"})
    assert any("duplicate pin for torch" in error for error in errors)
