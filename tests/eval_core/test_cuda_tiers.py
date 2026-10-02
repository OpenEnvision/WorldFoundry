from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.runtime import cuda_tiers
from worldfoundry.runtime.cuda_tiers import (
    DEFAULT_CUDA_TIER,
    best_cuda_tier_for_driver,
    cuda_tier_report,
    normalize_cuda_profile,
    resolve_install_tier,
    resolve_cuda_tier,
    torch_specs_for_tier,
    unified_env_name,
)
from worldfoundry.runtime.conda import (
    RuntimeCondaEnvSpec,
    apply_unified_env_override,
    load_runtime_conda_env_spec,
    unified_env_blocker,
)


def test_normalize_cuda_profile_strips_legacy_suffixes() -> None:
    assert normalize_cuda_profile("cu118") == "cu118"
    assert normalize_cuda_profile("cu128_recommended") == "cu128"
    assert normalize_cuda_profile("prepare_only") == "prepare_only"


def test_resolve_cuda_tier_prefers_cu128_on_modern_driver() -> None:
    assert resolve_cuda_tier("cu113", driver_cuda="13.0") == "cu128"
    assert resolve_cuda_tier("cu118", driver_cuda="13.0") == "cu128"
    assert resolve_cuda_tier("cu124", driver_cuda="13.0") == "cu124"
    assert resolve_cuda_tier("cu118", driver_cuda="12.4", preferred_tier="cu124") == "cu124"
    assert resolve_cuda_tier("cu118", driver_cuda="12.2", preferred_tier="cu128") == "cu121"
    assert resolve_cuda_tier("cu118", driver_cuda="12.5", preferred_tier="cu128") == "cu124"


def test_best_cuda_tier_for_driver() -> None:
    assert best_cuda_tier_for_driver("13.0") == "cu128"
    assert best_cuda_tier_for_driver("12.6") == "cu124"
    assert best_cuda_tier_for_driver("12.4") == "cu124"
    assert best_cuda_tier_for_driver("12.1") == "cu121"
    assert best_cuda_tier_for_driver("") == "cu128"
    with pytest.raises(ValueError, match="No NVIDIA driver CUDA version detected"):
        best_cuda_tier_for_driver("", allow_missing_driver=False)


def test_resolve_install_tier_supports_auto_and_driver_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_install_tier("auto", driver_cuda="12.9") == "cu128"
    assert resolve_install_tier("auto", driver_cuda="12.5") == "cu124"
    assert resolve_install_tier("cu128", driver_cuda="12.5") == "cu124"
    assert resolve_install_tier("cu129", driver_cuda="13.0") == "cu128"
    monkeypatch.setattr(cuda_tiers, "detect_nvidia_driver_cuda", lambda: None)
    assert resolve_install_tier("cu128", driver_cuda=None) == "cu128"
    with pytest.raises(ValueError, match="No NVIDIA driver CUDA version detected"):
        resolve_install_tier("auto", driver_cuda="")


def test_cuda_tier_report_is_user_facing() -> None:
    report = cuda_tier_report("auto", driver_cuda="12.9")
    assert report["tier"] == "cu128"
    assert report["env_name"] == "worldfoundry-unified-cu128"
    assert report["torch_index_url"] == "https://download.pytorch.org/whl/cu128"
    assert report["torch_specs"] == torch_specs_for_tier("cu128")


def test_cpu_bootstrap_selects_explicit_wheel_tier_and_skips_flash_attn() -> None:
    script = Path("scripts/setup/bootstrap_worldfoundry.sh").read_text(encoding="utf-8")
    assert "if ! command -v nvidia-smi" in script
    assert "ALLOW_NO_CUDA=1" in script
    assert "SKIP_FLASH_ATTN=1" in script
    assert "CUDA_TIER=cu128" in script


def test_unified_env_name() -> None:
    assert unified_env_name() == "worldfoundry-unified-cu128"
    assert unified_env_name("cu124") == "worldfoundry-unified-cu124"


def test_apply_unified_env_override_routes_runnable_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "conda_envs"
    prefix = env_root / "worldfoundry-unified-cu128"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(env_root))
    monkeypatch.setenv("WORLDFOUNDRY_DETECTED_DRIVER_CUDA", "13.0")

    spec = RuntimeCondaEnvSpec(
        model_id="scope",
        env_name="worldfoundry-scope-cu128",
        cuda_profile="cu128",
        driver_status="compatible",
        env_root=env_root,
    )
    routed = apply_unified_env_override(spec)
    assert routed.env_name == "worldfoundry-unified-cu128"
    assert routed.cuda_profile == "cu128"


def test_apply_unified_env_override_keeps_prepare_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "conda_envs"
    prefix = env_root / "worldfoundry-unified-cu128"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(env_root))

    spec = RuntimeCondaEnvSpec(
        model_id="dreamzero",
        env_name="worldfoundry-dreamzero-prepare",
        cuda_profile="prepare_only",
        driver_status="blocked",
        env_root=env_root,
    )
    assert apply_unified_env_override(spec).env_name == "worldfoundry-dreamzero-prepare"


def test_apply_unified_env_override_keeps_hard_abi_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "conda_envs"
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(env_root))

    torch_pinned = RuntimeCondaEnvSpec(
        model_id="legacy-torch",
        env_name="worldfoundry-legacy-torch",
        cuda_profile="cu118",
        driver_status="compatible",
        pip_packages=("torch==2.5.1+cu118", "transformers>=4.57.0,<5"),
        env_root=env_root,
    )
    jax_runtime = RuntimeCondaEnvSpec(
        model_id="jax-action",
        env_name="worldfoundry-jax-action",
        cuda_profile="cu118",
        driver_status="compatible",
        pip_packages=("jax==0.4.25", "jaxlib==0.4.25+cuda11.cudnn86"),
        env_root=env_root,
    )

    assert unified_env_blocker(torch_pinned) == "torch_exact_abi_pin"
    assert apply_unified_env_override(torch_pinned).env_name == "worldfoundry-legacy-torch"
    assert unified_env_blocker(jax_runtime) == "jax_runtime_requires_isolated_env"
    assert apply_unified_env_override(jax_runtime).env_name == "worldfoundry-jax-action"


def test_load_runtime_conda_env_spec_can_route_scope_to_unified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "conda_envs"
    prefix = env_root / "worldfoundry-unified-cu128"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(env_root))

    spec = load_runtime_conda_env_spec("scope", env_root=env_root)
    assert spec is not None
    assert spec.env_name == "worldfoundry-unified-cu128"
    assert spec.cuda_profile == "cu128"


def test_default_cuda_tier_is_cu128() -> None:
    assert DEFAULT_CUDA_TIER == "cu128"
