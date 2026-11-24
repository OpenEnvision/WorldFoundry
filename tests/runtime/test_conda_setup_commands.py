# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Environment setup preserves dependency ordering and isolated worker ABIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.runtime.conda import (
    RuntimeCondaEnvSpec,
    apply_unified_env_override,
    load_runtime_conda_env_spec,
)

pytestmark = pytest.mark.unit


def test_setup_commands_survive_loading_serialization_and_unified_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "environment.yaml"
    manifest.write_text(
        "envs:\n- model_id: example\n  env_name: example\n  python: '3.11'\n  cuda_profile: cu128\n"
        "  pip_packages: [torch]\n  setup_commands: ['python -m pip install --no-build-isolation extension']\n"
    )
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_DETECTED_DRIVER_CUDA", "13.0")
    monkeypatch.setenv("WORLDFOUNDRY_CUDA_PROFILE", "cu128")
    spec = load_runtime_conda_env_spec("example", manifest, env_root=tmp_path)
    assert spec is not None
    assert spec.setup_commands == ("python -m pip install --no-build-isolation extension",)
    routed = apply_unified_env_override(spec, env_root=tmp_path)
    assert routed.resolved_env_name == "worldfoundry-unified-cu128"
    assert routed.to_dict(check_exists=False)["setup_commands"] == list(spec.setup_commands)


def test_environment_without_setup_commands_keeps_its_serialized_contract(tmp_path: Path) -> None:
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="example", env_root=tmp_path)
    assert spec.setup_commands == ()
    assert "setup_commands" not in spec.to_dict(check_exists=False)


def test_kairos_builds_sageattention_only_after_its_torch_dependencies() -> None:
    spec = load_runtime_conda_env_spec("kairos-sensenova")
    assert spec is not None
    assert "torch==2.12.0" in spec.pip_packages
    assert not any("sageattention" in package for package in spec.pip_packages)
    assert len(spec.setup_commands) == 1
    assert "--no-build-isolation" in spec.setup_commands[0]
    assert "40d3fcb518c3ff894501bf4b080ccaa589363adc" in spec.setup_commands[0]
    assert "sageattention" in spec.validation_imports


def test_magi2_text_worker_keeps_its_transformers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    spec = load_runtime_conda_env_spec("magi2-preview")
    assert spec is not None
    assert spec.resolved_env_name == "magi2-preview"
    assert "torch==2.5.1" in spec.pip_packages
    assert "transformers==5.5.0" in spec.pip_packages
    assert spec.setup_commands == ()
    assert "worldfoundry.base_models.diffusion_model.models.encoders.magi2_qwen35" in spec.validation_imports
