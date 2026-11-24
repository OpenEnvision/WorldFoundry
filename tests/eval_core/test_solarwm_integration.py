from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
PIPELINE_RUNNER_TARGET = "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"


def test_solarwm_runtime_spec_resolves_in_tree_launcher() -> None:
    from worldfoundry.synthesis.visual_generation.runtime_manifest import (
        resolve_runtime_manifest,
        runtime_spec,
    )
    from worldfoundry.synthesis.visual_generation.solarwm import worldfoundry_runtime

    spec = runtime_spec("solarwm")
    runtime_root, entrypoint, blocked_reason = resolve_runtime_manifest(spec)

    assert runtime_root == worldfoundry_runtime.RUNTIME_DIR
    assert entrypoint is not None and entrypoint.name == "infer.py"
    assert "worldfoundry/synthesis/visual_generation/solarwm" in entrypoint.as_posix()
    assert blocked_reason == ""


def test_solarwm_launcher_normalizes_the_generated_artifact(tmp_path: Path, monkeypatch) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm import infer

    def fake_run(command, **_kwargs):
        overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
        output = next(value.split("=", 1)[1] for value in overrides if value.startswith("runtime.output_dir="))
        artifact = Path(output) / "generation" / "model_self_forcing_nfe4" / "slot-000000" / "video.mp4"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"fake-solarwm-mp4")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(infer.subprocess, "run", fake_run)
    output = tmp_path / "artifacts" / "solarwm.mp4"
    args = argparse.Namespace(
        config=tmp_path / "infer.yaml",
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        output_path=output,
        test_index=None,
        sample_count=1,
        seed=17,
    )

    assert infer.run(args) == output.resolve()
    assert output.read_bytes() == b"fake-solarwm-mp4"


def test_solarwm_command_uses_explicit_inference_only_overrides(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.infer import build_official_command
    from worldfoundry.synthesis.visual_generation.solarwm.runtime import SOLARWM_MODULE

    args = argparse.Namespace(
        config=tmp_path / "infer.yaml",
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        test_index="recipes/custom/test-index.jsonl.gz",
        sample_count=1,
        seed=9,
    )
    command = build_official_command(args, run_root=(tmp_path / "run").resolve())

    assert command[:4] == [sys.executable, "-m", SOLARWM_MODULE, "infer"]
    assert "inference.output_layout=transaction_v1" in command
    assert "validation.sample_count=1" in command
    assert "validation.selection_seed=9" in command
    assert "data.test_index=recipes/custom/test-index.jsonl.gz" in command
    assert "train" not in command
    assert "preencode" not in command


def test_solarwm_binding_rejects_multi_sample_and_absolute_index(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.infer import build_official_command

    args = argparse.Namespace(
        config=tmp_path / "infer.yaml",
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        test_index="recipes/custom/test-index.jsonl.gz",
        sample_count=2,
        seed=9,
    )
    with pytest.raises(ValueError, match="single-MP4"):
        build_official_command(args, run_root=(tmp_path / "run").resolve())

    args.sample_count = 1
    args.test_index = "/tmp/test-index.jsonl.gz"
    with pytest.raises(ValueError, match="relative POSIX"):
        build_official_command(args, run_root=(tmp_path / "run").resolve())


def test_solarwm_public_wan_runtime_uses_official_wan22_environment() -> None:
    from worldfoundry.cli.model_run import load_model_run_schema
    from worldfoundry.runtime.conda import load_runtime_conda_env_spec
    from worldfoundry.zoo.runtime.profiles import load_runtime_profile

    spec = load_runtime_conda_env_spec("solarwm")
    assert spec is not None
    assert spec.model_id == "solarwm"
    assert spec.env_name == "worldfoundry-solarwm-wan22-cu124"
    assert spec.pip_extra_index_url == "https://download.pytorch.org/whl/cu124"
    assert "cuda-nvcc=12.4" in spec.conda_packages
    assert "torch==2.5.1+cu124" in spec.pip_packages
    assert "einops>=0.8" in spec.pip_packages
    assert "decord>=0.6" in spec.pip_packages
    assert "flash-attn==2.8.3" not in spec.pip_packages
    assert "flash_attn" in spec.validation_imports

    profile = load_runtime_profile("solarwm", check_conda_env_exists=False)
    assert profile.execution["environment"] == "solarwm"
    assert profile.conda_env["env_name"] == spec.env_name

    for model_id in ("solarwm-wan2.2-5b", "solarwm-wan22-5b", "solarwm-wan2.2-14b"):
        schema = load_model_run_schema(model_id)
        assert schema.model_id == "solarwm"
        assert load_runtime_conda_env_spec(schema.model_id) == spec


@pytest.mark.parametrize("variant", ["ltx", "h3-bidirectional"])
def test_solarwm_non_wan_routes_keep_explicit_python(tmp_path: Path, variant: str) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm import SolarWMRuntime

    python = tmp_path / f"{variant}-python"
    runtime = SolarWMRuntime(
        source_root=tmp_path,
        python_executable=python,
        variant=variant,
        device="cuda",
    )

    assert runtime.python_executable == str(python)
