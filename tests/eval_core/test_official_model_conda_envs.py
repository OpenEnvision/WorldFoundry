from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evals.utils import load_manifest_collection
from worldfoundry.pipelines.component_pipelines import OpenVLAPipeline
from worldfoundry.runtime.conda import (
    is_cuda_profile_supported,
    load_runtime_conda_env_specs_with_overrides,
)
from worldfoundry.synthesis.visual_generation.animatediff.worldfoundry_runtime import DEFAULT_ANIMATEDIFF_REPO_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_conda_envs_do_not_install_from_cache_or_external_model_repos() -> None:
    payload = load_manifest_collection(
        REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "environments",
        item_key="envs",
    )
    forbidden = (
        "cache/",
        str(REPO_ROOT / "cache"),
        str(Path("/", "share", "project", "bench", "model")),
    )

    for env in payload["envs"]:
        for key in ("source_requirement_files", "editable_install_dirs"):
            for item in env.get(key, []):
                assert not any(marker in str(item) for marker in forbidden), (env["model_id"], key, item)


def test_animatediff_default_official_repo_root_uses_integrated_code() -> None:
    expected = REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "animatediff"

    assert DEFAULT_ANIMATEDIFF_REPO_ROOT.resolve() == expected
    assert "cache" not in DEFAULT_ANIMATEDIFF_REPO_ROOT.resolve().parts


def test_cuda_profile_support_matches_driver_capability() -> None:
    assert is_cuda_profile_supported("cu113", "11.4") is True
    assert is_cuda_profile_supported("cu118", "11.4") is False
    assert is_cuda_profile_supported("prepare_only", "11.4") is True


def test_runtime_profile_plan_records_per_model_conda_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENV_ROOT", "/tmp/worldfoundry-conda-test")
    pipe = OpenVLAPipeline.from_pretrained(
        {"model_id": "openvla", "plan_only": True},
        device="cpu",
    )

    result = pipe(
        prompt="a robot picks up a cube",
        images="memory://rgb.png",
        interactions=[{"delta": [0.0] * 7}],
        output_path=tmp_path / "openvla_action_trace.json",
        return_dict=True,
    )

    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
    env_name = plan["profile"]["conda_env"]["env_name"]
    assert env_name.startswith("worldfoundry-unified-")
    assert plan["context"]["conda_env_name"] == env_name
    assert plan["context"]["conda_env_prefix"] == f"/tmp/worldfoundry-conda-test/{env_name}"
    assert plan["context"]["conda_env_driver_status"] == "compatible"
    assert plan["context"]["python"]


def test_model_env_install_script_has_current_open_source_contract() -> None:
    script = REPO_ROOT / "scripts" / "setup" / "model_env_install.sh"
    result = subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

    help_result = subprocess.run(["bash", str(script), "--help"], cwd=REPO_ROOT, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "--model MODEL" in help_result.stdout
    assert "--verify-only" in help_result.stdout
    assert "unified env is the default" in help_result.stdout
    # A bare "dry" substring check false-positives on "worldfoundry"; the
    # contract is that the script exposes no dry-run flag.
    assert "--dry-run" not in help_result.stdout.lower()


def test_model_env_install_list_uses_unified_default_for_runtime_profiles(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "PYTHONPATH": str(REPO_ROOT),
            "CONDA_EXE": "true",
            "WORLDFOUNDRY_CONDA_ENVS_ROOT": str(tmp_path / "envs"),
            "WORLDFOUNDRY_CUDA_PROFILE": "cu128",
            "WORLDFOUNDRY_ALLOW_NO_CUDA": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/setup/model_env_install.sh", "--list", "--env-root", str(tmp_path / "envs")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    rows = result.stdout.splitlines()
    assert any(row.startswith("animatediff\tworldfoundry-unified-") for row in rows)
    assert any(row.startswith("matrix-game-2\tworldfoundry-unified-") for row in rows)


def test_runtime_conda_resolver_routes_compatible_profiles_to_unified_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CUDA_PROFILE", "cu128")
    specs = load_runtime_conda_env_specs_with_overrides(env_root=tmp_path / "envs")

    for model_id in ("animatediff", "zeroscope", "matrix-game-2"):
        assert specs[model_id].env_name == "worldfoundry-unified-cu128"
    # openpi's JAX cuda12 stack conflicts with the unified torch env, so it
    # keeps its isolated per-model env even when unified routing is enabled.
    # The runtime is a flattened package now: no jax 0.4 pin and no
    # source-tree PYTHONPATH injection.
    assert specs["openpi"].env_name == "worldfoundry-openpi-cu12"
    assert "jax[cuda12]==0.5.3" in specs["openpi"].pip_packages
    assert specs["openpi"].pythonpath_dirs == ()
