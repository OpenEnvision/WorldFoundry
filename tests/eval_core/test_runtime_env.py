from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worldfoundry.runtime.assets import (
    expand_worldfoundry_path,
    load_local_assets,
    resolve_asset_manifest_path,
)
from worldfoundry.runtime.env import (
    WorldFoundryEnv,
    capture_runtime_environment,
    check_required_env,
    redact_env_for_manifest,
    resolve_artifact_dir,
    resolve_cache_dir,
    resolve_data_dir,
    resolve_hf_cache_dir,
    resolve_model_dir,
)
from worldfoundry.runtime.jobs import run_bounded_command


def test_worldfoundry_env_resolves_default_paths_from_home() -> None:
    env = {"WORLDFOUNDRY_HOME": "/tmp/worldfoundry-home"}
    runtime_env = WorldFoundryEnv(env)

    assert runtime_env.resolve_cache_dir() == Path("/tmp/worldfoundry-home/cache")
    assert runtime_env.resolve_data_dir() == Path("/tmp/worldfoundry-home/data")
    assert runtime_env.resolve_model_dir() == Path("/tmp/worldfoundry-home/models")
    assert runtime_env.resolve_artifact_dir() == Path("/tmp/worldfoundry-home/artifacts")
    assert runtime_env.resolve_hf_cache_dir() == Path("/tmp/worldfoundry-home/cache/huggingface/hub")


def test_worldfoundry_env_respects_explicit_overrides() -> None:
    env = {
        "WORLDFOUNDRY_CACHE_DIR": "/cache",
        "WORLDFOUNDRY_DATA_DIR": "/data",
        "WORLDFOUNDRY_MODEL_DIR": "/models",
        "WORLDFOUNDRY_ARTIFACT_DIR": "/artifacts",
        "HF_HOME": "/hf-home",
    }

    assert resolve_cache_dir(env) == Path("/cache")
    assert resolve_data_dir(env) == Path("/data")
    assert resolve_model_dir(env) == Path("/models")
    assert resolve_artifact_dir(env) == Path("/artifacts")
    assert resolve_hf_cache_dir(env) == Path("/hf-home/hub")


def test_hf_cache_resolution_prefers_specific_cache_variables() -> None:
    env = {
        "WORLDFOUNDRY_HOME": "/home/worldfoundry",
        "HF_HOME": "/hf-home",
        "TRANSFORMERS_CACHE": "/transformers-cache",
        "HF_DATASETS_CACHE": "/datasets-cache",
        "WORLDFOUNDRY_HF_CACHE_DIR": "/worldfoundry-hf-cache",
    }

    assert resolve_hf_cache_dir(env) == Path("/worldfoundry-hf-cache")
    assert resolve_hf_cache_dir({"HF_DATASETS_CACHE": "/datasets-cache", "HF_HOME": "/hf-home"}) == Path(
        "/datasets-cache"
    )
    assert resolve_hf_cache_dir({"TRANSFORMERS_CACHE": "/transformers-cache", "HF_HOME": "/hf-home"}) == Path(
        "/transformers-cache"
    )


def test_redact_env_for_manifest_records_secret_presence_only() -> None:
    env = {
        "WORLDFOUNDRY_HOME": "/runtime",
        "HF_TOKEN": "placeholder-token",
        "HUGGING_FACE_HUB_TOKEN": "placeholder-token-2",
        "OPENAI_API_KEY": "placeholder-openai-key",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }

    redacted = redact_env_for_manifest(env, keys=("WORLDFOUNDRY_HOME", "HF_TOKEN", "OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES"))

    assert redacted == {
        "WORLDFOUNDRY_HOME": "/runtime",
        "HF_TOKEN": {"present": True},
        "OPENAI_API_KEY": {"present": True},
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    assert "placeholder-openai-key" not in str(redacted)
    assert "placeholder-token" not in str(redacted)


def test_check_required_env_reports_presence_without_values() -> None:
    report = check_required_env(("HF_TOKEN", "OPENAI_API_KEY", "MISSING"), {"HF_TOKEN": "secret", "MISSING": ""})

    assert report.ok is False
    assert report.present == ("HF_TOKEN",)
    assert report.missing == ("OPENAI_API_KEY", "MISSING")
    assert report.to_dict() == {
        "ok": False,
        "missing": ["OPENAI_API_KEY", "MISSING"],
        "present": ["HF_TOKEN"],
    }


def test_capture_runtime_environment_is_manifest_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "placeholder-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    payload = capture_runtime_environment(include_torch=False, include_nvidia_smi=False)

    assert payload["environment"]["HF_TOKEN"] == {"present": True}
    assert payload["cuda"]["visible_devices"] == "0,1"
    assert "python" in payload
    assert payload["torch"] == {"checked": False}
    assert "placeholder-token" not in str(payload)


def test_worldfoundry_asset_paths_expand_runtime_tokens() -> None:
    env = {
        "WORLDFOUNDRY_DATA_DIR": "/data-root",
        "WORLDFOUNDRY_MODEL_DIR": "/model-root",
    }

    assert expand_worldfoundry_path("$WORLDFOUNDRY_DATA_DIR/datasets/example", env) == Path(
        "/data-root/datasets/example"
    )
    assert expand_worldfoundry_path("${WORLDFOUNDRY_MODEL_DIR}/checkpoints/example", env) == Path(
        "/model-root/checkpoints/example"
    )
    assert expand_worldfoundry_path("tmp/asset", env).is_absolute()


def test_local_asset_manifest_resolves_status(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "ExampleOrg" / "ExampleData"
    dataset_dir.mkdir(parents=True)
    missing_model_dir = tmp_path / "models" / "checkpoints" / "missing-model"
    manifest_path = tmp_path / "local_assets.yaml"
    manifest_path.write_text(
        """
schema_version: worldfoundry-local-assets-v1
benchmarks:
  - id: example-benchmark
    assets:
      - id: dataset
        kind: dataset
        path: $WORLDFOUNDRY_DATA_DIR/datasets/ExampleOrg/ExampleData
      - id: checkpoint
        kind: checkpoint
        path: $WORLDFOUNDRY_MODEL_DIR/checkpoints/missing-model
""".strip(),
        encoding="utf-8",
    )
    env = {
        "WORLDFOUNDRY_DATA_DIR": str(tmp_path),
        "WORLDFOUNDRY_MODEL_DIR": str(tmp_path / "models"),
    }

    assets = load_local_assets(manifest_path, env)

    assert assets[0].benchmark_id == "example-benchmark"
    assert assets[0].path == dataset_dir
    assert assets[0].ready is True
    assert assets[0].status == "available"
    assert assets[1].path == missing_model_dir
    assert assets[1].ready is False
    assert assets[1].status == "missing"


def test_local_asset_manifest_env_override(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    env = {"WORLDFOUNDRY_LOCAL_ASSET_MANIFEST": str(manifest_path)}

    assert resolve_asset_manifest_path(env=env) == manifest_path


def test_run_bounded_command_returns_success_payload() -> None:
    result = run_bounded_command(
        [sys.executable, "-c", "print('bounded-ok')"],
        timeout=5,
    )

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["kill_stuck"] is False
    assert result["stdout"].strip() == "bounded-ok"


def test_run_bounded_command_writes_timeout_payload() -> None:
    result = run_bounded_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=1,
        kill_timeout=1,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "TimeoutExpired" in result["stderr"]
