from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from worldfoundry.evaluation.api import GenerationRequest
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET
from worldfoundry.evaluation.utils import (
    RUN_FINGERPRINT_SCHEMA_VERSION,
    VERSION_CONTEXT_SCHEMA_VERSION,
    build_run_fingerprint,
    build_version_context,
    git_metadata,
    stable_hash,
    stable_json_dumps,
)


def test_stable_hash_is_order_independent_for_mappings() -> None:
    left = {"b": [2, {"z": 1}], "a": "same"}
    right = {"a": "same", "b": [2, {"z": 1}]}

    assert stable_json_dumps(left) == stable_json_dumps(right)
    assert stable_hash(left) == stable_hash(right)


def test_run_fingerprint_excludes_run_id_and_time_varying_fields() -> None:
    request = GenerationRequest(sample_id="sample-a", task_name="unit")
    context = build_version_context(
        runner="unit-runner",
        benchmark={"benchmark_name": "unit"},
        model={"model_name": "fake"},
        dataset={"sample_count": 1},
        extra={"profile": "cpu"},
    )

    first = build_run_fingerprint(version_context=context, requests=[request])
    second = build_run_fingerprint(version_context=context, requests=[request])

    assert context["schema_version"] == VERSION_CONTEXT_SCHEMA_VERSION
    assert first["schema_version"] == RUN_FINGERPRINT_SCHEMA_VERSION
    assert first["hash"] == second["hash"]
    assert first["version_context_hash"] == stable_hash(context)


def test_git_metadata_does_not_scan_untracked_files(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        if command[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[-3:] == ("status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(command, 0, stdout=" M file.py\n", stderr="")
        if "diff" in command:
            kwargs["stdout"].write(b"tracked diff bytes\n")
            return subprocess.CompletedProcess(command, 0, stdout=None, stderr=None)
        raise AssertionError(command)

    import worldfoundry.evaluation.utils as versioning

    monkeypatch.setattr(versioning.subprocess, "run", fake_run)

    metadata = git_metadata(Path("/repo"))

    assert metadata == {
        "available": True,
        "commit": "abc123",
        "dirty": True,
        "dirty_diff_sha256": hashlib.sha256(b"tracked diff bytes\n").hexdigest(),
    }
    assert ("git", "-C", "/repo", "status", "--porcelain", "--untracked-files=no") in calls
    diff_call = next(command for command in calls if "diff" in command)
    assert diff_call[-2:] == ("HEAD", "--")
    assert "--binary" in diff_call


def test_git_metadata_skips_diff_for_clean_tree(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        if command[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[-3:] == ("status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    import worldfoundry.evaluation.utils as versioning

    monkeypatch.setattr(versioning.subprocess, "run", fake_run)

    assert git_metadata(Path("/repo")) == {
        "available": True,
        "commit": "abc123",
        "dirty": False,
        "dirty_diff_sha256": None,
    }
    assert not any("diff" in command for command in calls)


def test_git_metadata_preserves_dirty_state_when_diff_times_out(monkeypatch) -> None:
    def fake_run(command: tuple[str, ...], **kwargs) -> subprocess.CompletedProcess[str]:
        if command[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[-3:] == ("status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(command, 0, stdout=" M file.py\n", stderr="")
        if "diff" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise AssertionError(command)

    import worldfoundry.evaluation.utils as versioning

    monkeypatch.setattr(versioning.subprocess, "run", fake_run)

    assert git_metadata(Path("/repo")) == {
        "available": True,
        "commit": "abc123",
        "dirty": True,
        "dirty_diff_sha256": None,
    }


def test_version_context_does_not_include_python_executable(monkeypatch) -> None:
    import worldfoundry.evaluation.utils as versioning

    monkeypatch.setattr(versioning, "git_metadata", lambda _root=None: {"available": False})

    context = build_version_context(runner="unit")

    assert context["python"] == {"version": versioning.platform.python_version()}


def test_eval_runner_writes_version_context_and_fingerprint(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    from worldfoundry.evaluation.runner import EvaluateRunRequest, execute_evaluate_run

    execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            mode="model",
            model_id=CONTRACT_FIXTURE_MODEL_ID,
            model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
            requests=[{"sample_id": "sample-a", "inputs": {"prompt": "go"}, "output_schema": {"generated_video": {}}}],
            metrics=("artifact_count",),
        )
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((output_dir / "execution_plan.json").read_text(encoding="utf-8"))
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert manifest["version_context"]["schema_version"] == VERSION_CONTEXT_SCHEMA_VERSION
    assert manifest["run_fingerprint"]["schema_version"] == RUN_FINGERPRINT_SCHEMA_VERSION
    assert plan["run_fingerprint"]["hash"] == manifest["run_fingerprint"]["hash"]
    assert scorecard["run"]["run_fingerprint"]["hash"] == manifest["run_fingerprint"]["hash"]
    assert scorecard["run"]["version_context"]["contract_versions"]["generation_request"]
