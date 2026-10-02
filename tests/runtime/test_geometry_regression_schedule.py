"""Baseline import and scheduling must preserve accepted evidence and other jobs."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"
spec = importlib.util.spec_from_file_location("schedule_under_test", TOOLS / "geometry_regression_schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)
suite = schedule.suite


@pytest.fixture
def accepted(tmp_path):
    source = tmp_path / "original"
    case = source / "model"
    case.mkdir(parents=True)
    exports = case / "export"
    exports.mkdir()
    np.save(exports / "depth.npy", np.array([2.0, 3.0], dtype=np.float32))
    np.savez(case / "arrays.npz", depth=np.array([2.0, 3.0], dtype=np.float32))
    suite.write_json(
        case / "manifest.json",
        {
            "status": "passed",
            "case": {"id": "model", "seed": 42},
            "assets": {"weights": "sha"},
            "runtime": {},
            "exported_files": [str(exports / "depth.npy")],
            "arrays_sha256": suite.replay.sha256(case / "arrays.npz"),
        },
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"model": {"id": "model"}}))
    return source, matrix, tmp_path / "durable" / "reference"


def test_import_relocates_exports_preserves_original_and_records_every_file(accepted):
    source, matrix, destination = accepted
    original_sha = suite.replay.sha256(source / "model/manifest.json")
    result = schedule.import_reference(source, destination, matrix)
    assert result["cases"] == ["model"]
    assert suite.replay.sha256(source / "model/manifest.json") == original_sha
    assert suite.replay.compare_runs(source / "model", destination / "model")["status"] == "passed"
    index = json.loads((destination / "accepted-index.json").read_text())
    assert index["cases"]["model"]["original_manifest_sha256"] == original_sha
    assert set(index["cases"]["model"]["files"]) == {"arrays.npz", "manifest.json", "export/depth.npy"}
    suite.verify_reference(destination / "model", "model", index)
    (destination / "model/export/depth.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        suite.verify_reference(destination / "model", "model", index)
    with pytest.raises(ValueError, match="new and separate"):
        schedule.import_reference(source, destination, matrix)


@pytest.mark.parametrize("kind", ["failed", "arrays", "symlink", "export-escape"])
def test_invalid_original_cannot_be_promoted_and_leaves_no_reference(accepted, kind):
    source, matrix, destination = accepted
    manifest = source / "model/manifest.json"
    data = json.loads(manifest.read_text())
    if kind == "failed":
        data["status"] = "failed"
    elif kind == "arrays":
        (source / "model/arrays.npz").write_bytes(b"modified")
    elif kind == "symlink":
        (source / "model/link").symlink_to(source / "model/arrays.npz")
    else:
        data["exported_files"] = ["/tmp/outside/arrays.npy"]
    suite.write_json(manifest, data)
    with pytest.raises(ValueError):
        schedule.import_reference(source, destination, matrix)
    assert not destination.exists()
    assert not list(destination.parent.glob(".accepted-*"))


def test_import_into_original_is_rejected(accepted):
    source, matrix, _ = accepted
    with pytest.raises(ValueError):
        schedule.import_reference(source, source / "nested", matrix)


def test_cron_install_is_idempotent_preserves_jobs_and_restores_timezone(tmp_path):
    existing = 'MAILTO=""\nCRON_TZ=UTC\nSHELL=/bin/bash\n5 4 * * * echo existing\n'
    paths = (sys.executable, tmp_path / "runner.py", tmp_path / "profile.json", tmp_path / "cron.log")
    updated = schedule.cron_text(existing, *paths)
    assert updated.startswith(existing)
    assert updated.count(schedule.BEGIN) == 1
    assert "CRON_TZ=Asia/Shanghai\n0 3 * * *" in updated
    assert "SHELL=/bin/bash\nCRON_TZ=UTC\n" + schedule.END in updated
    assert schedule.cron_text(updated, *paths) == updated
    assert schedule.remove_block(updated) == existing


def test_cron_quotes_literal_paths_and_escapes_percent(tmp_path):
    runner = tmp_path / "runner ' $name%script.py"
    runner.write_text(
        "import sys,json,subprocess; subprocess.run(['git','--version'],check=True,capture_output=True); print(json.dumps(sys.argv[1:]))\n"
    )
    profile = tmp_path / "profile ' $name%.json"
    log = tmp_path / "cron ' $name%.log"
    text = schedule.cron_text("", sys.executable, runner, profile, log)
    command = next(line.split(maxsplit=5)[5] for line in text.splitlines() if line.startswith("0 3 "))
    assert "\\%" in command
    subprocess.run(["/bin/sh", "-c", command.replace("\\%", "%")], env={"PATH": "/usr/bin:/bin"}, check=True)
    assert json.loads(log.read_text()) == ["--profile", str(profile)]


@pytest.mark.parametrize(
    "existing",
    [schedule.BEGIN + "\n", schedule.END + "\n", schedule.BEGIN + "\n" + schedule.BEGIN + "\n" + schedule.END + "\n"],
)
def test_malformed_cron_block_is_not_silently_overwritten(existing):
    with pytest.raises(ValueError):
        schedule.remove_block(existing)


def test_cron_read_permission_failure_is_not_treated_as_empty(monkeypatch):
    monkeypatch.setattr(
        schedule.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(["crontab", "-l"], 1, "", "permission denied"),
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        schedule.read_crontab()


def test_retention_deletes_only_owned_runs_preserves_baselines_and_reports(tmp_path):
    state = tmp_path / "state"
    current = state / "runs/20261002T030000Z-01234567"
    old = state / "runs/20261001T030000Z-01234567"
    for path in (old, current):
        path.mkdir(parents=True)
        suite.write_json(path / "report.json", {"status": "passed"})
        (path / "large-artifact.bin").write_bytes(b"numerical outputs")
    outside = tmp_path / "accepted"
    outside.mkdir()
    (state / "runs/20260929T030000Z-01234567").symlink_to(outside, target_is_directory=True)
    assert suite.prune_runs(state, 1, current) == [old.name]
    assert current.exists() and outside.exists()
    assert not old.exists()
    assert json.loads((state / "history" / (old.name + ".json")).read_text())["status"] == "passed"


def test_gpu_holder_restarts_original_args_in_detached_session_and_waits_for_readiness(tmp_path):
    pid_file = tmp_path / "holder.pid"
    args_file = tmp_path / "args.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys,json,os,signal,pathlib\n"
        "pathlib.Path(os.environ['FAKE_PID']).write_text(str(os.getpid()))\n"
        "pathlib.Path(os.environ['FAKE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('holder ready',flush=True)\nsignal.pause()\n"
    )
    script = tmp_path / "gg"
    script.write_text(
        '#!/bin/bash\ncase "$1" in\n'
        ' --status) test -f "$FAKE_PID" && kill -0 "$(cat "$FAKE_PID")" 2>/dev/null; exit $?;;\n'
        ' --stop) if test -f "$FAKE_PID"; then kill -TERM "$(cat "$FAKE_PID")" 2>/dev/null || true; rm "$FAKE_PID"; fi; exit 0;;\n'
        'esac\nexec "$FAKE_PYTHON" "$FAKE_WORKER" "$@"\n'
    )
    holder = suite.GpuHolder(
        {
            "script": str(script),
            "start_args": ["4", "2400", "0,1,2,3"],
            "ready_patterns": ["holder ready"],
            "env": {
                "FAKE_PID": str(pid_file),
                "FAKE_ARGS": str(args_file),
                "FAKE_PYTHON": sys.executable,
                "FAKE_WORKER": str(worker),
            },
        },
        tmp_path / "holder.log",
    )
    holder.stop()
    try:
        holder.restore()
        pid = int(pid_file.read_text())
        assert os.getpgid(pid) == pid
        assert json.loads(args_file.read_text()) == ["4", "2400", "0,1,2,3"]
        assert holder.running()
    finally:
        holder.stop()


def test_schedule_install_copies_committed_controller_and_remove_preserves_jobs(accepted, monkeypatch):
    original, matrix, reference = accepted
    manifest_file = original / "model/manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["case"]["target"] = "worldfoundry.model:Model"
    suite.write_json(manifest_file, manifest)
    matrix.write_text(json.dumps({"model": manifest["case"]}))
    imported = schedule.import_reference(original, reference, matrix)
    source = original.parent / "checkout"
    tools = source / "tests/manual"
    tools.mkdir(parents=True)
    for name in schedule.CONTROLLERS:
        shutil.copyfile(TOOLS / name, tools / name)
    (source / "worldfoundry").mkdir()
    (source / "worldfoundry/model.py").write_text("class Model: pass\n")
    shutil.copyfile(matrix, source / "matrix.json")
    suite.write_json(
        source / "dependencies.json",
        {
            "schema_version": 1,
            "shared_paths": ["worldfoundry/core/**"],
            "ignored_paths": ["docs/**"],
            "components": {"model": ["worldfoundry/model.py"]},
            "cases": {"model": ["model"]},
        },
    )
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-qm", "accepted controller"),
    ):
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    state = original.parent / "state"
    profile_path = original.parent / "profile.json"
    suite.write_json(
        profile_path,
        {
            "source_root": str(source),
            "python": sys.executable,
            "reference": str(reference),
            "state_root": str(state),
            "matrix": "matrix.json",
            "dependencies": "dependencies.json",
            "cuda_visible_devices": "0",
            "reference_index_sha256": imported["reference_index_sha256"],
        },
    )
    original_controller = (tools / "geometry_regression_suite.py").read_bytes()
    (tools / "geometry_regression_suite.py").write_text("dirty uncommitted controller must not execute")
    original_cron = "15 2 * * * echo preserve-existing-job\n"
    cron = [original_cron]
    monkeypatch.setattr(schedule, "read_crontab", lambda: cron[0])
    monkeypatch.setattr(schedule, "write_crontab", lambda value: cron.__setitem__(0, value))
    result = schedule.install(profile_path)
    assert result["status"] == "installed"
    stable = state / "controllers" / result["controller_revision"] / "geometry_regression_suite.py"
    assert stable.read_bytes() == original_controller
    assert cron[0].startswith(original_cron)
    assert schedule.install(profile_path)["cron"] == cron[0]
    assert schedule.status(profile_path)["installed"]
    assert schedule.uninstall(profile_path)["status"] == "removed"
    assert cron[0] == original_cron
    assert not schedule.status(profile_path)["installed"]
