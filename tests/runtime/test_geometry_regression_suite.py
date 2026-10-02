"""Exercise replay orchestration, immutable evidence and process cleanup on CPU."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"
spec = importlib.util.spec_from_file_location("suite_under_test", TOOLS / "geometry_regression_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


@pytest.fixture
def replay_host(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a", "b"):
        module = source / "worldfoundry" / (name + ".py")
        module.parent.mkdir(exist_ok=True)
        module.write_text("class Model: pass\n")
    helper = source / "worldfoundry/core/execution/process.py"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(TOOLS.parents[1] / "worldfoundry/core/execution/process.py", helper)
    cases = {name: {"id": name, "target": f"worldfoundry.{name}:Model", "assets": {}} for name in ("a", "b")}
    (source / "matrix.json").write_text(json.dumps(cases))
    (source / "dependencies.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_paths": ["worldfoundry/core/**"],
                "ignored_paths": ["docs/**"],
                "components": {"a": ["worldfoundry/a.py"], "b": ["worldfoundry/b.py"]},
                "cases": {"a": ["a"], "b": ["b"]},
            }
        )
    )
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Regression test"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    reference = tmp_path / "reference"
    index = {"schema_version": 1, "cases": {}}
    for name, case in cases.items():
        root = reference / name
        root.mkdir(parents=True)
        np.savez(root / "arrays.npz", depth=np.array([1.0, 2.0]))
        suite.write_json(
            root / "manifest.json",
            {
                "case": case,
                "status": "passed",
                "assets": {},
                "runtime": {},
                "arrays_sha256": suite.replay.sha256(root / "arrays.npz"),
            },
        )
        index["cases"][name] = {"files": {p.name: suite.replay.sha256(p) for p in root.iterdir()}}
    suite.write_json(reference / "accepted-index.json", index)
    profile = {
        "source_root": str(source),
        "python": sys.executable,
        "source_ref": "HEAD",
        "reference": str(reference),
        "state_root": str(tmp_path / "state"),
        "matrix": "matrix.json",
        "dependencies": "dependencies.json",
        "cuda_visible_devices": "0",
        "reference_index_sha256": suite.replay.sha256(reference / "accepted-index.json"),
    }
    return suite.validate_profile(profile)


def no_gpu(monkeypatch, *, failure=None):
    events = []
    lease = SimpleNamespace(release=lambda: events.append("release"))
    monkeypatch.setattr(suite, "acquire_gpu", lambda *args: lease)
    monkeypatch.setattr(suite, "probe_cuda", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite.GpuHolder, "stop", lambda self: (setattr(self, "entered", True), events.append("stop")))
    monkeypatch.setattr(suite.GpuHolder, "restore", lambda self: events.append("restore"))
    monkeypatch.setattr(suite, "run_child", lambda *args, **kwargs: (_ for _ in ()).throw(failure) if failure else 0)
    monkeypatch.setattr(
        suite.replay,
        "audit_matrix",
        lambda matrix, *args: {
            "status": "passed",
            "cases": {name: {"status": "passed"} for name in json.loads(matrix.read_text())},
        },
    )
    return events, lease


def test_snapshot_is_committed_and_helpers_ignore_import_cache(replay_host, tmp_path):
    source = Path(replay_host["source_root"])
    (source / "worldfoundry/a.py").write_text("dirty production edit\n")
    (source / "untracked.py").write_text("not accepted\n")
    target = tmp_path / "snapshot"
    meta = suite.snapshot_source(replay_host, target)
    assert len(meta["source_revision"]) == 40
    assert "class Model" in (target / "worldfoundry/a.py").read_text()
    assert not (target / "untracked.py").exists()
    assert Path(suite.process_helpers(target).__file__).is_relative_to(target)


@pytest.mark.parametrize("ref", ["--help", "HEAD^{tree}", "does-not-exist"])
def test_source_ref_must_be_a_commit(replay_host, tmp_path, ref):
    replay_host["source_ref"] = ref
    with pytest.raises(ValueError, match="committed revision"):
        suite.snapshot_source(replay_host, tmp_path / "snapshot")


def test_saved_plan_cannot_omit_affected_model_or_lie_about_empty_selection(replay_host):
    source = Path(replay_host["source_root"])
    matrix, deps = source / "matrix.json", source / "dependencies.json"
    plan = suite.impact.select_cases(matrix, deps, source, ["worldfoundry/core/new.py"])
    assert suite.validate_plan(plan, matrix, deps, source, plan["source_tree"]) == ["a", "b"]
    plan["selected_cases"] = ["a"]
    with pytest.raises(ValueError, match="omits required"):
        suite.validate_plan(plan, matrix, deps, source, plan["source_tree"])
    plan["selected_cases"] = []
    with pytest.raises(ValueError, match="contradicts"):
        suite.validate_plan(plan, matrix, deps, source, plan["source_tree"])
    plan["status"] = "no_inference_changes"
    with pytest.raises(ValueError, match="omits required"):
        suite.validate_plan(plan, matrix, deps, source, plan["source_tree"])


def test_stale_policy_plan_is_rejected(replay_host):
    source = Path(replay_host["source_root"])
    matrix, deps = source / "matrix.json", source / "dependencies.json"
    plan = suite.impact.select_cases(matrix, deps, source, ["worldfoundry/a.py"])
    deps.write_text(deps.read_text() + "\n")
    with pytest.raises(ValueError, match="stale"):
        suite.validate_plan(plan, matrix, deps, source, plan["source_tree"])


def test_only_documented_recipe_defaults_and_scope_can_change(replay_host):
    root = Path(replay_host["reference"]) / "a"
    case = json.loads((root / "manifest.json").read_text())["case"]
    portable = {**case, "seed": 42, "deterministic": False, "scope": "documentation"}
    assert suite.materialize_case(portable, root, {}) == case
    with pytest.raises(ValueError, match="accepted recipe"):
        suite.materialize_case({**portable, "seed": 43}, root, {})
    assert suite.resolve_case("${ASSET}/${OUTPUT_DIR}", {"ASSET": "path"}) == "path/${OUTPUT_DIR}"
    with pytest.raises(ValueError, match="Missing regression"):
        suite.resolve_case("${MISSING}", {})


@pytest.mark.parametrize("name", ["manifest.json", "arrays.npz", "extra-export.png"])
def test_modified_baseline_files_are_rejected_before_gpu(replay_host, monkeypatch, name):
    events, _ = no_gpu(monkeypatch)
    (Path(replay_host["reference"]) / "a" / name).write_bytes(b"modified")
    report = suite.run_suite(replay_host, case_ids=["a"])
    assert report["status"] == "failed"
    assert "reference" in report["error"]
    assert "stop" not in events


@pytest.mark.parametrize(
    "failure", [RuntimeError("worker failed"), subprocess.TimeoutExpired("worker", 1), KeyboardInterrupt("shutdown")]
)
def test_failure_timeout_and_interrupt_release_then_restore_holder(replay_host, monkeypatch, failure):
    events, _ = no_gpu(monkeypatch, failure=failure)
    report = suite.run_suite(replay_host, case_ids=["a"])
    assert report["status"] == "failed"
    assert events == ["stop", "release", "restore"]
    latest = json.loads((Path(replay_host["state_root"]) / "latest.json").read_text())
    assert latest == report


def test_teardown_failure_aborts_remaining_cases_and_defers_holder(replay_host, monkeypatch):
    events, _ = no_gpu(monkeypatch, failure=suite.TeardownError("resources still owned"))
    report = suite.run_suite(replay_host)
    assert report["status"] == "failed"
    assert set(report["cases"]) == {"a"}
    assert events == ["stop", "release"]
    assert "deferred" in report["holder_error"]


def test_failed_lease_release_defers_holder(replay_host, monkeypatch):
    events, lease = no_gpu(monkeypatch)
    lease.release = lambda: (_ for _ in ()).throw(RuntimeError("descendant alive"))
    report = suite.run_suite(replay_host, case_ids=["a"])
    assert report["status"] == "failed"
    assert "cleanup_error" in report
    assert events == ["stop"]


def test_running_status_is_written_before_expensive_work(replay_host, monkeypatch):
    original = suite.snapshot_source

    def observe(profile, destination):
        latest = json.loads((Path(profile["state_root"]) / "latest.json").read_text())
        assert latest["status"] == "running"
        return original(profile, destination)

    monkeypatch.setattr(suite, "snapshot_source", observe)
    no_gpu(monkeypatch)
    assert suite.run_suite(replay_host, case_ids=["a"])["status"] == "passed"


@pytest.mark.parametrize(
    "change",
    [
        {"cuda_visible_devices": "0,0"},
        {"case_timeout_seconds": 0},
        {"suite_timeout_seconds": float("inf")},
        {"retain_runs": False},
        {"env": {"INVALID=KEY": "value"}},
        {"source_ref": "--help"},
        {"gpu_holder": {"script": "relative", "start_args": []}},
        {"case_environments": {"a": {"python": "relative"}}},
        {"case_environments": {"a": {"env": {"BAD=KEY": "value"}}}},
        {"case_environments": {"a": {"unknown": "value"}}},
    ],
)
def test_unsafe_profiles_are_rejected(replay_host, change):
    with pytest.raises(ValueError):
        suite.validate_profile({**replay_host, **change})


def test_timeout_kills_sigterm_ignoring_descendant_after_it_is_ready(tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    child = (
        "import signal,time,pathlib,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(60)"
        % str(ready)
    )
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(60)" % child
    helper = suite.process_helpers(TOOLS.parents[1])
    original = subprocess.Popen

    class ReadyProcess(original):
        def wait(self, timeout=None):
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "descendant failed to announce readiness"
            return super().wait(timeout=timeout)

    monkeypatch.setattr(suite.subprocess, "Popen", ReadyProcess)
    with pytest.raises(subprocess.TimeoutExpired):
        suite.run_child(
            [sys.executable, "-c", parent], dict(os.environ), tmp_path, tmp_path / "worker.log", 0.05, helpers=helper
        )
    pid = int(ready.read_text())
    identity = helper.process_identity(pid)
    assert identity is None or not helper.process_group_alive(identity["pgid"])


def test_repeated_interrupts_cannot_interrupt_cleanup():
    before = signal.getsignal(signal.SIGTERM)
    with suite.interruption_cleanup():
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize("stop_signal", ["TERM", "INT"])
def test_restored_holder_can_stop_after_interrupted_cleanup(tmp_path, stop_signal):
    script = tmp_path / "holder.sh"
    pid_file = tmp_path / "holder.pid"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        '  --status) test -f "$HOLDER_TEST_PID" && kill -0 "$(cat "$HOLDER_TEST_PID")"; exit $?;;\n'
        '  --stop) kill -"$HOLDER_TEST_SIGNAL" "$(cat "$HOLDER_TEST_PID")"\n'
        '    for _ in $(seq 1 80); do test -f "$HOLDER_TEST_PID" || exit 0; sleep 0.025; done; exit 1;;\n'
        "esac\n"
        "trap 'rm -f \"$HOLDER_TEST_PID\"; exit 0' TERM INT\n"
        'echo "$$" > "$HOLDER_TEST_PID"\n'
        'echo "holder ready"\n'
        "while true; do sleep 0.025; done\n"
    )
    holder = suite.GpuHolder(
        {
            "script": str(script),
            "start_args": ["start"],
            "env": {"HOLDER_TEST_PID": str(pid_file), "HOLDER_TEST_SIGNAL": stop_signal},
            "ready_patterns": ["holder ready"],
        },
        tmp_path / "holder.log",
    )
    holder.entered = True
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    pid = None
    try:
        try:
            for sig in before:
                signal.signal(sig, signal.SIG_IGN)
            holder.restore()
            assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in before)
        finally:
            for sig, handler in before.items():
                signal.signal(sig, handler)
            if pid_file.exists():
                pid = int(pid_file.read_text())
        holder.stop()
        assert not holder.running()
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


def test_real_failed_preflight_has_log_and_never_stops_holder(replay_host, monkeypatch):
    events, _ = no_gpu(monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(
        suite,
        "acquire_gpu",
        lambda *args: SimpleNamespace(release=lambda: events.append("release"), file_descriptors=()),
    )
    monkeypatch.setattr(suite.GpuHolder, "stop", lambda self: events.append("stop"))
    # Force the CUDA probe to fail on GPU hosts as well as CPU-only CI. A real
    # visible GPU otherwise reaches the unrelated accepted-environment guard.
    replay_host["cuda_visible_devices"] = "999999"
    report = suite.run_suite(replay_host, case_ids=["a"])
    assert report["status"] == "failed"
    assert "preflight failed" in report["error"]
    log = (Path(report["run_root"]) / "cuda-preflight.log").read_text()
    assert "GPU regression requires CUDA" in log or "No module named 'torch'" in log
    assert events == ["release"]


def test_host_lock_prevents_overlapping_suites(replay_host, monkeypatch):
    import fcntl

    state = Path(replay_host["state_root"])
    state.mkdir()
    with (state / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            suite.run_suite(replay_host)


def test_no_inference_plan_never_probes_gpu(replay_host, tmp_path, monkeypatch):
    source = Path(replay_host["source_root"])
    plan = suite.impact.select_cases(source / "matrix.json", source / "dependencies.json", source, ["docs/guide.md"])
    plan_file = tmp_path / "plan.json"
    suite.write_json(plan_file, plan)
    monkeypatch.setattr(suite, "acquire_gpu", lambda *args: pytest.fail("no inference changes must not acquire GPU"))
    assert suite.run_suite(replay_host, plan=plan_file)["status"] == "no_inference_changes"


@pytest.mark.parametrize("field", ["cudnn", "packages"])
def test_preflight_rejects_changed_runtime_before_loading_weights(replay_host, tmp_path, monkeypatch, field):
    manifest = Path(replay_host["reference"]) / "a/manifest.json"
    data = json.loads(manifest.read_text())
    data["runtime"] = {field: "accepted"}
    suite.write_json(manifest, data)
    suite.write_json(tmp_path / "recipes.json", {"a": {}})
    monkeypatch.setattr(suite, "process_helpers", lambda source: None)

    def fake_probe(*args, **kwargs):
        suite.write_json(tmp_path / "cuda-preflight-metadata.json", {field: "different"})
        return 0

    monkeypatch.setattr(suite, "run_child", fake_probe)
    with pytest.raises(RuntimeError, match="environment differs.*" + field):
        suite.probe_cuda(replay_host, {}, tmp_path)


@pytest.mark.parametrize("reference_inventory", ["plain", "activated"])
def test_preflight_accepts_exact_inventory_with_or_without_setuptools_vendor_activation(
    replay_host, tmp_path, monkeypatch, reference_inventory
):
    plain = {"cudnn": 9000, "packages": {"torch": "2.10", "packaging": "26.3"}}
    activated = {"cudnn": 9000, "packages": {"torch": "2.10", "packaging": "26.0", "jaraco.text": "4.0"}}
    manifest = Path(replay_host["reference"]) / "a/manifest.json"
    data = json.loads(manifest.read_text())
    data["runtime"] = plain if reference_inventory == "plain" else activated
    suite.write_json(manifest, data)
    suite.write_json(tmp_path / "recipes.json", {"a": {}})
    monkeypatch.setattr(suite, "process_helpers", lambda source: None)

    def fake_probe(*args, **kwargs):
        suite.write_json(tmp_path / "cuda-preflight-metadata.json", activated)
        suite.write_json(tmp_path / "cuda-preflight-plain-metadata.json", plain)
        return 0

    monkeypatch.setattr(suite, "run_child", fake_probe)
    suite.probe_cuda(replay_host, {}, tmp_path)


@pytest.mark.parametrize("change", ["package_version", "cuda_version", "mixed_inventories"])
def test_preflight_never_ignores_real_drift_or_combines_parts_of_two_inventories(
    replay_host, tmp_path, monkeypatch, change
):
    expected = {"cudnn": 9000, "cuda": "12.8", "packages": {"torch": "2.10", "packaging": "26.3"}}
    plain = json.loads(json.dumps(expected))
    activated = {**expected, "packages": {"torch": "2.10", "packaging": "26.0", "jaraco.text": "4.0"}}
    if change == "package_version":
        plain["packages"]["torch"] = activated["packages"]["torch"] = "2.11"
    elif change == "cuda_version":
        plain["cuda"] = activated["cuda"] = "13.0"
    else:
        activated.update(packages=expected["packages"], cudnn=8000)
        plain["packages"]["packaging"] = "wrong"
    manifest = Path(replay_host["reference"]) / "a/manifest.json"
    data = json.loads(manifest.read_text())
    data["runtime"] = expected
    suite.write_json(manifest, data)
    suite.write_json(tmp_path / "recipes.json", {"a": {}})
    monkeypatch.setattr(suite, "process_helpers", lambda source: None)

    def fake_probe(*args, **kwargs):
        suite.write_json(tmp_path / "cuda-preflight-metadata.json", activated)
        suite.write_json(tmp_path / "cuda-preflight-plain-metadata.json", plain)
        return 0

    monkeypatch.setattr(suite, "run_child", fake_probe)
    with pytest.raises(RuntimeError, match="environment differs"):
        suite.probe_cuda(replay_host, {}, tmp_path)


def test_each_case_uses_its_accepted_environment_and_groups_preflight(replay_host, tmp_path, monkeypatch):
    no_gpu(monkeypatch)
    replay_host["env"] = {"COMMON_ENV": "shared", "CASE_ENV": "default"}
    other_python = str(tmp_path / "other-model-env/bin/python")
    replay_host["case_environments"] = {"b": {"python": other_python, "env": {"CASE_ENV": "special"}}}
    probes, children = [], []

    def probe(profile, env, root, **kwargs):
        probes.append((profile["python"], kwargs["case_ids"], env["CASE_ENV"]))

    def child(command, env, *args, **kwargs):
        children.append((command[0], env["CASE_ENV"], env["COMMON_ENV"]))
        return 0

    monkeypatch.setattr(suite, "probe_cuda", probe)
    monkeypatch.setattr(suite, "run_child", child)
    report = suite.run_suite(replay_host)
    assert report["status"] == "passed"
    assert probes == [(sys.executable, ["a"], "default"), (other_python, ["b"], "special")]
    assert children == [(sys.executable, "default", "shared"), (other_python, "special", "shared")]
    assert replay_host["env"]["CASE_ENV"] == "default"


def test_misspelled_case_environment_cannot_be_silently_ignored(replay_host, monkeypatch):
    events, _ = no_gpu(monkeypatch)
    replay_host["case_environments"] = {"misspelled-model": {"python": sys.executable}}
    report = suite.run_suite(replay_host)
    assert report["status"] == "failed"
    assert "unknown case id" in report["error"]
    assert "stop" not in events


def test_preflight_only_is_never_successful_inference_evidence(replay_host, monkeypatch):
    events, _ = no_gpu(monkeypatch)
    monkeypatch.setattr(suite, "run_child", lambda *args, **kwargs: pytest.fail("preflight-only must not load weights"))
    report = suite.run_suite(replay_host, preflight_only=True)
    assert report["status"] == "preflight_passed"
    assert report["mode"] == "preflight" and report["cases"] == {}
    assert "stop" not in events
