"""Run selected or scheduled 3D cases from a committed, isolated source snapshot."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def load_module(path: Path):
    name = "geometry_tool_" + path.stem + "_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


impact = load_module(Path(__file__).with_name("geometry_regression_impact.py"))
replay = load_module(Path(__file__).with_name("geometry_regression.py"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def validate_profile(profile: dict) -> dict:
    for name in ("source_root", "python", "reference", "state_root"):
        value = profile.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"Profile {name} must be an absolute path")
    for name in ("matrix", "dependencies"):
        impact.relative_path(profile[name])
    for name, default in (("case_timeout_seconds", 1800), ("suite_timeout_seconds", 7200), ("lease_wait_seconds", 60)):
        value = profile.setdefault(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Profile {name} must be finite and positive")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", profile.get("cuda_visible_devices", "")):
        raise ValueError("Declare physical CUDA indices in cuda_visible_devices")
    tokens = profile["cuda_visible_devices"].split(",")
    if len({int(t) for t in tokens}) != len(tokens):
        raise ValueError("CUDA indices must be unique")
    validate_env(profile.get("env", {}))
    overrides = profile.get("case_environments", {})
    if not isinstance(overrides, dict):
        raise ValueError("case_environments must map case ids to python/env overrides")
    for name, override in overrides.items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) or not isinstance(override, dict):
            raise ValueError("Invalid case environment override")
        if set(override) - {"python", "env"}:
            raise ValueError("Case environment overrides only support python and env")
        if "python" in override and (
            not isinstance(override["python"], str) or not Path(override["python"]).is_absolute()
        ):
            raise ValueError("Case environment python must be absolute")
        validate_env(override.get("env", {}))
    profile.setdefault("source_ref", "HEAD")
    if not isinstance(profile["source_ref"], str) or not profile["source_ref"] or profile["source_ref"].startswith("-"):
        raise ValueError("Invalid source_ref")
    retention = profile.setdefault("retain_runs", 7)
    if isinstance(retention, bool) or not isinstance(retention, int) or retention < 1:
        raise ValueError("retain_runs must be a positive integer")
    reference = Path(profile["reference"]).resolve()
    if reference.is_relative_to(Path(profile["state_root"]).resolve() / "runs"):
        raise ValueError("Accepted reference must be outside disposable runs")
    if not re.fullmatch(r"[0-9a-f]{64}", profile.get("reference_index_sha256", "")):
        raise ValueError("Pin reference_index_sha256 after explicitly importing an accepted reference")
    holder = profile.get("gpu_holder")
    if holder is not None:
        if not isinstance(holder, dict) or not Path(holder.get("script", "")).is_absolute():
            raise ValueError("gpu_holder.script must be absolute")
        args = holder.get("start_args")
        if not isinstance(args, list) or not args or not all(isinstance(v, str) and "\0" not in v for v in args):
            raise ValueError("gpu_holder.start_args must be a nonempty string list")
        validate_env(holder.get("env", {}))
        patterns = holder.get("ready_patterns", [])
        if not isinstance(patterns, list) or not all(isinstance(v, str) and v for v in patterns):
            raise ValueError("gpu_holder.ready_patterns must contain nonempty strings")
    return profile


def validate_env(env: dict) -> None:
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and re.fullmatch(r"[a-zA-Z_]\w*", k) and isinstance(v, str) and "\0" not in v
        for k, v in env.items()
    ):
        raise ValueError("Environment must contain valid string keys and values")


def read_profile(path: Path) -> dict:
    return validate_profile(json.loads(path.read_text()))


def case_configuration(profile: dict, name: str) -> dict:
    override = profile.get("case_environments", {}).get(name, {})
    return {
        **profile,
        "python": override.get("python", profile["python"]),
        "env": {**profile.get("env", {}), **override.get("env", {})},
    }


def child_environment(profile: dict) -> dict:
    return {
        **os.environ,
        **profile.get("env", {}),
        "CUDA_VISIBLE_DEVICES": profile["cuda_visible_devices"],
        "PYTHONPATH": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def reference_index(profile: dict) -> dict:
    path = Path(profile["reference"]) / "accepted-index.json"
    if replay.sha256(path) != profile["reference_index_sha256"]:
        raise ValueError("Accepted reference index was modified")
    index = json.loads(path.read_text())
    if index.get("schema_version") != 1 or not isinstance(index.get("cases"), dict):
        raise ValueError("Invalid accepted reference index")
    return index


def verify_reference(root: Path, name: str, index: dict) -> None:
    files = index["cases"][name]["files"]
    if not isinstance(files, dict) or not {"manifest.json", "arrays.npz"}.issubset(files):
        raise ValueError(f"Incomplete accepted reference index: {name}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Accepted reference contains a symlink: {name}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError(f"Accepted reference file inventory changed: {name}")
    for path, expected in files.items():
        impact.relative_path(path)
        if replay.sha256(root / path) != expected:
            raise ValueError(f"Accepted reference file was modified: {name}/{path}")


def resolve_case(value, env: dict):
    """Resolve asset variables while keeping the output-directory placeholder."""
    if isinstance(value, dict):
        return {key: resolve_case(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_case(item, env) for item in value]
    if isinstance(value, str):

        def replace(match):
            name = match.group(1)
            if name == "OUTPUT_DIR":
                return match.group(0)
            if name not in env:
                raise ValueError(f"Missing regression environment variable: {name}")
            return env[name]

        return re.sub(r"\$\{([a-zA-Z_]\w*)\}", replace, value)
    return value


def semantic_case(case: dict, env: dict) -> dict:
    result = resolve_case(case, env)
    result.pop("scope", None)  # Documentation-only coverage annotation.
    result.setdefault("seed", 42)
    result.setdefault("deterministic", False)
    return result


def materialize_case(case: dict, reference: Path, env: dict) -> dict:
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest.get("status") != "passed":
        raise ValueError(f"Reference did not pass: {case['id']}")
    if semantic_case(case, env) != semantic_case(manifest["case"], env):
        raise ValueError(f"Current case differs from its accepted recipe: {case['id']}")
    if manifest.get("arrays_sha256") != replay.sha256(reference / "arrays.npz"):
        raise ValueError(f"Accepted arrays were modified: {case['id']}")
    for path in resolve_case(case, env).get("assets", {}).values():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing replay asset for {case['id']}: {path}")
    # Retain the original representation of explicit defaults and asset paths.
    # The semantic equality check above prevents replaying an obsolete recipe.
    return manifest["case"]


def snapshot_source(profile: dict, destination: Path) -> dict:
    root = Path(profile["source_root"])
    revision = impact.revision(root, profile["source_ref"] + "^{commit}")
    if revision is None:
        raise ValueError("Cannot resolve source_ref to a committed revision")
    tree = impact.revision(root, revision + "^{tree}")
    archive = destination.with_suffix(".tar")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.required=false",
                "archive",
                "--format=tar",
                "--output=" + str(archive),
                revision,
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                impact.relative_path(member.name.rstrip("/"))
                if not member.isfile() and not member.isdir():
                    raise ValueError(f"Unsupported source archive link: {member.name}")
            bundle.extractall(destination, members=members)
    finally:
        archive.unlink(missing_ok=True)
    return {"source_revision": revision, "source_tree": tree}


def process_helpers(source: Path):
    # Load this exact file even when the calling test process has cached imports.
    sys.path.insert(0, str(source))
    return load_module(source / "worldfoundry/core/execution/process.py")


class TeardownError(RuntimeError):
    """Abort the suite when a case's descendants could still own the GPU."""


def run_child(command: list[str], env: dict, cwd: Path, log: Path, timeout: float, *, lease=None, helpers=None) -> int:
    """Always drain/kill the owned process group before releasing its lease."""
    if helpers is None:
        from worldfoundry.core.execution import process as helpers
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=lease.file_descriptors if lease is not None else (),
        )
        try:
            if lease is not None:
                lease.process_group_id = process.pid
            return process.wait(timeout=timeout)
        finally:
            try:
                helpers.terminate_process_group(process, grace_seconds=0.5)
            except Exception as exc:
                raise TeardownError(f"Replay teardown failed: {exc}") from exc
            if helpers.process_group_alive(process.pid):
                raise TeardownError("Replay descendants still own GPU resources after teardown")


def probe_cuda(profile: dict, env: dict, run_root: Path, lease=None, *, case_ids=None, label="cuda-preflight") -> None:
    metadata_path = run_root / (label + "-metadata.json")
    replay_path = str(Path(__file__).with_name("geometry_regression.py"))
    code = (
        "import importlib.util,json,pathlib,torch; "
        # Tensor runtimes import setuptools through torch's extension helpers.
        # Mirror its vendor-path activation before enumerating distributions.
        "import setuptools; "
        "assert torch.cuda.is_available(), 'GPU regression requires CUDA'; torch.cuda.init(); "
        f"spec=importlib.util.spec_from_file_location('preflight_replay', {replay_path!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        f"pathlib.Path({str(metadata_path)!r}).write_text(json.dumps(module.runtime_metadata(torch)))"
    )
    helpers = process_helpers(run_root / "source")
    result = run_child(
        [profile["python"], "-c", code],
        env,
        run_root,
        run_root / (label + ".log"),
        60,
        helpers=helpers,
        lease=lease,
    )
    if result:
        raise RuntimeError(f"GPU preflight failed; see {label}.log")
    actual = json.loads(metadata_path.read_text())
    recipes = json.loads((run_root / "recipes.json").read_text())
    for name in list(recipes) if case_ids is None else case_ids:
        expected = json.loads((Path(profile["reference"]) / name / "manifest.json").read_text())["runtime"]
        # Algorithm flags and seeds are set individually by run_case. These
        # environment fields must already match before spending time on weights.
        differences = [
            field
            for field in ("python", "torch", "cuda", "cudnn", "gpu", "packages")
            if actual.get(field) != expected.get(field)
        ]
        if differences:
            raise RuntimeError(
                f"GPU replay environment differs from accepted reference for {name}: "
                + ", ".join(differences)
                + "; see "
                + metadata_path.name
            )


def acquire_gpu(profile: dict, env: dict, source: Path):
    pool_type = load_module(source / "worldfoundry/runtime/device_pool.py").CudaDeviceLeasePool
    tokens = tuple(profile["cuda_visible_devices"].split(","))
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    keys = dict(tuple(item.strip() for item in line.split(",", 1)) for line in query.stdout.splitlines())
    if any(token not in keys for token in tokens):
        raise ValueError("Requested physical GPU was not discovered")
    directory = Path(
        env.get("WORLDFOUNDRY_GPU_LEASE_DIR")
        or str(Path(tempfile.gettempdir()) / f"worldfoundry-gpu-leases-{os.getuid()}")
    )
    pool = pool_type(tokens, lock_dir=directory, lock_keys=keys)
    return pool.acquire(tokens=tokens, deadline=time.monotonic() + profile["lease_wait_seconds"])


class GpuHolder:
    """Stop only the explicitly configured holder, and restore it on every exit."""

    def __init__(self, config: dict | None, log: Path):
        self.config, self.log = config, log
        self.entered = False

    def running(self) -> bool:
        result = subprocess.run(
            ["bash", self.config["script"], "--status"],
            env={**os.environ, **self.config.get("env", {})},
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0

    def stop(self):
        if self.config:
            self.entered = True
            subprocess.run(
                ["bash", self.config["script"], "--stop"],
                env={**os.environ, **self.config.get("env", {})},
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )

    def restore(self):
        if not self.config or not self.entered or self.running():
            return
        with self.log.open("a") as stream:
            offset = stream.tell()
            process = subprocess.Popen(
                ["bash", self.config["script"], *self.config["start_args"]],
                env={**os.environ, **self.config.get("env", {})},
                cwd=Path(self.config["script"]).parent,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("GPU holder exited during restoration")
            with self.log.open() as stream:
                stream.seek(offset)
                output = stream.read()
            if self.running() and all(item in output for item in self.config.get("ready_patterns", [])):
                return
            time.sleep(0.1)
        raise RuntimeError("GPU holder restoration did not become ready")


@contextlib.contextmanager
def interruption_cleanup():
    def interrupt(_signum, _frame):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt("GPU regression interrupted")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def validate_plan(selection: dict, matrix: Path, dependencies: Path, source: Path, tree: str) -> list[str]:
    if selection.get("status") not in {"planned", "no_inference_changes"}:
        raise ValueError("Uncovered or failed impact plan cannot authorize replays")
    if (
        selection.get("source_tree") != tree
        or selection.get("matrix_sha256") != impact.digest(matrix)
        or selection.get("dependencies_sha256") != impact.digest(dependencies)
    ):
        raise ValueError("Impact plan is stale for this snapshot")
    selected = selection.get("selected_cases")
    if (
        not isinstance(selected, list)
        or not all(isinstance(v, str) for v in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Invalid plan selected_cases")
    if bool(selected) != (selection["status"] == "planned"):
        raise ValueError("Impact plan status contradicts its selected cases")
    if not isinstance(selection.get("changed_paths"), list):
        raise ValueError("Impact plan must declare its changed_paths")
    minimum = impact.select_cases(matrix, dependencies, source, selection["changed_paths"])
    if minimum["status"] == "uncovered" or not set(minimum["selected_cases"]).issubset(selected):
        raise ValueError("Impact plan omits required cases or has uncovered changes")
    return selected


def prune_runs(state: Path, retain: int, current: Path) -> list[str]:
    removed = []
    runs = sorted(
        (p for p in (state / "runs").iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name, reverse=True
    )
    for path in runs[retain:]:
        if path == current or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", path.name):
            continue
        # Keep the compact report after removing source snapshots and artifacts.
        report = path / "report.json"
        if report.is_file():
            write_json(state / "history" / (path.name + ".json"), json.loads(report.read_text()))
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def run_suite(
    profile: dict, *, plan: Path | None = None, case_ids: list[str] | None = None, preflight_only: bool = False
) -> dict:
    profile = validate_profile(profile)
    state = Path(profile["state_root"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        root = state / "runs" / run_id
        root.mkdir(parents=True)
        report = {
            "status": "running",
            "run_id": run_id,
            "run_root": str(root),
            "cases": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": "preflight" if preflight_only else "replay",
        }
        write_json(root / "report.json", report)
        write_json(state / "latest.json", report)
        started = time.monotonic()
        holder = GpuHolder(profile.get("gpu_holder"), root / "gpu-holder.log")
        lease = None
        cleanup_safe = True
        try:
            report.update(snapshot_source(profile, root / "source"))
            source = root / "source"
            matrix, dependencies = source / profile["matrix"], source / profile["dependencies"]
            cases, _ = impact.load_definitions(matrix, dependencies)
            if set(profile.get("case_environments", {})) - set(cases):
                raise ValueError("Case environment override has an unknown case id")
            selected = list(cases) if case_ids is None else list(dict.fromkeys(case_ids))
            if plan is not None:
                if case_ids is not None:
                    raise ValueError("Do not combine a saved plan with explicit case ids")
                selection = json.loads(plan.read_text())
                selected = validate_plan(selection, matrix, dependencies, source, report["source_tree"])
                write_json(root / "impact-plan.json", selection)
            if not selected:
                if plan is None:
                    raise ValueError("A GPU suite cannot be empty")
                report["status"] = "no_inference_changes"
                return report
            if any(name not in cases for name in selected):
                raise ValueError("Selected case is not declared in the current matrix")
            report["selected_cases"] = selected
            env = child_environment(profile)
            configurations = {name: case_configuration(profile, name) for name in selected}
            recipes = {}
            index = reference_index(profile)
            for name in selected:
                verify_reference(Path(profile["reference"]) / name, name, index)
                recipes[name] = materialize_case(
                    cases[name], Path(profile["reference"]) / name, child_environment(configurations[name])
                )
            write_json(root / "recipes.json", recipes)
            helpers = process_helpers(source)
            lease = acquire_gpu(profile, env, source)
            groups = {}
            for name, configuration in configurations.items():
                key = (configuration["python"], json.dumps(configuration["env"], sort_keys=True))
                groups.setdefault(key, []).append(name)
            report["environments"] = []
            for index, names in enumerate(groups.values()):
                configuration = configurations[names[0]]
                label = "cuda-preflight" if index == 0 else f"cuda-preflight-{index}"
                report["environments"].append({"python": configuration["python"], "cases": names, "preflight": label})
                probe_cuda(
                    configuration, child_environment(configuration), root, lease=lease, case_ids=names, label=label
                )
            if preflight_only:
                report["status"] = "preflight_passed"
                return report
            holder.stop()
            for name in selected:
                remaining = profile["suite_timeout_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Whole GPU suite exceeded its time budget")
                candidate = root / "candidate" / name
                command = [
                    configurations[name]["python"],
                    str(Path(__file__).with_name("geometry_regression.py")),
                    "run",
                    "--case",
                    str(root / "recipes.json"),
                    "--case-id",
                    name,
                    "--source-root",
                    str(source),
                    "--output-dir",
                    str(candidate),
                ]
                try:
                    exit_code = run_child(
                        command,
                        child_environment(configurations[name]),
                        source,
                        root / f"{name}.log",
                        min(remaining, profile["case_timeout_seconds"]),
                        lease=lease,
                        helpers=helpers,
                    )
                    if exit_code:
                        raise RuntimeError(f"Replay process exited with status {exit_code}")
                    single = root / f"{name}-matrix.json"
                    write_json(single, {name: recipes[name]})
                    result = replay.audit_matrix(single, Path(profile["reference"]), root / "candidate", source, [])
                    if result["status"] != "passed":
                        raise ValueError(json.dumps(result["cases"][name]))
                    report["cases"][name] = result["cases"][name]
                except TeardownError as exc:
                    report["cases"][name] = {"status": "failed", "error": str(exc)}
                    cleanup_safe = False
                    raise
                except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    report["cases"][name] = {"status": "failed", "error": str(exc)}
                write_json(root / "report.json", report)
                write_json(state / "latest.json", report)
            if len(report["cases"]) == len(selected) and all(c["status"] == "passed" for c in report["cases"].values()):
                report["status"] = "passed"
            else:
                report["status"] = "failed"
            if time.monotonic() - started > profile["suite_timeout_seconds"]:
                raise TimeoutError("Whole GPU suite exceeded its time budget")
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if lease is not None:
                    lease.release()
            except Exception as exc:
                cleanup_safe = False
                report.update(status="failed", cleanup_error=str(exc))
            try:
                if cleanup_safe:
                    holder.restore()
                elif holder.entered:
                    report["holder_error"] = "GPU holder restoration deferred: replay resources may still be in use"
            except Exception as exc:
                report.update(status="failed", holder_error=str(exc))
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(root / "report.json", report)
            write_json(state / "latest.json", report)
            try:
                prune_runs(state, profile["retain_runs"], root)
            except Exception as exc:
                report["retention_error"] = str(exc)
                write_json(root / "report.json", report)
                write_json(state / "latest.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Check assets and accepted environments without model inference"
    )
    args = parser.parse_args()
    try:
        with interruption_cleanup():
            report = run_suite(
                read_profile(args.profile), plan=args.plan, case_ids=args.case_id, preflight_only=args.preflight_only
            )
    except (Exception, KeyboardInterrupt) as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    print(
        json.dumps({key: report[key] for key in ("status", "run_root", "elapsed_seconds", "error") if key in report}),
        flush=True,
    )
    return 0 if report["status"] in {"passed", "no_inference_changes", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
