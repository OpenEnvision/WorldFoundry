"""Fail-closed readiness checks for benchmark runtime profiles.

The runtime profile is the executable contract for an official benchmark
runner.  This module validates only requirements declared by that contract;
it does not download assets or treat a successful preflight as benchmark
score evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.evaluation.utils import write_json

SCHEMA_VERSION = "worldfoundry-runtime-preflight-v1"
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PROFILE_DIR = REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official"
_ENV_PATTERN = re.compile(
    r"\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
    r"|\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)
_ENV_NAME_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]+\b")
_SECRET_PARTS = ("API_KEY", "AUTH", "CREDENTIAL", "PASSWORD", "PRIVATE", "SECRET", "TOKEN")


def _expand_env(value: str, environ: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    """Expand shell-style environment references without invoking a shell."""

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("plain") or match.group("braced") or ""
        current = environ.get(name)
        if current not in (None, ""):
            return str(current)
        default = match.group("default")
        if default is not None:
            return default
        missing.add(name)
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value), tuple(sorted(missing))


def _redact_text(value: str, environ: Mapping[str, str]) -> str:
    redacted = value
    for name, secret in environ.items():
        if secret and any(part in name.upper() for part in _SECRET_PARTS):
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _resolve_path(raw_path: str, *, repo_root: Path, environ: Mapping[str, str]) -> tuple[Path, tuple[str, ...]]:
    expanded, missing = _expand_env(raw_path, environ)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path, missing


def _resolve_python(configured: str, environ: Mapping[str, str]) -> dict[str, Any]:
    expanded, missing = _expand_env(configured, environ)
    executable: str | None = None
    if not missing:
        candidate = Path(expanded).expanduser()
        if candidate.parent != Path(".") or candidate.is_absolute():
            if candidate.is_file() and os.access(candidate, os.X_OK):
                executable = str(candidate.resolve())
        else:
            executable = shutil.which(expanded)
    return {
        "configured": configured,
        "resolved": expanded,
        "executable": executable,
        "missing_expansion_env": list(missing),
        "ok": executable is not None,
    }


def _environment_checks(requirements: Sequence[Any], environ: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(requirements, start=1):
        if isinstance(item, Mapping):
            raw_names = item.get("any_of") or item.get("names") or item.get("name") or item.get("env")
            if isinstance(raw_names, str):
                names = _ENV_NAME_PATTERN.findall(raw_names)
            elif isinstance(raw_names, Sequence):
                names = [str(name) for name in raw_names]
            else:
                names = []
        else:
            names = _ENV_NAME_PATTERN.findall(str(item))
        names = list(dict.fromkeys(names))
        present = [name for name in names if environ.get(name) not in (None, "")]
        rows.append(
            {
                "id": f"required_env_{index}",
                "any_of": names,
                "present": present,
                "values_redacted": True,
                "ok": bool(names and present),
            }
        )
    return rows


def _path_checks(requirements: Sequence[Any], *, repo_root: Path, environ: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(requirements, start=1):
        descriptor = dict(item) if isinstance(item, Mapping) else {"path": item}
        identifier = str(descriptor.get("id") or descriptor.get("name") or f"required_path_{index}")
        required = bool(descriptor.get("required", descriptor.get("required_for_env", True)))
        source_env = descriptor.get("env") or descriptor.get("source_env")
        raw_path = descriptor.get("path") or descriptor.get("value")
        if source_env and environ.get(str(source_env)) not in (None, ""):
            raw_path = environ[str(source_env)]
        if raw_path in (None, ""):
            rows.append(
                {
                    "id": identifier,
                    "path": None,
                    "required": required,
                    "exists": False,
                    "ok": not required,
                    "error": "path is not configured",
                }
            )
            continue
        resolved, missing = _resolve_path(str(raw_path), repo_root=repo_root, environ=environ)
        exists = not missing and resolved.exists()
        rows.append(
            {
                "id": identifier,
                "path": str(resolved),
                "required": required,
                "source_env": None if source_env is None else str(source_env),
                "missing_expansion_env": list(missing),
                "exists": exists,
                "ok": exists or not required,
            }
        )
    return rows


def _pythonpath(
    roots: Sequence[Any], *, repo_root: Path, environ: Mapping[str, str]
) -> tuple[list[str], list[dict[str, Any]]]:
    paths = [str(repo_root)]
    rows: list[dict[str, Any]] = []
    for index, root in enumerate(roots, start=1):
        resolved, missing = _resolve_path(str(root), repo_root=repo_root, environ=environ)
        exists = not missing and resolved.exists()
        rows.append(
            {
                "id": f"pythonpath_root_{index}",
                "path": str(resolved),
                "missing_expansion_env": list(missing),
                "exists": exists,
                "ok": exists,
            }
        )
        if exists:
            paths.append(str(resolved))
    inherited = environ.get("PYTHONPATH")
    if inherited:
        paths.extend(part for part in inherited.split(os.pathsep) if part)
    return list(dict.fromkeys(paths)), rows


def _run_python(
    executable: str,
    code: str,
    args: Sequence[str],
    *,
    child_env: Mapping[str, str],
    timeout: float,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [executable, "-c", code, *args],
            check=False,
            capture_output=True,
            text=True,
            env=dict(child_env),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "error": f"timed out after {timeout:g}s"}
    stderr = _redact_text(completed.stderr.strip(), environ)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "error": stderr[-2000:] if stderr else None,
        "stdout": _redact_text(completed.stdout.strip(), environ)[-2000:] or None,
    }


def _import_checks(
    modules: Sequence[Any],
    *,
    executable: str | None,
    child_env: Mapping[str, str],
    timeout: float,
    environ: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    code = "import importlib,sys; importlib.import_module(sys.argv[1])"
    for item in modules:
        module = str(item).strip()
        if not module:
            rows.append({"module": module, "ok": False, "error": "empty module name"})
        elif executable is None:
            rows.append({"module": module, "ok": False, "error": "python executable is unavailable"})
        else:
            rows.append(
                {
                    "module": module,
                    **_run_python(
                        executable,
                        code,
                        [module],
                        child_env=child_env,
                        timeout=timeout,
                        environ=environ,
                    ),
                }
            )
    return rows


def _cuda_check(
    required: bool,
    *,
    executable: str | None,
    child_env: Mapping[str, str],
    timeout: float,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    if not required:
        return {"required": False, "ok": True, "checked": False}
    if executable is None:
        return {"required": True, "ok": False, "checked": False, "error": "python executable is unavailable"}
    code = (
        "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),"
        "'device_count':torch.cuda.device_count()})); raise SystemExit(0 if torch.cuda.is_available() else 3)"
    )
    result = _run_python(
        executable,
        code,
        [],
        child_env=child_env,
        timeout=timeout,
        environ=environ,
    )
    payload: dict[str, Any] = {
        "required": True,
        "checked": True,
        "cuda_visible_devices_set": bool(environ.get("CUDA_VISIBLE_DEVICES")),
        **result,
    }
    if result.get("stdout"):
        try:
            payload["runtime"] = json.loads(str(result["stdout"]).splitlines()[-1])
        except json.JSONDecodeError:
            pass
    return payload


def _base_model_checks(value: Any, *, repo_root: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, Mapping) else {}
    missing = [str(item) for item in payload.get("missing") or ()]
    ready_env = payload.get("ready_env")
    ready_rows: list[dict[str, Any]] = []
    if isinstance(ready_env, Mapping):
        ready_rows = _path_checks(
            [{"id": str(name), "path": path} for name, path in ready_env.items()],
            repo_root=repo_root,
            environ=environ,
        )
    return {
        "declared": bool(payload),
        "status": payload.get("status"),
        "missing": missing,
        "ready_paths": ready_rows,
        "ok": not missing and all(row["ok"] for row in ready_rows),
    }


def check_profile(
    profile: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    repo_root: str | Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
    import_timeout: float = 30.0,
) -> dict[str, Any]:
    """Check one runtime profile and return a redacted JSON-compatible report."""

    env = dict(os.environ if environ is None else environ)
    root = Path(repo_root).resolve()
    env.setdefault("WORLDFOUNDRY_REPO_ROOT", str(root))
    profile_id = str(profile.get("id") or "").strip()
    python = _resolve_python(str(profile.get("python_path") or sys.executable), env)
    pythonpath, pythonpath_rows = _pythonpath(profile.get("pythonpath_roots") or (), repo_root=root, environ=env)
    child_env = dict(env)
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env_rows = _environment_checks(profile.get("required_env") or (), env)
    path_rows = _path_checks(profile.get("required_paths") or (), repo_root=root, environ=env)
    import_rows = _import_checks(
        profile.get("required_imports") or (),
        executable=python.get("executable"),
        child_env=child_env,
        timeout=import_timeout,
        environ=env,
    )
    cuda = _cuda_check(
        bool(profile.get("requires_cuda_visibility")),
        executable=python.get("executable"),
        child_env=child_env,
        timeout=import_timeout,
        environ=env,
    )
    base_models = _base_model_checks(profile.get("base_model_dependency_preflight"), repo_root=root, environ=env)
    groups = (env_rows, path_rows, pythonpath_rows, import_rows)
    ok = bool(profile_id and python["ok"] and cuda["ok"] and base_models["ok"])
    ok = ok and all(bool(row.get("ok")) for group in groups for row in group)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id or None,
        "manifest": str(Path(manifest_path).resolve()),
        "environment_id": profile.get("environment_id"),
        "status": profile.get("status"),
        "ok": ok,
        "python": python,
        "checks": {
            "required_env": env_rows,
            "required_paths": path_rows,
            "pythonpath_roots": pythonpath_rows,
            "required_imports": import_rows,
            "cuda": cuda,
            "base_models": base_models,
        },
        "summary": {
            "missing_env_groups": sum(not row["ok"] for row in env_rows),
            "missing_required_paths": sum(not row["ok"] for row in path_rows),
            "missing_pythonpath_roots": sum(not row["ok"] for row in pythonpath_rows),
            "failed_imports": sum(not row["ok"] for row in import_rows),
        },
    }


def _profile_paths(profile: str, manifest: Path) -> tuple[Path, ...]:
    if manifest.is_file():
        return (manifest,)
    if not manifest.is_dir():
        raise FileNotFoundError(f"runtime profile manifest does not exist: {manifest}")
    if profile == "all":
        return tuple(sorted((*manifest.glob("*.yaml"), *manifest.glob("*.yml"))))
    candidates = (manifest / f"{profile}.yaml", manifest / f"{profile}.yml")
    for candidate in candidates:
        if candidate.is_file():
            return (candidate,)
    raise FileNotFoundError(f"runtime profile {profile!r} was not found under {manifest}")


def run_preflight(
    *,
    profile: str,
    manifest: str | Path = DEFAULT_PROFILE_DIR,
    output_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
    import_timeout: float = 30.0,
) -> dict[str, Any]:
    """Run one or all profile checks and persist the report."""

    source = Path(manifest)
    reports: list[dict[str, Any]] = []
    for path in _profile_paths(profile, source):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise TypeError(f"runtime profile must be a YAML mapping: {path}")
        profile_id = str(payload.get("id") or "")
        benchmark_ids = [str(item) for item in payload.get("benchmark_ids") or ()]
        if profile != "all" and profile not in {profile_id, *benchmark_ids}:
            raise ValueError(f"requested profile {profile!r} does not match manifest id {profile_id!r}: {path}")
        reports.append(
            check_profile(
                payload,
                manifest_path=path,
                repo_root=repo_root,
                environ=environ,
                import_timeout=import_timeout,
            )
        )
    result: dict[str, Any]
    if len(reports) == 1:
        result = reports[0]
    else:
        result = {
            "schema_version": f"{SCHEMA_VERSION}-collection",
            "profile_id": "all",
            "ok": bool(reports) and all(report["ok"] for report in reports),
            "profile_count": len(reports),
            "ready_count": sum(bool(report["ok"]) for report in reports),
            "profiles": reports,
        }
    destination = Path(output_dir) / "preflight_report.json"
    write_json(destination, result)
    result["report_path"] = str(destination.resolve())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Runtime profile id, or 'all'.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--import-timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        profile=args.profile,
        manifest=args.manifest,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        import_timeout=args.import_timeout,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        state = "ready" if report["ok"] else "not ready"
        print(f"runtime preflight: {state}; report: {report['report_path']}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
