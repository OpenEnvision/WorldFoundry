"""Build and write run manifests and environment snapshots.

A *run manifest* captures the full reproducibility context for an evaluation
run — configuration, environment, and requirements — so that results can be
independently verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import jsonable, write_json
from worldfoundry.runtime.env import check_required_env
from worldfoundry.evaluation.utils import git_metadata, package_version, stable_hash

# ── Schema version identifiers ──────────────────────────────
RUN_MANIFEST_SCHEMA_VERSION = "worldfoundry-run-manifest"
ENVIRONMENT_SCHEMA_VERSION = "worldfoundry-environment"
ENV_REQUIREMENTS_SCHEMA_VERSION = "worldfoundry-env-requirements"

# ── Secret redaction ─────────────────────────────────────────
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "bearer_token",
        "client_secret",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    }
)


def _is_sensitive_key(key: str) -> bool:
    """Return whether *key* looks like a secret field that should be redacted."""
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def redact_secrets(value: Any) -> Any:
    """Return a JSON payload with secret-like mapping values removed.

    Args:
        value: Configuration payload that may contain token, API key, password, or secret fields.
    """

    payload = jsonable(value)
    if isinstance(payload, Mapping):
        return {
            str(key): "<redacted>" if _is_sensitive_key(str(key)) else redact_secrets(item)
            for key, item in payload.items()
        }
    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]
    return payload


def _revision_from(payload: Mapping[str, Any]) -> str | None:
    """Extract a revision string from common key names in *payload*."""
    for key in ("revision", "repo_revision", "commit", "sha", "version"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


# ── Environment and path requirement rows ────────────────────
def _environment_row(environ: Mapping[str, str], name: str) -> dict[str, Any]:
    """Build a redacted environment-variable evidence row."""
    return {
        "name": name,
        "present": name in environ and environ[name] != "",
        "redacted": True,
    }


def _path_requirement_row(item: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Build a path existence evidence row from a requirement descriptor."""
    if isinstance(item, Mapping):
        name = str(item.get("name") or item.get("key") or item.get("path") or "path")
        raw_path = item.get("path") or item.get("value") or ""
    else:
        raw_path = item
        name = Path(item).name or str(item)
    path = Path(raw_path).expanduser() if raw_path not in (None, "") else Path("")
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
    }


def _installed_packages(package_names: Sequence[str] | None) -> dict[str, str]:
    """Collect installed package versions, optionally filtered by *package_names*."""
    requested = {item.lower().replace("_", "-") for item in package_names} if package_names else None
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get("Name") or "")
        normalized = name.lower().replace("_", "-")
        if not name or (requested is not None and normalized not in requested):
            continue
        packages[name] = str(distribution.version)
    if package_names and "worldfoundry" not in packages:
        packages["worldfoundry"] = package_version()
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def build_env_requirements(
    *,
    required_env: Sequence[str | Mapping[str, Any]] = (),
    required_paths: Sequence[str | Path | Mapping[str, Any]] = (),
    required_extras: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build redacted environment requirement evidence for a run.

    Args:
        required_env: Required environment variable names or mappings.
        required_paths: Required local paths checked for existence only.
        required_extras: Required optional dependency groups or runtime bundles.
        environ: Environment mapping used for deterministic tests; defaults to ``os.environ``.
    """

    env = os.environ if environ is None else environ
    env_names = [
        str(item.get("name") or item.get("env") or item.get("key"))
        if isinstance(item, Mapping)
        else str(item)
        for item in required_env
    ]
    required_env_rows = [_environment_row(env, name) for name in env_names if name]
    required_path_rows = [_path_requirement_row(item) for item in required_paths]
    env_report = check_required_env(tuple(row["name"] for row in required_env_rows), env)
    missing_env = list(env_report.missing)
    missing_paths = [
        {"name": row["name"], "path": row["path"]}
        for row in required_path_rows
        if not row["exists"]
    ]
    return {
        "schema_version": ENV_REQUIREMENTS_SCHEMA_VERSION,
        "required_env": required_env_rows,
        "missing_env": missing_env,
        "required_paths": required_path_rows,
        "missing_paths": missing_paths,
        "required_extras": [str(item) for item in required_extras],
    }


def build_environment(
    *,
    repo_root: str | Path | None = None,
    cache_paths: Mapping[str, Any] | None = None,
    package_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build reproducibility metadata for Python, packages, git, and cache paths.

    Args:
        repo_root: Repository root used for git metadata.
        cache_paths: Cache directories or files relevant to the run.
        package_names: Optional distribution names to record; all installed distributions are recorded when omitted.
    """

    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": git_metadata(repo_root),
        "packages": _installed_packages(package_names),
        "cache_paths": redact_secrets(cache_paths or {}),
    }


def build_run_manifest(
    *,
    base_manifest: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    env_requirements: Mapping[str, Any] | None = None,
    seed: int | str | None = None,
    cache_paths: Mapping[str, Any] | None = None,
    environment_path: str | Path | None = None,
    env_requirements_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the canonical run manifest payload.

    Args:
        base_manifest: Existing runner manifest fields to preserve.
        config: Redacted run configuration snapshot.
        environment: Environment payload from ``build_environment``.
        env_requirements: Requirement payload from ``build_env_requirements``.
        seed: Run seed when the caller has one.
        cache_paths: Cache path mapping relevant to model, dataset, or runner behavior.
        environment_path: Optional path to the separately written ``environment.json``.
        env_requirements_path: Optional path to the separately written ``env_requirements.json``.
    """

    manifest = dict(redact_secrets(base_manifest))
    model = dict(manifest.get("model") or {})
    dataset = dict(manifest.get("dataset") or {})
    requirement_payload = dict(env_requirements or {})
    manifest["schema_version"] = RUN_MANIFEST_SCHEMA_VERSION
    manifest["config"] = redact_secrets(config or {})
    manifest["environment"] = dict(environment or {})
    manifest["env_requirements"] = requirement_payload
    manifest["seed"] = None if seed is None else seed
    manifest["cache_paths"] = redact_secrets(cache_paths or {})
    manifest["model_revision"] = _revision_from(model)
    manifest["dataset_revision"] = _revision_from(dataset)
    manifest["reproducibility"] = {
        "git_commit": dict(environment or {}).get("git", {}).get("commit") if isinstance(environment, Mapping) else None,
        "python_version": dict(environment or {}).get("python", {}).get("version") if isinstance(environment, Mapping) else None,
        "package_count": len(dict(dict(environment or {}).get("packages") or {})) if isinstance(environment, Mapping) else 0,
    }
    manifest["artifacts"] = dict(manifest.get("artifacts") or {})
    if environment_path is not None:
        manifest["artifacts"]["environment"] = str(Path(environment_path).resolve())
    if env_requirements_path is not None:
        manifest["artifacts"]["env_requirements"] = str(Path(env_requirements_path).resolve())
    manifest["manifest_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    return manifest


def write_run_manifest_artifacts(
    *,
    output_dir: str | Path,
    base_manifest: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    required_env: Sequence[str | Mapping[str, Any]] = (),
    required_paths: Sequence[str | Path | Mapping[str, Any]] = (),
    required_extras: Sequence[str] = (),
    seed: int | str | None = None,
    cache_paths: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    package_names: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    manifest_path: str | Path | None = None,
    environment_path: str | Path | None = None,
    env_requirements_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write ``run_manifest.json``, ``environment.json``, and ``env_requirements.json``.

    Args:
        output_dir: Run output directory.
        base_manifest: Existing runner manifest fields to enrich.
        config: Run configuration snapshot with secret-like values redacted.
        required_env: Required environment variable names or mappings.
        required_paths: Required local paths verified before execution.
        required_extras: Required optional dependency groups or runtime bundles.
        seed: Run seed when available.
        cache_paths: Cache path mapping relevant to reproducibility.
        repo_root: Repository root used for git metadata.
        package_names: Optional distribution names to record.
        environ: Environment mapping used for deterministic tests.
        manifest_path: Optional destination for ``run_manifest.json``.
        environment_path: Optional destination for ``environment.json``.
        env_requirements_path: Optional destination for ``env_requirements.json``.
    """

    root = Path(output_dir)
    resolved_manifest_path = Path(manifest_path or root / "run_manifest.json")
    resolved_environment_path = Path(environment_path or root / "environment.json")
    resolved_env_requirements_path = Path(env_requirements_path or root / "env_requirements.json")
    environment = build_environment(repo_root=repo_root, cache_paths=cache_paths, package_names=package_names)
    env_requirements = build_env_requirements(
        required_env=required_env,
        required_paths=required_paths,
        required_extras=required_extras,
        environ=environ,
    )
    manifest = build_run_manifest(
        base_manifest=base_manifest,
        config=config,
        environment=environment,
        env_requirements=env_requirements,
        seed=seed,
        cache_paths=cache_paths,
        environment_path=resolved_environment_path,
        env_requirements_path=resolved_env_requirements_path,
    )
    write_json(resolved_environment_path, environment)
    write_json(resolved_env_requirements_path, env_requirements)
    write_json(resolved_manifest_path, manifest)
    return {
        "run_manifest": resolved_manifest_path.resolve(),
        "environment": resolved_environment_path.resolve(),
        "env_requirements": resolved_env_requirements_path.resolve(),
    }


__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "ENV_REQUIREMENTS_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_env_requirements",
    "build_environment",
    "build_run_manifest",
    "redact_secrets",
    "write_run_manifest_artifacts",
]
