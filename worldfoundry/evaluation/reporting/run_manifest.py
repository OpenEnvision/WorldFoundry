"""Build and write run manifests and environment snapshots.

A *run manifest* captures the full reproducibility context for an evaluation
run — configuration, environment, and requirements — so that results can be
independently verified.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from collections.abc import Sequence
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import git_metadata, jsonable, package_version, stable_hash, write_json
from worldfoundry.runtime.env import check_required_env

# ── Schema version identifiers ──────────────────────────────
RUN_MANIFEST_SCHEMA_VERSION = "worldfoundry-run-manifest"
ENVIRONMENT_SCHEMA_VERSION = "worldfoundry-environment"
ENV_REQUIREMENTS_SCHEMA_VERSION = "worldfoundry-env-requirements"
REPRODUCIBILITY_PACKAGES = ("worldfoundry", "numpy", "pandas", "torch")

# ── Secret redaction ─────────────────────────────────────────
# Single word segments that mark a key as sensitive when they appear as a
# whole segment (word-boundary match). Deliberately excludes "tokens"
# (max_new_tokens, num_tokens) and "tokenizer" — those are reproducibility
# parameters the manifest must preserve.
_SENSITIVE_KEY_SEGMENTS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "passwd",
        "password",
        "pwd",
        "secret",
        "secrets",
        "token",
    }
)
# Adjacent segment pairs that are sensitive even though the individual
# segments (e.g. "key") are too generic to match on their own.
_SENSITIVE_KEY_SEGMENT_PAIRS = frozenset(
    {
        ("access", "key"),
        ("api", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("session", "key"),
        ("ssh", "key"),
    }
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEGMENT_SEPARATOR = re.compile(r"[^a-z0-9]+")
_ARG_VALUE_PATTERN = re.compile(
    r"(?P<flag>--[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"(?P<separator>=|[ \t]+)"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s]+)"
)
_KEY_VALUE_PATTERN = re.compile(
    r"(?P<prefix>(?P<key_quote>[\"']?)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"
    r"(?P=key_quote)[ \t]*(?:=|:)[ \t]*)"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,}\]]+)"
)
_URL_USERINFO_PATTERN = re.compile(
    r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)(\bbearer\s+)([^\s,;]+)")
_KNOWN_TOKEN_PATTERN = re.compile(r"\b(?:sk|hf)[_-][A-Za-z0-9_-]{8,}\b")


def is_sensitive_key(key: str) -> bool:
    """Return whether *key* looks like a secret field that should be redacted.

    Matches on whole word segments (split on ``_``/``-``/camelCase) instead of
    raw substrings, so ``max_new_tokens`` and ``tokenizer`` stay visible while
    ``api_key``, ``hf_token``, and ``authToken`` are still redacted.
    """
    segments = tuple(
        segment
        for segment in _KEY_SEGMENT_SEPARATOR.split(_CAMEL_BOUNDARY.sub("_", key).lower())
        if segment
    )
    if any(segment in _SENSITIVE_KEY_SEGMENTS for segment in segments):
        return True
    return any(pair in _SENSITIVE_KEY_SEGMENT_PAIRS for pair in zip(segments, segments[1:]))


def _redacted_literal(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return f"{value[0]}<redacted>{value[-1]}"
    return "<redacted>"


def redact_secret_text(value: str, environ: Mapping[str, str] | None = None) -> str:
    """Redact credentials embedded in log, command, JSON, or URL text.

    ``environ`` is also inspected so a secret printed without its key can be
    removed. Key classification uses :func:`is_sensitive_key`; similarly
    named reproducibility parameters such as ``MAX_NEW_TOKENS`` and
    ``TOKENIZERS_PARALLELISM`` remain intact.
    """

    redacted = str(value)
    env = os.environ if environ is None else environ
    secrets = {
        str(secret)
        for name, secret in env.items()
        if secret not in (None, "") and is_sensitive_key(str(name))
    }
    for secret in sorted(secrets, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")

    redacted = _URL_USERINFO_PATTERN.sub(r"\g<scheme><redacted>@", redacted)
    redacted = _BEARER_TOKEN_PATTERN.sub(r"\1<redacted>", redacted)
    redacted = _KNOWN_TOKEN_PATTERN.sub("<redacted>", redacted)

    def redact_arg(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group("flag").lstrip("-")):
            return match.group(0)
        return f"{match.group('flag')}{match.group('separator')}{_redacted_literal(match.group('value'))}"

    redacted = _ARG_VALUE_PATTERN.sub(redact_arg, redacted)

    def redact_assignment(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group("key")):
            return match.group(0)
        return f"{match.group('prefix')}{_redacted_literal(match.group('value'))}"

    return _KEY_VALUE_PATTERN.sub(redact_assignment, redacted)


def _redact_sequence(value: Sequence[Any]) -> list[Any]:
    redacted: list[Any] = []
    index = 0
    while index < len(value):
        item = value[index]
        if isinstance(item, str) and item.startswith("--") and "=" not in item:
            if is_sensitive_key(item.lstrip("-")) and index + 1 < len(value):
                redacted.extend((item, "<redacted>"))
                index += 2
                continue
        redacted.append(redact_secrets(item))
        index += 1
    return redacted


def redact_secrets(value: Any) -> Any:
    """Return a JSON payload with secret-like mapping values removed.

    Args:
        value: Configuration payload that may contain token, API key, password, or secret fields.
    """

    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if is_sensitive_key(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_sequence(value))
    if isinstance(value, list):
        return _redact_sequence(value)
    payload = jsonable(value)
    if isinstance(payload, str):
        return redact_secret_text(payload, {})
    if payload is not value:
        return redact_secrets(payload)
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


@lru_cache(maxsize=64)
def _installed_packages_filtered(package_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    packages: dict[str, str] = {}
    for requested_name in package_names:
        try:
            distribution = metadata.distribution(requested_name)
        except metadata.PackageNotFoundError:
            continue
        canonical_name = str(distribution.metadata.get("Name") or requested_name)
        packages[canonical_name] = str(distribution.version)
    if not any(name.lower().replace("_", "-") == "worldfoundry" for name in packages):
        packages["worldfoundry"] = package_version()
    return tuple(sorted(packages.items(), key=lambda item: item[0].lower()))


def _installed_packages(package_names: Sequence[str] | None) -> dict[str, str]:
    """Collect installed package versions, optionally filtered by *package_names*."""
    if package_names:
        requested = tuple(sorted({str(item).lower().replace("_", "-") for item in package_names}))
        return dict(_installed_packages_filtered(requested))

    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get("Name") or "")
        if not name:
            continue
        packages[name] = str(distribution.version)
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _torch_environment() -> dict[str, Any]:
    """Collect optional Torch accelerator metadata without making it mandatory."""

    torch_payload: dict[str, Any] = {"version": None}
    cuda_payload: dict[str, Any] = {
        "build_version": None,
        "available": None,
        "device_count": None,
        "cudnn_version": None,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001 - optional runtime probing must not fail a run.
        return {"torch": torch_payload, "cuda": cuda_payload}

    torch_payload["version"] = str(getattr(torch, "__version__", "") or "") or None
    torch_cuda = getattr(torch, "cuda", None)
    torch_version = getattr(torch, "version", None)
    cuda_payload["build_version"] = getattr(torch_version, "cuda", None)
    if torch_cuda is not None:
        try:
            cuda_payload["available"] = bool(torch_cuda.is_available())
            cuda_payload["device_count"] = int(torch_cuda.device_count())
        except Exception:  # noqa: BLE001 - CUDA probing is optional metadata.
            pass
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        try:
            cuda_payload["cudnn_version"] = cudnn.version()
        except Exception:  # noqa: BLE001 - cuDNN probing is optional metadata.
            pass
    return {"torch": torch_payload, "cuda": cuda_payload}


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

    accelerator_environment = _torch_environment()
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
        **accelerator_environment,
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
    environment: Mapping[str, Any] | None = None,
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
        environment: Optional precomputed environment payload to reuse across related artifacts.
    """

    root = Path(output_dir)
    resolved_manifest_path = Path(manifest_path or root / "run_manifest.json")
    resolved_environment_path = Path(environment_path or root / "environment.json")
    resolved_env_requirements_path = Path(env_requirements_path or root / "env_requirements.json")
    environment_payload = (
        dict(environment)
        if environment is not None
        else build_environment(repo_root=repo_root, cache_paths=cache_paths, package_names=package_names)
    )
    env_requirements = build_env_requirements(
        required_env=required_env,
        required_paths=required_paths,
        required_extras=required_extras,
        environ=environ,
    )
    manifest = build_run_manifest(
        base_manifest=base_manifest,
        config=config,
        environment=environment_payload,
        env_requirements=env_requirements,
        seed=seed,
        cache_paths=cache_paths,
        environment_path=resolved_environment_path,
        env_requirements_path=resolved_env_requirements_path,
    )
    write_json(resolved_environment_path, environment_payload)
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
    "REPRODUCIBILITY_PACKAGES",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_env_requirements",
    "build_environment",
    "build_run_manifest",
    "is_sensitive_key",
    "redact_secret_text",
    "redact_secrets",
    "write_run_manifest_artifacts",
]
