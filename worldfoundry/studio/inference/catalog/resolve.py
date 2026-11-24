"""Checkpoint, local-path, and interpreter resolution shared by catalog defaults.

Other catalog modules call these helpers through the module (``resolve.<name>``) so a
single ``monkeypatch.setattr(resolve, ...)`` affects every caller."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.core.io.paths import (
    checkpoint_root_candidates,
    checkpoint_root_path,
)
from worldfoundry.studio.inference.paths import studio_hfd_cache_roots


@dataclass
class _CatalogPathSnapshot:
    roots: tuple[Path, ...]
    children: dict[Path, frozenset[str]]
    exists_cache: dict[Path, bool] = field(default_factory=dict)

    def missing(self, path: Path) -> bool:
        for root, names in self.children.items():
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                continue
            if parts and ".." not in parts and parts[0] not in names:
                return True
        return False

    def exists(self, path: Path) -> bool:
        if path not in self.exists_cache:
            self.exists_cache[path] = not self.missing(path) and path.exists()
        return self.exists_cache[path]


_CATALOG_CACHE_ROOTS: ContextVar[_CatalogPathSnapshot | None] = ContextVar("studio_catalog_cache_roots", default=None)

PIPELINES_ROOT = Path(__file__).resolve().parents[3] / "pipelines"


@lru_cache(maxsize=1)
def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[5]


def _project_root_candidates() -> tuple[Path, ...]:
    current = _project_root().resolve()
    current_text = str(current)
    if current_text.startswith("/share/project/"):
        candidates: list[Path] = [Path("/bench-workspace") / current_text[len("/share/project/") :], current]
    else:
        candidates = [current]
        if current_text.startswith("/bench-workspace/"):
            candidates.append(Path("/share/project") / current_text[len("/bench-workspace/") :])
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_path(*parts: str | Path) -> Path:
    return _package_root().joinpath("data", *(Path(part) for part in parts))


@cache
def _runtime_profile_call_defaults(model_id: str) -> tuple[tuple[str, Any], ...]:
    """Generation defaults declared by ``data/models/runtime/profiles/<model_id>.yaml``.

    The runtime profile is the shared source of call defaults; Studio overrides
    only add Studio-specific values on top of it.
    """
    path = _data_path("models", "runtime", "profiles", f"{model_id}.yaml")
    if not path.is_file():
        return ()
    execution = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("execution") or {}
    defaults = execution.get("default_parameters") or execution.get("defaults") or {}
    return tuple(defaults.items()) if isinstance(defaults, dict) else ()


def _cache_candidates(*repo_dir_names: str) -> list[str]:
    candidates: list[str] = []
    snapshot = _CATALOG_CACHE_ROOTS.get()
    roots = snapshot.roots if snapshot is not None else studio_hfd_cache_roots(project_roots=_project_root_candidates())
    for root in roots:
        for repo_dir_name in repo_dir_names:
            repo_paths = [root / repo_dir_name]
            if not repo_dir_name.startswith("models--"):
                repo_paths.append(root / f"models--{repo_dir_name}")
            for repo_path in repo_paths:
                if snapshot is not None and snapshot.missing(repo_path):
                    candidates.append(str(repo_path))
                    continue
                refs_main = repo_path / "refs" / "main"
                snapshots_dir = repo_path / "snapshots"
                if refs_main.is_file():
                    revision = refs_main.read_text(encoding="utf-8").strip()
                    if revision:
                        candidates.append(str(snapshots_dir / revision))
                if snapshots_dir.is_dir():
                    candidates.extend(
                        str(path) for path in snapshots_dir.iterdir() if path.is_dir() or path.is_symlink()
                    )
                candidates.append(str(repo_path))
    return candidates


def _coerce_aliases(*values: str) -> tuple[str, ...]:
    deduped: list[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return tuple(deduped)


def _resolve_override_value(value: Any) -> Any:
    return value() if callable(value) else value


def _prefer_existing_model_ref(*candidates: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _checkpoint_model_ref(*repo_dir_names: str, fallback: str = "") -> str:
    snapshot = _CATALOG_CACHE_ROOTS.get()
    candidates: list[str] = []
    for repo_dir_name in repo_dir_names:
        if not repo_dir_name:
            continue
        candidates.extend(str(path) for path in checkpoint_root_candidates(*Path(repo_dir_name).parts))
        candidates.extend(_cache_candidates(repo_dir_name))
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            exists = snapshot.exists(path) if snapshot is not None else path.exists()
            if exists:
                return str(path)
    if fallback:
        return fallback
    return next((candidate for candidate in candidates if candidate), "")


def _local_root_with_files(
    candidates: Iterable[str | Path],
    required_files: Sequence[str | Path],
) -> str:
    """Return the first local component root containing every required file."""

    for raw_candidate in candidates:
        if not raw_candidate:
            continue
        candidate = Path(raw_candidate).expanduser()
        if candidate.is_dir() and all((candidate / Path(name)).is_file() for name in required_files):
            return str(candidate.resolve())
    return ""


def _hf_checkpoint_model_ref(repo_dir_name: str, repo_id: str) -> str:
    """Return a local HF cache path only when it exists, otherwise use the repo id."""
    candidates = [
        *(str(path) for path in checkpoint_root_candidates(*Path(repo_dir_name).parts)),
        *_cache_candidates(repo_dir_name),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    return repo_id


def _hf_checkpoint_model_ref_at_revision(repo_dir_name: str, repo_id: str, revision: str) -> str:
    """Return a local checkpoint only when its immutable revision is verifiable."""

    candidates = [
        *(str(path) for path in checkpoint_root_candidates(f"{repo_dir_name}-{revision[:8]}")),
        *(str(path) for path in checkpoint_root_candidates(*Path(repo_dir_name).parts)),
        *_cache_candidates(repo_dir_name, repo_id.replace("/", "--")),
    ]
    for raw_candidate in candidates:
        candidate = Path(raw_candidate).expanduser()
        if not candidate.exists():
            continue
        if candidate.parent.name == "snapshots" and candidate.name == revision:
            return str(candidate)
        metadata_path = candidate / ".hfd" / "repo_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("sha") or "").strip() == revision:
            return str(candidate)
    return repo_id


def _current_python_executable() -> str:
    return str(Path(sys.executable).expanduser())


def _worldfoundry_unified_python_executable() -> str:
    override = os.getenv("WORLDFOUNDRY_UNIFIED_PYTHON_EXECUTABLE", "").strip()
    if override:
        return override
    candidate = _project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    opt_candidate = Path("/opt/conda/bin/python3.11")
    return str(opt_candidate) if opt_candidate.is_file() else _current_python_executable()


def _first_existing_path(*candidates: str | Path) -> str:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return str(Path(candidates[0]).expanduser()) if candidates else ""


def _checkpoint_file_if_present(*relative_paths: str) -> str | None:
    for relative_path in relative_paths:
        path = checkpoint_root_path(*Path(relative_path).parts)
        if path.is_file():
            return str(path)
    return None


def _joyai_echo_python_executable(environment_name: str) -> str:
    workspace_root = _project_root().parent
    return _first_existing_path(
        workspace_root / "envs" / environment_name / "bin" / "python",
        workspace_root / "conda" / "envs" / environment_name / "bin" / "python",
    )
