"""Shared filesystem resolution for WorldFoundry Studio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from worldfoundry.core.io.paths import project_root
from worldfoundry.runtime.env import (
    first_env_path,
    resolve_cache_dir,
    resolve_ckpt_dir,
    resolve_data_dir,
    resolve_hf_cache_dir,
    resolve_hfd_root,
    resolve_model_dir,
)

EnvMapping = Mapping[str, str]


def studio_repo_root() -> Path:
    """Return the repository root used by in-tree Studio defaults.

    Args:
        None.
    """

    return project_root(__file__)


def _dedupe_paths(paths: Sequence[Path | str | None]) -> tuple[Path, ...]:
    """Normalize path candidates while preserving priority order.

    Args:
        paths: Ordered path candidates, including optional empty values.
    """

    deduped: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        key = path.resolve().as_posix() if path.exists() else path.as_posix()
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return tuple(deduped)


def _mirror_project_roots(project_roots: Sequence[Path] | None = None) -> tuple[Path, ...]:
    """Return project root candidates including the bench workspace mirror.

    Args:
        project_roots: Optional explicit roots supplied by callers that already know their repo root.
    """

    roots = list(project_roots or (studio_repo_root(),))
    mirrored: list[Path] = []
    for root in roots:
        root = Path(root).expanduser()
        root_text = root.as_posix()
        mirrored.append(root)
        if root_text.startswith("/share/project/"):
            mirrored.append(Path("/bench-workspace") / root_text[len("/share/project/") :])
        elif root_text.startswith("/bench-workspace/"):
            mirrored.append(Path("/share/project") / root_text[len("/bench-workspace/") :])
    return _dedupe_paths(mirrored)


def studio_workspace_root(env: EnvMapping | None = None) -> Path:
    """Resolve the directory where Studio writes run artifacts.

    Args:
        env: Optional environment mapping; defaults to the process environment.
    """

    environ = os.environ if env is None else env
    explicit = first_env_path(
        (
            "WORLDFOUNDRY_STUDIO_WORKSPACE_DIR",
        ),
        environ,
    )
    if explicit is not None:
        return explicit
    artifact_dir = first_env_path(
        ("WORLDFOUNDRY_ARTIFACT_DIR", "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
        environ,
    )
    if artifact_dir is not None:
        return artifact_dir / "studio"
    return studio_repo_root() / "tmp" / "worldfoundry_studio"


def studio_model_roots(env: EnvMapping | None = None, *, repo_root: Path | None = None) -> tuple[Path, ...]:
    """Return candidate roots for local model repos and checkpoints.

    Args:
        env: Optional environment mapping; defaults to the process environment.
        repo_root: Optional repository root for in-tree fallbacks.
    """

    environ = os.environ if env is None else env
    explicit_root = first_env_path(
        (
            "WORLDFOUNDRY_STUDIO_MODEL_ROOT",
        ),
        environ,
    )
    model_dir = resolve_model_dir(environ)
    cache_dir = resolve_cache_dir(environ)
    root = repo_root or studio_repo_root()
    return _dedupe_paths(
        (
            explicit_root,
            model_dir,
            model_dir / "checkpoints",
            model_dir / "repos",
            cache_dir / "repos",
            root / "thirdparty",
        )
    )


def studio_hfd_cache_roots(
    env: EnvMapping | None = None,
    *,
    project_roots: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    """Return Hugging Face download roots checked by Studio catalog defaults.

    Args:
        env: Optional environment mapping; defaults to the process environment.
        project_roots: Optional project roots used for in-tree cache fallbacks.
    """

    environ = os.environ if env is None else env
    explicit_cache_dir = first_env_path(("WORLDFOUNDRY_CACHE_DIR",), environ)
    explicit_hf_cache_dir = first_env_path(
        (
            "WORLDFOUNDRY_HF_CACHE_DIR",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ),
        environ,
    )
    cache_dir = resolve_cache_dir(environ)
    ckpt_dir = resolve_ckpt_dir(environ)
    hfd_root = resolve_hfd_root(environ)
    hf_cache_dir = resolve_hf_cache_dir(environ)
    project_cache_roots = [root / "cache" / "hfd" for root in _mirror_project_roots(project_roots)]
    project_ckpt_roots = [
        root.parent / directory
        for root in _mirror_project_roots(project_roots)
        for directory in ("ckpt", "ckpts")
    ]
    return _dedupe_paths(
        (
            explicit_hf_cache_dir,
            explicit_cache_dir / "hfd" if explicit_cache_dir is not None else None,
            hfd_root,
            ckpt_dir,
            ckpt_dir / "hfd_models",
            ckpt_dir / "hfd",
            *project_ckpt_roots,
            *(root / "hfd_models" for root in project_ckpt_roots),
            *(root / "hfd" for root in project_ckpt_roots),
            *(root / ".hf_home" / "hub" for root in project_ckpt_roots),
            *project_cache_roots,
            hf_cache_dir,
            cache_dir / "hfd",
        )
    )


def studio_path_summary(env: EnvMapping | None = None) -> dict[str, str | list[str]]:
    """Build a startup-safe summary of Studio filesystem roots.

    Args:
        env: Optional environment mapping; defaults to the process environment.
    """

    environ = os.environ if env is None else env
    return {
        "workspace_root": str(studio_workspace_root(environ)),
        "data_dir": str(resolve_data_dir(environ)),
        "model_dir": str(resolve_model_dir(environ)),
        "cache_dir": str(resolve_cache_dir(environ)),
        "model_roots": [str(path) for path in studio_model_roots(environ)],
        "hfd_cache_roots": [str(path) for path in studio_hfd_cache_roots(environ)],
    }
