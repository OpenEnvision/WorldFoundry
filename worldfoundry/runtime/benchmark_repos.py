"""Canonical benchmark upstream repository path resolution.

Resolution order for a benchmark checkout:

1. ``WORLDFOUNDRY_<BENCH>_ROOT`` environment override
2. ``WORLDFOUNDRY_LOCAL_ASSET_MANIFEST`` repo entry for the benchmark id
3. ``${WORLDFOUNDRY_CACHE_DIR}/repos/<github-slug>`` (canonical default)
4. Legacy ``tmp/benchmark_zoo/repos/<slug>`` and known ``thirdparty/*`` layouts
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import project_root

from .assets import load_local_assets
from .env import EnvMapping, benchmark_repo_cache_root

# Resolved from core so the runtime layer never imports
# worldfoundry.evaluation (SA-10).
REPO_ROOT = project_root()

_GITHUB_SLUG_RE = re.compile(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)")

# Extra legacy roots keyed by cache slug (beyond tmp/benchmark_zoo/repos/<slug>).
_LEGACY_EXTRA_REPO_ROOTS: dict[str, tuple[Path, ...]] = {
    "github.com_haoyi-duan_WorldScore": (REPO_ROOT / "thirdparty" / "WorldScore",),
}

# Known benchmark ids -> repo cache slug for env-only resolution.
_BENCHMARK_REPO_SLUGS: dict[str, str] = {
    "vbench": "github.com_Vchitect_VBench",
    "worldscore": "github.com_haoyi-duan_WorldScore",
    "camerabench": "github.com_sy77777en_CameraBench",
    "t2v-compbench": "github.com_linzhiqiu_t2v_metrics",
}


def github_repo_cache_slug(repo_url: str) -> str:
    """Convert a GitHub URL to the canonical cache directory slug."""

    match = _GITHUB_SLUG_RE.search(repo_url.strip())
    if not match:
        raise ValueError(f"unsupported GitHub repo url for cache slug: {repo_url}")
    owner = match.group("owner")
    repo = match.group("repo").removesuffix(".git")
    return f"github.com_{owner}_{repo}"


def benchmark_root_env_name(benchmark_id: str) -> str:
    """Return ``WORLDFOUNDRY_<BENCH>_ROOT`` for a benchmark id."""

    token = benchmark_id.upper().replace("-", "_")
    return f"WORLDFOUNDRY_{token}_ROOT"


def canonical_benchmark_repo_path(
    *,
    slug: str,
    env: EnvMapping | None = None,
) -> Path:
    """Return the canonical repo checkout path under the shared cache layout."""

    return benchmark_repo_cache_root(env) / slug


def legacy_benchmark_repo_paths(slug: str) -> tuple[Path, ...]:
    """Return deprecated repo checkout locations kept for local migration."""

    legacy = (
        REPO_ROOT / "tmp" / "benchmark_zoo" / "repos" / slug,
        *_LEGACY_EXTRA_REPO_ROOTS.get(slug, ()),
    )
    return legacy


def _manifest_repo_path(benchmark_id: str, env: EnvMapping | None) -> Path | None:
    try:
        assets = load_local_assets(env=env)
    except (OSError, ValueError):
        return None
    for asset in assets:
        if asset.benchmark_id == benchmark_id and asset.kind == "repo" and asset.path is not None:
            return asset.path
    return None


def resolve_benchmark_repo_root(
    *,
    benchmark_id: str,
    repo_url: str | None = None,
    slug: str | None = None,
    env: EnvMapping | None = None,
    prefer_existing: bool = True,
) -> Path:
    """Resolve a benchmark upstream repo root using the shared precedence rules."""

    environ = os.environ if env is None else env
    env_name = benchmark_root_env_name(benchmark_id)
    explicit = environ.get(env_name)
    if explicit:
        return Path(explicit).expanduser()

    manifest_path = _manifest_repo_path(benchmark_id, env)
    if manifest_path is not None:
        return manifest_path

    cache_slug = slug or _BENCHMARK_REPO_SLUGS.get(benchmark_id)
    if cache_slug is None and repo_url:
        cache_slug = github_repo_cache_slug(repo_url)
    if cache_slug is None:
        raise ValueError(
            f"cannot resolve benchmark repo root for {benchmark_id!r}; "
            f"set {env_name} or provide repo_url/slug"
        )

    candidates: list[Path] = [canonical_benchmark_repo_path(slug=cache_slug, env=env)]
    if prefer_existing:
        candidates.extend(legacy_benchmark_repo_paths(cache_slug))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return candidates[0]


def resolve_benchmark_repo_root_from_profile_path(
    raw_path: str,
    *,
    profile: dict[str, Any] | None = None,
    env: EnvMapping | None = None,
) -> Path | None:
    """Infer a repo root env override from a runtime-profile path entry."""

    del profile  # reserved for future profile-scoped slug hints
    normalized = raw_path.replace("tmp/benchmark_zoo/repos/", "${WORLDFOUNDRY_CACHE_DIR}/repos/")
    slug = Path(normalized).name
    if not slug.startswith("github.com_"):
        return None
    for benchmark_id, known_slug in _BENCHMARK_REPO_SLUGS.items():
        if known_slug == slug:
            return resolve_benchmark_repo_root(benchmark_id=benchmark_id, slug=slug, env=env)
    return canonical_benchmark_repo_path(slug=slug, env=env)


__all__ = [
    "benchmark_root_env_name",
    "canonical_benchmark_repo_path",
    "github_repo_cache_slug",
    "legacy_benchmark_repo_paths",
    "resolve_benchmark_repo_root",
    "resolve_benchmark_repo_root_from_profile_path",
]
