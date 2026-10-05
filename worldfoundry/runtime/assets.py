"""Local asset discovery and manifest resolution for WorldFoundry benchmarks.

This module loads YAML manifests describing locally staged benchmark and model
assets, resolves ``$WORLDFOUNDRY_*`` path tokens inside manifest entries, and
produces :class:`LocalAsset` instances that report whether each asset path
exists on disk.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.manifests import load_manifest
from worldfoundry.core.io.paths import (
    hfd_dataset_root_path,
    package_data_root,
    project_root,
    resolve_worldfoundry_path,
)
from worldfoundry.core.io.paths import (
    worldfoundry_path_tokens as core_worldfoundry_path_tokens,
)

from .env import (
    EnvMapping,
    benchmark_repo_cache_root,
    resolve_artifact_dir,
    resolve_cache_dir,
    resolve_ckpt_dir,
    resolve_data_dir,
    resolve_hfd_root,
    resolve_model_dir,
)

# ── Constants ────────────────────────────────────────────────────────────────

LOCAL_ASSET_MANIFEST_ENV = "WORLDFOUNDRY_LOCAL_ASSET_MANIFEST"

# Same values evaluation.utils derives; resolved from core so the runtime
# layer never imports worldfoundry.evaluation (SA-10).
REPO_ROOT = project_root()
BENCHMARKS_DATA_ROOT = package_data_root() / "benchmarks"


def _has_huggingface_incomplete_download(path: Path) -> bool:
    """Return whether a local-dir HF cache records an unfinished download."""

    download_root = path / ".cache" / "huggingface" / "download"
    if not download_root.is_dir():
        return False
    return next(download_root.rglob("*.incomplete"), None) is not None


def local_file_is_ready(path: str | Path) -> bool:
    """Check for nonempty files without an unfinished-download sidecar.

    Downloaders can preallocate the complete file size before writing its
    contents. In particular, an aria2 control file means that transfer has
    not completed, even when the target has its expected byte length.
    This is a readiness check, not a checksum verification.
    """
    path = Path(path)
    try:
        return (
            path.is_file()
            and path.stat().st_size > 0
            and not any(Path(str(path) + suffix).exists() for suffix in (".aria2", ".incomplete", ".part"))
        )
    except OSError:
        return False


def _path_is_ready(path: Path | None) -> bool:
    """Return whether a resolved asset path looks usable.

    Plain existence is not enough for file assets: interrupted downloads and
    copies commonly leave zero-byte files behind, which then fail much later
    inside model loading instead of at asset resolution time.
    """
    if path is None:
        return False
    try:
        if path.is_file():
            return local_file_is_ready(path)
        if not path.is_dir():
            return False
        # ``huggingface_hub.snapshot_download(local_dir=...)`` creates this
        # cache layout.  The visible target directory can already contain
        # valid files while an interrupted transfer remains underneath it;
        # treating that directory as ready would expose a partial snapshot.
        return not _has_huggingface_incomplete_download(path)
    except OSError:
        return False


@dataclass(frozen=True)
class LocalAsset:
    """Describe one locally staged benchmark or model asset.

    Args:
        benchmark_id: Optional benchmark or integration id that owns the asset.
        asset_id: Stable asset id inside the benchmark group.
        kind: Asset kind, such as dataset, checkpoint, repo, manifest, or artifact.
        path: Resolved local path recorded by the manifest.
        canonical_path: Preferred path under the WorldFoundry root layout.
        status: Current path status computed at load time.
        ready: Whether the resolved path exists locally.
        metadata: Extra manifest fields preserved for consumers.
    """

    benchmark_id: str | None
    asset_id: str
    kind: str
    path: Path | None
    canonical_path: Path | None
    status: str
    ready: bool
    metadata: Mapping[str, Any]

    @classmethod
    def from_manifest_item(
        cls,
        item: Mapping[str, Any],
        *,
        benchmark_id: str | None = None,
        env: EnvMapping | None = None,
    ) -> "LocalAsset":
        """Build a local asset view from a manifest item.

        Args:
            item: Manifest asset mapping.
            benchmark_id: Optional parent benchmark id.
            env: Optional environment mapping used for path token expansion.
        """

        raw_env = item.get("env")
        if isinstance(raw_env, str) and raw_env.strip() and os.environ.get(raw_env.strip()):
            raw_path = os.environ[raw_env.strip()]
        else:
            raw_path = item.get("path") or item.get("local_path")
        raw_canonical_path = item.get("canonical_path")
        path = expand_worldfoundry_path(raw_path, env) if raw_path else None
        canonical_path = expand_worldfoundry_path(raw_canonical_path, env) if raw_canonical_path else None
        ready = _path_is_ready(path)
        status = "available" if ready else "missing"
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "asset_id", "kind", "path", "local_path", "canonical_path", "status"}
        }
        return cls(
            benchmark_id=benchmark_id,
            asset_id=str(item.get("id") or item.get("asset_id") or "asset"),
            kind=str(item.get("kind") or "asset"),
            path=path,
            canonical_path=canonical_path,
            status=status,
            ready=ready,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved asset status for logs and diagnostics.

        Args:
            None.
        """

        payload: dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "id": self.asset_id,
            "kind": self.kind,
            "path": str(self.path) if self.path is not None else None,
            "canonical_path": str(self.canonical_path) if self.canonical_path is not None else None,
            "status": self.status,
            "ready": self.ready,
        }
        payload.update(self.metadata)
        return payload


def _repo_root() -> Path:
    """Resolve the WorldFoundry repository root from the installed source tree.

    Args:
        None.
    """

    return REPO_ROOT


def worldfoundry_path_tokens(env: EnvMapping | None = None) -> dict[str, str]:
    """Return manifest path tokens backed by runtime environment helpers.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    repo_root = _repo_root()
    tokens = core_worldfoundry_path_tokens(environ)
    tokens.update({
        "WORLDFOUNDRY_BENCH_ROOT": str(Path(environ.get("WORLDFOUNDRY_BENCH_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_REPO_ROOT": str(Path(environ.get("WORLDFOUNDRY_REPO_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_CACHE_DIR": str(resolve_cache_dir(environ)),
        "WORLDFOUNDRY_DATA_DIR": str(resolve_data_dir(environ)),
        "WORLDFOUNDRY_MODEL_DIR": str(resolve_model_dir(environ)),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(resolve_artifact_dir(environ)),
        "WORLDFOUNDRY_CKPT_DIR": str(resolve_ckpt_dir(environ)),
        "WORLDFOUNDRY_HFD_ROOT": str(resolve_hfd_root(environ)),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(hfd_dataset_root_path(env=environ)),
        "WORLDFOUNDRY_BENCHMARK_REPO_ROOT": str(benchmark_repo_cache_root(environ)),
    })
    return tokens


def expand_worldfoundry_path(value: str | Path, env: EnvMapping | None = None) -> Path:
    """Expand a manifest path containing WorldFoundry environment tokens.

    Args:
        value: Path string or ``Path`` with optional ``$VAR`` or ``${VAR}`` tokens.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    path = resolve_worldfoundry_path(value, worldfoundry_path_tokens(env))
    if path.is_absolute():
        return path
    return _repo_root() / path


def default_asset_manifest_candidates(env: EnvMapping | None = None) -> tuple[Path, ...]:
    """Return local asset manifest candidates in preferred lookup order.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    repo_root = _repo_root()
    explicit = environ.get(LOCAL_ASSET_MANIFEST_ENV)
    candidates = [
        resolve_cache_dir(environ) / "manifests" / "local_assets_manifest.yaml",
        repo_root / "tmp" / "benchmark_zoo" / "local_assets_manifest.yaml",
        BENCHMARKS_DATA_ROOT / "local_assets_manifest.yaml",
        BENCHMARKS_DATA_ROOT / "local_assets.example.yaml",
    ]
    if explicit:
        candidates.insert(0, Path(explicit).expanduser())
    return tuple(candidates)


def resolve_asset_manifest_path(path: str | Path | None = None, env: EnvMapping | None = None) -> Path:
    """Resolve the local asset manifest path without touching benchmark runtimes.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    if path is not None:
        return Path(path).expanduser()
    environ = os.environ if env is None else env
    explicit = environ.get(LOCAL_ASSET_MANIFEST_ENV)
    if explicit:
        return Path(explicit).expanduser()
    candidates = default_asset_manifest_candidates(env)
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    return existing or candidates[0]


def load_local_asset_manifest(path: str | Path | None = None, env: EnvMapping | None = None) -> dict[str, Any]:
    """Read the local benchmark/model asset manifest as YAML.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping used for default path resolution.
    """

    manifest_path = resolve_asset_manifest_path(path, env)
    payload = load_manifest(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("local asset manifest must be a YAML mapping")
    return payload


def iter_manifest_asset_items(manifest: Mapping[str, Any]) -> Iterator[tuple[str | None, Mapping[str, Any]]]:
    """Yield benchmark id and asset item pairs from a local asset manifest.

    Args:
        manifest: Loaded local asset manifest mapping.
    """

    root_assets = manifest.get("assets")
    if isinstance(root_assets, list):
        for item in root_assets:
            if isinstance(item, Mapping):
                yield None, item
    benchmarks = manifest.get("benchmarks")
    if isinstance(benchmarks, list):
        for benchmark in benchmarks:
            if not isinstance(benchmark, Mapping):
                continue
            benchmark_id = benchmark.get("id") or benchmark.get("benchmark_id")
            assets = benchmark.get("assets")
            if not isinstance(assets, list):
                continue
            for item in assets:
                if isinstance(item, Mapping):
                    yield str(benchmark_id) if benchmark_id is not None else None, item


def load_local_assets(path: str | Path | None = None, env: EnvMapping | None = None) -> tuple[LocalAsset, ...]:
    """Load local asset entries with resolved path status.

    Args:
        path: Optional explicit manifest path.
        env: Optional environment mapping used for path expansion.
    """

    manifest = load_local_asset_manifest(path, env)
    return tuple(
        LocalAsset.from_manifest_item(item, benchmark_id=benchmark_id, env=env)
        for benchmark_id, item in iter_manifest_asset_items(manifest)
    )
