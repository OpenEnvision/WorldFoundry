from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from worldfoundry.core.io.paths import resolve_worldfoundry_path

from . import normalize as _normalize


def resolve_local_path(
    value: Path | str,
    *,
    label: str = "OpenPI asset",
    require_directory: bool | None = None,
) -> Path:
    """Resolve an OpenPI inference asset without contacting a remote store.

    Checkpoint acquisition is deliberately outside the model runtime.  Accepting
    a URI here would make a typo or an upstream remote default silently turn model
    construction into a network operation, so every inference path shares
    this strict local resolver.
    """

    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} path must not be empty")
    parsed = urlparse(text)
    if parsed.scheme:
        raise ValueError(f"{label} must be a staged local path, not a URI: {text!r}")
    path = resolve_worldfoundry_path(text).resolve()
    if require_directory is True and not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if require_directory is False and not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if require_directory is None and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_norm_stats(assets_dir: Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    root = resolve_local_path(assets_dir, label="OpenPI assets", require_directory=True)
    norm_stats_dir = resolve_local_path(
        root / asset_id,
        label="OpenPI normalization statistics",
        require_directory=True,
    )
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info("Loaded norm stats from %s", norm_stats_dir)
    return norm_stats


__all__ = ["load_norm_stats", "resolve_local_path"]
