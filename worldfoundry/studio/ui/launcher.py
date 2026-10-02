"""Launcher for standalone WorldFoundry Studio browser frontends."""

from __future__ import annotations

import os
from typing import Sequence

from worldfoundry.studio.inference.catalog import CatalogEntry, discover_catalog, find_entry
from worldfoundry.studio.inference.variants import resolve_cli_variant_id as _resolve_shared_cli_variant_id
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.ui.launch_config import parse_launch_config as _parse_launch_config_core
from worldfoundry.studio.visualization.backends.frontends import (
    NATIVE_FRONTENDS,
    resolve_frontend_mode,
    serve_native_frontend,
)


def _studio_catalog(entries: Sequence[CatalogEntry] | None = None) -> tuple[CatalogEntry, ...]:
    return tuple(entries) if entries is not None else tuple(discover_catalog())


def parse_launch_config(argv: Sequence[str] | None = None) -> StudioLaunchConfig:
    return _parse_launch_config_core(
        argv,
        studio_catalog=_studio_catalog,
        resolve_cli_variant_id=_resolve_shared_cli_variant_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Launch a standalone Studio browser frontend."""

    os.environ.setdefault("WORLDFOUNDRY_STUDIO_SKIP_RUNTIME_PROFILES", "1")
    launch_config = parse_launch_config(argv)
    entry = find_entry(launch_config.model_id)
    frontend_mode = resolve_frontend_mode(entry, launch_config.frontend, launch_config.asset_path or None)
    if frontend_mode not in NATIVE_FRONTENDS:
        raise SystemExit(f"Unsupported standalone Studio browser frontend: {frontend_mode}")
    serve_native_frontend(entry, launch_config, frontend_mode)


if __name__ == "__main__":
    main()
