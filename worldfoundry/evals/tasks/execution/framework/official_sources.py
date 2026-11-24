# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Official benchmark sources pinned by catalog runner runtimes.

A benchmark whose ``runner.runtime`` declares ``repo_url``, ``repo_revision``
and optionally ``repo_subdir`` executes that upstream tree from the
WorldFoundry source cache rather than from a vendored copy. The optional
``runtime_root_env`` names an environment variable that points at the
``repo_subdir`` directory of an existing local checkout of the same revision.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import env_path
from .source_cache import SourceSpec, SourceUnavailableError, ensure_source


def benchmark_source_spec(benchmark_id: str, runtime: Mapping[str, Any]) -> SourceSpec:
    """Build the pinned upstream source declared by a catalog ``runner.runtime`` block.

    Raises:
        ValueError: The runtime does not pin ``repo_url`` and ``repo_revision``.
    """
    url = runtime.get("repo_url")
    revision = runtime.get("repo_revision")
    if not url or not revision:
        raise ValueError(f"{benchmark_id}: runner.runtime must declare repo_url and repo_revision")
    return SourceSpec(
        source_id=benchmark_id,
        url=str(url),
        revision=str(revision),
        subdir=str(runtime.get("repo_subdir") or ""),
    )


def resolve_benchmark_source(
    benchmark_id: str,
    *,
    explicit: Path | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Return the directory holding the official upstream sources of ``benchmark_id``.

    Resolution order: ``explicit``, the catalog runtime's ``runtime_root_env``
    override, then the pinned revision in the source cache, fetched on first use.

    Args:
        benchmark_id: Canonical benchmark id; also the source cache key.
        explicit: Existing directory with the official sources, used as is.
        runtime: The benchmark's ``runner.runtime`` block; loaded from the
            packaged benchmark catalog when omitted.

    Raises:
        SourceUnavailableError: The pinned revision is not cached and could not be fetched.
    """
    if explicit is not None:
        return explicit.expanduser().resolve()

    if runtime is None:
        from worldfoundry.evals.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

        runtime = load_benchmark_zoo_registry().get(benchmark_id).runner_runtime or {}
    env_name = runtime.get("runtime_root_env")
    override = env_path(str(env_name)) if env_name else None
    if override is not None:
        return override.expanduser().resolve()

    spec = benchmark_source_spec(benchmark_id, runtime)
    try:
        return ensure_source(spec)
    except SourceUnavailableError as exc:
        where = f"the {spec.subdir} directory" if spec.subdir else "the root"
        hint = f"; point {env_name} at {where} of a local checkout of that revision" if env_name else ""
        raise SourceUnavailableError(f"{benchmark_id}: {exc}{hint}") from exc


__all__ = ["benchmark_source_spec", "resolve_benchmark_source"]
