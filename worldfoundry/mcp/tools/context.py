"""Shared execution context for WorldFoundry MCP tool payloads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR, REPO_ROOT
from worldfoundry.runtime.jobs import AsyncCommandJobStore

# ── Defaults ───────────────────────────────────────────────────

# Retention bound for the MCP job store: terminal jobs beyond this count are
# evicted oldest-first on submission, so a long-lived server cannot grow its
# in-memory registry (and on-disk index) without bound (CM-28).
DEFAULT_MCP_MAX_TRACKED_JOBS = 256

# Filename of the persisted job index kept under the MCP output root (CM-28).
MCP_JOB_INDEX_FILENAME = "jobs-index.json"


def resolve_mcp_output_root() -> Path:
    """Resolve the MCP run root to a stable absolute path.

    MCP stdio servers are spawned by clients with an arbitrary working
    directory (often ``/``), so a CWD-relative default would scatter run
    outputs in unpredictable places (CM-29). ``WORLDFOUNDRY_MCP_RUN_ROOT``
    still overrides; a relative override is resolved against the CWD at call
    time, so a value captured at import or server start stays stable for the
    server's lifetime.
    """

    override = os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs" / "mcp"


# NOTE: ``DEFAULT_MCP_OUTPUT_ROOT`` can be overridden via the
# ``WORLDFOUNDRY_MCP_RUN_ROOT`` environment variable.
DEFAULT_MCP_OUTPUT_ROOT = resolve_mcp_output_root()


@dataclass
class MCPToolContext:
    """Manifest roots, output root, and job store shared by MCP tool payloads.

    Attributes:
        output_root: Directory where MCP-triggered evaluation runs write
            their outputs. Defaults to ``DEFAULT_MCP_OUTPUT_ROOT``.
        model_manifest_dir: Root directory of the model manifest zoo, or
            ``None`` if not configured.
        benchmark_manifest_dir: Root directory of the benchmark manifest zoo.
        job_store: :class:`AsyncCommandJobStore` used to track active and
            completed evaluation runs. When omitted, a store is created that
            persists its job index under ``output_root`` and restores /
            pid-reconciles previously submitted runs on startup (CM-28).
    """

    output_root: Path = DEFAULT_MCP_OUTPUT_ROOT
    model_manifest_dir: Path | None = MODEL_ZOO_DIR
    benchmark_manifest_dir: Path = BENCHMARK_ZOO_DIR
    job_store: AsyncCommandJobStore | None = None

    def __post_init__(self) -> None:
        if self.job_store is None:
            self.job_store = AsyncCommandJobStore(
                max_jobs=DEFAULT_MCP_MAX_TRACKED_JOBS,
                state_path=Path(self.output_root) / MCP_JOB_INDEX_FILENAME,
            )


DEFAULT_CONTEXT = MCPToolContext()
_current_default_context = DEFAULT_CONTEXT


def get_default_context() -> MCPToolContext:
    """Return the process-wide default MCP tool context.

    Payload functions resolve their fallback context through this accessor
    (never through a direct ``DEFAULT_CONTEXT`` import binding), so a context
    installed via :func:`set_default_context` is observed everywhere.
    """

    return _current_default_context


def set_default_context(context: MCPToolContext) -> MCPToolContext:
    """Install *context* while synchronizing the public default singleton.

    Copying fields onto ``DEFAULT_CONTEXT`` keeps existing import aliases in
    sync.  The late-bound accessor still returns the actual installed context,
    preserving context identity for callers and allowing a previous context to
    be restored.  Registered tools and direct payload calls therefore share
    one job store and output root (CM-29).
    """

    global _current_default_context
    if context is DEFAULT_CONTEXT:
        _current_default_context = DEFAULT_CONTEXT
        return DEFAULT_CONTEXT
    DEFAULT_CONTEXT.output_root = context.output_root
    DEFAULT_CONTEXT.model_manifest_dir = context.model_manifest_dir
    DEFAULT_CONTEXT.benchmark_manifest_dir = context.benchmark_manifest_dir
    DEFAULT_CONTEXT.job_store = context.job_store
    _current_default_context = context
    return context


__all__ = [
    "DEFAULT_CONTEXT",
    "DEFAULT_MCP_MAX_TRACKED_JOBS",
    "DEFAULT_MCP_OUTPUT_ROOT",
    "MCP_JOB_INDEX_FILENAME",
    "MCPToolContext",
    "get_default_context",
    "resolve_mcp_output_root",
    "set_default_context",
]
