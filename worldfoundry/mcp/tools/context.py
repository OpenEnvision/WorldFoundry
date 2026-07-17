"""Shared execution context for WorldFoundry MCP tool payloads."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR
from worldfoundry.runtime.jobs import AsyncCommandJobStore

# ── Defaults ───────────────────────────────────────────────────

# NOTE: ``DEFAULT_MCP_OUTPUT_ROOT`` can be overridden via the
# ``WORLDFOUNDRY_MCP_RUN_ROOT`` environment variable.
DEFAULT_MCP_OUTPUT_ROOT = Path(os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", "runs/mcp"))


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
            completed evaluation runs.
    """

    output_root: Path = DEFAULT_MCP_OUTPUT_ROOT
    model_manifest_dir: Path | None = MODEL_ZOO_DIR
    benchmark_manifest_dir: Path = BENCHMARK_ZOO_DIR
    job_store: AsyncCommandJobStore = field(default_factory=AsyncCommandJobStore)


DEFAULT_CONTEXT = MCPToolContext()


__all__ = ["DEFAULT_CONTEXT", "DEFAULT_MCP_OUTPUT_ROOT", "MCPToolContext"]
