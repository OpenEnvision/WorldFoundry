"""Workspace-facing dispatch for in-tree benchmark runners."""

from worldfoundry.evals.tasks.catalog.dispatch import CLI_RUNNERS as CLI_RUNNERS
from worldfoundry.evals.tasks.catalog.dispatch import (
    OfficialRunnerSpec as WorkspaceRunnerSpec,  # noqa: F401 - Re-exported for workspace registry callers.
)

from .dispatch import *  # noqa: F403 - preserve the existing Workspace registry exports

__all__ = [name for name in globals() if not name.startswith("_")]
