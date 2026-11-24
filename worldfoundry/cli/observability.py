# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""CLI logging setup and per-run lifecycle events for commands that own an output tree."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def extract_logging_flags(
    argv: list[str],
) -> tuple[str | None, str | None, bool | None, bool, list[str]]:
    """Pre-scan ``argv`` for the global logging/verbosity flags, returning them
    plus the remaining argv with those flags stripped.

    Recognizes both ``--flag value`` and ``--flag=value`` forms; the flags are
    accepted in any position (before *or* after the subcommand) because the
    real application happens here, ahead of argparse.
    """
    level = log_file = log_json = None
    verbose = False
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--log-level" and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            level = argv[i + 1]
            i += 2
            continue
        if token.startswith("--log-level="):
            level = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--log-file" and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            log_file = argv[i + 1]
            i += 2
            continue
        if token.startswith("--log-file="):
            log_file = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--log-json":
            log_json = True
            i += 1
            continue
        if token in ("-v", "--verbose"):
            verbose = True
            i += 1
            continue
        out.append(token)
        i += 1
    return level, log_file, log_json, verbose, out


def configure_cli_logging(
    level: str | None,
    log_file: str | None,
    log_json: bool | None,
) -> None:
    """Apply ``configure_logging`` for the CLI and export the resolved values to
    the environment so framework-owned child processes (which re-enter
    ``main``) inherit the same configuration. A bad level falls back to INFO
    rather than aborting the whole command."""
    from worldfoundry.core import configure_logging

    try:
        configure_logging(level=level, log_file=log_file, json=log_json)
    except ValueError as exc:
        sys.stderr.write(f"warning: {exc} Falling back to INFO.\n")
        configure_logging(level="INFO", log_file=log_file, json=log_json, force=True)

    if level is not None:
        os.environ["WORLDFOUNDRY_LOG_LEVEL"] = str(level)
    if log_file is not None:
        os.environ["WORLDFOUNDRY_LOG_FILE"] = str(log_file)
    if log_json:
        os.environ["WORLDFOUNDRY_LOG_JSON"] = "1"


def _bind_cli_log_context(args: argparse.Namespace, *, run_id: str | None = None) -> None:
    """Attach command identity to logs emitted by this CLI process."""

    from worldfoundry.core import bind_log_context

    benchmark_id = getattr(args, "benchmark_id", None) or getattr(args, "benchmark_name", None)
    bind_log_context(
        run_id=run_id or getattr(args, "run_id", None),
        benchmark_id=benchmark_id,
        model_id=getattr(args, "model_id", None),
        phase="cli",
        command=getattr(args, "command", None),
    )


def _new_run_id() -> str:
    """Return a sortable, collision-resistant ID for a CLI-owned output run."""

    return f"wf-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _safe_run_log_component(run_id: str) -> str:
    """Convert a caller-provided run ID into one safe path component."""

    normalized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in run_id)
    return normalized.strip(".-") or "run"


def prepare_cli_run_observability(
    args: argparse.Namespace,
    *,
    explicit_log_file: str | None,
) -> tuple[Path, str, float] | None:
    """Create an isolated log directory for commands that own an output tree."""

    output_dir = getattr(args, "output_dir", None)
    if output_dir in (None, ""):
        _bind_cli_log_context(args)
        return None
    if getattr(args, "plan_only", False) or getattr(args, "print_config", False):
        # Dry-run flows must not create ``logs/`` under the output tree or
        # rewrite the process logging environment (CM-09).
        _bind_cli_log_context(args)
        return None

    run_id = str(getattr(args, "run_id", None) or _new_run_id())
    # Only some command parsers expose --run-id, but every output-owning CLI
    # operation should still carry a generated ID in its logs and child env.
    args.run_id = run_id
    _bind_cli_log_context(args, run_id=run_id)

    from worldfoundry.core import configure_logging, log_context_environment, write_jsonl_event

    root = Path(output_dir).expanduser().resolve()
    if bool(getattr(args, "_requires_exclusive_output_dir", False)):
        # Some commands atomically claim their output directory as a run
        # transaction.  Keep lifecycle logs in an isolated sibling so logging
        # cannot pre-create that directory and defeat the exclusivity check.
        event_path = (
            root.parent
            / ".worldfoundry-cli-logs"
            / _safe_run_log_component(root.name)
            / _safe_run_log_component(run_id)
            / "events.jsonl"
        )
    else:
        event_path = root / "logs" / _safe_run_log_component(run_id) / "events.jsonl"
    # A caller-selected --log-file / env file remains authoritative. Otherwise
    # make the per-run JSONL artifact the default sink for this command.
    if explicit_log_file is None and not os.environ.get("WORLDFOUNDRY_LOG_FILE"):
        configure_logging(log_file=event_path, json=True, force=True)
        os.environ["WORLDFOUNDRY_LOG_FILE"] = str(event_path)
        os.environ["WORLDFOUNDRY_LOG_JSON"] = "1"
    os.environ["WORLDFOUNDRY_RUN_ID"] = run_id
    os.environ.update(log_context_environment())
    write_jsonl_event(
        event_path,
        level="INFO",
        event="run.started",
        message="WorldFoundry command started",
        logger_name=__name__,
        output_dir=str(root),
        command=getattr(args, "command", None),
    )
    return event_path, run_id, time.monotonic()


def write_cli_run_lifecycle(
    state: tuple[Path, str, float] | None,
    *,
    event: str,
    level: str,
    message: str,
    exit_code: int | None = None,
    exception: BaseException | None = None,
) -> None:
    """Persist a terminal CLI lifecycle event even with a custom log sink."""

    if state is None:
        return
    from worldfoundry.core import write_jsonl_event

    event_path, _, started = state
    fields: dict[str, Any] = {"duration_seconds": round(time.monotonic() - started, 6)}
    if exit_code is not None:
        fields["exit_code"] = exit_code
    write_jsonl_event(
        event_path,
        level=level,
        event=event,
        message=message,
        logger_name=__name__,
        exception=exception,
        **fields,
    )


# Long-lived or workload-owning commands whose in-process logging must run
# through the configured pipeline even when the parsed namespace does not
# carry an ``output_dir`` value.
_EAGER_LOGGING_COMMANDS = frozenset(
    {
        "tui",
        "mcp",
    }
)


def logging_flags_requested(level: str | None, log_file: str | None, log_json: bool | None) -> bool:
    """Whether the caller explicitly asked for configured logging (flag or env)."""

    if level is not None or log_file is not None or log_json is not None:
        return True
    return any(
        os.environ.get(name) for name in ("WORLDFOUNDRY_LOG_LEVEL", "WORLDFOUNDRY_LOG_FILE", "WORLDFOUNDRY_LOG_JSON")
    )


def command_wants_eager_logging(args: argparse.Namespace) -> bool:
    """Whether the selected command should configure logging before dispatch.

    Pure query commands skip ``configure_logging`` entirely: its
    ``core.distributed`` reparenting currently drags ``torch`` into every
    process (CM-01), which query paths must not pay for.  Workload commands
    (anything owning an output tree, TUI, MCP server) keep the
    configured pipeline.  Dry-run variants (``--plan-only`` /
    ``--print-config`` / ``--model-status``) stay unconfigured like queries.
    """

    if getattr(args, "plan_only", False) or getattr(args, "print_config", False):
        return False
    if getattr(args, "command", None) in _EAGER_LOGGING_COMMANDS:
        return True
    return getattr(args, "output_dir", None) not in (None, "")
