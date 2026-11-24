"""Top-level CLI composition for WorldFoundry command execution."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from types import ModuleType
from typing import Any

from worldfoundry.evals.utils import REPO_ROOT

from .help import WorldFoundryArgumentParser
from .observability import (
    command_wants_eager_logging,
    configure_cli_logging,
    extract_logging_flags,
    logging_flags_requested,
    prepare_cli_run_observability,
    write_cli_run_lifecycle,
)
from .presentation import (
    BOLD,
    MUTED,
    Panel,
    Row,
    paint,
    print_hint,
    print_notice,
    render_panels,
    terminal_enabled,
    terminal_width,
)
from .utils import CliUsageError, cli_prog_name, json_dump

# ── CLI banners and public command surface ──────────────────────

_FIRST_RUN_BANNER = """\
WorldFoundry evaluation CLI

Start with GPU-ready discovery and validation commands:
  {prog} zoo benchmarks --json
  {prog} zoo models
  {prog} tasks list

Then run a selected model x benchmark cell:
  {prog} run --benchmark <id> --model <model-id> --output-dir tmp/worldfoundry_run --json

Interactive and help:
  {prog} tui
  {prog} <command> --help
"""

_PUBLIC_ROOT_COMMANDS = (
    "tui",
    "tasks",
    "suites",
    "task",
    "dataset",
    "config",
    "plan",
    "metric",
    "compare-runs",
    "index-runs",
    "validate-artifact",
    "preflight",
    "models",
    "zoo",
    "mcp",
    "evaluate",
    "score",
    "generate-score",
    "reproduce",
    "embodied",
    "validate",
    "run",
)


def _print_first_run_banner() -> None:
    """Print the first-run discovery banner when no subcommand is selected."""
    if terminal_enabled():
        prog = cli_prog_name()
        print(paint("WorldFoundry", BOLD))
        print(paint("World models · Inference · Evaluation", MUTED))
        print()
        panels = (
            Panel(
                "Explore",
                (
                    Row(f"{prog} zoo models", "Browse models and their requirements."),
                    Row(f"{prog} zoo benchmarks", "Find benchmarks and evaluation surfaces."),
                    Row(f"{prog} tui", "Open the interactive terminal workspace."),
                ),
            ),
            Panel(
                "Run and evaluate",
                (
                    Row(f"{prog} run <model> --help", "Inspect typed inputs, pipeline settings and defaults."),
                    Row(
                        f"{prog} run <model> --prompt 'A quiet coastal town'",
                        "Generate with a configured model and its checkpoints.",
                    ),
                    Row(f"{prog} score --help", "Score generated artifacts or existing results."),
                ),
            ),
        )
        print(render_panels(panels, terminal_width(), min_column_width=48))
        print_hint(f"Workflow templates: {prog} config list\nAll commands: {prog} --help")
        return
    print(_FIRST_RUN_BANNER.format(prog=cli_prog_name()).rstrip())


# ── Module and data loading helpers ─────────────────────────────


def _curate_root_subparser_help(subparsers: argparse._SubParsersAction) -> None:
    """Keep the root help focused on the supported public command surface."""

    subparsers.metavar = "{" + ",".join(_PUBLIC_ROOT_COMMANDS) + "}"
    # help.py reads this optional grouping through getattr when rendering terminal help.
    subparsers.help_groups = {  # ty: ignore[unresolved-attribute]
        "Start here": ("run", "tui", "zoo", "models"),
        "Evaluate": ("evaluate", "score", "generate-score", "reproduce", "embodied"),
        "Catalogs and workflows": ("tasks", "suites", "task", "dataset", "config", "plan", "metric"),
        "Results and validation": ("compare-runs", "index-runs", "validate-artifact", "validate", "preflight"),
        "Services": ("mcp",),
    }

    def describe_children(action: argparse._SubParsersAction) -> None:
        for command in action._get_subactions():
            child = action.choices[command.dest]
            if not child.description and command.help != argparse.SUPPRESS:
                child.description = command.help
        for child in set(action.choices.values()):
            for nested in child._actions:
                if isinstance(nested, argparse._SubParsersAction):
                    describe_children(nested)

    describe_children(subparsers)


def _load_repo_script(relative_path: str) -> ModuleType:
    """Dynamically import a repo script as a module by its relative path.

    Args:
        relative_path: Path relative to ``REPO_ROOT`` pointing to a Python script.

    Raises:
        ImportError: If the script cannot be loaded.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"_worldfoundry_cli_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── Argument-group helpers ──────────────────────────────────────


# ── Dataset command bridges ──────────────────────────────────────


# ── TUI bridges ─────────────────────────────────────────────────


# ── Evaluate command ─────────────────────────────────────────────


# ── Validate command ─────────────────────────────────────────────


# ── Run routing helpers ──────────────────────────────────────────


# ── Run command ──────────────────────────────────────────────────


def _build_parser(model_run_schema: Any | None = None) -> argparse.ArgumentParser:
    """Build all CLI commands, keeping parser wiring near handler selection."""
    parser = WorldFoundryArgumentParser(
        prog=cli_prog_name(),
        description="Discover world models, run inference, and evaluate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Command areas:\n"
            "  Discovery:   zoo benchmarks, zoo models, tasks list\n"
            "  Run:         run\n"
            "  Score:       evaluate, compare-runs, index-runs, validate-artifact\n"
            "  Extend:      model and benchmark manifests, task YAML, runtime profiles\n"
            "  Maintain:    task validate, preflight, dataset, plan, metric\n"
        ),
    )
    # Global logging flags. They are pre-scanned and stripped from ``argv`` in
    # ``main`` (so they may appear before *or* after the subcommand) and applied
    # via ``configure_logging`` before the handler runs. Registered here so
    # ``--help`` documents them; argparse never enforces them.
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (env: WORLDFOUNDRY_LOG_LEVEL). Default INFO.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="Write a rotating log to this path (env: WORLDFOUNDRY_LOG_FILE).",
    )
    parser.add_argument(
        "--log-json",
        dest="log_json",
        action="store_true",
        default=False,
        help="Emit the file sink as JSON Lines (env: WORLDFOUNDRY_LOG_JSON).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Show the full traceback when a command fails.",
    )
    subparsers = parser.add_subparsers(dest="command")

    from .tui import register_tui_subparser

    register_tui_subparser(subparsers)

    from .tasks import register_task_subparsers

    register_task_subparsers(subparsers)

    from .dataset import register_dataset_subparser

    register_dataset_subparser(subparsers)

    from .config import register_config_subparser

    register_config_subparser(subparsers)

    from .plan_metric import register_plan_metric_subparsers

    register_plan_metric_subparsers(subparsers)

    from .evaluation_intents import register_evaluation_intent_subparsers

    register_evaluation_intent_subparsers(subparsers)

    from .reporting import register_reporting_subparsers

    register_reporting_subparsers(subparsers)

    from .models import register_model_subparsers

    register_model_subparsers(subparsers)

    from .preflight import register_preflight_subparser

    register_preflight_subparser(subparsers)

    from .zoo import register_zoo_subparser

    register_zoo_subparser(subparsers)

    from .mcp import add_mcp_parser

    add_mcp_parser(subparsers)

    from .embodied import register_embodied_subparser

    register_embodied_subparser(subparsers)

    from .evaluate import register_evaluate_subparsers

    register_evaluate_subparsers(subparsers)

    from .run import register_run_subparser

    register_run_subparser(subparsers, model_run_schema)

    _curate_root_subparser_help(subparsers)
    return parser


def _format_cli_error(exc: BaseException) -> str:
    """Render one concise, typed error line for terminal users."""

    message = str(exc).strip()
    type_name = type(exc).__name__
    if not message:
        return type_name
    # ``str(KeyError("x"))`` is just ``"'x'"``; always carry the type so bare
    # messages stay interpretable.
    return f"{type_name}: {message}"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse args and dispatch to the selected command handler."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _log_level, _log_file, _log_json, _verbose, raw_argv = extract_logging_flags(raw_argv)
    # CM-01: defer configure_logging so ``--help`` and pure query commands do
    # not pull the heavy core.distributed/torch import chain. Explicit flags
    # or WORLDFOUNDRY_LOG_* environment values keep today's eager behaviour.
    if logging_flags_requested(_log_level, _log_file, _log_json):
        configure_cli_logging(_log_level, _log_file, _log_json)
    model_run_schema = None
    if raw_argv and raw_argv[0] == "run":
        from .model_run import (
            ModelRunSchemaError,
            load_model_run_schema,
            model_id_from_run_argv,
            option_from_argv,
        )

        model_ref = model_id_from_run_argv(raw_argv)
        positional_model = len(raw_argv) > 1 and not raw_argv[1].startswith("-")
        wants_model_schema = positional_model or any(
            item in {"-h", "--help", "--print-config", "--model-status"}
            or item.startswith(("--pipeline.", "--runtime."))
            for item in raw_argv[1:]
        )
        if model_ref and wants_model_schema:
            try:
                model_run_schema = load_model_run_schema(
                    model_ref,
                    variant_id=option_from_argv(raw_argv, "--model-variant"),
                    task_id=option_from_argv(raw_argv, "--pipeline.task-profile"),
                )
            except ModelRunSchemaError as exc:
                parser = _build_parser()
                parser.error(str(exc))

    parser = _build_parser(model_run_schema)
    args = parser.parse_args(raw_argv)
    if not hasattr(args, "func"):
        _print_first_run_banner()
        return 0
    if command_wants_eager_logging(args):
        configure_cli_logging(_log_level, _log_file, _log_json)
    run_observability = prepare_cli_run_observability(args, explicit_log_file=_log_file)

    try:
        exit_code = args.func(args)
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "INFO" if exit_code == 0 else "ERROR",
                "cli.command.finished",
                "CLI command finished",
                command=getattr(args, "command", None),
                exit_code=exit_code,
            )
        write_cli_run_lifecycle(
            run_observability,
            event="run.finished",
            level="INFO" if exit_code == 0 else "ERROR",
            message="WorldFoundry command finished",
            exit_code=exit_code,
        )
        return exit_code
    except KeyboardInterrupt:
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "WARNING",
                "cli.command.cancelled",
                "CLI command interrupted",
                command=getattr(args, "command", None),
                exit_code=130,
            )
        write_cli_run_lifecycle(
            run_observability,
            event="run.cancelled",
            level="WARNING",
            message="WorldFoundry command interrupted",
            exit_code=130,
        )
        parser.exit(130, "Interrupted.\n")
    except CliUsageError as exc:
        # Usage mistakes (bad flag combinations, missing required values) keep
        # the argparse convention: exit 2, one concise stderr line, no stack
        # trace or stack-carrying log event (CM-08).  Runtime failures below
        # stay on exit 1.  Returning (not parser.exit) preserves the historic
        # ``return 2`` handler behaviour for in-process callers of ``main``.
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "ERROR",
                "cli.command.usage_error",
                "CLI command rejected: usage error",
                command=getattr(args, "command", None),
                error=str(exc),
            )
        write_cli_run_lifecycle(
            run_observability,
            event="run.failed",
            level="ERROR",
            message="WorldFoundry command rejected: usage error",
            exit_code=2,
            exception=exc,
        )
        if getattr(args, "json", False):
            json_dump(
                {
                    "status": "error",
                    "command": getattr(args, "command", None),
                    "error": {"type": "usage", "message": str(exc)},
                    "exit_code": 2,
                }
            )
        print_notice(
            str(exc),
            level="error",
            stream=sys.stderr,
            hint=f"Run {cli_prog_name()} {getattr(args, 'command', '')} --help for usage.",
        )
        return 2
    except Exception as exc:
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "ERROR",
                "cli.command_failed",
                "CLI command failed",
                exc_info=_verbose,
                command=getattr(args, "command", None),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        write_cli_run_lifecycle(
            run_observability,
            event="run.failed",
            level="ERROR",
            message="WorldFoundry command failed",
            exit_code=1,
            exception=exc,
        )
        # Machine consumers asked for JSON: keep stdout a parseable contract
        # even on failure (CM-07), with the runtime-failure exit code 1 kept
        # distinct from argparse usage errors (exit 2).
        if getattr(args, "json", False):
            json_dump(
                {
                    "status": "error",
                    "command": getattr(args, "command", None),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "exit_code": 1,
                }
            )
        if _verbose:
            import traceback

            traceback.print_exc()
        print_notice(_format_cli_error(exc), level="error", stream=sys.stderr)
        parser.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
