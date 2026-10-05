"""CLI commands for listing and executing checked-in WorldFoundry YAML workflow templates."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from worldfoundry.evaluation.utils import REPO_ROOT

from .presentation import print_table, terminal_enabled
from .utils import json_dump, parse_key_value_mapping

# ── Config constants ────────────────────────────────────────────

CONFIG_SCHEMA_VERSION = "worldfoundry-example-config"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_SUPPORTED_ENTRYPOINTS = {"worldfoundry", "worldfoundry-eval"}


# ── Config loading and expansion ────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    """Load and validate a WorldFoundry YAML workflow config file.

    Args:
        path: Path to the YAML config file.

    Raises:
        ValueError: If the config schema version, command format, or entrypoint is invalid.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML object: {path}")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version in {path}: {payload.get('schema_version')}")
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("config command must be a non-empty list of strings")
    if command[0] not in _SUPPORTED_ENTRYPOINTS:
        raise ValueError(
            "config command must start with one of "
            f"{', '.join(sorted(_SUPPORTED_ENTRYPOINTS))}; got {command[0]!r}"
        )
    return payload


def _expand_env_token(token: str, env: Mapping[str, str]) -> str:
    """Expand ``${VAR}``, ``${VAR:-default}``, and ``$VAR`` references in a token string.

    Args:
        token: Raw token potentially containing environment variable references.
        env: Environment mapping used for substitution.
    """
    def replace(match: re.Match[str]) -> str:
        braced_name = match.group(1)
        default_value = match.group(2)
        bare_name = match.group(3)
        name = braced_name or bare_name
        if name in env and env[name] != "":
            return env[name]
        if braced_name and default_value is not None:
            return default_value
        return match.group(0)

    return _ENV_PATTERN.sub(replace, token)


def _expanded_command(payload: Mapping[str, Any], env: Mapping[str, str]) -> list[str]:
    """Expand all environment references in the config command list."""
    return [_expand_env_token(str(token), env) for token in payload["command"]]


def _with_output_dir(command: Sequence[str], output_dir: Path | None) -> list[str]:
    """Insert or replace ``--output-dir`` in the expanded command list."""
    tokens = list(command)
    if output_dir is None:
        return tokens
    text = str(output_dir)
    if "--output-dir" in tokens:
        index = tokens.index("--output-dir")
        if index + 1 >= len(tokens):
            raise ValueError("config command contains --output-dir without a value")
        tokens[index + 1] = text
        return tokens
    tokens.extend(["--output-dir", text])
    return tokens


def _command_payload(path: Path, payload: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    """Build a structured payload describing the expanded config command."""
    return {
        "schema_version": "worldfoundry-config-command",
        "config_path": str(path),
        "name": payload.get("name"),
        "kind": payload.get("kind"),
        "description": payload.get("description"),
        "requirements": payload.get("requirements", {}),
        "command": list(command),
        "argv": list(command[1:]),
        "shell": shlex.join(command),
    }


def _display_path(path: Path) -> str:
    """Show a path relative to ``REPO_ROOT`` when possible, otherwise absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _config_list_items(config_dir: Path) -> list[dict[str, Any]]:
    """Enumerate all YAML workflow configs under a directory."""
    if not config_dir.exists() or not config_dir.is_dir():
        raise ValueError(f"config directory does not exist: {config_dir}")
    items: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        payload = _load_config(path)
        command = list(payload["command"])
        items.append(
            {
                "schema_version": "worldfoundry-config-list-item",
                "path": _display_path(path),
                "name": payload.get("name"),
                "kind": payload.get("kind"),
                "description": payload.get("description"),
                "requirements": dict(payload.get("requirements", {})),
                "command": command,
                "shell": shlex.join(command),
                "run_command": f"worldfoundry-eval config run {shlex.quote(_display_path(path))}",
            }
        )
    return items


def _requirements_label(requirements: Mapping[str, Any]) -> str:
    """Render enabled requirement keys as a comma-separated label."""
    enabled = sorted(str(key).replace("_", "-") for key, value in requirements.items() if bool(value))
    return ",".join(enabled) if enabled else "no-extra-assets"


# ── Config command handlers ─────────────────────────────────────


def _handle_config_list(args: argparse.Namespace) -> int:
    """List checked-in WorldFoundry workflow templates with their requirements."""
    items = _config_list_items(args.config_dir)
    if args.json:
        json_dump(
            {
                "schema_version": "worldfoundry-config-list",
                "config_dir": _display_path(args.config_dir),
                "count": len(items),
                "items": items,
            }
        )
        return 0

    if terminal_enabled():
        print_table(
            "Workflow templates",
            ("Config", "Name", "Requirements", "Run"),
            [
                (item["path"], item.get("name") or "-", _requirements_label(item["requirements"]), item["run_command"])
                for item in items
            ],
        )
        return 0

    for item in items:
        print(f"{item['path']}: {item.get('name') or '-'} [{item.get('kind') or '-'}]")
        print(f"  requirements: {_requirements_label(item['requirements'])}")
        if item.get("description"):
            print(f"  {item['description']}")
        print(f"  run: {item['run_command']}")
    return 0


def _handle_config_run(args: argparse.Namespace) -> int:
    """Expand and execute a WorldFoundry YAML workflow template through the CLI."""
    payload = _load_config(args.config)
    env = {**os.environ, **parse_key_value_mapping(args.env)}
    command = _with_output_dir(_expanded_command(payload, env), args.output_dir)
    command_payload = _command_payload(args.config, payload, command)

    if args.plan_only:
        if args.json:
            json_dump(command_payload)
        else:
            print(command_payload["shell"])
        return 0

    if args.print_command:
        print(command_payload["shell"], file=sys.stderr)

    from .main import main as cli_main

    return cli_main(list(command[1:]))


def register_config_subparser(subparsers: argparse._SubParsersAction) -> None:
    config_parser = subparsers.add_parser(
        "config",
        help="Run checked-in WorldFoundry YAML workflow templates through the public CLI",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    list_parser = config_subparsers.add_parser(
        "list",
        help="List checked-in WorldFoundry workflow templates and their requirements",
    )
    list_parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=_handle_config_list)

    run_parser = config_subparsers.add_parser(
        "run",
        help="Expand and execute the command list from a WorldFoundry example config",
    )
    run_parser.add_argument("config", type=Path, help="Path to a WorldFoundry YAML workflow template.")
    run_parser.add_argument("--env", action="append", default=None, metavar="KEY=VALUE")
    run_parser.add_argument("--output-dir", type=Path, help="Override or append the config command's --output-dir.")
    run_parser.add_argument("--plan-only", action="store_true", help="Print the expanded command without executing it.")
    run_parser.add_argument("--print-command", action="store_true", help="Print the expanded command to stderr before execution.")
    run_parser.add_argument("--json", action="store_true", help="With --plan-only, print the expanded command as JSON.")
    run_parser.set_defaults(func=_handle_config_run)


__all__ = ["CONFIG_SCHEMA_VERSION", "register_config_subparser"]
