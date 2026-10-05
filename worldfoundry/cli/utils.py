"""CLI utility helpers for JSON serialization, key-value parsing, and zoo-id resolution."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


class CliUsageError(ValueError):
    """User-facing command-usage error raised by CLI handlers.

    Handlers raise this for invalid flag combinations, missing required
    values, and other mistakes the user can fix by changing the command
    line.  ``cli.main`` renders it as a single ``error: <message>`` line on
    stderr (JSON envelope in ``--json`` mode) and returns exit code 2 — the
    argparse usage-error convention — keeping runtime failures (exit 1)
    distinguishable from usage mistakes (CM-08).
    """

def cli_prog_name() -> str:
    """Match usage/help branding to the console script that was invoked.

    ``worldfoundry`` and ``worldfoundry-eval`` share this parser; anything
    else (``python -m worldfoundry.cli``, tests, ...) keeps the historical
    ``worldfoundry-eval`` branding so existing help-contract tests stay stable.
    """

    name = Path(sys.argv[0]).name
    if name in {"worldfoundry", "worldfoundry-eval"}:
        return name
    return "worldfoundry-eval"


def json_dump(payload: object) -> None:
    """Print the sole machine-readable payload for a CLI ``--json`` response.

    Operational progress, warnings, and logs must use stderr/the logging
    pipeline instead; callers use this helper only for their final result.

    Args:
        payload: JSON-serializable command response payload.
    """
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)


def append_optional_arg(argv: list[str], flag: str, value: object | None) -> None:
    """Append ``flag str(value)`` to *argv* when *value* is not ``None``.

    Shared by the command-builder helpers in ``zoo.py`` and
    ``tui_discovery.py`` that previously carried verbatim copies (CM-24).
    """
    if value is not None:
        argv.extend([flag, str(value)])


def task_roots_from_args(args: argparse.Namespace) -> tuple[Path, ...]:
    """Collect all task-root directories from CLI args and environment variables.

    Merges paths from ``--task-root``, ``--include-path``, and the
    ``WORLDFOUNDRY_TASK_ROOTS`` / ``WORLDFOUNDRY_BENCHMARK_INCLUDE_PATH`` env
    vars, deduplicating by path value.  Shared by the ``task`` and ``plan``
    command families so the environment-merge rules cannot drift.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Deduplicated tuple of :class:`Path` directories to search for
        task definitions.
    """
    from worldfoundry.evaluation.utils import BENCHMARK_TASK_ROOT

    roots = list(args.task_root or ())
    if not roots:
        roots = [BENCHMARK_TASK_ROOT] if BENCHMARK_TASK_ROOT.exists() else []
    for env_name in ("WORLDFOUNDRY_TASK_ROOTS", "WORLDFOUNDRY_BENCHMARK_INCLUDE_PATH"):
        for item in os.environ.get(env_name, "").split(os.pathsep):
            if item.strip():
                roots.append(Path(item))
    roots.extend(getattr(args, "include_path", None) or ())
    return tuple(dict.fromkeys(Path(root) for root in roots))


def parse_json_value(value: str) -> object:
    """Parse a CLI scalar as JSON only when it is clearly JSON.

    Args:
        value: Raw command-line value.
    """
    stripped = value.strip()
    if stripped in {"true", "false", "null"} or stripped[:1] in {'"', "{", "["} or _JSON_NUMBER_RE.fullmatch(stripped):
        return json.loads(value)
    return value


def parse_key_value_mapping(values: list[str] | None) -> dict[str, object]:
    """Parse repeated `KEY=VALUE` flags.

    Args:
        values: Repeated key-value CLI flag values.
    """
    payload: dict[str, object] = {}
    for item in values or ():
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        payload[key.strip()] = parse_json_value(value)
    return payload


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON object from an optional path.

    Args:
        path: Optional JSON file path.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return payload


def canonical_model_zoo_id(value: str | None, manifest_dir: Path | None) -> str | None:
    """Resolve a model-zoo id or alias when a manifest directory is available.

    Args:
        value: Model id or alias.
        manifest_dir: Optional model-zoo manifest directory.
    """
    if value is None:
        return None
    if manifest_dir is None or not manifest_dir.exists():
        return value
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

    return load_model_zoo_registry(manifest_dir).get(value).model_id


def canonical_benchmark_zoo_id(value: str | None, manifest_dir: Path | None) -> str | None:
    """Resolve a benchmark-zoo id or alias.

    Args:
        value: Benchmark id or alias.
        manifest_dir: Optional benchmark-zoo manifest directory or file.
    """
    if value is None:
        return None
    if manifest_dir is None or not manifest_dir.exists():
        return value
    from worldfoundry.evaluation.tasks.catalog.schema import load_entries
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import BenchmarkZooRegistry, load_benchmark_zoo_registry

    if manifest_dir.is_file():
        return BenchmarkZooRegistry(load_entries(manifest_dir)).get(value).benchmark_id
    return load_benchmark_zoo_registry(manifest_dir).get(value).benchmark_id


def resolve_cli_benchmark_for_materialize(task_type: str, benchmark_name: str) -> Any:
    """Resolve a benchmark adapter for legacy task-type/materialize CLI flows."""
    prog = cli_prog_name()
    raise ValueError(
        "Task-type/benchmark-name materialization is retired for benchmark-zoo entries. "
        f"Use `{prog} run --benchmark <id> --model <id>` or "
        f"`{prog} task materialize` with a filesystem task YAML."
    )
