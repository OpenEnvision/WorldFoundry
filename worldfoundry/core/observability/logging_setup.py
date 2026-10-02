"""Central logging configuration for WorldFoundry.

This module is the single opt-in entry point for wiring the framework's
loguru-backed ``log`` facade (:mod:`worldfoundry.core.distributed.runtime.logging`)
together with the stdlib :mod:`logging` hierarchy — vendored model runtimes,
the sequence-parallel infrastructure (``init_logger``), and the rank-aware
``distributed_logger``. Calling :func:`configure_logging` once at process
entry gives every log source a consistent level, format, and optional
rotating file sink.

Design
------
*When loguru is installed* the stdlib hierarchy is bridged *into* loguru via
:class:`_StdlibToLoguruHandler` on the root logger, so there is a single
output pipeline: stdlib records → root → handler → loguru → configured sinks,
and the facade writes to loguru directly.

*When loguru is unavailable* (e.g. a minimal runtime environment) the same
guarantees are provided by configuring the stdlib root logger directly with a
unified :class:`logging.StreamHandler` and optional
:class:`~logging.handlers.RotatingFileHandler`. In both paths the
``distributed_logger`` singleton is reparented onto the root logger
(``propagate=True``, own handlers cleared) so :func:`print_rank_0` /
:func:`print_per_rank` share the pipeline.

Importing this module has no side effects; callers opt in explicitly so
library use is not surprised by a reconfigured logger. The only process-wide
logging side effect that runs at ``worldfoundry.core`` import time remains the
benign Inductor autotuner-fallback demotion in
:mod:`worldfoundry.core.observability.log_filters`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "get_log_context",
    "get_logger",
    "is_configured",
    "log_context_environment",
    "log_context",
    "write_jsonl_event",
]

# --- environment knobs (inherited by framework-owned child processes) ---------
_LOG_LEVEL_ENV = "WORLDFOUNDRY_LOG_LEVEL"
_LOG_FILE_ENV = "WORLDFOUNDRY_LOG_FILE"
_LOG_JSON_ENV = "WORLDFOUNDRY_LOG_JSON"
_LOG_CONTEXT_ENV = "WORLDFOUNDRY_LOG_CONTEXT"
# The sequence-parallel stack reconfigures the *root*
# logger via ``logging.config.dictConfig`` when this is truthy (its default).
# Setting it to "0" before ``sp.logger`` is imported makes that takeover a
# no-op, so the central config owns the root logger exclusively.
_SP_CONFIGURE_ENV = "TRAINER_CONFIGURE_LOGGING"

# Unified record layout, mirrored across the loguru format string and the
# stdlib Formatter so both pipelines emit byte-for-byte identical prefixes.
_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "<level>{level:<7}</level> "
    "<cyan>[{name}]</cyan> {message}"
)
_STDLIB_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False
_CONFIGURED_LEVEL: int = logging.INFO
_CONFIGURE_LOCK = threading.Lock()

# ``ContextVar`` makes one run's correlation fields flow across asyncio tasks
# without leaking into concurrent runs in the same process.  The fields are
# attached by the formatter, which means stdlib and facade call sites receive
# the same context without every caller having to pass it explicitly.
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("worldfoundry_log_context", default={})
_SCHEMA_VERSION = "worldfoundry.log.v1"
_CONTEXT_FIELDS = (
    "run_id",
    "job_id",
    "benchmark_id",
    "model_id",
    "phase",
    "sample_id",
    "rank",
)
_SENSITIVE_FIELD_NAME = re.compile(
    r"(?:^|[_\-.])(?:api[_\-.]?key|access[_\-.]?token|auth(?:orization)?|secret|"
    r"password|passwd|cookie|credential|private[_\-.]?key|token)(?:$|[_\-.])",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization|secret|password|passwd|"
    r"cookie|credential|token)\b\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)(\bbearer\s+)([^\s,;]+)")
_KNOWN_TOKEN = re.compile(
    r"\b(?:"
    r"(?:sk|hf)[_-][A-Za-z0-9_-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")\b"
)


def bind_log_context(**fields: Any) -> Token[dict[str, Any]]:
    """Bind correlation fields for the current execution context.

    The returned token can be passed to :meth:`contextvars.ContextVar.reset`,
    or callers can prefer :func:`log_context` for scoped use.
    """

    current = dict(_LOG_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    return _LOG_CONTEXT.set(current)


def clear_log_context() -> None:
    """Clear all correlation fields for the current execution context."""

    _LOG_CONTEXT.set({})


def get_log_context() -> dict[str, Any]:
    """Return a copy of the current correlation fields."""

    return dict(_LOG_CONTEXT.get())


def log_context_environment(**fields: Any) -> dict[str, str]:
    """Serialize current correlation fields for a framework-owned child process.

    This deliberately carries only structured context, never logger handlers or
    raw stdout/stderr.  It is safe to merge into a child ``env`` mapping.
    """

    payload = get_log_context()
    payload.update({key: value for key, value in fields.items() if value is not None})
    return {_LOG_CONTEXT_ENV: json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)}


def _bind_log_context_from_environment() -> None:
    """Import parent correlation fields when this process is a child worker."""

    raw = os.environ.get(_LOG_CONTEXT_ENV)
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if isinstance(payload, Mapping):
        bind_log_context(**dict(payload))


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Temporarily bind correlation fields for a block of work."""

    token = bind_log_context(**fields)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)

try:  # loguru is an optional dependency; the stdlib path is fully supported.
    from loguru import logger as _loguru_logger
    _LOGURU_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal environments.
    _loguru_logger = None
    _LOGURU_AVAILABLE = False


def _is_sensitive_field_name(name: object) -> bool:
    """Return whether *name* conventionally holds a credential value."""

    return isinstance(name, str) and bool(_SENSITIVE_FIELD_NAME.search(name))


def _redact_text(value: str) -> str:
    """Remove common credential forms from unstructured log text."""

    value = _BEARER_TOKEN.sub(r"\1[REDACTED]", value)
    value = _SENSITIVE_ASSIGNMENT.sub(r"\1[REDACTED]", value)
    return _KNOWN_TOKEN.sub("[REDACTED]", value)


def _json_safe(value: Any, *, field_name: str | None = None) -> Any:
    """Convert fields to JSON-safe values while redacting credentials."""

    if _is_sensitive_field_name(field_name):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return _redact_text(str(value))


def _sanitize_record(record: logging.LogRecord) -> None:
    """Redact record fields in place before any configured sink sees them."""

    if getattr(record, "_worldfoundry_sanitized", False):
        return
    record._worldfoundry_sanitized = True  # type: ignore[attr-defined]
    if isinstance(record.msg, str):
        record.msg = _redact_text(record.msg)
    if isinstance(record.args, Mapping):
        record.args = {key: _json_safe(value, field_name=str(key)) for key, value in record.args.items()}
    elif isinstance(record.args, tuple):
        record.args = tuple(_json_safe(value) for value in record.args)
    fields = getattr(record, "_worldfoundry_fields", None)
    if isinstance(fields, Mapping):
        record._worldfoundry_fields = _json_safe(fields)  # type: ignore[attr-defined]


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Merge task-local context with optional fields bound to one log record."""

    fields = dict(_LOG_CONTEXT.get())
    supplied = getattr(record, "_worldfoundry_fields", None)
    if isinstance(supplied, Mapping):
        fields.update(supplied)
    # Native stdlib callers can adopt individual standard context fields while
    # migration to ``get_logger(...).event(...)`` happens incrementally.
    for name in _CONTEXT_FIELDS:
        value = getattr(record, name, None)
        if value is not None:
            fields[name] = value
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if fields.get("rank") is None and rank is not None:
        fields["rank"] = rank
    return _json_safe(fields)


def _json_event_payload(
    *,
    level: str,
    logger_name: str,
    message: str,
    fields: Mapping[str, Any] | None = None,
    exception: str | None = None,
    timestamp: datetime | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    """Build the canonical JSONL object shared by every structured sink."""

    values = _json_safe(fields or {})
    event = values.pop("event", None)
    resolved_timestamp = timestamp or datetime.now(timezone.utc)
    return {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": resolved_timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": str(level),
        "logger": str(logger_name),
        "event": event,
        "message": _redact_text(str(message)),
        "run_id": values.pop("run_id", None),
        "job_id": values.pop("job_id", None),
        "benchmark_id": values.pop("benchmark_id", None),
        "model_id": values.pop("model_id", None),
        "phase": values.pop("phase", None),
        "sample_id": values.pop("sample_id", None),
        "rank": values.pop("rank", os.environ.get("RANK") or os.environ.get("LOCAL_RANK")),
        "pid": os.getpid() if pid is None else int(pid),
        "exception": None if exception is None else _redact_text(exception),
        "fields": values,
    }


def write_jsonl_event(
    path: str | os.PathLike[str],
    *,
    level: str,
    event: str,
    message: str,
    logger_name: str = "worldfoundry",
    exception: BaseException | str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Append one canonical lifecycle event to a run-owned JSONL artifact.

    Use this for process boundaries where a child may not import the central
    logger.  Normal in-process application events should use
    :meth:`WorldFoundryLoggerAdapter.event` instead.
    """

    exception_text: str | None
    if isinstance(exception, BaseException):
        exception_text = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    elif exception is None:
        exception_text = None
    else:
        exception_text = str(exception)
    event_fields = get_log_context()
    event_fields.update(fields)
    event_fields["event"] = event
    payload = _json_event_payload(
        level=level,
        logger_name=logger_name,
        message=message,
        fields=event_fields,
        exception=exception_text,
    )
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


class _RedactingTextFormatter(logging.Formatter):
    """Normal text formatter that also redacts exception bodies."""

    def format(self, record: logging.LogRecord) -> str:
        """Sanitize extras, then redact secrets that may appear in the rendered line."""
        _sanitize_record(record)
        return _redact_text(super().format(record))


class WorldFoundryLoggerAdapter(logging.LoggerAdapter):
    """Small stdlib-compatible facade for named structured events.

    Existing ``logger.info(...)`` use remains valid.  New owned code should use
    ``get_logger(__name__).event("INFO", "run.started", ...)`` so event names
    and fields stay separate from human-readable messages.
    """

    def process(self, message: object, kwargs: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        """Merge adapter-bound fields with per-call extras without dropping caller ``extra``."""
        extra = dict(kwargs.get("extra") or {})
        supplied = extra.pop("_worldfoundry_fields", {})
        fields = dict(self.extra.get("_worldfoundry_fields", {}))
        if isinstance(supplied, Mapping):
            fields.update(supplied)
        extra["_worldfoundry_fields"] = fields
        kwargs["extra"] = extra
        return message, kwargs

    def bind(self, **fields: Any) -> "WorldFoundryLoggerAdapter":
        """Return a sibling logger with fields attached to every record."""

        current = dict(self.extra.get("_worldfoundry_fields", {}))
        current.update(fields)
        return type(self)(self.logger, {"_worldfoundry_fields": current})

    def event(
        self,
        level: int | str,
        event: str,
        message: str | None = None,
        *,
        exc_info: bool | BaseException | tuple[type[BaseException], BaseException, Any] | None = None,
        **fields: Any,
    ) -> None:
        """Emit a named event with JSON-safe structured fields."""

        resolved_level = _resolve_level(level) if isinstance(level, str) else level
        self.log(
            resolved_level,
            message or event,
            extra={"_worldfoundry_fields": {"event": event, **fields}},
            exc_info=exc_info,
        )


def get_logger(name: str | None = None) -> WorldFoundryLoggerAdapter:
    """Return the WorldFoundry structured-event facade for a stdlib logger."""

    return WorldFoundryLoggerAdapter(logging.getLogger(name), {})


# --------------------------------------------------------------------------- #
# Small parsing helpers for the stdlib-only path.                              #
# --------------------------------------------------------------------------- #
_BYTES_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}


def _parse_bytes(value: str | int, default: int) -> int:
    """Parse ``"10 MB"`` / ``"100m"`` / ``1024`` into a byte count."""
    if isinstance(value, int):
        return value
    text = value.strip().lower()
    if not text:
        return default
    num = ""
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] == "."):
        num += text[idx]
        idx += 1
    unit = text[idx:].strip()
    if not num:
        # A bare unit like "mb" has no magnitude; multiplying the default by
        # the unit factor would inflate it, so return the default unchanged.
        return default
    try:
        base = float(num)
    except ValueError:
        return default
    factor = _BYTES_UNITS.get(unit, 1)
    return int(base * factor)


def _parse_retention(value: str | int, default: int) -> int:
    """Parse ``"1 week"`` / ``"7 days"`` / ``5`` into a backup-file count."""
    if isinstance(value, int):
        return value
    text = value.strip().lower()
    if not text:
        return default
    num = ""
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] == "."):
        num += text[idx]
        idx += 1
    unit = text[idx:].strip()
    try:
        count = float(num) if num else float(default)
    except ValueError:
        return default
    if "week" in unit:
        count *= 7
    elif "day" in unit:
        count *= 1
    return max(1, int(count))


_LEVEL_ALIASES = {
    "WARN": logging.WARNING,
    "FATAL": logging.CRITICAL,
    "TRACE": logging.DEBUG,
}


def _resolve_level(level: str | int | None, default: int = logging.INFO) -> int:
    """Accept int, name, or WARN/FATAL/TRACE aliases; unknown names fail instead of becoming ``Level N``."""
    if level is None or level == "":
        return default
    if isinstance(level, int):
        return level
    upper = level.strip().upper()
    if upper in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[upper]
    resolved = logging.getLevelName(upper)
    if isinstance(resolved, int) and 0 <= resolved <= logging.CRITICAL:
        return resolved
    raise ValueError(
        f"Unknown log level {level!r}. Expected one of DEBUG/INFO/WARNING/ERROR/CRITICAL."
    )


# --------------------------------------------------------------------------- #
# stdlib -> loguru bridge (only used when loguru is available).                #
# --------------------------------------------------------------------------- #
class _StdlibToLoguruHandler(logging.Handler):
    """Forward stdlib :class:`logging.LogRecord`\\ s into loguru's pipeline.

    This is the standard loguru-intercept recipe: map the record's level to a
    loguru level name, then walk up the stack past the stdlib ``logging``
    module frames so loguru attributes the record to the original caller.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one stdlib record into loguru; never raise — failures go to ``handleError``."""
        try:
            _sanitize_record(record)
            try:
                level = _loguru_logger.level(record.levelname).name  # type: ignore[union-attr]
            except (ValueError, TypeError):
                level = record.levelno
            frame, depth = logging.currentframe(), 0
            while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1
            _loguru_logger.bind(  # type: ignore[union-attr]
                _worldfoundry_fields=_record_fields(record),
                _worldfoundry_logger_name=record.name,
            ).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
        except Exception:  # never raise from a logging handler
            self.handleError(record)

    def flush(self) -> None:
        """Honor stdlib flush/shutdown for Loguru's queued file sinks."""
        if _loguru_logger is not None:
            _loguru_logger.complete()


class _JsonFormatter(logging.Formatter):
    """Emit the stable WorldFoundry JSONL schema for stdlib records."""

    def format(self, record: logging.LogRecord) -> str:
        """One JSON object per line; keys sorted so Studio/eval parsers can diff logs."""
        _sanitize_record(record)
        exception = self.formatException(record.exc_info) if record.exc_info else None
        payload = _json_event_payload(
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            fields=_record_fields(record),
            exception=exception,
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc),
            pid=record.process,
        )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loguru_exception_text(exception: Any) -> str | None:
    """Convert Loguru's exception record to text without depending on Loguru types."""

    if exception is None:
        return None
    try:
        return _redact_text("".join(traceback.format_exception(exception.type, exception.value, exception.traceback)))
    except (AttributeError, TypeError):
        return _redact_text(str(exception))


def _loguru_json_format(record: Mapping[str, Any]) -> str:
    """Format direct Loguru records using the same JSONL schema as stdlib."""

    extra = record.get("extra")
    fields = dict(_LOG_CONTEXT.get())
    if isinstance(extra, Mapping):
        bound = extra.get("_worldfoundry_fields")
        if isinstance(bound, Mapping):
            fields.update(bound)
        for key, value in extra.items():
            if key != "_worldfoundry_fields":
                fields.setdefault(str(key), value)
    record_time = record.get("time")
    if isinstance(record_time, datetime):
        timestamp = record_time
    else:
        timestamp = datetime.now(timezone.utc)
    level = record.get("level")
    level_name = getattr(level, "name", str(level))
    process = record.get("process")
    payload = _json_event_payload(
        level=level_name,
        logger_name=fields.pop("_worldfoundry_logger_name", str(record.get("name", ""))),
        message=str(record.get("message", "")),
        fields=fields,
        exception=_loguru_exception_text(record.get("exception")),
        timestamp=timestamp,
        pid=getattr(process, "id", os.getpid()),
    )
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")
    # JSON escapes preserve literal markup through Loguru color-tag parsing.
    # A Loguru callable formatter returns a format template, not final text.
    # Escape JSON braces so keys such as ``benchmark_id`` are not interpreted
    # as replacement fields by ``str.format_map`` inside the sink.
    return rendered.replace("{", "{{").replace("}", "}}") + "\n"


def _loguru_text_format(record: dict[str, Any]) -> str:
    """Render exception text through the same redaction policy as JSON sinks."""
    record["_worldfoundry_exception_text"] = _loguru_exception_text(record.get("exception")) or ""
    # Callable formats do not append Loguru's unredacted exception rendering.
    # Store the text outside `extra` so it cannot leak into structured fields.
    return _LOGURU_FORMAT + "\n{_worldfoundry_exception_text}"


def _redact_loguru_record(record: dict[str, Any]) -> bool:
    """Apply the same redaction policy to direct Loguru sink records."""

    record["message"] = _redact_text(str(record.get("message", "")))
    extra = record.get("extra")
    if isinstance(extra, Mapping):
        record["extra"] = _json_safe(extra)
        if "_worldfoundry_logger_name" in extra:
            record["name"] = _redact_text(str(extra["_worldfoundry_logger_name"]))
    return True


def _make_text_formatter() -> logging.Formatter:
    """Stdlib text formatter whose msec prefix matches the loguru sink (``.`` not ``,``)."""
    formatter = _RedactingTextFormatter(_STDLIB_FORMAT, datefmt=_DATEFMT)
    # Render sub-second as ``.123`` (not the stdlib default ``,123``) so the
    # prefix matches the loguru sink byte-for-byte.
    formatter.default_msec_format = "%s.%03d"  # type: ignore[assignment]
    return formatter


def _rank_suffix() -> str:
    """Append ``_rankN`` to the log file in distributed/spawn processes."""
    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("LOCAL_RANK")
    if rank is None:
        return ""
    try:
        int(rank)
    except ValueError:
        return ""
    return f"_rank{rank}"


def _apply_distributed_logger(level: int) -> None:
    """Reparent the rank-aware ``distributed_logger`` onto the root pipeline.

    Lazily imported so this module never eagerly pulls ``torch`` / the
    distributed stack; if it is unavailable we simply skip reparenting.
    """
    try:
        from worldfoundry.core.distributed.runtime.logging import distributed_logger
    except Exception:
        return
    distributed_logger.handlers.clear()
    distributed_logger.propagate = True
    distributed_logger.setLevel(level)


def _configure_stdlib(
    *,
    level: int,
    log_file: str | os.PathLike | None,
    json_file: bool,
    rotation: str,
    retention: str,
) -> None:
    """Replace root handlers with stderr + optional rotating file; no loguru in this path."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    text_formatter = _make_text_formatter()
    file_formatter: logging.Formatter = _JsonFormatter() if json_file else text_formatter

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(text_formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(os.path.expanduser(str(log_file)))
        suffix = _rank_suffix()
        if suffix and suffix not in path.name:
            path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=_parse_bytes(rotation, 10 * 1024 * 1024),
            backupCount=_parse_retention(retention, 8),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)


def _configure_loguru(
    *,
    level: int,
    level_name: str,
    log_file: str | os.PathLike | None,
    json_file: bool,
    rotation: str,
    retention: str,
    colorize: bool | None,
) -> None:
    """Reset loguru sinks and intercept stdlib so vendored loggers share one pipeline."""
    _loguru_logger.remove()  # type: ignore[union-attr]
    # Bridge stdlib logging into loguru so vendored / sequence-parallel loggers
    # share the same sinks as the facade.
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler = _StdlibToLoguruHandler(level=logging.DEBUG)
    root.addHandler(handler)

    console_color = True if colorize is None else colorize
    _loguru_logger.add(  # type: ignore[union-attr]
        sys.stderr,
        level=level_name,
        format=_loguru_text_format,
        colorize=console_color,
        filter=_redact_loguru_record,
    )
    if log_file is not None:
        path = Path(os.path.expanduser(str(log_file)))
        suffix = _rank_suffix()
        if suffix and suffix not in path.name:
            path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(  # type: ignore[union-attr]
            str(path),
            level=level_name,
            format=_loguru_json_format if json_file else _loguru_text_format,
            colorize=False,
            rotation=rotation,
            retention=retention,
            serialize=False,
            enqueue=True,
            filter=_redact_loguru_record,
        )


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | os.PathLike | None = None,
    json: bool | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    colorize: bool | None = None,
    force: bool = False,
) -> None:
    """Configure WorldFoundry's unified logging pipeline.

    Idempotent: a second call without ``force=True`` is a no-op, so it is safe
    to invoke from every framework entry point (CLI, model-runtime worker, …)
    without duplicating handlers/sinks.

    Args:
        level: Threshold for *every* log source. Accepts a name
            (``"DEBUG"``/``"INFO"``/…, case-insensitive; ``"WARN"``/``"TRACE"``
            accepted as aliases) or an ``int`` level. ``None`` falls back to the
            ``WORLDFOUNDRY_LOG_LEVEL`` env var, then INFO.
        log_file: Optional rotating file sink. ``None`` falls back to the
            ``WORLDFOUNDRY_LOG_FILE`` env var, else no file sink.
        json: When true the file sink emits JSON Lines (the console stays
            colored text). ``None`` falls back to ``WORLDFOUNDRY_LOG_JSON``.
        rotation: Size threshold for file rotation (e.g. ``"10 MB"``).
        retention: How many rotated files to keep (e.g. ``"1 week"`` → 7
            backups; only the count is honoured by the stdlib path).
        colorize: Override console colorization (loguru path only).
        force: Reconfigure even if already configured.
    """
    global _CONFIGURED, _CONFIGURED_LEVEL

    # Serialize concurrent first calls: the idempotence check and the
    # remove()/add() sink surgery below are not atomic on their own, and two
    # racing configurations can otherwise duplicate sinks.
    with _CONFIGURE_LOCK:
        if _CONFIGURED and not force:
            return

        _bind_log_context_from_environment()

        # Child processes inherit these from the parent CLI's env; an explicit
        # argument always wins, ``None`` means "consult the env then the default".
        if level is None:
            level = os.environ.get(_LOG_LEVEL_ENV, "INFO")
        if log_file is None:
            log_file = os.environ.get(_LOG_FILE_ENV)
        if json is None:
            json = os.environ.get(_LOG_JSON_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

        resolved = _resolve_level(level, logging.INFO)
        level_name = logging.getLevelName(resolved)

        # Prevent the sequence-parallel stack from reconfiguring the root logger
        # later; ``sp.envs`` reads this lazily at import time.
        os.environ.setdefault(_SP_CONFIGURE_ENV, "0")

        if _LOGURU_AVAILABLE:
            _configure_loguru(
                level=resolved,
                level_name=level_name,
                log_file=log_file,
                json_file=json,
                rotation=rotation,
                retention=retention,
                colorize=colorize,
            )
        else:  # pragma: no cover - depends on loguru being uninstalled.
            _configure_stdlib(
                level=resolved,
                log_file=log_file,
                json_file=json,
                rotation=rotation,
                retention=retention,
            )

        _apply_distributed_logger(resolved)

        _CONFIGURED = True
        _CONFIGURED_LEVEL = resolved


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run in this process."""
    return _CONFIGURED
