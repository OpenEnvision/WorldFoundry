"""Rank-aware logging helpers for distributed runtime code.

``print_rank_0`` avoids N-way log spam. ``print_per_rank`` is for
debug of a single collective mismatch. Not a metrics path — use
``metric_sync`` for reduced scalars.

Public surface: :func:`print_rank_0`, :func:`print_per_rank`,
:class:`DistributedLogger`, and the :data:`log` facade.
"""

from __future__ import annotations

import logging

import torch

# ──────────────────────────────────────────────────────────────────────────
# Process-local logger — one StreamHandler, never propagate into root spam
# ──────────────────────────────────────────────────────────────────────────


class DistributedLogger:
    """Lazy stdlib logger used when loguru is not installed."""

    _logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls, *, level: int = logging.INFO) -> logging.Logger:
        """Return the singleton logger, installing a StreamHandler on first use.

        ``propagate`` is off so rank-filtered records do not also hit the root
        logger and double-print.
        """

        if cls._logger is None:
            cls._logger = logging.getLogger("worldfoundry_distributed")
            cls._logger.setLevel(level)
            cls._logger.propagate = False
            cls._logger.handlers.clear()
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)
        return cls._logger


distributed_logger = DistributedLogger.get_logger()

try:  # Keep the shared distributed helpers usable in minimal model environments.
    from loguru import logger as _backend_logger
except ImportError:  # pragma: no cover - exercised in isolated runtime environments.
    _backend_logger = distributed_logger


# ──────────────────────────────────────────────────────────────────────────
# Rank-filtered print — rank 0 for progress, every rank for collective mismatch
# ──────────────────────────────────────────────────────────────────────────


def print_per_rank(message: object) -> None:
    """Emit one informational log record from every calling rank."""
    distributed_logger.info(message)


def print_rank_0(message: object) -> None:
    """Emit one informational record on rank zero, or in a non-distributed process."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            distributed_logger.info(message)
    else:
        distributed_logger.info(message)


class _RankAwareLog:
    """Loguru-compatible facade for model code that supports rank-zero filtering."""

    logger = _backend_logger

    @staticmethod
    def _enabled(rank0_only: bool) -> bool:
        """True unless the caller asked for rank-0-only and this rank is not 0."""

        return not rank0_only or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def _write(self, level: str, message, *args, rank0_only: bool = False, **kwargs) -> None:
        """Dispatch to loguru or the stdlib fallback after rank filtering.

        ``success`` is remapped to ``info`` because the stdlib logger has no
        such level. Loguru ``{}`` placeholders are pre-formatted so the
        fallback logger does not treat them as ``%``-style args.
        """

        if self._enabled(rank0_only):
            method = getattr(_backend_logger, "info" if level == "success" else level)
            if _backend_logger is distributed_logger:
                # Loguru call sites commonly use ``{}`` placeholders and keyword
                # options that the stdlib logger does not accept.
                try:
                    message = str(message).format(*args)
                    args = ()
                except (IndexError, KeyError, ValueError):
                    pass
                kwargs = {key: value for key, value in kwargs.items() if key in {"exc_info", "stack_info", "stacklevel", "extra"}}
            method(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        """Emit a debug record, honoring optional ``rank0_only``."""

        self._write("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        """Emit an info record, honoring optional ``rank0_only``."""

        self._write("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        """Emit a warning record, honoring optional ``rank0_only``."""

        self._write("warning", message, *args, **kwargs)

    warn = warning

    def error(self, message, *args, **kwargs):
        """Emit an error record, honoring optional ``rank0_only``."""

        self._write("error", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        """Emit a critical record, honoring optional ``rank0_only``."""

        self._write("critical", message, *args, **kwargs)

    def success(self, message, *args, **kwargs):
        """Emit a success/info record, honoring optional ``rank0_only``."""

        self._write("success", message, *args, **kwargs)


log = _RankAwareLog()


__all__ = ["DistributedLogger", "distributed_logger", "log", "print_per_rank", "print_rank_0"]
