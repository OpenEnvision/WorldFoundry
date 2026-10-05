# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/logging_utils/formatter.py
"""Formatters that prefix wrapped lines so multi-line SP logs stay aligned.

:class:`NewLineFormatter` repeats the logging prefix after each newline
so continuation lines line up in rank-prefixed output.
:func:`setup_for_distributed` replaces builtin ``print`` so non-master
ranks stay quiet unless ``force=True``.

This is a stdlib ``logging`` adapter, not the WorldFoundry loguru setup.

Public surface: :class:`NewLineFormatter`, :func:`setup_for_distributed`.
"""

import logging


# ──────────────────────────────────────────────────────────────────────────
# Continuation-line prefix — wrapped NCCL dumps stay aligned under the rank
# ──────────────────────────────────────────────────────────────────────────


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        """Same arguments as :class:`logging.Formatter`."""
        logging.Formatter.__init__(self, fmt, datefmt, style)

    def format(self, record):
        """Format ``record``, repeating the prefix after each embedded newline."""
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


# ──────────────────────────────────────────────────────────────────────────
# Builtin print gate — process-wide; only master (or force=True) emits
# ──────────────────────────────────────────────────────────────────────────


def setup_for_distributed(is_master):
    """Mute builtin ``print`` on non-master ranks unless ``force=True`` is passed."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        """Builtin replacement: emit only on master, or when ``force=True``."""
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
