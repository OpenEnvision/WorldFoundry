# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel logging formatters and distributed print gating.

Re-exports :class:`NewLineFormatter` and :func:`setup_for_distributed`
from :mod:`.formatter`. Multi-line SP logs keep a rank/prefix aligned
on wrapped lines; non-master ranks can have builtin ``print`` muted.

This package does not configure the WorldFoundry application logger.

Public surface: :class:`NewLineFormatter`, :func:`setup_for_distributed`.
"""

from .formatter import NewLineFormatter, setup_for_distributed

__all__ = [
    "NewLineFormatter",
    "setup_for_distributed",
]
