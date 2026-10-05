"""Deterministic leaderboard ordering for known model ids.

The evaluator itself is model-agnostic: it consumes whatever generation results a
model wrote under ``models/<MODEL_ID>/``. This module only fixes the *display*
order of models that have already been evaluated, so that repeated report builds
produce byte-identical output.

IDs that are not listed here still appear in reports; ``report.build_report``
appends unknown ids in sorted order after the known ones.
"""

from __future__ import annotations


# Publication order of the models evaluated in the HarnessEval leaderboard.
MODEL_REGISTRY: tuple[str, ...] = (
    "seedance-2.0-standard",
)
