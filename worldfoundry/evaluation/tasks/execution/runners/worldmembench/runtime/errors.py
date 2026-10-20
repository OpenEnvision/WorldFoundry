# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Structured errors mapped to WorldMemBench result statuses."""


class WorldMemBenchError(Exception):
    """Base class for expected WorldMemBench errors."""


class DataInvalidError(WorldMemBenchError):
    """The benchmark item or its required annotations are invalid."""


class EvaluationError(WorldMemBenchError):
    """The evaluator failed independently of the generated model output."""


class VideoDecodeError(EvaluationError):
    """A required video frame could not be decoded."""


class NotApplicableError(WorldMemBenchError):
    """The requested metric is not applicable to the item."""
