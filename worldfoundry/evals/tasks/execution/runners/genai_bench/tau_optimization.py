# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is adapted for WorldFoundry from the tau optimization helper used by
# t2v_metrics. It is intentionally dependency-light and contains no benchmark
# data loading or model inference code.

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing


class TauSufficientStats:
    """Sufficient statistics for tau variants with metric-score tie handling."""

    def __init__(
        self,
        con: int = 0,
        dis: int = 0,
        ties_human: int = 0,
        ties_metric: int = 0,
        ties_both: int = 0,
    ) -> None:
        self.con = con
        self.dis = dis
        self.ties_human = ties_human
        self.ties_metric = ties_metric
        self.ties_both = ties_both
        self.num_pairs = con + dis + ties_human + ties_metric + ties_both

    def tau_23(self) -> float:
        if self.num_pairs == 0:
            return 0.0
        return (self.con + self.ties_both - self.dis - self.ties_human - self.ties_metric) / self.num_pairs

    def acc_23(self) -> float:
        if self.num_pairs == 0:
            return 0.0
        return (self.con + self.ties_both) / self.num_pairs

    def acc_ignore_tie(self) -> float:
        denominator = self.num_pairs - self.ties_human
        if denominator == 0:
            return 1.0
        return self.con / denominator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TauSufficientStats):
            return False
        return (
            self.con,
            self.dis,
            self.ties_human,
            self.ties_metric,
            self.ties_both,
        ) == (
            other.con,
            other.dis,
            other.ties_human,
            other.ties_metric,
            other.ties_both,
        )

    def __iadd__(self, other: TauSufficientStats) -> TauSufficientStats:
        self.con += other.con
        self.dis += other.dis
        self.ties_human += other.ties_human
        self.ties_metric += other.ties_metric
        self.ties_both += other.ties_both
        self.num_pairs += other.num_pairs
        return self

    def __isub__(self, other: TauSufficientStats) -> TauSufficientStats:
        self.con -= other.con
        self.dis -= other.dis
        self.ties_human -= other.ties_human
        self.ties_metric -= other.ties_metric
        self.ties_both -= other.ties_both
        self.num_pairs -= other.num_pairs
        return self

    def __repr__(self) -> str:
        return (
            "("
            + ",".join(
                [
                    f"C={self.con}",
                    f"D={self.dis}",
                    f"T_h={self.ties_human}",
                    f"T_m={self.ties_metric}",
                    f"T_hm={self.ties_both}",
                ]
            )
            + ")"
        )


@dataclasses.dataclass(frozen=True)
class TauOptimizationResult:
    thresholds: list[float]
    taus: list[float]
    best_threshold: float
    best_tau: float


class _RankedPair:
    def __init__(self, h1: float, h2: float, m1: float, m2: float, row: int) -> None:
        self.row = row
        self.diff = abs(m1 - m2)
        if h1 == h2 and m1 == m2:
            self.stats = TauSufficientStats(ties_both=1)
        elif h1 == h2:
            self.stats = TauSufficientStats(ties_human=1)
        elif m1 == m2:
            self.stats = TauSufficientStats(ties_metric=1)
        elif (h1 > h2 and m1 > m2) or (h1 < h2 and m1 < m2):
            self.stats = TauSufficientStats(con=1)
        else:
            self.stats = TauSufficientStats(dis=1)

        self.tie_stats = TauSufficientStats(ties_both=1) if h1 == h2 else TauSufficientStats(ties_metric=1)


def _enumerate_pairs(
    human_scores: np.ndarray,
    metric_scores: np.ndarray,
    sample_rate: float,
    filter_nones: bool = True,
) -> tuple[list[_RankedPair], set[int]]:
    pairs: list[_RankedPair] = []
    rows: set[int] = set()
    for row, (human_row, metric_row) in enumerate(zip(human_scores, metric_scores, strict=True)):
        if filter_nones:
            filtered = [
                (human_value, metric_value)
                for human_value, metric_value in zip(human_row, metric_row, strict=True)
                if human_value is not None and metric_value is not None
            ]
            if not filtered:
                continue
            human_row, metric_row = zip(*filtered, strict=True)
        for left in range(len(human_row)):
            for right in range(left + 1, len(human_row)):
                if sample_rate == 1.0 or np.random.random() <= sample_rate:
                    pairs.append(
                        _RankedPair(human_row[left], human_row[right], metric_row[left], metric_row[right], row)
                    )
                    rows.add(row)
    return pairs, rows


def tau_optimization(
    metric_scores: numpy.typing.ArrayLike,
    human_scores: numpy.typing.ArrayLike,
    tau_fn: Callable[[TauSufficientStats], float],
    sample_rate: float = 1.0,
) -> TauOptimizationResult:
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError(f"`sample_rate` must be in the range (0, 1]. Found {sample_rate}")

    metric_scores = np.array(metric_scores)
    human_scores = np.array(human_scores)
    if metric_scores.ndim == 1:
        metric_scores = np.expand_dims(metric_scores, 0)
    if human_scores.ndim == 1:
        human_scores = np.expand_dims(human_scores, 0)
    if human_scores.shape != metric_scores.shape:
        raise ValueError("Human and metric scores must have the same shape.")

    pairs, rows = _enumerate_pairs(human_scores, metric_scores, sample_rate)
    if not pairs or not rows:
        return TauOptimizationResult([0.0], [0.0], 0.0, 0.0)

    row_to_stats = {row: TauSufficientStats() for row in rows}
    for pair in pairs:
        row_to_stats[pair.row] += pair.stats

    thresholds = [0.0]
    total_tau = sum(tau_fn(stats) for stats in row_to_stats.values())
    taus = [total_tau / len(rows)]
    pairs.sort(key=lambda pair: pair.diff)
    for pair in pairs:
        total_tau -= tau_fn(row_to_stats[pair.row])
        row_to_stats[pair.row] -= pair.stats
        row_to_stats[pair.row] += pair.tie_stats
        total_tau += tau_fn(row_to_stats[pair.row])
        overall_tau = total_tau / len(rows)
        if thresholds[-1] == pair.diff:
            taus[-1] = overall_tau
        else:
            thresholds.append(pair.diff)
            taus.append(overall_tau)

    max_index = int(np.nanargmax(taus))
    return TauOptimizationResult(thresholds, taus, thresholds[max_index], taus[max_index])
