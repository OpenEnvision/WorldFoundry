"""Correlation helpers for GenAI-Bench score alignment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import tau_optimization


def calc_pearson(metric1_scores: Any, metric2_scores: Any) -> float:
    left = np.asarray(metric1_scores, dtype=float)
    right = np.asarray(metric2_scores, dtype=float)
    if left.size < 2 or right.size < 2 or left.shape != right.shape:
        return 0.0
    value = 100 * np.corrcoef(left, right)[0, 1]
    return 0.0 if math.isnan(float(value)) else float(value)


def _matrix_sufficient_statistics(x: Any, y: Any, epsilon: float) -> tuple[int, int, int, int, int]:
    x = np.asarray(x)
    x1, x2 = np.meshgrid(x, x.T)
    x_diffs = x1 - x2
    x_is_tie = np.abs(x_diffs) <= epsilon
    x_diffs[x_is_tie] = 0.0

    y = np.asarray(y)
    y1, y2 = np.meshgrid(y, y.T)
    y_diffs = y1 - y2
    y_is_tie = y_diffs == 0.0

    n = len(y)
    num_pairs = n * (n - 1) // 2
    concordant = int((((x_diffs > 0) & (y_diffs > 0)) | ((x_diffs < 0) & (y_diffs < 0))).sum() / 2)
    tied_x_only = int((x_is_tie & ~y_is_tie).sum() / 2)
    tied_y_only = int((~x_is_tie & y_is_tie).sum() / 2)
    tied_both = int(((x_is_tie & y_is_tie).sum() - n) / 2)
    discordant = num_pairs - (concordant + tied_x_only + tied_y_only + tied_both)
    return concordant, discordant, tied_x_only, tied_y_only, tied_both


def kendall_variants(
    gold_scores: Any,
    metric_scores: Any,
    *,
    variant: str = "acc23",
    epsilon: float = 0.0,
) -> tuple[float, float]:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if epsilon > 0 and variant == "c":
        raise ValueError("non-zero epsilon with tau-c is not supported")

    x = np.asarray(metric_scores, dtype=float)
    y = np.asarray(gold_scores, dtype=float)
    if x.shape != y.shape or x.size < 2 or np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return float("nan"), 0.0

    concordant, discordant, tied_x_only, tied_y_only, tied_both = _matrix_sufficient_statistics(x, y, epsilon)
    size = y.size
    x_ties = tied_x_only + tied_both
    y_ties = tied_y_only + tied_both
    total = concordant + discordant + tied_x_only + tied_y_only + tied_both
    if total == 0:
        return float("nan"), 0.0
    if variant in {"b", "c"} and (x_ties == total or y_ties == total):
        return float("nan"), 0.0
    if variant == "b":
        tau = (concordant - discordant) / math.sqrt(total - x_ties) / math.sqrt(total - y_ties)
    elif variant == "c":
        minclasses = min(len(set(x.tolist())), len(set(y.tolist())))
        tau = 0.0 if minclasses <= 1 else 2 * (concordant - discordant) / (size**2 * (minclasses - 1) / minclasses)
    elif variant == "23":
        tau = (concordant + tied_both - discordant - tied_x_only - tied_y_only) / total
    elif variant == "acc23":
        tau = (concordant + tied_both) / total
    else:
        raise ValueError("variant must be one of 'b', 'c', '23', or 'acc23'")
    return float(tau), 0.0


def calc_metric(
    gold_scores: Any,
    metric_scores: Any,
    *,
    variant: str = "pairwise_acc_with_tie_optimization",
    sample_rate: float = 1.0,
) -> float | tuple[float, float]:
    gold = np.array(gold_scores)
    metric = np.array(metric_scores)
    if gold.shape != metric.shape:
        raise ValueError("gold_scores and metric_scores must have the same shape")
    if gold.ndim == 1:
        gold = gold.reshape(1, -1)
        metric = metric.reshape(1, -1)

    if variant == "pairwise_acc_with_tie_optimization":
        result = tau_optimization.tau_optimization(
            metric, gold, tau_optimization.TauSufficientStats.acc_23, sample_rate=sample_rate
        )
        return result.best_tau, result.best_threshold
    if variant == "pairwise_acc_ignore_tie":
        result = tau_optimization.tau_optimization(
            metric, gold, tau_optimization.TauSufficientStats.acc_ignore_tie, sample_rate=sample_rate
        )
        return result.taus[0], result.thresholds[0]
    if variant == "tau_with_tie_optimization":
        result = tau_optimization.tau_optimization(
            metric, gold, tau_optimization.TauSufficientStats.tau_23, sample_rate=sample_rate
        )
        return result.best_tau, result.best_threshold

    if variant not in {"tau_b", "tau_c"}:
        raise ValueError(f"unknown calc_metric variant: {variant}")
    kind = "b" if variant == "tau_b" else "c"
    taus = [
        kendall_variants(gold_row, metric_row, variant=kind)[0]
        for gold_row, metric_row in zip(gold, metric, strict=True)
    ]
    value = float(np.nanmean(np.asarray(taus, dtype=float))) if taus else 0.0
    return 0.0 if math.isnan(value) else value
