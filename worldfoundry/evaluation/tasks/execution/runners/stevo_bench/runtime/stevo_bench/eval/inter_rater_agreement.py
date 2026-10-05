#!/usr/bin/env python3
# eval/inter_rater_agreement.py
"""
Inter-rater agreement across a user-specified set of annotators.

Computes per-metric agreement statistics from a summary.json produced by
the eval pipeline.  Seven metrics are supported:

    state_evol  occlusion_done  trigger_applied  physical_inaccuracy
    control_success  task_success

control_success and task_success are derived fields written by
compute_level_stats.py; run that script first.

For each metric, only tasks where ALL listed raters provided a value are
included (the valid set can differ per metric).

Annotator 'llm' reads from the task's top-level fields (task[metric]).
VLM judge slugs (e.g. 'gemini__gemini-3-1-pro-preview', 'openai__gpt-4o')
  read from task["llm_evals"][slug][metric].
All other names read from task["annotations"][name][metric].

Special pseudo-rater 'human_aggregate':
    Must be used with exactly one other rater: 'llm'.
    Aggregates every human annotator found in task["annotations"] via
    strict majority vote (True if >50% said True; ties are excluded).

Agreement is measured as simple percent agreement (fraction of tasks where
both raters gave the same label), averaged over all relevant rater pairs.

Output table (metrics as columns):

    human-only / VLM-only mode  (llm not listed):
        pairwise - percent   — avg pairwise agreement among all rater pairs
        pairwise - ROC/AUC   — avg symmetric ROC-AUC among all rater pairs

    LLM-included mode  (llm listed):
        n_tasks         — tasks included per metric
        llm_vs_human    — avg pairwise agreement between LLM and each human
        human_vs_human  — avg pairwise agreement among all human pairs
                          (absent when only one human / human_aggregate used)

Usage:
    python -m eval.inter_rater_agreement runs/veo_run --raters llm rater1 rater2

    python -m eval.inter_rater_agreement runs/veo_run --raters rater1 rater2 rater3

    python -m eval.inter_rater_agreement runs/veo_run --raters llm human_aggregate

    python -m eval.inter_rater_agreement runs/veo_run \\
        --raters gemini__gemini-3-1-pro-preview openai__gpt-4o
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# METRICS = ["state_evol", "physical_inaccuracy", "task_success",
#            "occlusion_done", "trigger_applied", "control_success"]

METRICS = ["physical_inaccuracy",]

# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def _get_rater_value(task: dict, rater: str, metric: str) -> Optional[int]:
    """Return 0, 1, or None for one rater × one metric on one task."""
    if rater == "human_aggregate":
        votes = [
            int(bool(ann[metric]))
            for ann in task.get("annotations", {}).values()
            if metric in ann
        ]
        if not votes:
            return None
        s, n = sum(votes), len(votes)
        if s * 2 == n:   # exact tie — excluded
            return None
        return int(s * 2 > n)  # strict majority
    if rater == "llm":
        val = task.get(metric)
        return int(bool(val)) if val is not None else None
    # VLM judge slug (e.g. "gemini__gemini-3-1-pro-preview"): check llm_evals first
    llm_evals = task.get("llm_evals", {})
    if rater in llm_evals:
        val = llm_evals[rater].get(metric)
        return int(bool(val)) if val is not None else None
    # Fall back to human annotations
    ann = task.get("annotations", {}).get(rater, {})
    val = ann.get(metric)
    return int(bool(val)) if val is not None else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(summary_path: Path, raters: list) -> tuple:
    """Return (data, task_counts, tie_counts, task_ids).

    data[metric][rater] = list of 0/1 for tasks where every rater provided a value.
    task_counts[metric] = length of those lists.
    tie_counts[metric]  = tasks skipped because human_aggregate was tied.
    task_ids[metric]    = task_id strings in the same order as the data lists.
    """
    tasks = json.loads(summary_path.read_text(encoding="utf-8"))["tasks"]
    data = {m: {r: [] for r in raters} for m in METRICS}
    task_counts = {m: 0 for m in METRICS}
    tie_counts  = {m: 0 for m in METRICS}
    task_ids    = {m: [] for m in METRICS}
    has_aggregate = "human_aggregate" in raters

    for task in tasks:
        for metric in METRICS:
            # Detect and count human_aggregate ties before computing other values.
            if has_aggregate:
                votes = [
                    int(bool(ann[metric]))
                    for ann in task.get("annotations", {}).values()
                    if metric in ann
                ]
                if votes and sum(votes) * 2 == len(votes):
                    tie_counts[metric] += 1
                    continue

            vals = {r: _get_rater_value(task, r, metric) for r in raters}
            if any(v is None for v in vals.values()):
                continue
            for r in raters:
                data[metric][r].append(vals[r])
            task_ids[metric].append(task.get("task_id", ""))
            task_counts[metric] += 1

    return data, task_counts, tie_counts, task_ids


# ---------------------------------------------------------------------------
# Agreement metric
# ---------------------------------------------------------------------------

def pairwise_agreement(a: list, b: list) -> float:
    """Fraction of tasks where rater A and rater B gave the same label."""
    n = len(a)
    if n == 0:
        return float("nan")
    return float(np.mean(np.array(a) == np.array(b)))


def _avg_agreement(data_m: dict, rater_pairs: list) -> float:
    """Average pairwise agreement over a list of (r1, r2) pairs for one metric."""
    scores = [pairwise_agreement(data_m[r1], data_m[r2]) for r1, r2 in rater_pairs]
    return round(float(np.nanmean(scores)), 3)


def _roc_auc_pair(pred: list, gt: list) -> float:
    """ROC-AUC treating gt as ground truth and pred as binary classifier output.

    With hard (yes/no) predictions the ROC curve has only one interior point, so
    AUC = (hit_rate + correct_rejection_rate) / 2  =  (TPR + TNR) / 2.
    Returns nan when gt contains only one class.
    """
    p = np.array(pred)
    g = np.array(gt)
    pos, neg = g == 1, g == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = float((p[pos] == 1).mean())
    tnr = float((p[neg] == 0).mean())
    return (tpr + tnr) / 2.0


def _avg_roc_auc(data_m: dict, llm_human_pairs: list) -> float:
    """Average ROC-AUC over (llm, human) pairs for one metric."""
    scores = [_roc_auc_pair(data_m["llm"], data_m[h]) for _, h in llm_human_pairs]
    return round(float(np.nanmean(scores)), 3)


def _avg_roc_auc_symmetric(data_m: dict, human_pairs: list) -> float:
    """Average ROC-AUC over human pairs, averaging both directions per pair.

    Since neither human is designated ground truth, each pair (h1, h2) contributes
    two scores: h1 as predictor/h2 as gt, and h2 as predictor/h1 as gt.
    """
    scores = []
    for h1, h2 in human_pairs:
        scores.append(_roc_auc_pair(data_m[h1], data_m[h2]))
        scores.append(_roc_auc_pair(data_m[h2], data_m[h1]))
    return round(float(np.nanmean(scores)), 3)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_agreement(summary_path: Path, raters: list) -> tuple:
    """Return (df, tie_counts, disagreements, distributions).

    df            — agreement table as a DataFrame.
    tie_counts    — per-metric count of tasks skipped due to human_aggregate ties.
    disagreements — dict metric → list of (task_id, {rater: val}) for tasks where
                    raters disagreed.
    distributions — dict metric → {rater: fraction_positive (0.0–1.0)}.
    """
    data, task_counts, tie_counts, task_ids = load_data(summary_path, raters)
    has_llm = "llm" in raters
    humans = [r for r in raters if r != "llm"]

    rows: dict = {}

    if not has_llm:
        # Pairwise mode (human-only or VLM-only).
        pairs = list(combinations(humans, 2))
        rows["pairwise - percent"] = {
            m: _avg_agreement(data[m], pairs) for m in METRICS
        }
        rows["pairwise - ROC/AUC"] = {
            m: _avg_roc_auc_symmetric(data[m], pairs) for m in METRICS
        }
    else:
        # LLM-included mode: full table.
        rows["n_tasks"] = {m: task_counts[m] for m in METRICS}

        if humans:
            llm_human_pairs = [("llm", h) for h in humans]
            rows["llm_vs_human - percent"] = {
                m: _avg_agreement(data[m], llm_human_pairs) for m in METRICS
            }
            rows["llm_vs_human - ROC/AUC"] = {
                m: _avg_roc_auc(data[m], llm_human_pairs) for m in METRICS
            }

        if len(humans) >= 2:
            human_pairs = list(combinations(humans, 2))
            rows["human_vs_human - percent"] = {
                m: _avg_agreement(data[m], human_pairs) for m in METRICS
            }
            rows["human_vs_human - ROC/AUC"] = {
                m: _avg_roc_auc_symmetric(data[m], human_pairs) for m in METRICS
            }

    # Disagreements: tasks where not all raters gave the same label.
    disagreements: dict = {}
    for metric in METRICS:
        disag = []
        for i, tid in enumerate(task_ids[metric]):
            vals = {r: data[metric][r][i] for r in raters}
            if len(set(vals.values())) > 1:
                disag.append((tid, vals))
        disagreements[metric] = disag

    # Label distributions: fraction of positive (True) labels per rater.
    distributions: dict = {}
    for metric in METRICS:
        distributions[metric] = {}
        n = task_counts[metric]
        for r in raters:
            distributions[metric][r] = (
                sum(data[metric][r]) / n if n > 0 else float("nan")
            )

    return pd.DataFrame(rows).T[METRICS], tie_counts, disagreements, distributions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inter-rater agreement from a STEVO-Bench summary.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=Path, help="Run folder containing summary.json")
    parser.add_argument(
        "--raters", nargs="+", required=True, metavar="NAME",
        help="Annotator names, e.g.  --raters llm rater1 rater2",
    )
    args = parser.parse_args()

    if "human_aggregate" in args.raters:
        others = [r for r in args.raters if r != "human_aggregate"]
        if others != ["llm"]:
            parser.error("'human_aggregate' can only be used with exactly one other rater: 'llm'")

    summary_json = args.run_dir / "summary.json"
    if not summary_json.exists():
        parser.error(f"summary.json not found in {args.run_dir}")

    df, tie_counts, disagreements, distributions = compute_agreement(summary_json, args.raters)

    def _fmt(val):
        if isinstance(val, float) and not np.isnan(val) and val == int(val):
            return str(int(val))
        return f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else str(val)

    print(df.to_string(formatters={m: _fmt for m in df.columns}))
    print()
    print("percent = fraction of tasks where both raters gave the same label, averaged over pairs.")

    # Label distributions
    print()
    print("Label distribution (fraction flagged True):")
    for metric in METRICS:
        print(f"  {metric}:")
        for rater, frac in distributions[metric].items():
            pct = f"{frac * 100:.1f}%" if not np.isnan(frac) else "n/a"
            print(f"    {rater}: {pct}")

    # Disagreement task listing
    for metric in METRICS:
        disag = disagreements[metric]
        if disag:
            print()
            print(f"Disagreements on {metric} ({len(disag)} / {sum(len(v) for v in [disag])} tasks):")
            for tid, vals in disag:
                parts = ", ".join(
                    f"{r}={'True' if v else 'False'}" for r, v in vals.items()
                )
                print(f"  {tid}: {parts}")
        else:
            print()
            print(f"No disagreements on {metric}.")

    if "human_aggregate" in args.raters and any(tie_counts.values()):
        print()
        print("Tasks skipped due to human_aggregate tie:")
        for m in METRICS:
            if tie_counts[m]:
                print(f"  {m}: {tie_counts[m]}")


if __name__ == "__main__":
    main()
