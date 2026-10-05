#!/usr/bin/env python3
# eval/model_ranking_agreement.py
"""
Pairwise model-ranking agreement across run directories.

For every pair of run directories (models A and B), for every task shared by
both, asks: does the judge agree on which model performed better on this task?

Scoring per task × model pair:
  full agreement             = 1.0
  one says tie, other doesn't = 0.5
  reversal (A vs B flipped)  = 0.0
  missing data               = excluded

Two rows are reported (one column per metric):
  llm_vs_human    — LLM ranking vs majority-vote human aggregate ranking
  human_vs_human  — pairwise agreement between individual human annotators

Usage:
    python -m eval.model_ranking_agreement runs/veo_run runs/sora_run runs/hy_run
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


METRICS = ["state_evol", "physical_inaccuracy", "task_success",
           "occlusion_done", "trigger_applied", "control_success"]


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def _load_tasks(run_dir: Path) -> Dict[str, dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    tasks = json.loads(summary_path.read_text(encoding="utf-8"))["tasks"]
    return {t["task_id"]: t for t in tasks if "task_id" in t}


def _llm_value(task: dict, metric: str) -> Optional[int]:
    val = task.get(metric)
    return int(bool(val)) if val is not None else None


def _human_aggregate(task: dict, metric: str) -> Optional[int]:
    """Strict majority vote. Returns None on tie or no data."""
    votes = [
        int(bool(ann[metric]))
        for ann in task.get("annotations", {}).values()
        if metric in ann
    ]
    if not votes:
        return None
    s, n = sum(votes), len(votes)
    if s * 2 == n:
        return None
    return int(s * 2 > n)


def _human_individual(task: dict, metric: str) -> Dict[str, int]:
    """Return {annotator_name: 0/1} for each annotator that has this metric."""
    return {
        name: int(bool(ann[metric]))
        for name, ann in task.get("annotations", {}).items()
        if metric in ann
    }


# ---------------------------------------------------------------------------
# Ranking comparison helpers
# ---------------------------------------------------------------------------

def _compare(val_a: Optional[int], val_b: Optional[int]) -> str:
    """Which model did better? Returns 'A_wins', 'B_wins', 'tie', or 'unknown'."""
    if val_a is None or val_b is None:
        return "unknown"
    if val_a == val_b:
        return "tie"
    return "A_wins" if val_a > val_b else "B_wins"


def _ranking_score(gt: str, pred: str) -> Optional[float]:
    """Agreement score between two ranking judgements.
    Returns None if either is unknown (task excluded from average).
    """
    if gt == "unknown" or pred == "unknown":
        return None
    if gt == pred:
        return 1.0
    if gt == "tie" or pred == "tie":
        return 0.5
    return 0.0  # reversal


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_model_ranking_agreement(
    run_dirs: List[Path],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
        llm_vs_human[metric]   — weighted ranking agreement, LLM vs human aggregate
        human_vs_human[metric] — weighted ranking agreement, pairs of human annotators
    """
    run_tasks: List[Tuple[str, Dict[str, dict]]] = [
        (rd.name, _load_tasks(rd)) for rd in run_dirs
    ]

    llm_scores:   Dict[str, List[float]] = {m: [] for m in METRICS}
    human_scores: Dict[str, List[float]] = {m: [] for m in METRICS}

    for (name_a, tasks_a), (name_b, tasks_b) in combinations(run_tasks, 2):
        shared_ids = sorted(set(tasks_a) & set(tasks_b))
        for task_id in shared_ids:
            ta, tb = tasks_a[task_id], tasks_b[task_id]
            for metric in METRICS:

                # ── LLM vs human_aggregate ────────────────────────────────────
                llm_cmp  = _compare(_llm_value(ta, metric),       _llm_value(tb, metric))
                hagg_cmp = _compare(_human_aggregate(ta, metric), _human_aggregate(tb, metric))
                score = _ranking_score(hagg_cmp, llm_cmp)
                if score is not None:
                    llm_scores[metric].append(score)

                # ── human_vs_human ────────────────────────────────────────────
                # Each annotator's ranking of A vs B on this task:
                #   compare(annotator's score on model A, annotator's score on model B)
                # Then check if two annotators agree on who won.
                ann_a = _human_individual(ta, metric)
                ann_b = _human_individual(tb, metric)
                shared_annotators = sorted(set(ann_a) & set(ann_b))
                for h1, h2 in combinations(shared_annotators, 2):
                    h1_cmp = _compare(ann_a[h1], ann_b[h1])
                    h2_cmp = _compare(ann_a[h2], ann_b[h2])
                    s = _ranking_score(h1_cmp, h2_cmp)
                    if s is not None:
                        human_scores[metric].append(s)

    def _mean(vals: List[float]) -> float:
        return round(float(np.nanmean(vals)), 3) if vals else float("nan")

    return (
        {m: _mean(llm_scores[m])   for m in METRICS},
        {m: _mean(human_scores[m]) for m in METRICS},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise model-ranking agreement across run directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="Two or more eval run directories each containing summary.json.",
    )
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("At least two run directories are required.")

    run_dirs = [Path(d).expanduser().resolve() for d in args.run_dirs]
    for rd in run_dirs:
        if not rd.exists():
            parser.error(f"Run directory not found: {rd}")

    llm_vs_human, human_vs_human = compute_model_ranking_agreement(run_dirs)

    df = pd.DataFrame(
        {"llm_vs_human": llm_vs_human, "human_vs_human": human_vs_human}
    ).T[METRICS]

    def _fmt(val):
        return f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else "  nan"

    print(f"Runs ({len(run_dirs)}): " + ", ".join(d.name for d in run_dirs))
    print()
    print(df.to_string(formatters={m: _fmt for m in df.columns}))
    print()
    print("Score: full agreement=1.0 | one-side tie=0.5 | reversal=0.0")


if __name__ == "__main__":
    main()
