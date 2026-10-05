#!/usr/bin/env python3
# eval/analyze_llm_human_agreement.py
"""
Analyze LLM-vs-human judge agreement for a given metric.

Outputs:
  1. Confusion matrix (LLM prediction vs human ground truth), with per-run FP/FN breakdown
  2. Pairwise model-ranking agreement (when >= 2 run dirs are given):
     for each pair of models, does the LLM judge agree with the human judge
     on which model performs better on each task?
     Scoring: full agreement = 1.0 | tie vs non-tie = 0.5 | reversal = 0.0

Usage (single run):
    python -m eval.analyze_llm_human_agreement --run_dir runs/my_run --metric occlusion_done
    python -m eval.analyze_llm_human_agreement --run_dir runs/my_run --metric physical_inaccuracy --level 3

Usage (multiple runs — combined confusion matrix + pairwise agreement):
    python -m eval.analyze_llm_human_agreement --run_dir runs/run_a runs/run_b runs/run_c --metric physical_inaccuracy
    python -m eval.analyze_llm_human_agreement --run_dir runs/run_a runs/run_b --metric state_evol --output cm.png

Supported metrics  (LLM value source                    → human value source)
  occlusion_done      task["occlusion_done"]              task["human_occlusion_done"]
  trigger_applied     task["trigger_applied"]             task["human_trigger_applied"]
  physical_inaccuracy task["physical_inaccuracy"]         task["human_physical_inaccuracy"]
  state_evol          task["accuracy"] == 1.0             task["human_state_evol"]

Only tasks where both the LLM value and the human value are non-null are included.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Supported metrics
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = ["occlusion_done", "trigger_applied", "physical_inaccuracy", "state_evol"]


def _get_llm_value(task: dict, metric: str) -> Optional[bool]:
    """Return the LLM judge boolean for a task entry, or None if absent."""
    if metric == "state_evol":
        acc = task.get("accuracy")
        return (acc == 1.0) if acc is not None else None
    val = task.get(metric)
    return bool(val) if val is not None else None


def _get_human_value(task: dict, metric: str) -> Optional[bool]:
    """Return the human judge boolean for a task entry, or None if absent."""
    key = "human_state_evol" if metric == "state_evol" else f"human_{metric}"
    val = task.get(key)
    return bool(val) if val is not None else None


# ---------------------------------------------------------------------------
# Confusion matrix  (rows = Human ground truth, cols = LLM prediction)
#
#              LLM=T   LLM=F
#  Human=T  |   TP   |   FN  |
#  Human=F  |   FP   |   TN  |
# ---------------------------------------------------------------------------

def compute_confusion(
    tasks: List[dict],
    metric: str,
    level: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        cm : 2×2 int ndarray  —  cm[human_idx, llm_idx]
             row/col 0 = True, row/col 1 = False
        n  : number of tasks included
    """
    cm = np.zeros((2, 2), dtype=int)
    included = 0
    for t in tasks:
        if level is not None and str(t.get("task_level", "")) != level:
            continue
        llm   = _get_llm_value(t, metric)
        human = _get_human_value(t, metric)
        if llm is None or human is None:
            continue
        row = 0 if human else 1
        col = 0 if llm   else 1
        cm[row, col] += 1
        included += 1
    return cm, included


def _cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = (cm[0, 0] + cm[1, 1]) / n
    pe = (
        (cm[0, :].sum() * cm[:, 0].sum() + cm[1, :].sum() * cm[:, 1].sum())
        / (n * n)
    )
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def _roc_auc(cm: np.ndarray) -> float:
    """
    ROC-AUC for a binary LLM classifier against human ground truth.

    With binary (hard) predictions the ROC curve has only one interior point,
    so AUC = (TPR + TNR) / 2  (equivalent to balanced accuracy).

    cm layout:  cm[0,0]=TP  cm[0,1]=FN
                cm[1,0]=FP  cm[1,1]=TN
    """
    pos = cm[0, :].sum()   # all human-positive cases  (TP + FN)
    neg = cm[1, :].sum()   # all human-negative cases  (FP + TN)
    if pos == 0 or neg == 0:
        return float("nan")
    tpr = cm[0, 0] / pos   # sensitivity / recall
    tnr = cm[1, 1] / neg   # specificity
    return (tpr + tnr) / 2.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    metric: str,
    n: int,
    output_path: Path,
    level: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    agreement = (cm[0, 0] + cm[1, 1]) / n if n > 0 else float("nan")
    kappa = _cohen_kappa(cm)

    cell_colors = [
        ["#1a3a1a", "#3a1a1a"],  # Human=T row: TP green, FN red
        ["#3a1a1a", "#1a3a1a"],  # Human=F row: FP red, TN green
    ]
    labels = ["True (T)", "False (F)"]

    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_colors[i][j], zorder=0,
            ))
            count = cm[i, j]
            pct   = 100.0 * count / n if n > 0 else 0.0
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="#c9d1d9", zorder=1,
            )

    for v in [-0.5, 0.5, 1.5]:
        ax.axhline(v, color="#30363d", linewidth=1.2, zorder=2)
        ax.axvline(v, color="#30363d", linewidth=1.2, zorder=2)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#c9d1d9", fontsize=11)
    ax.set_yticklabels(labels, color="#c9d1d9", fontsize=11)
    ax.set_xlabel("LLM Judge", color="#58a6ff", fontsize=12, labelpad=8)
    ax.set_ylabel("Human Judge", color="#58a6ff", fontsize=12, labelpad=8)
    ax.tick_params(colors="#8b949e", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    level_str = f" · Level {level}" if level else ""
    main_title = title or f"Confusion Matrix: {metric}{level_str}"
    subtitle = f"N = {n}  |  Agreement = {agreement:.1%}  |  κ = {kappa:.3f}"

    fig.suptitle(main_title, color="#c9d1d9", fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(subtitle, color="#8b949e", fontsize=10, pad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Pairwise model-ranking agreement
# ---------------------------------------------------------------------------

def _compare_pairwise(val_a: Optional[bool], val_b: Optional[bool]) -> str:
    """Return 'A_wins', 'B_wins', 'tie', or 'unknown'."""
    if val_a is None or val_b is None:
        return "unknown"
    if val_a == val_b:
        return "tie"
    return "A_wins" if val_a > val_b else "B_wins"  # True > False


def _pairwise_agreement_score(gt_cmp: str, llm_cmp: str) -> Optional[float]:
    """
    1.0 = full agreement, 0.5 = one is tie and the other is not, 0.0 = reversal.
    None if either comparison is unknown (missing human or LLM label).
    """
    if gt_cmp == "unknown" or llm_cmp == "unknown":
        return None
    if gt_cmp == llm_cmp:
        return 1.0
    if gt_cmp == "tie" or llm_cmp == "tie":
        return 0.5
    return 0.0  # A_wins vs B_wins


def compute_pairwise_agreement(
    tasks_per_dir: List[Tuple[Path, List[dict]]],
    metric: str,
    level: Optional[str] = None,
) -> Dict:
    """
    For every pair of run dirs, compare LLM and human judge rankings across all
    shared tasks and compute weighted agreement scores.

    Returns a dict with keys:
        'overall'  : aggregate stats across all pairs
        'by_pair'  : per-pair stats keyed by "<run_a> vs <run_b>"
    """
    # Build task_id -> run_name -> {llm, human}
    task_map: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for run_dir, tasks in tasks_per_dir:
        for t in tasks:
            if level is not None and str(t.get("task_level", "")) != level:
                continue
            task_id = t.get("task_id")
            if not task_id:
                continue
            task_map[task_id][run_dir.name] = {
                "llm":   _get_llm_value(t, metric),
                "human": _get_human_value(t, metric),
            }

    run_names = [d.name for d, _ in tasks_per_dir]
    pairs = [
        (run_names[i], run_names[j])
        for i in range(len(run_names))
        for j in range(i + 1, len(run_names))
    ]

    overall: Dict = {"total_score": 0.0, "count": 0, "full": 0, "partial": 0, "reversal": 0, "unknown": 0}
    by_pair: Dict[str, Dict] = {}

    for name_a, name_b in pairs:
        stats: Dict = {
            "total_score": 0.0, "count": 0,
            "full": 0, "partial": 0, "reversal": 0, "unknown": 0,
            "reversals": [],
        }

        for task_id, run_data in task_map.items():
            if name_a not in run_data or name_b not in run_data:
                continue
            data_a = run_data[name_a]
            data_b = run_data[name_b]

            gt_cmp  = _compare_pairwise(data_a["human"], data_b["human"])
            llm_cmp = _compare_pairwise(data_a["llm"],   data_b["llm"])
            score   = _pairwise_agreement_score(gt_cmp, llm_cmp)

            if score is None:
                stats["unknown"]  += 1
                overall["unknown"] += 1
            else:
                stats["total_score"]  += score
                stats["count"]        += 1
                overall["total_score"] += score
                overall["count"]       += 1
                if score == 1.0:
                    stats["full"]   += 1;  overall["full"]   += 1
                elif score == 0.5:
                    stats["partial"] += 1; overall["partial"] += 1
                else:
                    stats["reversal"] += 1; overall["reversal"] += 1
                    stats["reversals"].append({"task_id": task_id, "gt": gt_cmp, "llm": llm_cmp})

        stats["weighted_agreement"] = (
            stats["total_score"] / stats["count"] if stats["count"] > 0 else float("nan")
        )
        by_pair[f"{name_a} vs {name_b}"] = stats

    overall["weighted_agreement"] = (
        overall["total_score"] / overall["count"] if overall["count"] > 0 else float("nan")
    )
    return {"overall": overall, "by_pair": by_pair}


def print_pairwise_agreement(analysis: Dict, metric: str, level: Optional[str] = None) -> None:
    level_str = f"  (level {level})" if level else ""
    print(f"\n{'='*70}")
    print(f"PAIRWISE MODEL-RANKING AGREEMENT: {metric}{level_str}")
    print(f"{'='*70}")
    print("Scoring: full agreement = 1.0  |  tie vs non-tie = 0.5  |  reversal = 0.0")
    print("(For each task: does the LLM judge agree with the human judge on which")
    print(" model performs better?)\n")

    overall = analysis["overall"]
    print(
        f"Overall  N={overall['count']}  "
        f"weighted agreement = {overall['weighted_agreement']:.3f}  "
        f"full={overall['full']}  partial={overall['partial']}  "
        f"reversal={overall['reversal']}  unknown={overall['unknown']}"
    )

    sorted_pairs = sorted(
        analysis["by_pair"].items(),
        key=lambda kv: kv[1]["weighted_agreement"] if not np.isnan(kv[1]["weighted_agreement"]) else -1,
        reverse=True,
    )

    col_w = max((len(p) for p, _ in sorted_pairs), default=20)
    print(f"\n  {'Pair':{col_w}s}   {'N':>5}   {'Agr':>6}   {'Full':>5}   {'Part':>5}   {'Rev':>5}")
    print(f"  {'-'*col_w}   {'-----'}   {'------'}   {'-----'}   {'-----'}   {'-----'}")
    for pair, stats in sorted_pairs:
        agr = stats["weighted_agreement"]
        agr_str = f"{agr:.3f}" if not np.isnan(agr) else "  N/A"
        print(
            f"  {pair:{col_w}s}   {stats['count']:>5}   {agr_str:>6}"
            f"   {stats['full']:>5}   {stats['partial']:>5}   {stats['reversal']:>5}"
        )

    # Show reversal examples for the worst-performing pair
    if sorted_pairs:
        worst_pair, worst_stats = sorted_pairs[-1]
        if worst_stats["reversals"]:
            print(f"\n  Reversal examples for worst pair ({worst_pair}):")
            for rev in worst_stats["reversals"][:3]:
                print(f"    - {rev['task_id']}: Human={rev['gt']}, LLM={rev['llm']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM-vs-human judge agreement for a given metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir", required=True, nargs="+",
        help="One or more eval run directories each containing summary.json. "
             "If multiple are given, tasks are pooled for the confusion matrix "
             "and pairwise ranking agreement is computed across all pairs of runs.",
    )
    parser.add_argument(
        "--metric", required=True, choices=SUPPORTED_METRICS,
        help="Metric name. One of: " + ", ".join(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--level", default=None,
        help="Restrict to a specific task level (e.g. 1, 2, …). Default: all levels.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path. Defaults to <run_dir>/confusion_<metric>[_level<N>].png "
             "for a single run, or ./confusion_<metric>[_level<N>].png for multiple runs.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom plot title (overrides the auto-generated title).",
    )
    args = parser.parse_args()

    # Load tasks from all run directories.
    run_dirs = [Path(d).expanduser().resolve() for d in args.run_dir]
    all_tasks: List[dict] = []
    tasks_per_dir: List[Tuple[Path, List[dict]]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tasks = summary.get("tasks", [])
        if not tasks:
            raise ValueError(f"summary.json contains no tasks: {summary_path}")
        tasks_per_dir.append((run_dir, tasks))
        all_tasks.extend(tasks)

    cm, n = compute_confusion(all_tasks, args.metric, level=args.level)

    if n == 0:
        level_str = f" at level {args.level}" if args.level else ""
        print(
            f"No tasks{level_str} have both LLM and human values for "
            f"'{args.metric}'. Nothing to plot."
        )
        return

    # -----------------------------------------------------------------------
    # Confusion matrix — text summary
    # -----------------------------------------------------------------------
    agree = (cm[0, 0] + cm[1, 1]) / n
    level_str = f"  (level {args.level})" if args.level else ""
    n_runs = len(run_dirs)
    print(f"\nConfusion matrix: {args.metric}{level_str}")
    print(f"  Runs ({n_runs}): " + ", ".join(d.name for d in run_dirs))
    print(f"  N = {n}")
    print(f"  {'':16s}  LLM=T   LLM=F")
    print(f"  {'Human=T':16s}  {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"  {'Human=F':16s}  {cm[1, 0]:5d}   {cm[1, 1]:5d}")
    print(f"  Agreement : {agree:.1%}")
    print(f"  Cohen's κ : {_cohen_kappa(cm):.3f}")
    print(f"  ROC-AUC   : {_roc_auc(cm):.3f}")

    # Per-run FP / FN breakdown
    col_w = max(len(d.name) for d, _ in tasks_per_dir)
    col_w = max(col_w, 8)
    print(f"\n  Per-run breakdown ({args.metric}{level_str or ''}):")
    print(f"  {'Run':{col_w}s}   {'N':>5}   {'FP':>5}   {'FN':>5}   {'TP':>5}   {'TN':>5}")
    print(f"  {'-' * col_w}   {'-----'}   {'-----'}   {'-----'}   {'-----'}   {'-----'}")
    for run_dir, run_tasks in tasks_per_dir:
        rcm, rn = compute_confusion(run_tasks, args.metric, level=args.level)
        tp, fn, fp, tn = rcm[0, 0], rcm[0, 1], rcm[1, 0], rcm[1, 1]
        print(f"  {run_dir.name:{col_w}s}   {rn:>5}   {fp:>5}   {fn:>5}   {tp:>5}   {tn:>5}")

    # -----------------------------------------------------------------------
    # Pairwise model-ranking agreement (only meaningful with >= 2 runs)
    # -----------------------------------------------------------------------
    pw_overall_agreement: Optional[float] = None
    if len(tasks_per_dir) >= 2:
        pw = compute_pairwise_agreement(tasks_per_dir, args.metric, level=args.level)
        print_pairwise_agreement(pw, args.metric, level=args.level)
        pw_overall_agreement = pw["overall"]["weighted_agreement"]

    # -----------------------------------------------------------------------
    # Summary — three headline numbers across all runs
    # -----------------------------------------------------------------------
    auc = _roc_auc(cm)
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
    pw_str  = f"{pw_overall_agreement:.3f}" if pw_overall_agreement is not None and not np.isnan(pw_overall_agreement) else "N/A (single run)"

    print(f"\n{'='*50}")
    print(f"SUMMARY  —  {args.metric}{level_str}  —  {n_runs} run(s)")
    print(f"{'='*50}")
    print(f"  Agreement with human  : {agree:.3f}  ({agree:.1%})")
    print(f"  Pairwise agreement    : {pw_str}")
    print(f"  ROC-AUC               : {auc_str}")
    print(f"{'='*50}")

    # -----------------------------------------------------------------------
    # Confusion matrix — plot
    # -----------------------------------------------------------------------
    level_suffix = f"_level{args.level}" if args.level else ""
    filename = f"confusion_{args.metric}{level_suffix}.png"
    if args.output:
        output_path = Path(args.output)
    elif len(run_dirs) == 1:
        output_path = run_dirs[0] / filename
    else:
        output_path = Path(filename)

    plot_confusion_matrix(
        cm, args.metric, n, output_path,
        level=args.level, title=args.title,
    )


if __name__ == "__main__":
    main()