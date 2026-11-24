#!/usr/bin/env python3
"""
Aggregate all `summary.json` files that live (possibly deeply-nested) under
a given directory. The script

1. Merges *details* from every run.
2. Concatenates and de-duplicates the `missing_ep_paths` lists.
3. Computes global / per-run statistics.
4. Saves everything in   <result_path>/summary_overall.json
5. Prints a tidy report with `tabulate`.

Usage
-----
python wiw_manip/aggregate_results.py \
       running/vlm-base/Qwen2.5-VL-72B-Instruct-AWQ/06.28_igen_q3_step5
"""

import argparse
import glob
import json
import os
import statistics
from typing import Any

from tabulate import tabulate  # pip install tabulate


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_summaries(root: str) -> list[dict]:
    """Return a list of dicts, one for every summary.json under *root*."""
    pattern = os.path.join(root, "**", "summary.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No summary.json found under: {root}")
    summaries: list[dict] = []
    for path in sorted(files):
        with open(path, encoding="utf-8") as fh:
            summaries.append(json.load(fh))
    return summaries


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(summaries: list[dict]) -> dict[str, Any]:
    """Merge details, concat missing lists, and compute overall statistics."""
    # ----- merge episode-level information ----------------------------------
    merged_details: dict[str, Any] = {}
    merged_missing: list[str] = []

    for run in summaries:
        merged_missing.extend(run.get("missing_ep_paths", []))

        for ep_id, ep_info in run.get("details", {}).items():
            # Later runs silently overwrite duplicates.  This is usually harmless
            # because episode ids are globally unique. Change if needed.
            merged_details[ep_id] = ep_info

    merged_missing = sorted(set(merged_missing))  # de-duplicate

    # ----- global numbers ---------------------------------------------------
    total_tasks = sum(r["total_num_tasks"] for r in summaries)
    total_success = sum(r["num_success"] for r in summaries)
    total_planner_steps = sum(r["avg_planner_steps"] * r["total_num_tasks"] for r in summaries)
    total_output_errors = sum(r["output_format_error"] for r in summaries)

    overall_sr = total_success / total_tasks if total_tasks else float("nan")
    overall_eps = total_planner_steps / total_tasks if total_tasks else float("nan")

    per_run_sr = [r["success_rate"] for r in summaries]
    per_run_eps = [r["avg_planner_steps"] for r in summaries]

    summary = {
        "num_runs": len(summaries),
        "total_num_tasks": total_tasks,
        "num_success": total_success,
        "success_rate": overall_sr,
        "avg_planner_steps": overall_eps,
        "output_format_error": total_output_errors,
        "mean_success_rate_per_run": statistics.mean(per_run_sr),
        "median_success_rate_per_run": statistics.median(per_run_sr),
        "mean_avg_planner_steps_per_run": statistics.mean(per_run_eps),
        "missing_ep_count": len(merged_missing),
    }

    return {
        "details": merged_details,
        "summary": summary,
        "missing_ep_paths": merged_missing,
    }


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_report(overall: dict[str, Any]) -> None:
    summary = overall["summary"]
    missing = overall["missing_ep_paths"]

    # ------------- global table --------------------------------------------
    rows_global = [
        ["Total tasks", summary["total_num_tasks"]],
        ["Total successes", summary["num_success"]],
        ["Success rate", f"{summary['success_rate']:.3%}"],
        ["Avg planner steps", f"{summary['avg_planner_steps']:.3f}"],
        ["Output-format errors", summary["output_format_error"]],
        ["Missing episodes (count)", summary["missing_ep_count"]],
    ]
    print(tabulate(rows_global, headers=["Metric", "Value"], tablefmt="github"))
    print()

    # ------------- per-run table -------------------------------------------
    rows_run = [  # noqa: F841 - Formats the per-run summary as upstream; can raise on missing keys.
        ["Mean success rate / run", f"{summary['mean_success_rate_per_run']:.3%}"],
        ["Median success rate / run", f"{summary['median_success_rate_per_run']:.3%}"],
        ["Mean avg steps   / run", f"{summary['mean_avg_planner_steps_per_run']:.3f}"],
    ]
    # print(tabulate(rows_run, headers=["Per-run statistic", "Value"], tablefmt="github"))

    # ------------- list missing episodes -----------------------------------
    if missing:
        print(f"\nMissing episode paths ({len(missing)}):")
        for path in missing:
            print(path)


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all summary.json files and write summary_overall.json")
    parser.add_argument(
        "result_path",
        help="Directory that contains (possibly nested) summary.json files.",
    )
    args = parser.parse_args()

    # 1) read every summary.json
    summaries = load_summaries(args.result_path)

    # 2) compute merged object
    overall = aggregate(summaries)

    # 4) print nicely
    print_report(overall)

    # 3) save
    overall_path = os.path.join(args.result_path, "summary_overall.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\nSaved merged results → {overall_path}\n")


if __name__ == "__main__":
    main()
