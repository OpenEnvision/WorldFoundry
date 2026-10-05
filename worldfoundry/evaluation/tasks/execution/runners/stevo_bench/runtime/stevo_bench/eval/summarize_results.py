# eval/summarize_results.py
"""
Post-pipeline script: computes per-level breakdown statistics and writes them
back into the by_level and overall sections of summary.json.

Also annotates each task entry in-place with derived fields:
  - Top-level (LLM):       control_success, task_success  (inside llm_evals[judge])
  - Per-annotator (human): control_success, task_success

Run after the eval pipeline:
    python -m eval.summarize_results --run_dir runs/my_run --judge gemini__gemini-3-1-pro-preview
    python -m eval.summarize_results --run_dir runs/my_run --judge gemini__gemini-3-1-pro-preview --print_llm
    python -m eval.summarize_results --run_dir runs/my_run --judge gemini__gemini-3-1-pro-preview --print_llm --print_human

If --judge is omitted, the script will auto-detect the judge from available llm_evals keys
(works when only one judge has been run). Falls back to legacy flat fields for old runs.

Stats added to each level in by_level[judge] and overall[judge]:
  num_tasks              : number of tasks in the group
  avg_occlusion_done     : fraction of tasks where occlusion was successfully applied (LLM) [Observation]
  avg_trigger_applied    : fraction of tasks where the trigger was applied (LLM) [Action]
  avg_physical_inaccuracy : fraction of tasks where a physical inaccuracy was detected (LLM)
  avg_control_success    : fraction of tasks where occlusion_done AND trigger_applied (LLM)
  avg_state_evol_success : fraction of tasks where state_evol=True (LLM) [State Progress]
  avg_task_success       : fraction of tasks where state_evol=True AND physical_inaccuracy=False (LLM)

Tasks missing the relevant fields are excluded from those averages.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _avg(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _get_task_fields(t: Dict[str, Any], judge: Optional[str]) -> Dict[str, Any]:
    """
    Return the metric dict for task t under the given judge.

    With the new nested format, values live in t["llm_evals"][judge].
    For backward compat with old flat-field runs (no llm_evals key), fall back
    to reading directly from t.
    """
    if judge and "llm_evals" in t:
        return t["llm_evals"].get(judge, {})
    # Legacy flat format
    return t


def _annotate_task(t: Dict[str, Any], judge: Optional[str]) -> None:
    """Write control_success and task_success into the judge's eval dict
    and into each task["annotations"][name] dict (per annotator)."""

    # ── LLM fields (nested under judge) ──────────────────────────────────────
    ev = _get_task_fields(t, judge)

    occ  = ev.get("occlusion_done")
    trig = ev.get("trigger_applied")
    pi   = ev.get("physical_inaccuracy")
    se   = ev.get("state_evol")

    if occ is not None and trig is not None:
        ev["control_success"] = bool(occ) and bool(trig)
    if se is not None and pi is not None:
        ev["task_success"] = bool(se) and not bool(pi)

    # When using nested format, write derived fields back into the nested dict.
    if judge and "llm_evals" in t and judge in t["llm_evals"]:
        t["llm_evals"][judge] = ev

    # ── Per-annotator fields ──────────────────────────────────────────────────
    for ann in t.get("annotations", {}).values():
        a_occ = ann.get("occlusion_done")
        a_trig = ann.get("trigger_applied")
        a_pi  = ann.get("physical_inaccuracy")
        a_se  = ann.get("state_evol")

        if a_occ is not None and a_trig is not None:
            ann["control_success"] = bool(a_occ) and bool(a_trig)
        if a_se is not None and a_pi is not None:
            ann["task_success"] = bool(a_se) and not bool(a_pi)


def _empty_buckets() -> Dict[str, List]:
    return {
        "occlusion_done":      [],
        "trigger_applied":     [],
        "physical_inaccuracy": [],
        "control_success":     [],
        "state_evol_success":  [],
        "task_success":        [],
    }


def _fill_bucket(bucket: Dict[str, List], t: Dict[str, Any], judge: Optional[str]) -> None:
    """Accumulate one task entry into a bucket dict (LLM metrics only)."""
    ev = _get_task_fields(t, judge)

    occlusion_done     = ev.get("occlusion_done")
    trigger_applied    = ev.get("trigger_applied")
    physical_inaccuracy = ev.get("physical_inaccuracy")
    state_evol         = ev.get("state_evol")
    control_success    = ev.get("control_success")
    task_success       = ev.get("task_success")

    if occlusion_done is not None:
        bucket["occlusion_done"].append(1.0 if occlusion_done else 0.0)
    if trigger_applied is not None:
        bucket["trigger_applied"].append(1.0 if trigger_applied else 0.0)
    if physical_inaccuracy is not None:
        bucket["physical_inaccuracy"].append(1.0 if physical_inaccuracy else 0.0)
    if control_success is not None:
        bucket["control_success"].append(1.0 if control_success else 0.0)
    if state_evol is not None:
        bucket["state_evol_success"].append(1.0 if state_evol else 0.0)
    if task_success is not None:
        bucket["task_success"].append(1.0 if task_success else 0.0)


def _bucket_to_stats(bucket: Dict[str, List], num_tasks: int) -> Dict[str, Any]:
    return {
        "num_tasks":               num_tasks,
        "avg_occlusion_done":      _avg(bucket["occlusion_done"]),
        "avg_trigger_applied":     _avg(bucket["trigger_applied"]),
        "avg_physical_inaccuracy": _avg(bucket["physical_inaccuracy"]),
        "avg_control_success":     _avg(bucket["control_success"]),
        "avg_state_evol_success":  _avg(bucket["state_evol_success"]),
        "avg_task_success":        _avg(bucket["task_success"]),
    }


def _stats_for_tasks(tasks: List[Dict[str, Any]], judge: Optional[str]) -> Dict[str, Any]:
    """Compute aggregate stats for an arbitrary list of task entries."""
    bucket = _empty_buckets()
    for t in tasks:
        _fill_bucket(bucket, t, judge)
    return _bucket_to_stats(bucket, len(tasks))


def _majority_vote(values: List) -> Optional[bool]:
    """Strict majority vote over boolean-ish values. Returns None on tie or empty."""
    if not values:
        return None
    n = len(values)
    s = sum(1 for v in values if v)
    if s * 2 == n:   # exact tie — skip
        return None
    return s * 2 > n


def _fill_bucket_human(bucket: Dict[str, List], t: Dict[str, Any]) -> None:
    """Accumulate one task into bucket via majority vote across human annotators."""
    anns = list(t.get("annotations", {}).values())
    if not anns:
        return
    # (annotation key, bucket key)
    fields = [
        ("occlusion_done",     "occlusion_done"),
        ("trigger_applied",    "trigger_applied"),
        ("physical_inaccuracy", "physical_inaccuracy"),
        ("control_success",    "control_success"),
        ("state_evol",         "state_evol_success"),
        ("task_success",       "task_success"),
    ]
    for ann_key, bucket_key in fields:
        vals = [ann[ann_key] for ann in anns if ann_key in ann]
        vote = _majority_vote(vals)
        if vote is not None:
            bucket[bucket_key].append(1.0 if vote else 0.0)


def _stats_for_tasks_human(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats using majority-voted human annotations."""
    bucket = _empty_buckets()
    for t in tasks:
        _fill_bucket_human(bucket, t)
    return _bucket_to_stats(bucket, len(tasks))


def compute_level_stats(
    tasks: List[Dict[str, Any]],
    judge: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-level and overall breakdown stats.

    Returns:
      (level_stats, overall_stats)
      level_stats   — dict keyed by level string
      overall_stats — same metrics aggregated across all levels
    """
    level_buckets: Dict[str, Dict[str, List]] = defaultdict(_empty_buckets)
    level_counts:  Dict[str, int] = defaultdict(int)
    overall_bucket = _empty_buckets()
    overall_count = 0

    for t in tasks:
        level = str(t.get("task_level", ""))
        if not level:
            continue
        _fill_bucket(level_buckets[level], t, judge)
        level_counts[level] += 1
        _fill_bucket(overall_bucket, t, judge)
        overall_count += 1

    level_stats = {
        level: _bucket_to_stats(level_buckets[level], level_counts[level])
        for level in level_buckets
    }
    overall_stats = _bucket_to_stats(overall_bucket, overall_count)

    return level_stats, overall_stats


def _detect_judge(tasks: List[Dict[str, Any]]) -> Optional[str]:
    """Auto-detect the judge slug when exactly one judge appears in llm_evals."""
    judges: set = set()
    for t in tasks:
        judges.update(t.get("llm_evals", {}).keys())
    if len(judges) == 1:
        return next(iter(judges))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-level breakdown stats and write them into summary.json."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Path to the eval run directory containing summary.json.",
    )
    parser.add_argument(
        "--judge",
        default=None,
        type=str,
        help=(
            "Judge slug to summarize (e.g. 'gemini__gemini-3-1-pro-preview' or "
            "'openai__gpt-4o'). Auto-detected when only one judge is present."
        ),
    )
    parser.add_argument(
        "--print_llm",
        action="store_true",
        help="Print the LLM judge summary table.",
    )
    parser.add_argument(
        "--print_human",
        action="store_true",
        help="Print the human annotation summary table (majority vote).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    tasks = summary.get("tasks", [])
    if not tasks:
        raise ValueError("summary.json contains no tasks.")

    # Determine which judge to summarize.
    judge = args.judge
    if judge is None:
        judge = _detect_judge(tasks)
        if judge:
            print(f"[INFO] Auto-detected judge: {judge}")
        else:
            any_nested = any("llm_evals" in t for t in tasks)
            if any_nested:
                available = sorted({k for t in tasks for k in t.get("llm_evals", {})})
                raise ValueError(
                    f"Multiple judges found in llm_evals; specify --judge. "
                    f"Available: {available}"
                )
            # Old flat-field run — judge=None means read from top-level fields.

    # Annotate each task with derived control_success / task_success fields.
    for t in tasks:
        _annotate_task(t, judge)

    level_stats, overall_stats = compute_level_stats(tasks, judge)

    # Write stats nested under the judge slug so multiple judges coexist in
    # summary.json. Falls back to flat keys for legacy (judge=None) runs.
    def _set_stats(section: dict, key: Optional[str], stats: dict) -> None:
        if key:
            section[key] = stats
        else:
            section.update(stats)

    by_level = summary.setdefault("by_level", {})
    for level, stats in level_stats.items():
        level_dict = by_level.setdefault(level, {})
        _set_stats(level_dict, judge, stats)

    _set_stats(summary.setdefault("overall", {}), judge, overall_stats)

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.print_llm and not args.print_human:
        return

    # Split tasks into baseline (_00) and occluded (non-_00) groups
    baseline_tasks = [t for t in tasks if str(t.get("task_id", "")).endswith("_00")]
    occluded_tasks = [t for t in tasks if not str(t.get("task_id", "")).endswith("_00")]

    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else f"{v:>5}" if isinstance(v, int) else "  n/a"

    header = (
        f"{'Group':>10}  {'N':>5}  {'St.Prog':>8}  {'Phys.Inac':>10}  "
        f"{'Success':>8}  {'Observ.':>8}  {'Action':>8}  {'Ctrl.Suc':>8}"
    )

    def _print_table(title: str, rows: list) -> None:
        print(title)
        print(header)
        print("-" * len(header))
        for label, s in rows:
            print(
                f"{label:>10}  "
                f"{s.get('num_tasks', '?'):>5}  "
                f"{fmt(s.get('avg_state_evol_success')):>8}  "
                f"{fmt(s.get('avg_physical_inaccuracy')):>10}  "
                f"{fmt(s.get('avg_task_success')):>8}  "
                f"{fmt(s.get('avg_occlusion_done')):>8}  "
                f"{fmt(s.get('avg_trigger_applied')):>8}  "
                f"{fmt(s.get('avg_control_success')):>8}"
            )

    if args.print_llm:
        all_stats      = _stats_for_tasks(tasks, judge)
        baseline_stats = _stats_for_tasks(baseline_tasks, judge)
        occluded_stats = _stats_for_tasks(occluded_tasks, judge)
        title = f"LLM Judge" + (f" ({judge})" if judge else "")
        _print_table(
            title,
            [("all", all_stats), ("baseline", baseline_stats), ("occluded", occluded_stats)],
        )

    if args.print_human and any(t.get("annotations") for t in tasks):
        all_human      = _stats_for_tasks_human(tasks)
        baseline_human = _stats_for_tasks_human(baseline_tasks)
        occluded_human = _stats_for_tasks_human(occluded_tasks)
        if args.print_llm:
            print()
        _print_table(
            "Human Annotations (majority vote)",
            [("all", all_human), ("baseline", baseline_human), ("occluded", occluded_human)],
        )


if __name__ == "__main__":
    main()
