#!/usr/bin/env python3
"""
migrate_runs.py

Migrates existing eval runs from the old flat format to the new nested format.

Changes per run
---------------
1. Per-task report files
   Reads provider/model from each untagged report and copies it to a
   judge-tagged name, e.g.:
     control_report.json  ->  control_report__gemini__gemini-3-1-pro-preview.json
   Skipped if:
     - provider/model fields are absent (very old schema)
     - the key metric field is null (init stub, not a real LLM result)
     - the tagged destination already exists

2. summary.json task entries
   Flat metric fields are moved into task["llm_evals"][judge_slug].
   Derived fields (control_success, task_success) are recomputed per slug.
   Flat fields are removed after migration.
   Already-migrated tasks (those with an "llm_evals" key) are left untouched.

Usage
-----
    # Preview without writing
    python migrate_runs.py --dry-run

    # Apply
    python migrate_runs.py

    # Single run
    python migrate_runs.py --run runs/veo_output_map_veo_run_7

After migration, recompute overall/by_level stats:
    python -m eval.summarize_results --run_dir runs/<run> --judge <slug>
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

# Key fields that distinguish a real LLM result from an init stub (null = stub)
_KEY_FIELDS: Dict[str, tuple] = {
    "control_report":   ("occlusion_done", "trigger_applied"),
    "artifact_report":  ("artifact",),
    "coherence_report": ("coherence",),
    "se_report":        ("state_evol",),
}

# Maps flat summary.json field -> report type it came from
_METRIC_SOURCE = {
    "occlusion_done":  "control_report",
    "trigger_applied": "control_report",
    "artifact":        "artifact_report",
    "coherence":       "coherence_report",
    "state_evol":      "se_report",
}

_DERIVED_FIELDS = {"control_success", "task_success"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slug(provider: str, model: str) -> str:
    return f"{provider}__{model.replace('.', '-')}"


def _slug_from_file(report_path: Path) -> Optional[str]:
    """Read provider/model from a report file and return the judge slug."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        provider = data.get("provider")
        model = data.get("model")
        if provider and model:
            return _make_slug(provider, model)
    except Exception:
        pass
    return None


def _is_real_result(report_path: Path, report_type: str) -> bool:
    """Return True only if the report has at least one non-null key metric."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return any(data.get(f) is not None for f in _KEY_FIELDS[report_type])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-file migration
# ---------------------------------------------------------------------------

def migrate_report_file(task_dir: Path, report_type: str, dry_run: bool) -> Optional[str]:
    """
    Copy <report_type>.json -> <report_type>__<slug>.json if it has real results.
    Returns slug on success, None if skipped.
    """
    src = task_dir / f"{report_type}.json"
    if not src.exists():
        return None

    slug = _slug_from_file(src)
    if slug is None:
        return None  # old schema without provider/model

    if not _is_real_result(src, report_type):
        return None  # init stub

    dst_name = f"{report_type}__{slug}.json"
    dst = task_dir / dst_name

    if dst.exists():
        return slug  # already migrated

    if not dry_run:
        shutil.copy2(src, dst)

    print(f"    {'[dry] ' if dry_run else ''}copy  {src.name}  ->  {dst_name}")
    return slug


# ---------------------------------------------------------------------------
# Summary migration
# ---------------------------------------------------------------------------

def migrate_summary(run_dir: Path, dry_run: bool) -> None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks", [])
    per_task_root = run_dir / "per_task"

    changed = False
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            continue

        if "llm_evals" in task:
            continue  # already migrated

        task_dir = per_task_root / task_id

        # Determine which slug each flat metric belongs to by reading its source report.
        slug_for: Dict[str, str] = {}
        for field, report_type in _METRIC_SOURCE.items():
            if task.get(field) is None:
                continue
            slug = _slug_from_file(task_dir / f"{report_type}.json")
            if slug:
                slug_for[field] = slug

        if not slug_for:
            continue

        # Group metric values by slug.
        llm_evals: Dict[str, dict] = {}
        for field, slug in slug_for.items():
            llm_evals.setdefault(slug, {})[field] = task[field]

        # Recompute derived fields per slug.
        for slug, ev in llm_evals.items():
            occ  = ev.get("occlusion_done")
            trig = ev.get("trigger_applied")
            art  = ev.get("artifact")
            coh  = ev.get("coherence")
            se   = ev.get("state_evol")
            if occ is not None and trig is not None:
                ev["control_success"] = bool(occ) and bool(trig)
            if se is not None and art is not None and coh is not None:
                ev["task_success"] = bool(se) and bool(coh) and not bool(art)

        task["llm_evals"] = llm_evals

        # Remove flat fields.
        for field in list(_METRIC_SOURCE.keys()) + list(_DERIVED_FIELDS):
            task.pop(field, None)

        changed = True

    if not changed:
        return

    if not dry_run:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"  {'[dry] ' if dry_run else ''}updated summary.json  ({sum(1 for t in tasks if 'llm_evals' in t)} tasks now nested)")


# ---------------------------------------------------------------------------
# Per-run migration
# ---------------------------------------------------------------------------

def migrate_run(run_dir: Path, dry_run: bool) -> None:
    per_task = run_dir / "per_task"
    if not per_task.exists():
        return

    task_dirs = sorted(d for d in per_task.iterdir() if d.is_dir())
    file_counts: Dict[str, int] = {rt: 0 for rt in _KEY_FIELDS}

    for task_dir in task_dirs:
        for report_type in _KEY_FIELDS:
            slug = migrate_report_file(task_dir, report_type, dry_run=dry_run)
            if slug:
                file_counts[report_type] += 1

    totals = ", ".join(f"{rt}: {n}" for rt, n in file_counts.items() if n)
    if totals:
        print(f"  files copied — {totals}")

    migrate_summary(run_dir, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate eval runs to nested llm_evals format.")
    parser.add_argument("--runs-dir", default="runs/", help="Root runs directory (default: runs/).")
    parser.add_argument("--run", default=None, help="Migrate a single run directory instead.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no files will be written ===\n")

    if args.run:
        run_dir = Path(args.run).expanduser().resolve()
        print(f"\n=== {run_dir.name} ===")
        migrate_run(run_dir, dry_run=args.dry_run)
    else:
        runs_dir = Path(args.runs_dir).expanduser().resolve()
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "per_task").exists():
                print(f"\n=== {run_dir.name} ===")
                migrate_run(run_dir, dry_run=args.dry_run)

    print("\nDone.")
    if not args.dry_run:
        print("Run  python -m eval.summarize_results --run_dir <run> --judge <slug>  to refresh overall/by_level stats.")


if __name__ == "__main__":
    main()
