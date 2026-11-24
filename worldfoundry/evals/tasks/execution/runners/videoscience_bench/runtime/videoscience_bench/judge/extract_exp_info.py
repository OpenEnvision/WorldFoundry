#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
from pathlib import Path

# ----- exact column names
REQUIRED_COLUMNS = [
    "Expected phenomenon",  # -> --phenomenon
    "Prompts",  # -> --description / text_prompt
    "Source",  # -> kept for schema; not used for path resolution
    "Unique ID",  # -> naming + ID mapping
]

# Files look like: vid_45_run_1.mp4, vid_050_run_2.mp4, etc.
RUN_RE = re.compile(r"^(vid_0*\d+)_run_(\d+)\.mp4$", re.IGNORECASE)
VID_KEY_RE = re.compile(r"vid_0*(\d+)", re.IGNORECASE)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "untitled"


def uid_to_folder(uid: str, prefix: str = "vid_", pad: int = 3) -> str:
    """Folder name from CSV UID; accepts '50', 'vid_50', 'vid_050', or arbitrary strings."""
    uid = (uid or "").strip()
    if uid.isdigit():
        return f"{prefix}{int(uid):0{pad}d}"
    m = VID_KEY_RE.search(uid)
    if m:
        return f"{prefix}{int(m.group(1)):0{pad}d}"
    if not uid:
        return f"{prefix}uid"
    return f"{prefix}{slugify(uid)}"


def vid_int_from_any(s: str) -> int | None:
    """Return integer ID from 'vid_50', 'vid_050', '50', etc."""
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        return int(s)
    m = VID_KEY_RE.search(s)
    if m:
        return int(m.group(1))
    return None


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}\nFound columns: {reader.fieldnames}")
        return [dict(r) for r in reader]


def infer_ref_path_for_vid(ref_dir: Path | None, vid_num: int) -> str | None:
    """Try both non-padded and padded reference names under ref_dir."""
    if not ref_dir:
        return None
    cand1 = ref_dir / f"vid_{vid_num}_ref.mp4"
    if cand1.exists():
        return str(cand1)
    cand2 = ref_dir / f"vid_{vid_num:03d}_ref.mp4"
    if cand2.exists():
        return str(cand2)
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract experiment info from CSV and emit run_all.sh for VLM judging.")
    ap.add_argument("--csv", required=True, help="Path to CSV with exact columns.")
    ap.add_argument("--provider", default="openai", help="Provider (default: openai).")
    ap.add_argument("--model", default="gpt-5-pro", help="Model (default: gpt-5-pro).")
    ap.add_argument(
        "--out-dir",
        default="judge/results/evaluation_videos",
        help="Base results dir (default: judge/results/evaluation_videos).",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Cap total runs (0=no cap).",
    )
    ap.add_argument(
        "--script",
        default="judge/vlm_as_a_judge.py",
        help="Judge script path.",
    )
    ap.add_argument(
        "--reference-dir",
        default="",
        help="Dir containing vid_{id}_ref.mp4 files (optional).",
    )

    # filters
    ap.add_argument(
        "--mode",
        choices=["ready", "all"],
        default="all",
        help="When 'ready', include only rows whose 'finalized?' is 'done'. Default: all.",
    )
    ap.add_argument(
        "--author",
        default="",
        help="Author or comma-separated authors (exact match). Omit for all authors.",
    )

    # source scanning
    ap.add_argument(
        "--source-dir",
        required=True,
        help="Root directory containing model/eval_target subfolders with run videos.",
    )
    ap.add_argument(
        "--eval_target",
        help="Subfolder under --source-dir to scan (e.g., sora2).",
    )

    # checklist integration
    ap.add_argument(
        "--checklist-dir",
        default="",
        help=(
            "Directory containing checklist_*.json files (optional). If omitted, will use CHECKLIST_DIR env var if set."
        ),
    )

    # CV metrics integration
    ap.add_argument(
        "--enable-cv",
        action="store_true",
        help=("If set, run cv_tools/cv_metrics.py for each run and pass its JSON as --cv_json to the judge."),
    )
    ap.add_argument(
        "--cv-metrics-script",
        default="judge/cv_tools/cv_metrics.py",
        help="Path to cv_metrics.py (default: judge/cv_tools/cv_metrics.py).",
    )
    ap.add_argument(
        "--cv-modules",
        default="grounding,bytetrack,raft,clip4clip,lpips",
        help="Comma-separated CV modules for cv_metrics.py (default: grounding,bytetrack,raft,clip4clip,lpips).",
    )

    args = ap.parse_args()

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)

    # ---- apply filters (exact column names only) ----
    if args.mode == "ready":
        rows = [r for r in rows if (r.get("finalized?") or "").strip().lower() == "done"]

    authors_filter: list[str] = []
    if args.author.strip():
        authors_filter = [a.strip() for a in args.author.split(",") if a.strip()]
        rows = [r for r in rows if (r.get("Author") or "").strip() in authors_filter]

    # Map from normalized vid integer -> CSV row
    selected_by_vid_int: dict[int, dict[str, str]] = {}
    for r in rows:
        uid_raw = (r.get("Unique ID") or "").strip()
        vid_num = vid_int_from_any(uid_raw)
        if vid_num is None:
            # optional fallback if Example Title contains vid_###
            vid_num = vid_int_from_any(r.get("Example Title", ""))
        if vid_num is None:
            continue
        selected_by_vid_int[vid_num] = r  # last one wins if duplicates

    # Prepare paths
    base_out = Path(args.out_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    model_slug = slugify(args.model)
    eval_slug = slugify(args.eval_target)

    # New runner filename pattern
    author_slug = slugify(args.author or "all")
    runner_filename = f"run_checklist_cv-method_{author_slug}_target_{eval_slug}.sh"
    runner_path = base_out / runner_filename

    ref_dir = Path(args.reference_dir).resolve() if args.reference_dir else None

    # Checklist dir: CLI > env > None
    checklist_dir: Path | None = None
    if args.checklist_dir:
        checklist_dir = Path(args.checklist_dir).resolve()
    else:
        env_cl = os.environ.get("CHECKLIST_DIR", "").strip()
        if env_cl:
            checklist_dir = Path(env_cl).resolve()

    # CV metrics script path (if enabled)
    cv_metrics_enabled = bool(args.enable_cv)
    cv_metrics_script_path: Path | None = None
    if cv_metrics_enabled:
        cv_metrics_script_path = Path(args.cv_metrics_script).resolve()
        if not cv_metrics_script_path.exists():
            raise SystemExit(f"--enable-cv was set but cv-metrics-script not found: {cv_metrics_script_path}")

    # Scan root: source_dir/<eval_target>
    source_root = Path(args.source_dir).resolve() / args.eval_target
    if not source_root.exists():
        raise SystemExit(f"--source-dir target not found: {source_root}")

    # -------- build run script --------
    run_lines: list[str] = []
    run_lines.append("#!/usr/bin/env bash")
    run_lines.append("set -euo pipefail")
    run_lines.append("")
    run_lines.append(f"PROVIDER={shlex.quote(args.provider)}")
    run_lines.append(f"MODEL={shlex.quote(args.model)}")
    run_lines.append(f"JUDGE_SCRIPT={shlex.quote(str(Path(args.script).resolve()))}")
    run_lines.append(f"BASE_OUT_DIR={shlex.quote(str(base_out))}")
    run_lines.append("")

    # CV env vars (only meaningful if --enable-cv was used)
    if cv_metrics_enabled and cv_metrics_script_path is not None:
        run_lines.append(f"CV_METRICS_SCRIPT={shlex.quote(str(cv_metrics_script_path))}")
        run_lines.append(f"CV_MODULES={shlex.quote(args.cv_modules)}")
    else:
        run_lines.append('CV_METRICS_SCRIPT=""')
        run_lines.append('CV_MODULES=""')

    if args.max_runs > 0:
        run_lines.append(f'MAX_RUNS="${{MAX_RUNS:-{args.max_runs}}}"')
    else:
        run_lines.append('MAX_RUNS="${MAX_RUNS:-0}"')
    run_lines.append("RUN_COUNT=0")
    run_lines.append("")
    run_lines.append("run() {")
    run_lines.append('  if [[ "${MAX_RUNS}" != "0" && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then')
    run_lines.append('    echo "[cap] Reached MAX_RUNS=${MAX_RUNS}, halting." >&2')
    run_lines.append("    exit 0")
    run_lines.append("  fi")
    run_lines.append("  RUN_COUNT=$((RUN_COUNT+1))")
    run_lines.append('  echo "[run ${RUN_COUNT}] $*"')
    run_lines.append('  eval "$@"')
    run_lines.append("}")
    run_lines.append("")
    run_lines.append("# ----- Generated runs -----")

    n_cmds = 0

    # Traverse only within source_dir/eval_target
    for p in sorted(source_root.rglob("*.mp4")):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        vid_key = m.group(1)  # e.g. "vid_50" or "vid_050"
        run_idx = m.group(2)  # e.g. "1"
        vid_num = vid_int_from_any(vid_key)
        if vid_num is None:
            continue

        # Keep only vids that passed CSV filters (author/mode)
        row = selected_by_vid_int.get(vid_num)
        if not row:
            continue

        phenomenon = (row.get("Expected phenomenon") or "").strip()
        description = (row.get("Prompts") or "").strip()
        if not (phenomenon and description):
            continue

        uid_raw = (row.get("Unique ID") or "").strip()

        # Output folder: {BASE}/{MODEL}/{EVAL_TARGET}/vid_xxx/run_i
        uid_folder = uid_to_folder(uid_raw, prefix="vid_", pad=3)
        out_dir = base_out / model_slug / eval_slug / uid_folder / f"run_{run_idx}"

        # Reference for this vid (try non-padded and padded)
        ref_path = infer_ref_path_for_vid(ref_dir, vid_num) if ref_dir else None
        ref_arg_judge = f" --ref_video {shlex.quote(ref_path)}" if ref_path else ""

        # Checklist JSON for this UID (if exists)
        checklist_arg = ""
        if checklist_dir is not None and uid_raw:
            cl_path = checklist_dir / f"checklist_{uid_raw}.json"
            if cl_path.exists():
                checklist_arg = f" --checklist_json {shlex.quote(str(cl_path))}"

        # CV metrics JSON path & prefix command (if enabled)
        cv_prefix = ""
        cv_arg = ""
        if cv_metrics_enabled and cv_metrics_script_path is not None:
            cv_json_path = out_dir / "cv_metrics.json"
            cv_json_quoted = shlex.quote(str(cv_json_path))
            # reuse ref for cv_metrics, if available
            cv_ref_arg = f" --ref_video {shlex.quote(ref_path)}" if ref_path else ""

            cv_prefix = (
                "python3 ${CV_METRICS_SCRIPT} "
                f"--video {shlex.quote(str(p))} "
                f"--text_prompt {shlex.quote(description)}"
                f"{cv_ref_arg} "
                '--modules "${CV_MODULES:-grounding,bytetrack,raft,clip4clip,lpips}" '
                "--max_frames 128 "
                f"--output_json {cv_json_quoted} && "
            )

            cv_arg = f" --cv_json {cv_json_quoted}"

        cmd = (
            "mkdir -p {od} && "
            "{cv_prefix}"
            "python3 {script} "
            "--provider ${{PROVIDER}} "
            "--model ${{MODEL}} "
            "--video {video} "
            "--description {desc} "
            "--phenomenon {phen} "
            "--md_out {md} "
            "--json_out {js}{checklist}{cv}{ref}"
        ).format(
            script="${JUDGE_SCRIPT}",
            video=shlex.quote(str(p)),
            desc=shlex.quote(description),
            phen=shlex.quote(phenomenon),
            md=shlex.quote(str(out_dir / "report.md")),
            js=shlex.quote(str(out_dir / "report.json")),
            od=shlex.quote(str(out_dir)),
            checklist=checklist_arg,
            ref=ref_arg_judge,
            cv=cv_arg,
            cv_prefix=cv_prefix,
        )
        run_lines.append(f"run {cmd}")
        n_cmds += 1

    runner_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    os.chmod(runner_path, 0o755)

    summary = {
        "csv": str(csv_path),
        "source_dir": str(source_root),
        "reference_dir": str(ref_dir) if ref_dir else "",
        "checklist_dir": str(checklist_dir) if checklist_dir else "",
        "out_dir": str(base_out),
        "eval_target": args.eval_target,
        "runner": str(runner_path),
        "num_runs": n_cmds,
        "provider": args.provider,
        "model": args.model,
        "max_runs_cap": args.max_runs,
        "mode": args.mode,
        "author_filter": [a for a in (args.author.split(",") if args.author else []) if a.strip()],
        "cv_metrics_enabled": cv_metrics_enabled,
        "cv_metrics_script": str(cv_metrics_script_path) if cv_metrics_script_path else "",
        "cv_modules": args.cv_modules if cv_metrics_enabled else "",
    }
    print(json.dumps(summary, indent=2))
    print(f"\nUsage:\n  {runner_path}           # uses built-in cap or MAX_RUNS env\n  MAX_RUNS=3 {runner_path}")


if __name__ == "__main__":
    main()
