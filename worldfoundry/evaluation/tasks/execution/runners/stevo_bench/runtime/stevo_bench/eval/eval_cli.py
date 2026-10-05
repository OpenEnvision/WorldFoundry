import argparse
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from eval.task_resolver import resolve_tasks_from_outputs_json, ResolvedTask
from eval.utils import _path_to_rel
from eval.control_judge import (
    evaluate_control_one_task, append_control_results_to_summary,
    _load_task_fields, _compute_requested_fields,
)
from eval.physics_judge import (
    evaluate_physics_one_task, append_physics_results_to_summary, PhysicsJudgeResult,
)
from eval.control_judge import ControlJudgeResult
from eval.se_judge import (
    evaluate_se_one_task, append_se_results_to_summary, SEJudgeResult,
)


def _make_judge_slug(provider: str, model: str) -> str:
    """Stable filesystem-safe identifier for a (provider, model) pair."""
    return f"{provider}__{model.replace('.', '-')}"


def _report_filename(report_type: str, judge_slug: str) -> str:
    """Return the per-task report filename for this judge."""
    return f"{report_type}_report__{judge_slug}.json"


def _init_summary_json(resolved_tasks: List[ResolvedTask], run_root: Path) -> None:
    """
    Ensure summary.json exists with a skeleton entry for every resolved task.

    - If the file does not exist, create it with num_tasks / overall / by_level / tasks.
    - If the file already exists, add entries for any task_ids not yet present
      (handles re-runs with --pattern or newly added tasks) without touching
      existing entries.
    """
    summary_path = run_root / "summary.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_ids = {t["task_id"] for t in summary.get("tasks", []) if "task_id" in t}
        new_entries = [
            {"task_id": t.task_id, "task_level": t.task_level}
            for t in resolved_tasks
            if t.task_id not in existing_ids
        ]
        if new_entries:
            summary.setdefault("tasks", []).extend(new_entries)
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    else:
        summary = {
            "num_tasks": len(resolved_tasks),
            "overall":   {},
            "by_level":  {},
            "tasks": [
                {"task_id": t.task_id, "task_level": t.task_level}
                for t in resolved_tasks
            ],
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _write_init_stubs(
    resolved_tasks: List[ResolvedTask],
    per_task_root: Path,
    camera_controlled: bool = False,
) -> None:
    """
    Write skeleton judge_report.json, control_report.json, and quality_report.json
    for each task so human_eval_server.py can list and display tasks without any LLM judging.

    Only creates files that don't yet exist — safe to re-run.

    Args:
        camera_controlled: Pass True for camera-controlled models (HY-WorldPlay,
            LingBot, GEN3C) so that requested_occlusion is set to "camera pan" rather than
            being extracted from video_WM.
    """
    for task in resolved_tasks:
        task_dir = per_task_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute task prompt fields once; used by both control and physics stubs.
        try:
            video_wm, camera_wm, camera_pose = _load_task_fields(task.task_yaml)
            requested_trigger, requested_occlusion = _compute_requested_fields(
                video_wm, camera_wm, camera_pose, camera_controlled=camera_controlled
            )
        except Exception:
            video_wm = ""
            requested_trigger = ""
            requested_occlusion = ""

        # control_report.json — skeleton with pre-computed request fields, no LLM judgments.
        # requested_trigger and requested_occlusion are derived from the task YAML using the
        # same logic as the real control judge, so the human eval UI displays them immediately.
        cr_path = task_dir / "control_report.json"
        if not cr_path.exists():
            cr = {
                "task_id":             task.task_id,
                "video_WM_prompt":     video_wm,
                "wm_video":            _path_to_rel(task.wm_video),
                "requested_occlusion": requested_occlusion,
                "requested_trigger":   requested_trigger,
                "occlusion_done":      None,
                "trigger_applied":     None,
                "notes":               "",
            }
            cr_path.write_text(json.dumps(cr, indent=2, ensure_ascii=False), encoding="utf-8")

        # physics_report.json — skeleton matching evaluate_physics_one_task output
        phys_path = task_dir / "physics_report.json"
        if not phys_path.exists():
            phys = {
                "task_id":             task.task_id,
                "wm_video":            _path_to_rel(task.wm_video),
                "requested_occlusion": requested_occlusion,
                "physical_inaccuracy": None,
                "notes":               "",
            }
            phys_path.write_text(json.dumps(phys, indent=2, ensure_ascii=False), encoding="utf-8")

        # se_report.json — skeleton matching evaluate_se_one_task output
        se_path = task_dir / "se_report.json"
        if not se_path.exists():
            se = {
                "task_id":   task.task_id,
                "wm_video":  _path_to_rel(task.wm_video),
                "state_evol": None,
                "notes":     "",
            }
            se_path.write_text(json.dumps(se, indent=2, ensure_ascii=False), encoding="utf-8")


def _eval_one_task(
    task: ResolvedTask,
    *,
    provider: str,
    control_model: str,
    physics_model: str,
    se_model: str,
    control_report_fn: str,
    physics_report_fn: str,
    se_report_fn: str,
    run_control: bool = True,
    run_physics: bool = True,
    run_state: bool = True,
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> Tuple[None, Optional[ControlJudgeResult], Optional[PhysicsJudgeResult], Optional[SEJudgeResult]]:
    judge_result = None  # binary judge questions (judge_report.json) are obsolete
    control_result = evaluate_control_one_task(task, provider=provider, model=control_model, report_filename=control_report_fn, camera_controlled=camera_controlled, ensemble_size=ensemble_size, ensemble_mode=ensemble_mode) if run_control else None
    physics_result = evaluate_physics_one_task(task, provider=provider, model=physics_model, report_filename=physics_report_fn, camera_controlled=camera_controlled, ensemble_size=ensemble_size, ensemble_mode=ensemble_mode) if run_physics else None
    se_result = evaluate_se_one_task(task, provider=provider, model=se_model, report_filename=se_report_fn, camera_controlled=camera_controlled, ensemble_size=ensemble_size, ensemble_mode=ensemble_mode) if run_state else None
    return judge_result, control_result, physics_result, se_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=str, help="Outputs folder containing the JSON map (task_id -> video path or URL).")

    parser.add_argument(
        "--task_root",
        default="benchmark/tasks/",
        type=str,
        help="Root directory containing task folders."
    )
    parser.add_argument(
        "--run_dir",
        default="runs/",
        type=str,
        help="Directory to write normalized run artifacts."
    )

    # Judge LLM provider info
    parser.add_argument(
        "--vlm_provider",
        default="gemini",
        type=str,
        help="Provider of the judge VLM, currently support OpenAI, Anthropic and Gemini. API key must be configured."
    )
    parser.add_argument(
        "--control_judge_model",
        default="gemini-3.1-pro-preview",
        type=str,
        help="Name of the control judge model."
    )
    parser.add_argument(
        "--physics_judge_model",
        default="gemini-3.1-pro-preview",
        type=str,
        help="Name of the physics judge model (physical inaccuracy detector).",
    )
    parser.add_argument(
        "--se_judge_model",
        default="gemini-3.1-pro-preview",
        type=str,
        help="Name of the SE judge model (state evolution two-step evaluator).",
    )

    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of parallel workers for per-task eval (judge + control judge).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        type=str,
        help="fnmatch pattern to filter task IDs (e.g. 'ice_on_burner*'). If omitted, all tasks are evaluated.",
    )
    parser.add_argument(
        "--exclude_baseline",
        action="store_true",
        help="Skip all tasks whose IDs end with '_00' (baseline / no-occlusion variants).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate tasks that already have both judge_report.json and control_report.json.",
    )
    parser.add_argument(
        "--download_urls",
        action="store_true",
        help="If set, download http(s) videos into the run directory.",
    )
    # parser.add_argument(                                    # OBSOLETE: binary judge questions
    #     "--state",
    #     action="store_true",
    #     help="Run the state evolution judge (judge_report).",
    # )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Run the control judge (control_report).",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Run the physics judge (physics_report).",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Run the state evolution judge (se_report).",
    )
    parser.add_argument(
        "--ensemble_size",
        default=1,
        type=int,
        help="Number of independent judge queries per task (ensemble). Default: 1 (no ensemble).",
    )
    parser.add_argument(
        "--ensemble_mode",
        default="majority",
        choices=["majority", "unanimous", "unanimous_true"],
        help=(
            "How to aggregate ensemble votes into a final decision. "
            "'majority': strict majority (ties go False). "
            "'unanimous': True only if ALL members vote True; default False. "
            "'unanimous_true': False only if ALL members vote False; default True "
            "(use for judges where false-negatives dominate, e.g. physics). "
            "Default: majority."
        ),
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Initialize the run directory only: resolve tasks, create per-task folders, "
            "summary.json, and skeleton report stubs. Exits without running any LLM judge. "
            "Use this to prepare a run for human-only evaluation."
        ),
    )
    args = parser.parse_args()

    any_selected = args.control or args.physics or args.state
    run_control = args.control or not any_selected
    run_physics = args.physics or not any_selected
    run_state = args.state or not any_selected

    # Judge slugs uniquely identify (provider, model) pairs in filenames and summary.json.
    control_slug = _make_judge_slug(args.vlm_provider, args.control_judge_model)
    physics_slug = _make_judge_slug(args.vlm_provider, args.physics_judge_model)
    se_slug = _make_judge_slug(args.vlm_provider, args.se_judge_model)

    control_report_fn = _report_filename("control", control_slug)
    physics_report_fn = _report_filename("physics", physics_slug)
    se_report_fn = _report_filename("se", se_slug)

    # ---------------------------------------------------------------------------
    # Build eval run directory
    # ---------------------------------------------------------------------------
    task_root = Path(args.task_root).expanduser().resolve()
    outputs_dir = Path(args.outputs).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()

    if not task_root.exists():
        raise FileNotFoundError(task_root)
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs folder not found: {outputs_dir}")

    json_files = list(outputs_dir.glob("*.json"))
    if len(json_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .json file in {outputs_dir}, found {len(json_files)}: "
            + ", ".join(f.name for f in json_files)
        )
    output_json = json_files[0]

    # Auto-detect camera-controlled models from the outputs folder name.
    # These models receive a camera trajectory instead of a text prompt, so
    # requested_occlusion is always "camera pan" rather than parsed from video_WM.
    _CAMERA_CONTROLLED_NAMES = {"worldplay", "genie", "lingbot", "gen3c"}
    camera_controlled = any(n in outputs_dir.name.lower() for n in _CAMERA_CONTROLLED_NAMES)
    if camera_controlled:
        print(f"[INFO] Camera-controlled model detected from folder name '{outputs_dir.name}' — requested_occlusion will be set to 'camera pan'.")

    run_dir.mkdir(parents=True, exist_ok=True)

    run_name = output_json.stem
    run_root = run_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    per_task_root = run_root / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Task resolution — sequential (final frame extraction happens here)
    # ---------------------------------------------------------------------------
    resolved_tasks: List[ResolvedTask] = resolve_tasks_from_outputs_json(
        tasks_root=task_root,
        outputs_json=output_json,
        run_dir=run_root,
        download_urls=args.download_urls,
        pattern=args.pattern,
        exclude_baseline=args.exclude_baseline,
    )

    print(f"[DONE] Resolved {len(resolved_tasks)} tasks into: {per_task_root}")

    # ---------------------------------------------------------------------------
    # Initialise summary.json skeleton (always, before any judging)
    # ---------------------------------------------------------------------------
    _init_summary_json(resolved_tasks, run_root)

    # ---------------------------------------------------------------------------
    # --init: write skeleton report stubs and exit — no LLM judging
    # ---------------------------------------------------------------------------
    if args.init:
        _write_init_stubs(resolved_tasks, per_task_root, camera_controlled=camera_controlled)
        print(f"[DONE] Initialized {len(resolved_tasks)} task(s) in: {run_root}")
        print(f"summary.json, judge_report.json, control_report.json, physics_report.json and se_report.json stubs written.")
        print(f"Start human_eval_server.py to begin human evaluation.")
        return

    # Keep the full list for the summary-append step at the end.
    all_resolved_tasks = resolved_tasks

    # ---------------------------------------------------------------------------
    # Skip tasks that are already fully evaluated (unless --overwrite)
    # ---------------------------------------------------------------------------
    if not args.overwrite:
        def _report_evaluated(path: Path, *fields: str, ensemble_size: int = 1, response_keys: tuple = ()) -> bool:
            """True if the file exists, every listed field is non-None, and (when ensemble_size > 1) each response_key array has >= ensemble_size entries."""
            if not path.exists():
                return False
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not all(data.get(f) is not None for f in fields):
                    return False
                if ensemble_size > 1:
                    for key in response_keys:
                        if len(data.get(key, [])) < ensemble_size:
                            return False
                return True
            except Exception:
                return False

        ens = args.ensemble_size
        pending, skipped = [], 0
        for task in resolved_tasks:
            task_dir = per_task_root / task.task_id
            control_done = _report_evaluated(
                task_dir / control_report_fn,
                "occlusion_done", "trigger_applied",
                ensemble_size=ens,
                response_keys=("occlusion_responses", "trigger_responses"),
            )
            physics_done = _report_evaluated(
                task_dir / physics_report_fn,
                "physical_inaccuracy",
                ensemble_size=ens,
                response_keys=("responses",),
            )
            se_done = _report_evaluated(
                task_dir / se_report_fn,
                "state_evol",
                ensemble_size=ens,
                response_keys=("responses",),
            )
            already_done = (
                (run_control and control_done or not run_control) and
                (run_physics and physics_done or not run_physics) and
                (run_state and se_done or not run_state)
            )
            if already_done:
                skipped += 1
            else:
                pending.append(task)
        if skipped:
            print(f"[SKIP] {skipped} already-evaluated tasks (use --overwrite to re-run)")
        resolved_tasks = pending

    # ---------------------------------------------------------------------------
    # Per-task eval — parallel (judge + control judge run together per task)
    # ---------------------------------------------------------------------------
    # judge_results: List[JudgeResult] = []  # OBSOLETE: binary judge questions
    control_results: List[ControlJudgeResult] = []
    physics_results: List[PhysicsJudgeResult] = []
    se_results: List[SEJudgeResult] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _eval_one_task,
                task,
                provider=args.vlm_provider,
                control_model=args.control_judge_model,
                physics_model=args.physics_judge_model,
                se_model=args.se_judge_model,
                control_report_fn=control_report_fn,
                physics_report_fn=physics_report_fn,
                se_report_fn=se_report_fn,
                run_control=run_control,
                run_physics=run_physics,
                run_state=run_state,
                camera_controlled=camera_controlled,
                ensemble_size=args.ensemble_size,
                ensemble_mode=args.ensemble_mode,
            ): task
            for task in resolved_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                jr, cr, phys, ser = future.result()
                if cr is not None: control_results.append(cr)
                if phys is not None: physics_results.append(phys)
                if ser is not None: se_results.append(ser)
            except Exception as e:
                print(f"[ERROR] {task.task_id}: {e}")
                failed += 1

    print(f"[DONE] Evaluated {len(resolved_tasks) - failed} tasks ({failed} failed)")

    # ---------------------------------------------------------------------------
    # Append control results to summary — sequential
    # Uses all_resolved_tasks so already-skipped tasks with existing
    # control_report.json are also merged into summary.json.
    # ---------------------------------------------------------------------------
    if run_control:
        append_control_results_to_summary(tasks=all_resolved_tasks, judge_slug=control_slug, report_filename=control_report_fn)
        print(f"[DONE] Controllability evaluation done")

    if run_physics:
        append_physics_results_to_summary(tasks=all_resolved_tasks, judge_slug=physics_slug, report_filename=physics_report_fn)
        print(f"[DONE] Physics evaluation done")

    if run_state:
        append_se_results_to_summary(tasks=all_resolved_tasks, judge_slug=se_slug, report_filename=se_report_fn)
        print(f"[DONE] State evolution evaluation done")

    return


if __name__ == "__main__":
    main()
