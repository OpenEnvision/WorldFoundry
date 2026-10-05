# eval/control_judge.py
"""
Controllability evaluator (provider-agnostic).

Input:
- ResolvedTask (assumes task_resolver already ran)
- Uses:
    - task.task_yaml to load prompts.video_WM
    - task.wm_video as the WM output video

Output (per task):
- writes per_task/<task_id>/control_report.json

Returns:
- ControlJudgeResult for downstream summary

Deps:
  pip install pyyaml google-genai
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.utils import _path_to_rel


def _ensemble_decide(votes_true: int, n: int, mode: str) -> bool:
    """
    Aggregate n binary votes into a single decision.

    mode="majority"       — True if strictly more than half vote True (ties → False).
    mode="unanimous"      — True only if ALL n vote True; default False otherwise.
    mode="unanimous_true" — False only if ALL n vote False; default True otherwise.
    """
    if mode == "unanimous_true":
        return votes_true > 0
    elif mode == "unanimous":
        return votes_true == n
    else:
        return votes_true * 2 > n

import yaml

from eval.judge_client import make_judge_client
from eval.task_resolver import ResolvedTask


@dataclass
class ControlJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    requested_occlusion: str
    requested_trigger: str
    occlusion_done: bool
    trigger_applied: bool

    raw_text: str


def _load_task_fields(task_yaml_path: Path) -> Tuple[str, str, str]:
    """Return (video_WM, camera_WM, camera_pose) from the task YAML. Empty string if absent."""
    data = yaml.safe_load(Path(task_yaml_path).read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", {}) or {}
    video_wm = prompts.get("video_WM", "") or ""
    camera_wm = prompts.get("camera_WM", "") or ""
    raw_pose = (data.get("camera_control") or {}).get("HY-WorldPlay") or ""
    camera_pose = str(raw_pose).strip() if raw_pose else ""
    return video_wm.strip(), camera_wm.strip(), camera_pose


_OCCLUSION_SIGNAL = "after a pause"  # case-insensitive; present in all occlusion prompts, never in baselines


def _extract_occlusion_clause(video_wm: str) -> str:
    """Extract just the occlusion description from video_WM.

    Canonical prompt templates (tasks_v4):

    image_implied (no semicolon):
      "<Occlusion — object held between camera and scene>, completely blocking <subject>
       from view for a prolonged duration. After a pause, <reveal>.
       The camera remains stationary; no scene cuts; no new people or objects enter the scene.
       Smooth, continuous evolution with only objects present in the initial scene."
      → strips everything from ". After a pause" onward (reveal + trailing directives);
        returns the occlusion description.

    simple_kickoff ("<kick-off>; while <process>, <occlusion> for a prolonged duration.
       After a pause, <reveal>. The camera remains stationary; …
       Smooth, continuous evolution with only objects present in the initial scene."):
      → strips from ". After a pause" onward, then takes the part after the first ";" —
        the "while <process>, <occlusion>" clause.

    Legacy patterns (tasks_v2) also handled:
      A: ", then, after a pause, ..."  (image_implied old style)
      B: ". After a pause, ..."        (simple_kickoff old style)
    """
    # Collapse any YAML-introduced line-wrapping whitespace
    text = " ".join(video_wm.split())

    # Strip the reveal + trailing continuity clause in both patterns:
    #   A: ", then, after a pause, ..."  (legacy image_implied)
    #   B: ". After a pause, ..."        (v4 style, both task types)
    text = re.sub(
        r",?\s*then[,\s]+after\s+a\s+pause\b.*", "", text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    text = re.sub(
        r"\.\s*after\s+a\s+pause\b.*", "", text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # For simple_kickoff prompts the kick-off and occlusion are separated by ";".
    # Take only the part after the first ";" — the "while process, occlusion" clause.
    if ";" in text:
        text = text.split(";", 1)[1].strip()

    return text


def _compute_requested_fields(
    video_wm: str,
    camera_wm: str,
    camera_pose: str,
    camera_controlled: bool = False,
) -> Tuple[str, str]:
    """Compute (requested_trigger, requested_occlusion) from task prompt fields.

    Args:
        camera_controlled: True for models that receive camera_WM + trajectory
            (e.g. HY-WorldPlay, LingBot, GEN3C) rather than video_WM. For these models
            the in-scene occlusion described in video_WM is irrelevant — the model
            can only implement a camera pan, so requested_occlusion is always
            derived from camera_pose, not from video_WM.

    requested_trigger:
      - camera_WM if non-empty (simple_kickoff tasks always set camera_WM to the
        kick-off sentence; image_implied tasks always leave it empty)
      - else "none"

    requested_occlusion (text model, camera_controlled=False):
      - occlusion clause extracted from video_WM if it contains the occlusion signal
        ("after a pause"); see _extract_occlusion_clause for the two supported shapes:
          * image_implied: single clause ending in ", then, after a pause, …"
          * simple_kickoff: "<kick-off>; while <process>, <occlusion>. After a pause, …"
      - else "camera pan" if a camera pose string is present
      - else "none" (pure baseline — no occlusion of any kind requested)

    requested_occlusion (camera model, camera_controlled=True):
      - always "camera pan" (these models always occlude via camera trajectory)
    """
    # --- requested_trigger ---
    # camera_WM is the authoritative source for the kick-off action.
    # image_implied tasks always have camera_WM empty (no kickoff needed).
    # simple_kickoff tasks always have camera_WM filled with the kickoff sentence.
    if camera_wm:
        requested_trigger = camera_wm
    else:
        requested_trigger = "none"

    # --- requested_occlusion ---
    if camera_controlled:
        # Camera model: video_WM was never sent to the model, so any in-scene
        # occlusion described there is irrelevant.  These models always implement
        # occlusion via a camera pan, regardless of whether camera_pose is set.
        requested_occlusion = "camera pan"
    else:
        # Text model: inspect video_WM for in-scene occlusion signal first.
        _has_occlusion = _OCCLUSION_SIGNAL in video_wm.lower()
        if _has_occlusion:
            requested_occlusion = _extract_occlusion_clause(video_wm)
        elif camera_pose:
            requested_occlusion = "camera pan"
        else:
            requested_occlusion = "none"

    return requested_trigger, requested_occlusion


def build_occlusion_judge_prompt(requested_occlusion: str) -> str:
    occlusion_display = requested_occlusion or "none"

    return (
        "You are evaluating a generated video for whether a partial observability mechanism was correctly applied.\n\n"

        "The following has been determined from the task specification:\n"
        f"  requested_occlusion: {occlusion_display}\n\n"

        "PARTIAL OBSERVABILITY:\n"
        "The video may hide the scene via in-scene occlusion (lights off, object placed in front,\n"
        "smoke filling the view, etc.) OR by camera pan (camera moves away, taking the subject\n"
        "fully out of frame). Both mechanisms are valid.\n"
        "Do NOT penalise a mismatch between requested_occlusion and the mechanism shown in the video.\n"
        "Occlusion success is judged solely by whether the subject became invisible, regardless of how.\n\n"

        "Evaluate ONE thing from the GENERATED video:\n\n"

        "occlusion_done — Were the key subject and dynamic successfully hidden from view\n"
        "   (became invisible) at the appropriate moment, by ANY mechanism?\n"
        "   Camera pan that moves the subject fully out of frame counts as occlusion_done = True,\n"
        "   even if requested_occlusion described a different in-scene method — and vice versa.\n"
        "   TRUE only if the main subject/area became clearly invisible or fully obscured.\n"
        "   FALSE if the dynamic process already completed BEFORE the occlusion took place.\n"
        "   FALSE if the scene was never hidden at all.\n"
        "   Set to true if requested_occlusion is \"none\".\n\n"

        "General guidance:\n"
        "- Use only visual evidence from the video.\n"
        "- Ignore timestamps, watermarks, subtitles, and UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        "  \"occlusion_done\": true/false,\n"
        "  \"notes\": \"optional short explanation\"\n"
        "}\n"
    )


def build_trigger_judge_prompt(requested_trigger: str) -> str:
    trigger_display = requested_trigger or "none"

    return (
        "You are evaluating a generated video for whether a trigger action was correctly applied.\n\n"

        "The following has been determined from the task specification:\n"
        f"  requested_trigger: {trigger_display}\n\n"

        "Evaluate ONE thing from the GENERATED video:\n\n"

        "trigger_applied — Did the requested kickoff action occur?\n"
        "   TRUE if the action itself happened or its physical effect is clearly visible.\n"
        "   Set to true if requested_trigger is \"none\".\n\n"

        "General guidance:\n"
        "- Use only visual evidence from the video.\n"
        "- Ignore timestamps, watermarks, subtitles, and UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        "  \"trigger_applied\": true/false,\n"
        "  \"notes\": \"optional short explanation\"\n"
        "}\n"
    )


# ---- robust JSON parsing (handles reasoning / fenced blocks / pure JSON) ----

def _unwrap_full_fence(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _extract_first_object(s: str) -> Optional[str]:
    m = re.search(r"\{.*\}", s or "", flags=re.DOTALL)
    return m.group(0).strip() if m else None

def _try_load_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def parse_control_judge_output(raw_text: str) -> Dict[str, Any]:
    raw = (raw_text or "").strip()

    # 1) pure JSON or fully fenced JSON
    data = _try_load_json(_unwrap_full_fence(raw))
    if data is None:
        data = _try_load_json(raw)

    # 2) extract first {...} anywhere (handles reasoning before JSON)
    if data is None:
        obj = _extract_first_object(raw)
        if obj:
            data = _try_load_json(_unwrap_full_fence(obj))

    if data is None:
        raise ValueError(f"Could not parse control judge JSON. Raw (truncated):\n{raw[:2000]}")
    return data


def evaluate_control_one_task(
    task: ResolvedTask,
    *,
    provider: str = "gemini",
    model: str = "gemini-3.1-pro-preview",
    report_filename: str = "control_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> ControlJudgeResult:
    print(f"[control_judge] task={task.task_id}  provider={provider!r}  model={model!r}")

    # Two independent clients — one per judge so they can run fully in parallel.
    occ_client  = make_judge_client(model=model, provider=provider)
    trig_client = make_judge_client(model=model, provider=provider)

    video_wm_prompt, camera_wm_prompt, camera_pose = _load_task_fields(Path(task.task_yaml))
    requested_trigger, requested_occlusion = _compute_requested_fields(
        video_wm_prompt, camera_wm_prompt, camera_pose, camera_controlled=camera_controlled
    )

    occlusion_prompt = build_occlusion_judge_prompt(requested_occlusion)
    trigger_prompt   = build_trigger_judge_prompt(requested_trigger)

    def _occ_query(_: int) -> Dict[str, Any]:
        raw    = occ_client.judge(prompt=occlusion_prompt, video_path=Path(task.wm_video))
        parsed = parse_control_judge_output(raw)
        return {
            "occlusion_done": bool(parsed.get("occlusion_done", False)),
            "notes":          str(parsed.get("notes", "")).strip(),
            "raw_text":       raw,
        }

    def _trig_query(_: int) -> Dict[str, Any]:
        raw    = trig_client.judge(prompt=trigger_prompt, video_path=Path(task.wm_video))
        parsed = parse_control_judge_output(raw)
        return {
            "trigger_applied": bool(parsed.get("trigger_applied", False)),
            "notes":           str(parsed.get("notes", "")).strip(),
            "raw_text":        raw,
        }

    n = max(1, ensemble_size)

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path  = run_task_dir / report_filename

    # Load existing responses so we only generate the delta needed.
    existing_occ_responses: list = []
    existing_trig_responses: list = []
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            existing_occ_responses  = existing.get("occlusion_responses", [])
            existing_trig_responses = existing.get("trigger_responses", [])
        except Exception:
            pass

    n_needed_occ  = max(0, n - len(existing_occ_responses))
    n_needed_trig = max(0, n - len(existing_trig_responses))

    new_occ_responses: list = []
    new_trig_responses: list = []
    max_workers = n_needed_occ + n_needed_trig
    if max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            occ_futures  = [ex.submit(_occ_query,  i) for i in range(n_needed_occ)]
            trig_futures = [ex.submit(_trig_query, i) for i in range(n_needed_trig)]
            new_occ_responses  = [f.result() for f in occ_futures]
            new_trig_responses = [f.result() for f in trig_futures]

    occ_responses  = existing_occ_responses  + new_occ_responses
    trig_responses = existing_trig_responses + new_trig_responses

    votes_occlusion = sum(1 for r in occ_responses  if r["occlusion_done"])
    votes_trigger   = sum(1 for r in trig_responses if r["trigger_applied"])
    occlusion_done  = _ensemble_decide(votes_occlusion, len(occ_responses), ensemble_mode)
    trigger_applied = _ensemble_decide(votes_trigger,   len(trig_responses), ensemble_mode)

    n_occ  = len(occ_responses)
    n_trig = len(trig_responses)
    if n_occ > 1 or n_trig > 1:
        notes = (
            f"Ensemble occlusion_done {votes_occlusion}/{n_occ}, "
            f"trigger_applied {votes_trigger}/{n_trig}."
        )
    else:
        parts = [p for p in (occ_responses[0]["notes"], trig_responses[0]["notes"]) if p]
        notes = "; ".join(parts)

    payload: Dict[str, Any] = {
        "task_id":             task.task_id,
        "provider":            occ_client.provider,
        "model":               model,
        "wm_video":            _path_to_rel(task.wm_video),
        "video_WM_prompt":     video_wm_prompt,
        "camera_WM_prompt":    camera_wm_prompt,
        "requested_occlusion": requested_occlusion,
        "requested_trigger":   requested_trigger,
        "ensemble_size":       n,
        "ensemble_mode":       ensemble_mode,
        "occlusion_done":      occlusion_done,
        "trigger_applied":     trigger_applied,
        "notes":               notes,
        "occlusion_prompt":    occlusion_prompt,
        "occlusion_raw_text":  occ_responses[0]["raw_text"],
        "trigger_prompt":      trigger_prompt,
        "trigger_raw_text":    trig_responses[0]["raw_text"],
    }
    if n > 1:
        payload["votes_occlusion_done"]  = votes_occlusion
        payload["occlusion_responses"]   = occ_responses
        payload["votes_trigger_applied"] = votes_trigger
        payload["trigger_responses"]     = trig_responses
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return ControlJudgeResult(
        task_id=task.task_id,
        provider=occ_client.provider,
        model=model,
        report_path=str(report_path),
        requested_occlusion=requested_occlusion,
        requested_trigger=requested_trigger,
        occlusion_done=occlusion_done,
        trigger_applied=trigger_applied,
        raw_text=occ_responses[0]["raw_text"],
    )


def evaluate_control_all_tasks(
    tasks: Sequence[ResolvedTask],
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "control_report.json",
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> List[ControlJudgeResult]:
    out: List[ControlJudgeResult] = []
    for t in tasks:
        out.append(
            evaluate_control_one_task(
                t,
                provider=provider,
                model=model,
                report_filename=report_filename,
                ensemble_size=ensemble_size,
                ensemble_mode=ensemble_mode,
            )
        )
    return out


def append_control_results_to_summary(
    tasks: List[ResolvedTask],
    *,
    judge_slug: str,
    report_filename: str,
) -> None:
    """
    For each task in the same run:
      - Load the judge-tagged control report (report_filename) from the per-task folder
      - Insert occlusion_done and trigger_applied into summary.json under
        task["llm_evals"][judge_slug]

    Assumes all tasks belong to the same run.
    """
    if not tasks:
        return

    first_task_dir = Path(tasks[0].final_frame).parent
    run_dir = first_task_dir.parent.parent
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "tasks" not in summary or not isinstance(summary["tasks"], list):
        raise ValueError("summary.json does not contain valid 'tasks' list.")

    task_entries = {t["task_id"]: t for t in summary["tasks"] if "task_id" in t}

    for task in tasks:
        task_id = task.task_id
        run_task_dir = Path(task.final_frame).parent
        report_path = run_task_dir / report_filename

        if not report_path.exists() or task_id not in task_entries:
            continue

        data = json.loads(report_path.read_text(encoding="utf-8"))
        occlusion_done  = bool(data.get("occlusion_done", False))
        trigger_applied = bool(data.get("trigger_applied", False))

        ev = task_entries[task_id].setdefault("llm_evals", {}).setdefault(judge_slug, {})
        ev["occlusion_done"]  = occlusion_done
        ev["trigger_applied"] = trigger_applied

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
