# eval/physics_judge.py
"""
Physical inaccuracy evaluator (provider-agnostic).

Replaces the deprecated artifact + coherence judges with a single, orthogonal metric.

A PHYSICAL INACCURACY is any event in the VISIBLE portions of the video that either
violates the laws of physics or indicates the video was not a genuine continuous
recording. Three sub-types are recognized:

  TYPE 1 — Instantaneous violation  (single frame / short moment)
  TYPE 2 — Dynamic violation        (cause is visible; wrong effect follows)
  TYPE 3 — Continuity violation     (impossible scene change)

The key design constraint is orthogonality to state_evol:
  - The intended change NOT happening → state_evol failure, NOT physical_inaccuracy.
  - Events during occlusion are not assessable → excluded from this metric entirely.

Input:
- ResolvedTask (assumes task_resolver already ran)
- Uses task.wm_video as the WM output video

Output (per task):
- writes per_task/<task_id>/physics_report.json

Returns:
- PhysicsJudgeResult for downstream summary aggregation

Deps:
  pip install google-genai
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from eval.control_judge import _load_task_fields, _compute_requested_fields, _ensemble_decide
from eval.judge_client import make_judge_client
from eval.task_resolver import ResolvedTask
from eval.utils import _path_to_rel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhysicsJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    physical_inaccuracy: bool  # True = a physical inaccuracy was detected
    intended_state_evolution: str  # judge's restatement of the intended SE

    raw_text: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _load_init2final_edit_prompt(task_yaml_path: Path) -> str:
    """Return init2final_edit_prompt from the task YAML; raises if missing or empty."""
    data = yaml.safe_load(Path(task_yaml_path).read_text(encoding="utf-8")) or {}
    eval_section = data.get("evaluation") or {}
    if not isinstance(eval_section, dict):
        raise ValueError(f"'evaluation' field is not a mapping in {task_yaml_path}")
    val = " ".join(str(eval_section.get("init2final_edit_prompt", "") or "").split())
    if not val:
        raise ValueError(f"Missing 'evaluation.init2final_edit_prompt' in {task_yaml_path}")
    return val


def build_physics_judge_prompt(requested_occlusion: str, init2final_edit_prompt: str) -> str:
    occ = (requested_occlusion or "").strip()
    has_occlusion = occ and occ.lower() != "none"

    # Occlusion context: explain what the hidden period is so the judge does not
    # attempt to assess events that occurred while the scene was not visible.
    if has_occlusion:
        occlusion_note = (
            f"OCCLUSION CONTEXT: This task intentionally hides the scene via:\n"
            f"  \"{occ}\"\n\n"
            f"CRITICAL: Do NOT evaluate anything that happens DURING the occlusion —\n"
            f"that hidden period is assessed by a separate metric. Only evaluate\n"
            f"what is directly visible BEFORE and AFTER the occlusion.\n"
            f"Do NOT flag the occlusion mechanism itself as a physical inaccuracy.\n\n"
        )
    else:
        occlusion_note = (
            "OCCLUSION CONTEXT: None. The scene should be fully visible throughout.\n\n"
        )

    # State-evolution note: the intended change not happening is NOT physical_inaccuracy.
    edit = (init2final_edit_prompt or "").strip()
    if edit:
        state_evol_note = (
            f"INTENDED STATE EVOLUTION: The following describes the physical change\n"
            f"that was supposed to occur in this scene:\n"
            f"  \"{edit}\"\n\n"
            f"Before judging, extract a one-sentence plain-language summary of this\n"
            f"intended change and record it as \"intended_state_evolution\" in your JSON.\n\n"
            f"CRITICAL DISTINCTION — physical inaccuracy vs. state evolution failure:\n"
            f"  PHYSICAL INACCURACY: something WRONG actively happens — the video shows\n"
            f"    an event that is physically impossible or that indicates the video was\n"
            f"    tampered/cut. → flag here.\n"
            f"  STATE EVOLUTION FAILURE: the intended change simply does NOT happen —\n"
            f"    the scene looks frozen or nothing moves. → evaluated separately;\n"
            f"    do NOT flag here.\n\n"
        )
        intended_se_field = (
            "  \"intended_state_evolution\": \"one-sentence summary of the intended physical change\",\n"
        )
    else:
        state_evol_note = ""
        intended_se_field = ""

    return (
        "You are evaluating a generated video for PHYSICAL INACCURACY.\n\n"

        "A PHYSICAL INACCURACY is any event in the VISIBLE portions of the video that\n"
        "violates the laws of physics or indicates the video was not a genuine,\n"
        "continuous, uninterrupted recording of a single physical scene.\n\n"

        + occlusion_note
        + state_evol_note +

        "Evaluate ONLY what is directly visible. There are three types:\n\n"

        "TYPE 1 — Instantaneous violation  (single frame or very short moment)\n"
        "Something is physically wrong at a moment in time, independent of what came\n"
        "before or after.\n"
        "Examples:\n"
        "  - A rigid object (cup, block) deforms or changes shape without any force\n"
        "  - An object's material, color, or texture changes suddenly with no cause\n"
        "  - Two solid objects overlap, pass through each other, or merge\n"
        "  - An object teleports: its position jumps discontinuously with no continuous\n"
        "    motion visible between the two locations\n"
        "  - An object floats in mid-air against gravity with no support or attachment\n\n"

        "TYPE 2 — Dynamic violation  (cause is shown; wrong effect follows)\n"
        "Individual frames look plausible but the evolution over the VISIBLE portion\n"
        "defies physics. The cause must be clearly visible in the video.\n"
        "Examples:\n"
        "  - Water is continuously poured into a glass but the water level drops\n"
        "  - A block slides down a ramp then reverses upward with no visible push\n"
        "  - An object is clearly pushed but accelerates in the opposite direction\n\n"
        "IMPORTANT: If the cause happened DURING the occlusion (not visible), do NOT\n"
        "flag the effect as a TYPE 2 violation — you cannot assess a hidden cause.\n\n"

        "TYPE 3 — Continuity violation  (impossible scene change)\n"
        "The background or setting changes in a way that is physically impossible for\n"
        "a genuine single-take recording.\n"
        "Examples:\n"
        "  - The background or wall color/texture changes INSTANTANEOUSLY (zero\n"
        "    transitional frames), implying a hidden scene cut or edit\n"
        "  - The scene briefly blacks out then resumes with objects in different\n"
        "    positions or states, with no physical explanation for the change\n"
        "    (a blackout that corresponds to the INTENDED OCCLUSION and resumes the\n"
        "    same scene is NOT a violation — only flag unexpected resets)\n\n"

        "SEVERITY THRESHOLD: Flag only violations that a casual viewer would find\n"
        "clearly jarring and impossible. Do NOT flag:\n"
        "  - The intended change not happening (state evolution failure — separate metric)\n"
        "  - Events during the occlusion period (not visible, not assessable)\n"
        "  - Subtle rendering imperfections (minor flickering, slightly uneven textures)\n"
        "  - Small quantitative discrepancies (smoke a bit faster, water a bit slower)\n"
        "  - Video quality issues (blur, noise, compression artifacts)\n"
        "  - Any event that, while imperfect, is physically plausible in direction\n\n"

        "General guidance:\n"
        "  - Watch the full video before judging.\n"
        "  - Flag only CLEAR, VISUALLY JARRING violations.\n"
        "  - Use only visual evidence. Ignore timestamps, watermarks, UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        + intended_se_field +
        "  \"physical_inaccuracy\": true/false,\n"
        "  \"violation_type\": \"instantaneous\" | \"dynamic\" | \"continuity\" | \"none\",\n"
        "  \"notes\": \"brief description of what was observed and why it is or is not a physical inaccuracy\"\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------

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

def _parse_judge_output(raw_text: str) -> Dict[str, Any]:
    raw = (raw_text or "").strip()
    data = _try_load_json(_unwrap_full_fence(raw))
    if data is None:
        data = _try_load_json(raw)
    if data is None:
        obj = _extract_first_object(raw)
        if obj:
            data = _try_load_json(_unwrap_full_fence(obj))
    if data is None:
        raise ValueError(f"Could not parse physics judge JSON. Raw (truncated):\n{raw[:2000]}")
    return data


# ---------------------------------------------------------------------------
# Per-task evaluator
# ---------------------------------------------------------------------------

def evaluate_physics_one_task(
    task: ResolvedTask,
    *,
    provider: str = "gemini",
    model: str = "gemini-3.1-pro-preview",
    report_filename: str = "physics_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> PhysicsJudgeResult:
    print(f"[physics_judge] task={task.task_id}  provider={provider!r}  model={model!r}")

    client = make_judge_client(model=model, provider=provider)

    video_wm, camera_wm, camera_pose = _load_task_fields(Path(task.task_yaml))
    _, requested_occlusion = _compute_requested_fields(
        video_wm, camera_wm, camera_pose, camera_controlled=camera_controlled
    )
    init2final = _load_init2final_edit_prompt(Path(task.task_yaml))
    prompt = build_physics_judge_prompt(
        requested_occlusion=requested_occlusion,
        init2final_edit_prompt=init2final,
    )

    def _single_query(_: int) -> Dict[str, Any]:
        raw = client.judge(prompt=prompt, video_path=Path(task.wm_video))
        parsed = _parse_judge_output(raw)
        return {
            "physical_inaccuracy":    bool(parsed.get("physical_inaccuracy", False)),
            "violation_type":         str(parsed.get("violation_type", "none")).strip(),
            "intended_state_evolution": str(parsed.get("intended_state_evolution", "")).strip(),
            "notes":                  str(parsed.get("notes", "")).strip(),
            "raw_text":               raw,
        }

    n = max(1, ensemble_size)

    run_task_dir = Path(task.final_frame).parent
    report_path = run_task_dir / report_filename

    # Load existing responses so we only generate the delta needed.
    existing_responses: list = []
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            existing_responses = existing.get("responses", [])
        except Exception:
            pass

    n_needed = max(0, n - len(existing_responses))
    if n_needed > 0:
        with ThreadPoolExecutor(max_workers=n_needed) as ex:
            new_responses = list(ex.map(_single_query, range(n_needed)))
    else:
        new_responses = []
    responses = existing_responses + new_responses

    votes_inaccurate = sum(1 for r in responses if r["physical_inaccuracy"])
    physical_inaccuracy = _ensemble_decide(votes_inaccurate, len(responses), ensemble_mode)

    intended_state_evolution = responses[0]["intended_state_evolution"] if responses else ""
    if len(responses) > 1:
        notes = f"Ensemble {votes_inaccurate}/{len(responses)} members flagged a physical inaccuracy."
    else:
        notes = responses[0]["notes"] if responses else ""

    payload: Dict[str, Any] = {
        "task_id":                   task.task_id,
        "provider":                  client.provider,
        "model":                     model,
        "wm_video":                  _path_to_rel(task.wm_video),
        "requested_occlusion":       requested_occlusion,
        "init2final_edit_prompt":    init2final,
        "ensemble_size":             n,
        "ensemble_mode":             ensemble_mode,
        "intended_state_evolution":  intended_state_evolution,
        "physical_inaccuracy":       physical_inaccuracy,
        "notes":                     notes,
        "prompt":                    prompt,
        "raw_text":                  responses[0]["raw_text"],
    }
    if n > 1:
        payload["votes_physical_inaccuracy"] = votes_inaccurate
        payload["responses"] = responses
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return PhysicsJudgeResult(
        task_id=task.task_id,
        provider=client.provider,
        model=model,
        report_path=str(report_path),
        physical_inaccuracy=physical_inaccuracy,
        intended_state_evolution=intended_state_evolution,
        raw_text=responses[0]["raw_text"],
    )


def evaluate_physics_all_tasks(
    tasks: Sequence[ResolvedTask],
    *,
    provider: str = "gemini",
    model: str = "gemini-3.1-pro-preview",
    report_filename: str = "physics_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> List[PhysicsJudgeResult]:
    out: List[PhysicsJudgeResult] = []
    for t in tasks:
        out.append(
            evaluate_physics_one_task(
                t,
                provider=provider,
                model=model,
                report_filename=report_filename,
                camera_controlled=camera_controlled,
                ensemble_size=ensemble_size,
                ensemble_mode=ensemble_mode,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def append_physics_results_to_summary(
    tasks: List[ResolvedTask],
    *,
    judge_slug: str,
    report_filename: str,
) -> None:
    """
    For each task in the same run:
      - Load the judge-tagged physics report (report_filename) from the per-task folder
      - Insert physical_inaccuracy into summary.json under task["llm_evals"][judge_slug]

    Assumes all tasks belong to the same run.
    """
    if not tasks:
        return

    run_dir = Path(tasks[0].final_frame).parent.parent.parent
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "tasks" not in summary or not isinstance(summary["tasks"], list):
        raise ValueError("summary.json does not contain valid 'tasks' list.")

    task_entries = {t["task_id"]: t for t in summary["tasks"] if "task_id" in t}

    for task in tasks:
        run_task_dir = Path(task.final_frame).parent
        report_path = run_task_dir / report_filename
        if not report_path.exists() or task.task_id not in task_entries:
            continue
        data = json.loads(report_path.read_text(encoding="utf-8"))
        ev = task_entries[task.task_id].setdefault("llm_evals", {}).setdefault(judge_slug, {})
        ev["physical_inaccuracy"] = bool(data.get("physical_inaccuracy", False))

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
