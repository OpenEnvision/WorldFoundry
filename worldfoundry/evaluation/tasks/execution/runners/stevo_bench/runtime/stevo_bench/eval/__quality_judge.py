# eval/quality_judge.py
"""
Visual quality evaluator (provider-agnostic): artifact detection and coherence scoring.

Each metric is a separate Gemini query with its own prompt.

Input:
- ResolvedTask (assumes task_resolver already ran)
- Uses task.wm_video as the WM output video

Output (per task):
- writes per_task/<task_id>/artifact_report.json   (artifact)
- writes per_task/<task_id>/coherence_report.json  (coherence)

Returns:
- ArtifactJudgeResult  for artifact
- CoherenceJudgeResult for coherence

Deps:
  pip install google-genai
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from eval.judge_client import make_judge_client
from eval.control_judge import _load_task_fields, _compute_requested_fields
from eval.task_resolver import ResolvedTask
from eval.utils import _path_to_rel


def _ensemble_decide(votes_true: int, n: int, mode: str) -> bool:
    """
    Aggregate n binary votes into a single decision.

    mode="majority"       — True if strictly more than half vote True (ties → False).
    mode="unanimous"      — True only if ALL n vote True; default False otherwise.
    mode="unanimous_true" — False only if ALL n vote False; default True otherwise.
                            Use this when false-negatives are the dominant error and
                            you want the ensemble to require unanimous agreement to
                            conclude the negative outcome (e.g. coherence=False).
    """
    if mode == "unanimous_true":
        return votes_true > 0          # False only if every member voted False
    elif mode == "unanimous":
        return votes_true == n         # True only if every member voted True
    else:                              # majority
        return votes_true * 2 > n     # strict majority; ties → False


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


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArtifactJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    artifact: bool
    intended_state_evolution: str  # Gemini's one-sentence restatement of the intended SE

    raw_text: str


@dataclass
class CoherenceJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    coherence: bool  # True = video IS coherent (no failures detected)

    raw_text: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_artifact_judge_prompt(requested_occlusion: str, init2final_edit_prompt: str) -> str:
    # Occlusion context note — identical logic to coherence judge
    occ = (requested_occlusion or "").strip()
    has_occlusion = occ and occ.lower() != "none"
    if has_occlusion:
        occlusion_note = (
            f"INTENDED OCCLUSION CONTEXT: This task intentionally includes the\n"
            f"following occlusion: \"{occ}\". Any scene darkening, blackout, or\n"
            f"view obstruction that corresponds to this description is expected\n"
            f"behaviour and must NOT be flagged as an artifact.\n"
            f"Do NOT flag any irregularities related to the intended occlusion as artifacts.\n"
            f"If the intended occlusion does not correctly happen, DO NOT flag it as artifact either.\n\n"
        )
    else:
        occlusion_note = (
            "INTENDED OCCLUSION CONTEXT: There is no intended occlusion in this\n"
            "task. You can ignore this reminder.\n\n"
        )

    # State evolution context note — the edit prompt describes exactly what
    # physical change should occur. If the change is absent (nothing happens),
    # that is a STATE EVOLUTION FAILURE, not an artifact.
    edit = (init2final_edit_prompt or "").strip()
    if edit:
        state_evol_note = (
            f"INTENDED STATE EVOLUTION: The following image-edit prompt describes the\n"
            f"main physical change that was intended to occur in the scene:\n"
            f"\"{edit}\"\n\n"
            f"Before judging artifacts, extract a one-sentence plain-language summary\n"
            f"of this intended change (e.g. \"the block slides to the bottom of the ramp\")\n"
            f"and record it as \"intended_state_evolution\" in your JSON response.\n\n"
            f"CRITICAL DISTINCTION — artifact vs. state evolution failure:\n"
            f"  - ARTIFACT: something WRONG actively happens (a physically impossible\n"
            f"    event occurs, e.g. an object deforms with no force, two solids pass\n"
            f"    through each other, water level drops while being poured in).\n"
            f"  - STATE EVOLUTION FAILURE: the intended change simply does NOT happen\n"
            f"    (e.g. the scene looks frozen, nothing moves, the intended effect is\n"
            f"    absent). This is evaluated separately and must NOT be flagged here.\n\n"
            f"Only flag as artifact if the video shows an active physical violation,\n"
            f"not merely a lack of the intended change.\n\n"
        )
    else:
        state_evol_note = ""

    intended_se_field = (
        "  \"intended_state_evolution\": \"one-sentence summary of the intended physical change\",\n"
        if edit else ""
    )

    return (
        "You are evaluating a generated video for physical plausibility.\n\n"

        "An ARTIFACT is any event or behavior in the video that actively violates the\n"
        "laws of physics or is otherwise physically impossible. There are two types:\n\n"

        "TYPE 1 — Instantaneous violations\n"
        "Something is physically wrong in a single frame or very short moment,\n"
        "independent of what came before or after.\n"
        "Examples:\n"
        "  - A rigid object (cup, block, bottle) deforms or changes shape without\n"
        "    any force being applied\n"
        "  - An object's color or texture changes suddenly without a physical cause\n"
        "  - Two solid objects overlap, pass through each other, or merge together\n"
        "  - An object abruptly moves or changes velocity with no visible contact,\n"
        "    force, or physical cause\n"
        "  - An object floats in mid-air against gravity, without support or attachment"

        "TYPE 2 — Dynamic violations\n"
        "Individual frames look plausible, but the evolution of the scene over time\n"
        "violates physics. The cause is shown, but the WRONG effect follows.\n"
        "Examples:\n"
        "  - Water is continuously poured into a glass but the level drops or stays\n"
        "    flat (the wrong direction of change given the cause).\n"
        "  - A block slides down a ramp but then reverses upward with no visible push.\n"
        "  - An object is set in motion but accelerates with no driving force.\n\n"

        "For TYPE 2, ask yourself: a cause is clearly shown — does the effect that\n"
        "follows actively defy physics? If yes, that is an artifact. If instead the\n"
        "scene simply does not change much (the cause has no visible effect), that is\n"
        "a state evolution failure, not an artifact — do NOT flag it here.\n\n"

        + occlusion_note
        + state_evol_note +

        "SEVERITY THRESHOLD: Only flag violations that a casual viewer would find\n"
        "clearly jarring and physically impossible. Do NOT flag:\n"
        "  - Subtle rendering imperfections (slightly uneven textures, minor flickering)\n"
        "  - Small quantitative discrepancies (smoke that diffuses slightly faster than\n"
        "    expected, a water level that rises a bit too slowly)\n"
        "  - Sand or granular material appearing or clumping in plausible ways\n"
        "  - Any effect that, while imperfect, is physically plausible\n"
        "  - Video quality issues such as blurriness or noise\n\n"

        "General guidance:\n"
        "- Watch the full video before making a judgment.\n"
        "- Flag only CLEAR, VISUALLY JARRING violations that a layperson would\n"
        "  immediately notice as impossible.\n"
        "- Use only visual evidence. Ignore timestamps, watermarks, and UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        + intended_se_field +
        "  \"artifact\": true/false,\n"
        "  \"notes\": \"brief description of what was observed and why it is or is not an artifact\"\n"
        "}\n"
    )


def build_coherence_judge_prompt(requested_occlusion: str) -> str:
    occ = (requested_occlusion or "").strip()
    is_camera_pan = occ.lower() == "camera pan"
    has_occlusion = occ and occ.lower() != "none"

    if is_camera_pan:
        occlusion_context = (
            "INTENDED OCCLUSION: This task uses a CAMERA PAN to hide the subject.\n"
            "The camera sweeps the subject out of frame, then pans back to reveal it.\n"
            "This is intentional and expected behaviour.\n\n"

            "OCCLUSION vs. DISAPPEARANCE for camera pan:\n"
            "  - Subject leaving the frame because the camera panned away = OCCLUSION.\n"
            "    Mark subject_disappears = false in this case.\n"
            "  - Only mark subject_disappears = true if the subject vanishes while\n"
            "    still within the camera's field of view, with no physical cause.\n\n"

            "Camera pan vs. scene cut:\n"
            "  A pan has continuous intermediate frames of spatial motion connecting\n"
            "  the two viewpoints. A cut jumps INSTANTANEOUSLY with zero transitional\n"
            "  frames. If any sliding motion is visible, it is a pan — not a cut.\n"
            "  Do NOT mark background_cut = true for a camera pan, even a fast or\n"
            "  direction-reversing one.\n\n"
        )
        no_flag_list = (
            "  - subject_disappears = false when the subject leaves frame due to camera pan.\n"
            "  - background_cut = false for any camera pan motion, including fast or\n"
            "    direction-reversing pans (pan away to hide, pan back to reveal).\n"
        )
    elif has_occlusion:
        occlusion_context = (
            f"INTENDED OCCLUSION: This task intentionally includes the following:\n"
            f"  \"{occ}\"\n"
            "The subject becoming hidden by this mechanism (lights off, curtain drawn,\n"
            "smoke filling the frame, object placed in front, etc.) is expected.\n\n"

            "OCCLUSION vs. DISAPPEARANCE:\n"
            "  - Subject hidden by the intended occlusion mechanism = OCCLUSION.\n"
            "    Mark subject_disappears = false in this case.\n"
            "  - Only mark subject_disappears = true if the subject vanishes with\n"
            "    no physical cause and no correspondence to the intended occlusion.\n"
            "  - A blackout or darkening that corresponds to the intended occlusion\n"
            "    and resumes the same scene is NOT a coherence failure.\n\n"
        )
        no_flag_list = (
            "  - subject_disappears = false when the intended occlusion accounts for\n"
            "    the subject becoming hidden.\n"
            "  - blackout_reset = false for a blackout that matches the intended\n"
            "    occlusion and resumes the same scene without repositioning objects.\n"
        )
    else:
        occlusion_context = (
            "INTENDED OCCLUSION: None. The scene should remain continuously visible\n"
            "with no intentional hiding of the subject.\n\n"

            "OCCLUSION vs. DISAPPEARANCE:\n"
            "  Since no occlusion is intended, any disappearance of the main subject\n"
            "  without a visible physical cause should be marked subject_disappears = true.\n\n"
        )
        no_flag_list = ""

    return (
        "You are evaluating a generated video for temporal and scene coherence.\n\n"

        "A COHERENCE FAILURE means the video does not appear to be a single,\n"
        "continuous, uninterrupted recording of one scene.\n\n"

        + occlusion_context +

        "STEP 1 — Identify the main subject.\n"
        "Identify the primary object or entity the video is about. Record it as main_subject.\n\n"

        "STEP 2 — Occlusion vs. disappearance.\n"
        "For this task, determine whether any disappearance of the main subject\n"
        "is explained by the INTENDED OCCLUSION described above.\n"
        "  OCCLUSION: subject becomes invisible because something hides it\n"
        "    (camera moves away, lights off, curtain drawn, smoke fills frame, etc.).\n"
        "    The subject is still physically present — just hidden. NOT a failure.\n"
        "  DISAPPEARANCE: subject is simply absent with no physical or occlusion cause.\n"
        "    This IS a coherence failure.\n\n"

        "STEP 3 — Fill in the checklist.\n"
        "Answer each item true or false based only on visual evidence:\n\n"

        "  subject_disappears — Did the main subject vanish from the scene with NO\n"
        "    physical explanation and NO correspondence to the intended occlusion?\n"
        "    (If the intended occlusion accounts for the disappearance, mark false.)\n\n"

        "  subject_reappears — After the subject was hidden by occlusion, did it\n"
        "    reappear in a state or position that cannot be explained by the visible\n"
        "    process? (A normal reveal after intended occlusion is fine — flag only\n"
        "    unexpected discontinuous jumps upon reappearance.)\n\n"

        "  background_cut — Did the background or environment change INSTANTANEOUSLY\n"
        "    (zero transitional frames), implying a hidden edit or scene cut?\n"
        "    Example: table surface or wall color abruptly changes; a window switches\n"
        "    sides. A camera pan has continuous motion — it is NOT a cut.\n\n"

        "  state_jump — Did the main subject's state change discontinuously, with no\n"
        "    visible transition? Example: a full glass is suddenly empty; an intact ice\n"
        "    cube is suddenly fully melted, with the melting process entirely skipped.\n\n"

        "  blackout_reset — Did the scene briefly black out and then resume in a visibly\n"
        "    different configuration (objects repositioned or in different states) with\n"
        "    no physical explanation? (A blackout matching the intended occlusion that\n"
        "    resumes the same scene is NOT a reset.)\n\n"

        "  teleport — Did any object's position jump discontinuously between frames,\n"
        "    with no continuous motion connecting the two locations?\n"
        "    Example: a block on the left of a ramp is suddenly at the bottom right.\n\n"

        "COHERENCE RULE:\n"
        "  coherence = false  if ANY checklist item above is true.\n"
        "  coherence = true   if ALL checklist items are false.\n\n"

        "Do NOT flag:\n"
        "  - Normal continuous physical processes, even surprising or fast ones.\n"
        "  - Video quality issues such as blurriness or noise.\n"
        + no_flag_list + "\n"

        "General guidance:\n"
        "  - Watch the full video before filling the checklist.\n"
        "  - Flag only CLEAR failures. Unusual but continuous events are not failures.\n"
        "  - Use only visual evidence. Ignore timestamps, watermarks, and UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        "  \"main_subject\": \"<primary object or entity in the video>\",\n"
        "  \"subject_disappears\": true/false,\n"
        "  \"subject_reappears\": true/false,\n"
        "  \"background_cut\": true/false,\n"
        "  \"state_jump\": true/false,\n"
        "  \"blackout_reset\": true/false,\n"
        "  \"teleport\": true/false,\n"
        "  \"coherence\": true/false,\n"
        "  \"notes\": \"brief explanation referencing specific observations\"\n"
        "}\n"
        "(coherence: true = video IS coherent — all checklist items false;\n"
        " coherence: false = a coherence failure was detected.)\n"
    )


# ---------------------------------------------------------------------------
# Robust JSON parsing (handles reasoning / fenced blocks / pure JSON)
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

def _parse_judge_output(raw_text: str, judge_name: str) -> Dict[str, Any]:
    raw = (raw_text or "").strip()
    data = _try_load_json(_unwrap_full_fence(raw))
    if data is None:
        data = _try_load_json(raw)
    if data is None:
        obj = _extract_first_object(raw)
        if obj:
            data = _try_load_json(_unwrap_full_fence(obj))
    if data is None:
        raise ValueError(
            f"Could not parse {judge_name} JSON. Raw (truncated):\n{raw[:2000]}"
        )
    return data


# ---------------------------------------------------------------------------
# Artifact judge (one Gemini query per task)
# ---------------------------------------------------------------------------

def evaluate_artifact_one_task(
    task: ResolvedTask,
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "artifact_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> ArtifactJudgeResult:
    client = make_judge_client(model=model, provider=provider)
    print(f"[artifact_judge] occ_client: model={client.model!r}")

    video_wm, camera_wm, camera_pose = _load_task_fields(Path(task.task_yaml))
    _, requested_occlusion = _compute_requested_fields(
        video_wm, camera_wm, camera_pose, camera_controlled=camera_controlled
    )
    init2final = _load_init2final_edit_prompt(Path(task.task_yaml))
    prompt = build_artifact_judge_prompt(
        requested_occlusion=requested_occlusion,
        init2final_edit_prompt=init2final,
    )

    def _single_query(_: int) -> Dict[str, Any]:
        raw = client.judge(prompt=prompt, video_path=Path(task.wm_video))
        parsed = _parse_judge_output(raw, "artifact")
        return {
            "artifact": bool(parsed.get("artifact", False)),
            "intended_state_evolution": str(parsed.get("intended_state_evolution", "")).strip(),
            "notes": str(parsed.get("notes", "")).strip(),
            "raw_text": raw,
        }

    n = max(1, ensemble_size)
    if n > 1:
        with ThreadPoolExecutor(max_workers=n) as ex:
            responses = list(ex.map(_single_query, range(n)))
    else:
        responses = [_single_query(0)]

    votes_artifact = sum(1 for r in responses if r["artifact"])
    artifact = _ensemble_decide(votes_artifact, n, ensemble_mode)

    intended_state_evolution = responses[0]["intended_state_evolution"]
    if n > 1:
        notes = f"Ensemble {votes_artifact}/{n} members flagged an artifact."
    else:
        notes = responses[0]["notes"]

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path = run_task_dir / report_filename

    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "provider": client.provider,
        "model": model,
        "wm_video": _path_to_rel(task.wm_video),
        "requested_occlusion": requested_occlusion,
        "init2final_edit_prompt": init2final,
        "ensemble_size": n,
        "ensemble_mode": ensemble_mode,
        "intended_state_evolution": intended_state_evolution,
        "artifact": artifact,
        "notes": notes,
        "prompt": prompt,
        "raw_text": responses[0]["raw_text"],
    }
    if n > 1:
        payload["votes_artifact"] = votes_artifact
        payload["responses"] = responses
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return ArtifactJudgeResult(
        task_id=task.task_id,
        provider=client.provider,
        model=model,
        report_path=str(report_path),
        artifact=artifact,
        intended_state_evolution=intended_state_evolution,
        raw_text=responses[0]["raw_text"],
    )


def evaluate_artifact_all_tasks(
    tasks: Sequence[ResolvedTask],
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "artifact_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> List[ArtifactJudgeResult]:
    out: List[ArtifactJudgeResult] = []
    for t in tasks:
        out.append(
            evaluate_artifact_one_task(
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


def append_artifact_results_to_summary(
    tasks: List[ResolvedTask],
    *,
    judge_slug: str,
    report_filename: str,
) -> None:
    """
    For each task in the same run:
      - Load the judge-tagged artifact report (report_filename) from the per-task folder
      - Insert artifact into summary.json under task["llm_evals"][judge_slug]

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
        ev["artifact"] = bool(data.get("artifact", False))

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Coherence judge
# ---------------------------------------------------------------------------

def evaluate_coherence_one_task(
    task: ResolvedTask,
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "coherence_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> CoherenceJudgeResult:
    client = make_judge_client(model=model, provider=provider)
    print(f"[coherence_judge] occ_client: model={client.model!r}")

    video_wm, camera_wm, camera_pose = _load_task_fields(Path(task.task_yaml))
    _, requested_occlusion = _compute_requested_fields(
        video_wm, camera_wm, camera_pose, camera_controlled=camera_controlled
    )
    prompt = build_coherence_judge_prompt(requested_occlusion)

    def _single_query(_: int) -> Dict[str, Any]:
        raw = client.judge(prompt=prompt, video_path=Path(task.wm_video))
        parsed = _parse_judge_output(raw, "coherence")
        # coherence: true = video IS coherent (default True if model omits the key)
        return {
            "main_subject":       str(parsed.get("main_subject", "")).strip(),
            "subject_disappears": bool(parsed.get("subject_disappears", False)),
            "subject_reappears":  bool(parsed.get("subject_reappears", False)),
            "background_cut":     bool(parsed.get("background_cut", False)),
            "state_jump":         bool(parsed.get("state_jump", False)),
            "blackout_reset":     bool(parsed.get("blackout_reset", False)),
            "teleport":           bool(parsed.get("teleport", False)),
            "coherence":          bool(parsed.get("coherence", True)),
            "notes":              str(parsed.get("notes", "")).strip(),
            "raw_text":           raw,
        }

    n = max(1, ensemble_size)
    if n > 1:
        with ThreadPoolExecutor(max_workers=n) as ex:
            responses = list(ex.map(_single_query, range(n)))
    else:
        responses = [_single_query(0)]

    votes_coherent = sum(1 for r in responses if r["coherence"])
    coherence = _ensemble_decide(votes_coherent, n, ensemble_mode)

    main_subject = responses[0]["main_subject"]
    if n > 1:
        notes = f"Ensemble {votes_coherent}/{n} members found the video coherent."
    else:
        notes = responses[0]["notes"]

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path = run_task_dir / report_filename

    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "provider": client.provider,
        "model": model,
        "wm_video": _path_to_rel(task.wm_video),
        "requested_occlusion": requested_occlusion,
        "ensemble_size": n,
        "ensemble_mode": ensemble_mode,
        "main_subject": main_subject,
        "coherence": coherence,
        "notes": notes,
        "prompt": prompt,
        "raw_text": responses[0]["raw_text"],
    }
    if n == 1:
        r0 = responses[0]
        payload["subject_disappears"] = r0["subject_disappears"]
        payload["subject_reappears"]  = r0["subject_reappears"]
        payload["background_cut"]     = r0["background_cut"]
        payload["state_jump"]         = r0["state_jump"]
        payload["blackout_reset"]     = r0["blackout_reset"]
        payload["teleport"]           = r0["teleport"]
    if n > 1:
        payload["votes_coherent"] = votes_coherent
        payload["responses"] = responses
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return CoherenceJudgeResult(
        task_id=task.task_id,
        provider=client.provider,
        model=model,
        report_path=str(report_path),
        coherence=coherence,
        raw_text=responses[0]["raw_text"],
    )


def evaluate_coherence_all_tasks(
    tasks: Sequence[ResolvedTask],
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "coherence_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> List[CoherenceJudgeResult]:
    out: List[CoherenceJudgeResult] = []
    for t in tasks:
        out.append(
            evaluate_coherence_one_task(
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


def append_coherence_results_to_summary(
    tasks: List[ResolvedTask],
    *,
    judge_slug: str,
    report_filename: str,
) -> None:
    """
    For each task in the same run:
      - Load the judge-tagged coherence report (report_filename) from the per-task folder
      - Insert coherence into summary.json under task["llm_evals"][judge_slug]

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
        ev["coherence"] = bool(data.get("coherence", True))

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
