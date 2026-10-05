# eval/se_judge.py
"""
Two-step state-evolution (SE) judge (provider-agnostic).

Step 1 — predict:
  Given the task's initial frame image and optional trigger prompt (camera_WM),
  call the judge VLM to predict what physical process should occur and produce
  a one-sentence concise description.

Step 2 — verify (ensemble-able):
  Given the WM output video and the predicted process, call the judge VLM to
  verify whether the process actually happened during the occlusion / camera movement.

Prompts and decision logic match eval/eval_se/gemini_evaluator.py.

Output (per task):
- writes per_task/<task_id>/se_report.json

Returns:
- SEJudgeResult for downstream summary aggregation
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.control_judge import _load_task_fields, _ensemble_decide
from eval.judge_client import make_judge_client
from eval.task_resolver import ResolvedTask
from eval.utils import _path_to_rel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SEJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    state_evol: bool
    expected_process_concise: str
    raw_text: str  # verification response (first member if ensemble)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_process_line(response_text: str) -> str:
    """Extract concise process from 'Process: ...' line (last occurrence).
    Falls back to the full response text if no such line is found."""
    for line in reversed(response_text.strip().split("\n")):
        stripped = line.strip()
        if stripped.lower().startswith("process:"):
            return stripped[8:].strip()
    return response_text.strip()


def _parse_verdict(response_text: str) -> bool:
    """Extract True/False from 'VERDICT: Yes/No'. Defaults to False if unclear."""
    for line in response_text.split("\n"):
        lower = line.lower().strip()
        if lower.startswith("verdict:"):
            verdict_part = lower[len("verdict:"):].strip()
            if "yes" in verdict_part:
                return True
            elif "no" in verdict_part:
                return False
    return False


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_se_predict_prompt(action_prompt: Optional[str]) -> str:
    """Prompt for step 1: init image (+ optional trigger) → expected physical process."""
    if action_prompt:
        return (
            f"You are analyzing a video generation task. You are given:\n"
            f"1. An initial image showing the starting state\n"
            f"2. An action/event that will occur: \"{action_prompt}\"\n"
            f"\n"
            f"Based on the initial image and the action described, what physical process or state evolution should happen?\n"
            f"Describe the expected changes in the scene, including:\n"
            f"- What objects or elements will change\n"
            f"- How they will change (e.g., shape, position, state)\n"
            f"- What the final state should look like\n"
            f"\n"
            f"Be specific and concrete in your description.\n"
            f"\n"
            f"After your full explanation, provide a one-sentence summary on the last line in this format:\n"
            f"Process: [concise one-sentence description of the expected evolution]"
        )
    else:
        return (
            "You are analyzing a video generation task. You are given an initial image showing the starting state.\n"
            "\n"
            "Based on what you see in the image, what physical process or state evolution appears to be set up or about to happen?\n"
            "Describe the expected changes in the scene, including:\n"
            "- What objects or elements will change\n"
            "- How they will change (e.g., shape, position, state)\n"
            "- What the final state should look like\n"
            "\n"
            "Be specific and concrete in your description.\n"
            "\n"
            "After your full explanation, provide a one-sentence summary on the last line in this format:\n"
            "Process: [concise one-sentence description of the expected evolution]"
        )


def build_se_verify_prompt(expected_process: str, camera_controlled: bool) -> str:
    """Prompt for step 2: video + expected process → VERDICT."""
    if camera_controlled:
        occlusion_context = (
            "\nIMPORTANT: This video includes CAMERA MOVEMENT where the camera turns away from the main object.\n"
            "You must specifically check: Did the expected process HAPPEN while the camera was turned away "
            "(when the main object was not visible)?\n"
            "This is critical - the process should evolve even when the camera is not looking at it."
        )
    else:
        occlusion_context = (
            "\nIMPORTANT: This video includes an OCCLUSION where an object blocks the view of the scene.\n"
            "You must specifically check: Did the expected process CONTINUE HAPPENING during the occlusion "
            "(when the view was blocked)?\n"
            "This is critical - the process should evolve even when we cannot see it."
        )

    return (
        "You are a CONSERVATIVE and SKEPTICAL video evaluator. Your job is to verify whether a video shows "
        "an expected physical process.\n"
        "\n"
        f"The expected process is:\n{expected_process}\n"
        f"{occlusion_context}\n"
        "\n"
        "Watch the video carefully and answer:\n"
        "1. Did the expected process happen in the video? (Yes/No)\n"
        "2. Provide a detailed explanation of what is the object state right before it becomes invisible, "
        "and right after it is seen again.\n"
        "3. Compare what happened in the video to what was expected\n"
        "4. SPECIFICALLY address: Did the process continue/happen during the occlusion or camera movement?\n"
        "5. If the process did not happen as expected, explain what was different or missing\n"
        "\n"
        "Please ignore physical violations or artifacts. Only focus on whether the key process happened "
        "DURING the occlusion or camera movement.\n"
        "\n"
        "Format your response as:\n"
        "VERDICT: [Yes/No]\n"
        "EXPLANATION: [Your detailed explanation]"
    )


# ---------------------------------------------------------------------------
# Per-task evaluator
# ---------------------------------------------------------------------------

def evaluate_se_one_task(
    task: ResolvedTask,
    *,
    provider: str = "gemini",
    model: str = "gemini-3-pro-preview",
    report_filename: str = "se_report.json",
    camera_controlled: bool = False,
    ensemble_size: int = 1,
    ensemble_mode: str = "majority",
) -> SEJudgeResult:
    client = make_judge_client(model=model, provider=provider)

    # Load camera_WM as the trigger-action context for step 1 prediction.
    # For image_implied tasks camera_WM is empty → action_prompt=None (image-only prediction).
    # For simple_trigger tasks camera_WM holds the kick-off action.
    _, camera_wm, _ = _load_task_fields(Path(task.task_yaml))
    action_prompt: Optional[str] = camera_wm.strip() or None

    n = max(1, ensemble_size)

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path = run_task_dir / report_filename

    # Load existing responses so we only generate the delta needed.
    # Also reuse step-1 output (expected_process_concise) to avoid re-running it.
    existing_responses: list = []
    concise_process = ""
    full_prediction = ""
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            existing_responses = existing.get("responses", [])
            concise_process    = existing.get("expected_process_concise", "")
            full_prediction    = existing.get("expected_process_full", "")
        except Exception:
            pass

    # Step 1: predict expected process from init frame — skip if already stored.
    predict_prompt = build_se_predict_prompt(action_prompt)
    if not concise_process:
        full_prediction = client.judge_image(
            prompt=predict_prompt,
            image_path=Path(task.init_frame),
        )
        concise_process = _parse_process_line(full_prediction)

    # Step 2: verify process in video (ensemble over this step)
    verify_prompt = build_se_verify_prompt(concise_process, camera_controlled=camera_controlled)

    def _single_verify(_: int) -> Dict[str, Any]:
        raw = client.judge(prompt=verify_prompt, video_path=Path(task.wm_video))
        return {
            "state_evol": _parse_verdict(raw),
            "raw_text": raw,
        }

    n_needed = max(0, n - len(existing_responses))
    if n_needed > 0:
        with ThreadPoolExecutor(max_workers=n_needed) as ex:
            new_responses = list(ex.map(_single_verify, range(n_needed)))
    else:
        new_responses = []
    responses = existing_responses + new_responses

    votes_true = sum(1 for r in responses if r["state_evol"])
    state_evol = _ensemble_decide(votes_true, len(responses), ensemble_mode)

    if len(responses) > 1:
        notes = f"Ensemble {votes_true}/{len(responses)} members found state evolution occurred."
    else:
        notes = ""

    payload: Dict[str, Any] = {
        "task_id":                    task.task_id,
        "provider":                   client.provider,
        "model":                      model,
        "wm_video":                   _path_to_rel(task.wm_video),
        "action_prompt":              action_prompt,
        "camera_controlled":          camera_controlled,
        "ensemble_size":              n,
        "ensemble_mode":              ensemble_mode,
        "expected_process_concise":   concise_process,
        "expected_process_full":      full_prediction,
        "state_evol":                 state_evol,
        "notes":                      notes,
        "predict_prompt":             predict_prompt,
        "verify_prompt":              verify_prompt,
        "raw_text":                   responses[0]["raw_text"],
    }
    if n > 1:
        payload["votes_state_evol"] = votes_true
        payload["responses"] = responses

    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return SEJudgeResult(
        task_id=task.task_id,
        provider=client.provider,
        model=model,
        report_path=str(report_path),
        state_evol=state_evol,
        expected_process_concise=concise_process,
        raw_text=responses[0]["raw_text"],
    )


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def append_se_results_to_summary(
    tasks: List[ResolvedTask],
    *,
    judge_slug: str,
    report_filename: str,
) -> None:
    """
    For each task in the same run:
      - Load the judge-tagged SE report (report_filename) from the per-task folder
      - Insert state_evol into summary.json under task["llm_evals"][judge_slug]

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
        ev["state_evol"] = bool(data.get("state_evol", False))

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
