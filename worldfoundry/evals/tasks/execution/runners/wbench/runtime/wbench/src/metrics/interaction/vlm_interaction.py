"""
Interaction adherence — VLM-based per-turn evaluation.

3 metrics:
- event_edit_adherence: 5 progressive binary Q&A per event_edit turn
- subject_action_adherence: 5 progressive binary Q&A per subject_action turn
- perspective_switch_adherence: 4 questions (Q1+Q3+Q4+Q_cam) per ps turn,
  with separate frame sets (early/late/full)

Score = correct / total per turn, averaged across turns.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..vlm.vlm_evaluator import VLMClient

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "You are evaluating AI-generated video segments from interactive world models. "
    "Each video segment corresponds to one interaction turn where a specific event "
    "was instructed to happen.\n\n"
    "Your task: answer each question with ONLY 'Yes' or 'No' based on what you "
    "observe in the provided frames. Follow these guidelines:\n"
    "- These are AI-generated videos, not real footage. Minor visual artifacts, "
    "slight inconsistencies, or imperfect rendering should NOT cause a 'No' "
    "if the overall intent is recognizable.\n"
    "- Focus on whether the CORE semantic meaning of the described event is "
    "visually present, not whether every literal detail matches perfectly.\n"
    "- A long or detailed event description may only be partially depicted. "
    "Judge based on whether the main action or change is recognizable.\n"
    "- When in doubt between Yes and No, lean toward the interpretation that "
    "better matches what you actually see, not what you expect to see."
)

REASON_FORMAT = (
    "Use this EXACT format (max 50 words):\n"
    "Reason: <only describe specific issues found; "
    "if score is not full marks you MUST name the exact problem; "
    "do not describe what looks good>\n"
)


# ════════════════════════════════════════════════════════════════
#  Question generators
# ════════════════════════════════════════════════════════════════


def generate_event_edit_questions(action: str) -> list[dict[str, str]]:
    """5 progressive questions for event_edit turns. Score = correct/5."""
    return [
        {
            "sub_id": "Q1",
            "question": (
                f"Does the scene remain largely unchanged, "
                f"with no clear indication of '{action}' occurring?\n"
                f"(Criterion: answer No if there is ANY recognizable change "
                f"related to this event. Only answer Yes if the scene is "
                f"completely static or changes are entirely unrelated.)"
            ),
            "expected": "no",
        },
        {
            "sub_id": "Q2",
            "question": (
                f"Does the video show something resembling or consistent "
                f"with the described event: '{action}'?\n"
                f"(Criterion: answer Yes if the core action or change is "
                f"visually recognizable, even if not every detail matches "
                f"perfectly. Partial depiction counts as Yes.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q3",
            "question": (
                f"By the end of this segment, has the described event "
                f"'{action}' reached a clear conclusion or outcome?\n"
                f"(Criterion: answer Yes only if the event has a visible "
                f"end state or result. If the event is still ongoing, "
                f"just beginning, or only partially shown, answer No.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q4",
            "question": (
                f"Are the key details of '{action}' accurately depicted, "
                f"such as the correct objects, agents, directions, "
                f"and quantities involved?\n"
                f"(Criterion: focus on the key entities and actions "
                f"mentioned in the description. Ignore minor background "
                f"differences or irrelevant details.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q5",
            "question": (
                f"Does any unexpected object, entity, or visual anomaly "
                f"appear in this segment that is clearly unrelated "
                f"to '{action}'?\n"
                f"(Criterion: only answer Yes if something obviously "
                f"out-of-place appears. Natural consequences of the event "
                f"do not count. Minor AI rendering artifacts like texture "
                f"flickering do not count.)"
            ),
            "expected": "no",
        },
    ]


def generate_subject_action_questions(action: str) -> list[dict[str, str]]:
    """5 progressive questions for subject_action turns. Score = correct/5."""
    return [
        {
            "sub_id": "Q1",
            "question": (
                f"Does the subject remain idle or stationary, with no attempt "
                f"to perform '{action}'?\n"
                f"(Criterion: answer No if the subject shows ANY movement or "
                f"gesture related to this action. Only answer Yes if the "
                f"subject is completely still or doing something unrelated.)"
            ),
            "expected": "no",
        },
        {
            "sub_id": "Q2",
            "question": (
                f"Does the video show the subject performing something "
                f"resembling or consistent with: '{action}'?\n"
                f"(Criterion: answer Yes if the core action is visually "
                f"recognizable, even if not every detail matches perfectly. "
                f"Partial execution counts as Yes.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q3",
            "question": (
                f"By the end of this segment, has the subject's action "
                f"'{action}' reached a clear conclusion or outcome?\n"
                f"(Criterion: answer Yes only if there is a visible end "
                f"state or result of the action. If the action is still "
                f"ongoing, just beginning, or only partially shown, "
                f"answer No.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q4",
            "question": (
                f"Are the key details of '{action}' accurately depicted, "
                f"such as the correct subject, objects being interacted "
                f"with, and manner of the action?\n"
                f"(Criterion: focus on the key entities and the manner of "
                f"action mentioned in the description. Ignore minor "
                f"background differences or irrelevant details.)"
            ),
            "expected": "yes",
        },
        {
            "sub_id": "Q5",
            "question": (
                f"Does the subject exhibit any unnatural body movement, "
                f"impossible pose, or physically implausible interaction "
                f"while performing '{action}'?\n"
                f"(Criterion: only answer Yes for clearly unnatural "
                f"movements, impossible poses, or physically impossible "
                f"interactions. Minor AI rendering artifacts like slight "
                f"jitter or imperfect hand shape do not count.)"
            ),
            "expected": "no",
        },
    ]


PERSPECTIVE_TYPE_DESC = {
    "fp": (
        "first-person — camera through the character's eyes, looking "
        "outward at the world; no part of the character's own body is "
        "visible except possibly hands or a held weapon/tool"
    ),
    "tp": (
        "third-person — an external camera showing a character's body "
        "(back, side, or full figure) from behind, above, or at an angle"
    ),
    "scope": (
        "scoped / ADS — a magnified view through a weapon scope, "
        "binoculars, or similar optic, typically with a circular "
        "vignette, crosshair, reticle, or other scope overlay"
    ),
}
PERSPECTIVE_TYPE_SHORT = {
    "fp": "first-person",
    "tp": "third-person",
    "scope": "scoped/magnified",
}


def _parse_ps_action(action: str) -> tuple[str, str]:
    """Parse perspective_switch action string → (before_type, after_type)."""
    action_lower = action.lower().strip()
    if ":" in action_lower:
        action_lower = action_lower.split(":")[0].strip()
    if "fp_to_tp" in action_lower:
        return "fp", "tp"
    elif "tp_to_fp" in action_lower:
        return "tp", "fp"
    elif "tp_to_tp" in action_lower:
        return "tp", "tp"
    elif "fp_to_fp" in action_lower:
        return "fp", "fp"
    elif "fp_to_scope" in action_lower:
        return "fp", "scope"
    elif "scope_to_fp" in action_lower:
        return "scope", "fp"
    parts = action_lower.split("_to_")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "fp", "tp"


def generate_perspective_switch_questions(action: str) -> list[dict[str, str]]:
    """4 questions for perspective_switch turns. Strict: Q1∧Q3∧Q4 all pass."""
    before_type, after_type = _parse_ps_action(action)
    before_short = PERSPECTIVE_TYPE_SHORT.get(before_type, before_type)
    after_short = PERSPECTIVE_TYPE_SHORT.get(after_type, after_type)
    after_desc = PERSPECTIVE_TYPE_DESC.get(after_type, after_type)
    same_type = before_type == after_type

    if same_type:
        q1 = (
            f"You are shown 6 frames: the first 3 are from the BEGINNING "
            f"and the last 3 from the END of this segment. "
            f"Does the camera viewpoint change noticeably? Even though "
            f"both before and after are {before_short}, the camera "
            f"position, the followed subject, or the viewing angle should "
            f"shift significantly.\n"
            f"(Answer No only if the viewpoint stays essentially the same "
            f"throughout — same position, same subject, same angle.)"
        )
    else:
        q1 = (
            f"You are shown 6 frames: the first 3 are from the BEGINNING "
            f"and the last 3 from the END of this segment. "
            f"Does the perspective clearly change from {before_short} "
            f"to {after_short}?\n"
            f"(Answer No only if the perspective type remains "
            f"{before_short} throughout with no visible transition.)"
        )

    q3 = (
        f"You are shown 3 frames from the very END of this segment "
        f"(AFTER the perspective switch). Is the perspective at this "
        f"point {after_short}?\n"
        f"Definition: {after_desc}.\n"
        f"(Answer Yes if these frames match this perspective type.)"
    )

    if after_type == "tp":
        q4 = (
            "You are shown 3 frames from the END of this segment where "
            "the view should be third-person. Evaluate whether this is "
            "a STANDARD third-person camera setup. ALL of these must "
            "be true:\n"
            "  1. The character is roughly CENTERED in the frame\n"
            "  2. The camera is positioned BEHIND the character "
            "(over-the-shoulder or directly behind, NOT from the side "
            "or front)\n"
            "  3. The camera is at an appropriate DISTANCE — the "
            "character's upper body or full body is visible, not an "
            "extreme close-up or a far-away wide shot\n"
            "(Answer Yes only if all three criteria are met.)"
        )
    elif after_type == "fp":
        q4 = (
            "You are shown 3 frames from the END of this segment where "
            "the view should be first-person. Check whether the view "
            "is a VALID first-person perspective with NO impossible "
            "self-view artifacts.\n"
            "A valid first-person view must NOT show any of:\n"
            "  - The character's own FACE (front of head, facial "
            "features)\n"
            "  - The character's own BACK, rear of head, or shoulders "
            "from behind\n"
            "  - The character's own full body seen from outside\n"
            "Acceptable: hands, held weapon/tool, feet when looking "
            "down — these are normal in first-person.\n"
            "(Answer Yes if the view is a clean first-person perspective "
            "with none of the above violations. Answer No if the "
            "character's face, back, or full body is visible.)"
        )
    else:  # scope
        q4 = (
            "You are shown 3 frames from the END of this segment where "
            "the view should be through a scope or optic. Check whether "
            "the scoped view is VALID.\n"
            "A valid scope view should show at least one of:\n"
            "  - Circular vignette or scope ring border\n"
            "  - Crosshair, reticle, or aiming markers\n"
            "  - Visible magnification (objects appear closer than "
            "normal)\n"
            "  - HUD overlay, distance readout, or targeting display\n"
            "(Answer Yes if any scope/optic visual element is present. "
            "Answer No if the view looks like a normal unscoped "
            "perspective.)"
        )

    q_cam = (
        "You are shown frames sampled evenly across the ENTIRE segment. "
        "A perspective switch should be achieved by CAMERA movement "
        "(pulls back, pushes in, pans, or cuts to a new angle) or by "
        "a CINEMATIC CUT.\n"
        "It should NOT be achieved by CHARACTER movement alone.\n"
        "Is the viewpoint change achieved through camera motion or "
        "a cinematic cut (rather than just character movement)?\n"
        "(Answer Yes if the camera clearly moves, zooms, or cuts.)"
    )

    return [
        {"sub_id": "Q1", "question": q1, "expected": "yes", "frame_set": "all"},
        {"sub_id": "Q3", "question": q3, "expected": "yes", "frame_set": "late"},
        {"sub_id": "Q4", "question": q4, "expected": "yes", "frame_set": "late"},
        {"sub_id": "Q_cam", "question": q_cam, "expected": "yes", "frame_set": "full"},
    ]


# ════════════════════════════════════════════════════════════════
#  Frame sampling utilities
# ════════════════════════════════════════════════════════════════


def _sample_turn_frames(video_path: str, n_turns: int, target_fps: float = 3.0):
    """Sample frames per turn at target_fps. Returns dict[turn_num] → list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_turn = total_frames // n_turns
    step = max(1, round(fps / target_fps))

    turn_frames = {}
    for t in range(1, n_turns + 1):
        start = (t - 1) * frames_per_turn
        end = t * frames_per_turn if t < n_turns else total_frames
        turn_frames[t] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        fid = start
        while fid < end:
            ret, frame = cap.read()
            if not ret:
                break
            if (fid - start) % step == 0:
                turn_frames[t].append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            fid += 1
    cap.release()
    return turn_frames


def _sample_early_late(
    video_path: str, n_turns: int, n_each: int = 3, margin_ratio: float = 0.18, min_margin_sec: float = 0.6
):
    """Sample n_each early + n_each late frames per turn for perspective_switch.

    Margin = max(min_margin_sec, seg_dur * margin_ratio), capped at 40% of segment.
    Early frames from [start, start+margin], late from [end-margin, end].
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_turn = total_frames // n_turns

    turn_early = {}
    turn_late = {}
    for t in range(1, n_turns + 1):
        start = (t - 1) * frames_per_turn
        end = t * frames_per_turn if t < n_turns else total_frames
        seg_len = end - start
        seg_dur = seg_len / fps

        margin_sec = max(min_margin_sec, seg_dur * margin_ratio)
        margin_sec = min(margin_sec, seg_dur * 0.40)
        margin_frames = int(margin_sec * fps)

        early_end = start + margin_frames
        late_start = end - margin_frames

        early_indices = np.linspace(start, min(early_end, end - 1), n_each, dtype=int)
        late_indices = np.linspace(max(late_start, start), end - 1, n_each, dtype=int)

        early_frames, late_frames = [], []
        for idx in early_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                early_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for idx in late_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                late_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        turn_early[t] = early_frames
        turn_late[t] = late_frames

    cap.release()
    return turn_early, turn_late


# ════════════════════════════════════════════════════════════════
#  Evaluation functions
# ════════════════════════════════════════════════════════════════


def evaluate_event_edit(client: VLMClient, video_path: str, case_data: dict, nproc: int = 1) -> dict[str, Any]:
    """Evaluate event_edit adherence: 5 binary Q&A per event_edit turn."""
    interactions = case_data.get("interactions", [])
    n_turns = len(interactions)
    ee_turns = [(it["turn"], it["action"]) for it in interactions if it.get("type") == "event_edit"]

    if not ee_turns:
        return {"score": None, "skipped": True, "reason": "no event_edit turns"}

    turn_frames = _sample_turn_frames(video_path, n_turns)

    tasks = []  # (turn, sub_id, question, expected, frames)
    for turn, action in ee_turns:
        frames = turn_frames.get(turn, [])
        if not frames:
            continue
        for q in generate_event_edit_questions(action):
            tasks.append((turn, q["sub_id"], q["question"], q["expected"], frames))

    return _execute_binary_tasks(client, tasks, ee_turns, nproc, "event_edit_adherence")


def evaluate_subject_action(client: VLMClient, video_path: str, case_data: dict, nproc: int = 1) -> dict[str, Any]:
    """Evaluate subject_action adherence: 5 binary Q&A per subject_action turn."""
    interactions = case_data.get("interactions", [])
    n_turns = len(interactions)
    sa_turns = [(it["turn"], it["action"]) for it in interactions if it.get("type") == "subject_action"]

    if not sa_turns:
        return {"score": None, "skipped": True, "reason": "no subject_action turns"}

    turn_frames = _sample_turn_frames(video_path, n_turns)

    tasks = []
    for turn, action in sa_turns:
        frames = turn_frames.get(turn, [])
        if not frames:
            continue
        for q in generate_subject_action_questions(action):
            tasks.append((turn, q["sub_id"], q["question"], q["expected"], frames))

    return _execute_binary_tasks(client, tasks, sa_turns, nproc, "subject_action_adherence")


PS_SYSTEM_PROMPT = (
    "You are evaluating perspective (camera viewpoint) switches in "
    "AI-generated video segments from interactive world models. "
    "Each segment corresponds to one interaction turn where a specific "
    "perspective change was instructed.\n\n"
    "Key definitions:\n"
    "- First-person (FP): The camera is the character's eyes. No part of "
    "the character's body is visible except possibly hands/weapon.\n"
    "- Third-person (TP): The camera follows the character from behind or "
    "above. The character's full or partial body is visible in the frame.\n"
    "- Scope/ADS: The view is through a weapon scope or binoculars, "
    "typically showing a circular vignette, crosshair/reticle, and "
    "magnified view.\n\n"
    "Your task: answer each question based on what you observe in the "
    "provided frames. Always give a one-sentence reason, then answer "
    "Yes or No.\n"
    "- These are AI-generated videos. Minor visual artifacts should NOT "
    "cause a 'No' if the overall intent is recognizable.\n"
    "- Focus on whether the PERSPECTIVE TYPE is correct and the "
    "transition is coherent, not whether every detail is perfect."
)

PS_REASON_FORMAT = (
    "Respond in this EXACT format (no extra text):\n"
    "Reason: <one short sentence describing what you observe>\n"
    "Answer: Yes or No"
)


def evaluate_perspective_switch(client: VLMClient, video_path: str, case_data: dict, nproc: int = 1) -> dict[str, Any]:
    """Evaluate perspective_switch: Q1+Q3+Q4+Q_cam with separate frame sets.

    Scoring: strict = Q1∧Q3∧Q4 all pass → 1, else 0.
    """
    import re as _re

    interactions = case_data.get("interactions", [])
    n_turns = len(interactions)
    ps_turns = [it for it in interactions if it.get("type") == "perspective_switch"]

    if not ps_turns:
        return {"score": None, "skipped": True, "reason": "no perspective_switch turns"}

    turn_early, turn_late = _sample_early_late(video_path, n_turns, n_each=3)
    turn_full = _sample_turn_frames(video_path, n_turns, target_fps=3.0)

    # Build tasks
    all_tasks = []  # (turn, sub_id, question, expected, frames)
    for it in ps_turns:
        turn = it["turn"]
        action = it.get("action", "")
        early = turn_early.get(turn, [])
        late = turn_late.get(turn, [])
        full = turn_full.get(turn, [])

        for q in generate_perspective_switch_questions(action):
            fs = q["frame_set"]
            if fs == "early":
                images = early
            elif fs == "late":
                images = late
            elif fs == "full":
                images = full
            else:  # "all" = early + late
                images = early + late
            all_tasks.append((turn, q["sub_id"], q["question"], q["expected"], images))

    if not all_tasks:
        return {"score": None, "error": "no tasks generated"}

    # Execute concurrently — each question independent, with reason + system prompt
    results = [None] * len(all_tasks)

    def _run(idx):
        _, sub_id, question, _, images = all_tasks[idx]
        full_q = f"{question}\n\n{PS_REASON_FORMAT}"
        try:
            resp = client.ask(full_q, images, max_tokens=300, system_prompt=PS_SYSTEM_PROMPT)
            reason = ""
            rm = _re.search(r"Reason:\s*(.+?)(?=\n\s*Answer|\Z)", resp, _re.DOTALL | _re.IGNORECASE)
            if rm:
                reason = rm.group(1).strip()
            am = _re.search(r"Answer[:\s]+(yes|no)", resp, _re.IGNORECASE)
            answer = am.group(1).lower() == "yes" if am else resp.lower().strip().startswith("yes")
            return idx, answer, reason
        except Exception as e:
            logger.warning(f"PS task {idx} ({sub_id}) failed: {e}")
            return idx, None, str(e)

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = {executor.submit(_run, i): i for i in range(len(all_tasks))}
        for fut in as_completed(futures):
            idx, answer, reason = fut.result()
            results[idx] = (answer, reason)

    # Aggregate per-turn: strict = Q1∧Q3∧Q4
    turn_details = []
    scores = []
    for it in ps_turns:
        turn = it["turn"]
        q_map = {}
        q_reasons = {}
        for i, (t, sub_id, _question, expected, _) in enumerate(all_tasks):
            if t != turn:
                continue
            result_item = results[i]
            if result_item is not None:
                answer, reason = result_item
                if answer is not None:
                    expected_bool = expected.lower() == "yes"
                    q_map[sub_id] = answer == expected_bool
                else:
                    q_map[sub_id] = None
                q_reasons[sub_id] = reason
            else:
                q_map[sub_id] = None
                q_reasons[sub_id] = ""

        q1_ok = q_map.get("Q1") is True
        q3_ok = q_map.get("Q3") is True
        q4_ok = q_map.get("Q4") is True
        s = 1.0 if (q1_ok and q3_ok and q4_ok) else 0.0
        scores.append(s)
        turn_details.append(
            {
                "turn": turn,
                "action": it.get("action"),
                "score": s,
                "q_results": q_map,
                "q_reasons": q_reasons,
            }
        )

    avg_score = sum(scores) / len(scores) if scores else None
    return {
        "score": round(avg_score, 4) if avg_score is not None else None,
        "num_turns": len(ps_turns),
        "turn_details": turn_details,
    }


def _execute_binary_tasks(
    client: VLMClient, tasks: list[tuple], turns: list[tuple], nproc: int, metric_name: str
) -> dict[str, Any]:
    """Execute binary Q&A tasks — each question as independent VLM call with reason."""
    import re

    if not tasks:
        return {"score": None, "error": "no tasks"}

    results = [None] * len(tasks)  # (answer_bool, reason)

    def _run_single(idx):
        _, sub_id, question, expected, images = tasks[idx]
        full_q = f"{question}\n\n{REASON_FORMAT}Answer: Yes or No"
        try:
            response = client.ask(full_q, images, max_tokens=300, system_prompt=EVAL_SYSTEM_PROMPT)
            reason = ""
            reason_match = re.search(r"Reason:\s*(.+?)(?=\n\s*Answer|\Z)", response, re.DOTALL | re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()
            answer_match = re.search(r"Answer[:\s]+(yes|no)", response, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).lower() == "yes"
            else:
                answer = response.lower().strip().startswith("yes")
            return idx, answer, reason
        except Exception as e:
            logger.warning(f"{metric_name} task {idx} ({sub_id}) failed: {e}")
            return idx, None, str(e)

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = {executor.submit(_run_single, i): i for i in range(len(tasks))}
        for fut in as_completed(futures):
            idx, answer, reason = fut.result()
            results[idx] = (answer, reason)

    # Aggregate per turn
    turn_scores = []
    turn_details = []
    for turn_num, action in turns:
        correct, total = 0, 0
        q_details = []
        for i, (t, sub_id, _question, expected, _) in enumerate(tasks):
            if t != turn_num:
                continue
            answer, reason = results[i] if results[i] else (None, "")
            if answer is not None:
                expected_bool = expected.lower() == "yes"
                is_correct = answer == expected_bool
                if is_correct:
                    correct += 1
                total += 1
                q_details.append(
                    {
                        "sub_id": sub_id,
                        "correct": is_correct,
                        "answer": "yes" if answer else "no",
                        "expected": expected,
                        "reason": reason,
                    }
                )
            else:
                total += 1
                q_details.append(
                    {"sub_id": sub_id, "correct": None, "answer": "error", "expected": expected, "reason": reason}
                )

        turn_score = correct / total if total > 0 else 0.0
        turn_scores.append(turn_score)
        turn_details.append(
            {
                "turn": turn_num,
                "action": action,
                "score": round(turn_score, 4),
                "correct": correct,
                "total": total,
                "questions": q_details,
            }
        )

    avg = sum(turn_scores) / len(turn_scores) if turn_scores else None
    return {
        "score": round(avg, 4) if avg is not None else None,
        "num_turns": len(turns),
        "turn_details": turn_details,
    }
