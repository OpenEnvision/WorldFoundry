"""
Causal fidelity — VLM-based dual-track physics evaluation.

Method: Track1 (overall physics + causal consistency) + Track2 (per-dimension scoring).
- Track1: VLM rates entire video 0-3 on rendering physics and causal logic
- Track2: VLM rates each annotated physics dimension 0-3
- Final score = (T1 + mean(T2)) / 2, or T1 alone if no Track2 dims
- Normalized: score / 3.0 → [0, 1]

Requires: case_data['causal_fidelity']['dims'] for Track2 dimension selection.
Frames: 3fps full video, original resolution (no resize).
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from PIL import Image

from ..vlm.vlm_evaluator import VLMClient

logger = logging.getLogger(__name__)

METRIC_NAME = "causal_fidelity"

PHYSICS_DIMENSIONS = {
    "fluid_dynamics": {
        "name": "Fluid & Smoke",
        "prompt": (
            "Rate fluid, smoke, and related effects (0-3).\n"
            "3 = Realistic flow, splash, smoke rise, underwater drift\n"
            "2 = Minor artifacts but behavior is recognizable\n"
            "1 = Notable issues: liquid defies gravity, smoke static, no bubbles\n"
            "0 = Fluid/smoke behavior completely unrealistic or absent"
        ),
    },
    "collision": {
        "name": "Collision & Clipping",
        "prompt": (
            "Rate collision and solid body integrity (0-3).\n"
            "ONLY evaluate collisions/contacts that actually occur.\n"
            "3 = All contacts are natural, no clipping or pass-through\n"
            "2 = Minor clipping but mostly plausible contacts\n"
            "1 = Obvious clipping: limbs pass through objects or surfaces\n"
            "0 = Severe clipping, objects completely ignore each other"
        ),
    },
    "surface_interaction": {
        "name": "Surface Tracks & Imprints",
        "prompt": (
            "Rate surface interaction and imprint accuracy (0-3).\n"
            "3 = Clear, appropriate marks visible (footprints in snow, tracks in sand)\n"
            "2 = Some marks visible but incomplete or inconsistent\n"
            "1 = No marks visible on a surface that should clearly show them\n"
            "0 = Surface shows no response at all to contact"
        ),
    },
    "deformation": {
        "name": "Deformation & Destruction",
        "prompt": (
            "Rate deformation and destruction effects (0-3).\n"
            "3 = Realistic shattering, crumbling, bending, debris scatter\n"
            "2 = Minor issues but deformation is recognizable\n"
            "1 = Deformation clearly unnatural or inconsistent\n"
            "0 = No deformation when expected, or completely unrealistic"
        ),
    },
    "wind_weather": {
        "name": "Wind & Environmental Forces",
        "prompt": (
            "Rate wind and environmental force effects (0-3).\n"
            "3 = Natural swaying, drifting, fluttering of affected objects\n"
            "2 = Minor issues but environmental effects are recognizable\n"
            "1 = Effects are stiff, static, or inconsistent with the scene\n"
            "0 = No environmental effects visible when expected"
        ),
    },
    "reflection": {
        "name": "Reflection & Lighting",
        "prompt": (
            "Rate the accuracy of SPECIAL reflection and lighting effects (0-3).\n"
            "Evaluate ONLY optical effects that are actually VISIBLE in the video. "
            "Do NOT penalize for effects that are absent from the scene.\n"
            "Check for:\n"
            "- Water/mirror reflections: do they match the shape and motion of the source?\n"
            "- Point light sources: is the illumination range and falloff plausible?\n"
            "- Shadows: do they point in the correct direction relative to the light?\n"
            "- Metallic/glass surfaces: do they show plausible environment reflections?\n\n"
            "3 = All visible optical effects are accurate and physically consistent\n"
            "2 = Minor errors in one effect (e.g. reflection slightly misaligned)\n"
            "1 = Clear errors: reflections show wrong content, shadows point wrong way, "
            "or lighting does not respond to scene changes\n"
            "0 = Optical effects are completely wrong or absent when clearly expected"
        ),
    },
    "human_physics": {
        "name": "Human Motion & Expression",
        "prompt": (
            "Rate human motion naturalness and facial expression quality (0-3).\n"
            "Focus on MOVEMENT ACROSS FRAMES, not just single-frame anatomy:\n"
            "- Does the character move fluidly between poses, or jerk/teleport?\n"
            "- Are body movements biomechanically plausible (natural gait, "
            "realistic reach, believable weight shifts)?\n"
            "- Do limbs move in coordinated, purposeful ways, or flail randomly?\n"
            "- Is the face coherent (no melting, distortion, or uncanny shifts)?\n"
            "Minor structural issues (slight hand distortion, brief extra finger) "
            "should NOT heavily impact the score if overall motion is natural.\n"
            "3 = Smooth, natural human motion throughout; face is coherent\n"
            "2 = Mostly natural motion with minor stiffness or brief glitches\n"
            "1 = Clearly unnatural: jerky limbs, puppet-like movement, "
            "frozen poses, or obvious face distortion\n"
            "0 = Severely broken: limbs flail impossibly, body contorts "
            "unnaturally, face melts or deforms grotesquely"
        ),
    },
}

TRACK1_SYSTEM_PROMPT = (
    "You are evaluating the RENDERING-LEVEL physical plausibility and "
    "causal consistency of an AI-generated video. You will see frames "
    "sampled from the ENTIRE video in temporal order.\n\n"
    "IMPORTANT RULES:\n"
    "- Do NOT judge whether depicted actions are possible in the real world. "
    "Fantasy/sci-fi elements are acceptable if the setting calls for them.\n"
    "- Do NOT penalize camera behavior (tilting, shaking, panning, zooming). "
    "Camera motion is NOT a physics issue.\n"
    "- ONLY evaluate the physics of OBJECTS and CHARACTERS.\n"
    "- ANY violation, no matter how brief or small in screen area, counts. "
    "A single clear clipping event or causal error is enough to deduct.\n\n"
    "Scan ALL frames for:\n"
    "A. RENDERING PHYSICS\n"
    "  1. Solid body integrity — no clipping\n"
    "  2. Consistent gravity\n"
    "  3. Motion continuity — no teleporting\n"
    "  4. Object permanence — no random appear/vanish\n"
    "  5. Natural deformation\n"
    "  6. Character physics — natural movement\n\n"
    "B. CAUSAL CONSISTENCY\n"
    "  NOTE: Instructed actions and their consequences are EXPECTED — "
    "do NOT penalize them. Only penalize UNINTENDED changes.\n"
    "  7. No effect without cause (for non-instructed changes)\n"
    "  8. No cause without effect\n"
    "  9. No something from nothing\n\n"
    "Scoring (0-3):\n"
    "3 = Good: physics and causality natural and consistent throughout\n"
    "2 = Fair: 1-2 noticeable problems anywhere in the video\n"
    "1 = Poor: multiple violations or one severe breakdown\n"
    "0 = Failure: physics/causality mostly broken"
)

REASON_FORMAT = (
    "Use this EXACT format (max 50 words):\n"
    "Reason: <only describe specific issues found; "
    "if score is not full marks you MUST name the exact problem; "
    "do not describe what looks good>\n"
)

SCORE_SUFFIX = "Score: a single integer from 0 to 3"

TRACK2_SYSTEM_PROMPT = (
    "You are evaluating a specific physics phenomenon across an ENTIRE "
    "AI-generated video. You will see frames from the full video.\n"
    "Score based ONLY on the specified physics dimension.\n"
    "ANY violation of this dimension anywhere in the video, no matter how "
    "brief or localized, should lower the score.\n"
    "Minor AI rendering artifacts are acceptable. Focus on whether the "
    "specific physical behavior is plausible throughout."
)


def _parse_score(response: str, max_score: int = 3) -> tuple[int | None, str]:
    """Extract score from 'Reason: ...\nScore: N' format."""
    reason = ""
    # Try to extract Reason:
    reason_match = re.search(r"Reason:\s*(.+?)(?=\n\s*Score|\Z)", response, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    # Extract Score:
    score_match = re.search(r"Score[:\s]+(\d)", response, re.IGNORECASE)
    if score_match:
        return min(int(score_match.group(1)), max_score), reason

    # Fallback: any single digit 0-3
    match = re.search(r"\b([0-3])\b", response)
    if match:
        return int(match.group(1)), reason or response
    return None, reason or response


def evaluate_case(
    client: VLMClient,
    frames: list[Image.Image],
    case_data: dict,
    nproc: int = 1,
) -> dict[str, Any]:
    """
    Dual-track physics evaluation on full video frames (3fps, original resolution).

    Args:
        client: VLMClient instance
        frames: sampled video frames (3fps, NO resize)
        case_data: case JSON with optional causal_fidelity.dims
        nproc: concurrent VLM calls
    """
    settings = case_data.get("settings", {})
    scene = settings.get("scene", {})
    scene_env = scene.get("environment", "")
    scene_name = scene.get("name", "")

    interactions = case_data.get("interactions", [])
    actions = [
        it.get("action", "") for it in interactions if it.get("type") != "perspective_switch" and it.get("action")
    ]
    actions_str = " -> ".join(actions) if actions else "navigation"

    # Get annotated dims
    cf_data = case_data.get("causal_fidelity", {})
    dims = cf_data.get("dims", [])
    valid_dims = [d for d in dims if d in PHYSICS_DIMENSIONS]

    # --- Track 1: overall physics + causal ---
    t1_question = ""
    if scene_name:
        t1_question += f"Scene context: {scene_env} / {scene_name}\n"
    t1_question += f"Actions in this video: {actions_str}\n"
    t1_question += (
        "\nRate the overall rendering physics AND causal consistency "
        "of this ENTIRE video from 0 to 3. Any physics violation at "
        "any point should be noted and penalized."
    )

    # --- Track 2: per-dimension ---
    with ThreadPoolExecutor(max_workers=max(1, nproc)) as executor:
        # Submit T1
        t1_future = executor.submit(_ask_score, client, t1_question, frames, TRACK1_SYSTEM_PROMPT)

        # Submit T2 for each dim
        t2_futures = {}
        for dim_key in valid_dims:
            dim = PHYSICS_DIMENSIONS[dim_key]
            q = f"Actions in this video: {actions_str}\n"
            q += f"\nEvaluating: {dim['name']}\n{dim['prompt']}"
            t2_futures[executor.submit(_ask_score, client, q, frames, TRACK2_SYSTEM_PROMPT)] = dim_key

        # Collect T1
        try:
            t1_score, t1_reason = t1_future.result()
        except Exception as e:
            logger.warning(f"Track1 failed: {e}")
            t1_score, t1_reason = None, str(e)

        # Collect T2
        t2_details = []
        valid_t2_scores = []
        for fut in as_completed(t2_futures):
            dim_key = t2_futures[fut]
            try:
                s, r = fut.result()
            except Exception as e:
                logger.warning(f"Track2 {dim_key} failed: {e}")
                s, r = None, str(e)
            t2_details.append(
                {
                    "dim": dim_key,
                    "dim_name": PHYSICS_DIMENSIONS[dim_key]["name"],
                    "score": s,
                    "max": 3,
                    "reason": r,
                }
            )
            if s is not None:
                valid_t2_scores.append(s)

    t2_details.sort(key=lambda x: x["dim"])

    has_t2 = len(valid_t2_scores) > 0
    t2_avg = sum(valid_t2_scores) / len(valid_t2_scores) if has_t2 else None

    if has_t2 and t1_score is not None and t2_avg is not None:
        final = (t1_score + t2_avg) / 2.0
    elif t1_score is not None:
        final = float(t1_score)
    else:
        final = None

    return {
        "score": round(final / 3.0, 4) if final is not None else None,
        "raw_score": round(final, 2) if final is not None else None,
        "max": 3,
        "num_frames": len(frames),
        "track1": {"score": t1_score, "max": 3, "reason": t1_reason},
        "track2": {
            "avg_score": round(t2_avg, 2) if t2_avg is not None else None,
            "max": 3,
            "num_dims": len(valid_dims),
            "details": t2_details,
        },
    }


def _ask_score(
    client: VLMClient, question: str, frames: list[Image.Image], system_prompt: str
) -> tuple[int | None, str]:
    """Ask VLM to score 0-3. System prompt as separate role, question + format in user."""
    full_question = f"{question}\n\n{REASON_FORMAT}{SCORE_SUFFIX}"
    response = client.ask(full_question, frames, max_tokens=300, system_prompt=system_prompt)
    return _parse_score(response)
