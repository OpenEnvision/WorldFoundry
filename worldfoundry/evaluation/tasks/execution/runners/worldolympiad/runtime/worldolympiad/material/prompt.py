"""Active material prompt engineering used by physical VLM judges.

This file is intentionally limited to the prompt assets that are reachable from
the `physical/` directory. It does not document standalone material prompts
that are not called by `physical/vlm_judge.py`.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


ACTIVE_ENTRYPOINTS = [
    "physical/vlm_judge.py",
    "physical/vlm_judge_prompt_engineering.py",
]
ACTIVE_ENTRYPOINT = ACTIVE_ENTRYPOINTS[0]

ACTIVE_LOCAL_SOURCES = [
    "material/vlm_judge.py:MATERIAL_SYSTEM_PROMPT",
    "problem/problem_set.py:MATERIAL_QUESTIONS",
]

EXCLUDED_LOCAL_PROMPT_SOURCES = [
    "material/vlm_judge.py:build_prompts",
]

MATERIAL_QUESTION_CONTEXTS: List[Dict[str, str]] = [
    {
        "question_id": "color_mixing",
        "question": "When different colored liquids or paints mix, do they produce the correct resulting color?",
        "success_condition": "Colored liquids, paints, powders, smoke/dye, pigments, or other visibly colored substances should blend into plausible resulting colors when they contact and mix. Red + Yellow -> Orange, Blue + Yellow -> Green, Red + Blue -> Purple, etc. If colored objects merely pass near each other without material transfer or blending, they should not change color.",
        "hardness_specific_note": "",
    },
    {
        "question_id": "solubility",
        "question": "Do soluble materials (sugar, salt) dissolve properly when placed in water or other solvents?",
        "success_condition": "Soluble or dispersible substances such as sugar, salt, powder, dye, tablets, or granular material should gradually disperse, dissolve, fade, or become suspended/invisible in a liquid solvent. Insoluble solids should remain visible or settle. Do not require perfect chemistry labels when the video visibly shows material entering liquid and changing concentration/visibility.",
        "hardness_specific_note": "",
    },
    {
        "question_id": "hardness",
        "question": "Do materials with different hardness levels behave correctly when grasped, pressed, cut, folded, or broken?",
        "success_condition": "Soft materials (paper, cloth, foam, food, plants, soil, loose dirt, powder) should bend, fold, tear, compress, scatter, or deform when appropriate; hard materials (metal, stone, glass, rigid vehicle bodies, tools, containers) should resist deformation unless force/collision is strong enough. In robot manipulation, human interaction, vehicle, racing, sports, or navigation scenes, grasping, pinching, pressing, stepping, tire-ground contact, placing, sliding, collision, or load-bearing can reveal rigidity versus softness. Shape change should follow visible contact or applied force, and the object/person/vehicle/material being acted on should remain visually consistent.",
        "hardness_specific_note": (
            "Hardness-specific note: rigid objects that suddenly show irregular deformation/buckling are violations; "
            "soft materials that stay stiff when they should bend/compress are also violations. "
            "Loose or granular materials may scatter, pile up, or leave dust/debris under contact; rigid bodies should not melt, wobble, or morph without cause. "
            "If there is no visible touching, pressing, cutting, support change, collision, load, or contact with the ground/surface, the material should not suddenly deform or abruptly change speed because of that interaction.\n"
        ),
    },
    {
        "question_id": "combustibility",
        "question": "Do flammable materials burn correctly, producing fire, smoke, or char?",
        "success_condition": "Wood, paper, fabric should ignite and produce flames/smoke; non-flammable materials should not.",
        "hardness_specific_note": "",
    },
]

MATERIAL_SYSTEM_PROMPT = """You are MaterialJudge, an expert at evaluating material physics in videos.
Determine if the observed behavior matches real-world expectations for the given question.
Use the heuristic metrics as hints only; rely on the video primarily.
Respond with strict JSON (no markdown fences):
{
  "compliant": true|false,
  "confidence": 0.0-1.0,
  "explanation": "2-4 sentences explaining reasoning",
  "observations": "key visual evidence you relied on"
}
If unsure or video is unclear, set compliant=false with low confidence."""

RELEVANCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are MaterialFilter. Decide if the video is testing material properties "
    "(hardness, rigidity, deformability, durability, color mixing, solubility, combustibility), even when "
    "the task looks mechanical (e.g., pulling, pressing, cutting, driving, sliding, stepping, grasping, pouring, or stirring). If you see any "
    "deformation under force, granular scattering, dust/debris, liquid/powder behavior, visible mixing, burning, smoke, charring, or a soft-looking object that resists "
    "bending/compression, treat it as potentially material-related. When uncertain for hardness/rigidity, "
    "err on the side of related=true so compliance can check it. In robot grasping, pick-and-place, human interaction, vehicle/racing, sports, or navigation scenes, treat touching, gripping, pinching, pressing, "
    "supporting, placing, stepping on, tire-ground contact, collision, sliding, or load-bearing as hardness/rigidity-related whenever the interaction "
    "could reveal whether an object or material stays rigid, bends, dents, scatters, compresses, breaks, or resists shape "
    "change; do not require obvious cutting or breakage. For color mixing or solubility, visible colored substances, liquids, powders, dyes, or materials entering liquid are enough for related=true. For combustibility, flame, smoke, char, glowing, sparks, or heat-damaged material are enough. Look for visual "
    "evidence that forces or contact reveal material traits (rigid vs soft, breaks vs stays "
    "intact, scatters vs stays solid, mixes vs stays separate, burns vs remains unchanged). Return JSON: {\"related\": true|false, "
    "\"confidence\": 0-1, \"reason\": \"short\"}."
)

RELEVANCE_USER_PROMPT_TEMPLATE = """Video description: {video_prompt}
Question: {question}
Success condition: {success_condition}
{hardness_specific_note}Inspect the reference video for material evidence: contact force, deformation, rigidity, bending, breaking, scattering, dust/debris, liquid/powder behavior, color transfer, mixing, dissolving, flame, smoke, char, or heat damage.
If any visible interaction could reveal this material property, mark related=true and cite the cue. If the scene only shows unrelated motion with no material evidence for this property, mark related=false.
Is this property being evaluated in the video?"""

COMPLIANCE_SYSTEM_PROMPT_TEMPLATE = MATERIAL_SYSTEM_PROMPT + (
    "\nYou are comparing a generated video to a ground-truth reference. "
    "Judge compliance of the generated video with the material rule and "
    "consistency with the reference. Pay extra attention to whether deformation "
    "or motion changes are caused by visible contact/force/load, whether liquid/powder/color/fire behavior follows plausible material properties, and whether the same "
    "object/person/vehicle/material stays identity-consistent across frames. "
    "Mention concrete evidence such as bending, dents, cracks, scattering, dust, color blending, dissolution, smoke, flame, char, rigidity, or impossible morphing."
)

COMPLIANCE_USER_PROMPT_TEMPLATE = """Two videos are provided: generated candidate and ground truth reference.
Video description: {video_prompt}
Question: {question}
Success condition: {success_condition}
{hardness_specific_note}Return only the JSON described by the system prompt."""


def get_prompt_engineering_record() -> Dict[str, object]:
    return {
        "active_entrypoint": ACTIVE_ENTRYPOINT,
        "active_entrypoints": ACTIVE_ENTRYPOINTS,
        "active_local_sources": ACTIVE_LOCAL_SOURCES,
        "excluded_local_prompt_sources": EXCLUDED_LOCAL_PROMPT_SOURCES,
        "question_contexts": MATERIAL_QUESTION_CONTEXTS,
        "shared_system_prompt": MATERIAL_SYSTEM_PROMPT,
        "relevance_prompt": {
            "system_template": RELEVANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": RELEVANCE_USER_PROMPT_TEMPLATE,
        },
        "compliance_prompt": {
            "system_template": COMPLIANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": COMPLIANCE_USER_PROMPT_TEMPLATE,
        },
    }


def get_question_context(question_id: str) -> Dict[str, str]:
    for context in MATERIAL_QUESTION_CONTEXTS:
        if context["question_id"] == question_id:
            return context
    raise ValueError(f"Unknown material question id: {question_id}")


def build_relevance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_question_context(question_id)
    user_prompt = RELEVANCE_USER_PROMPT_TEMPLATE.format(
        video_prompt=video_prompt,
        question=context["question"],
        success_condition=context["success_condition"],
        hardness_specific_note=context["hardness_specific_note"],
    )
    return RELEVANCE_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_compliance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_question_context(question_id)
    user_prompt = COMPLIANCE_USER_PROMPT_TEMPLATE.format(
        video_prompt=video_prompt,
        question=context["question"],
        success_condition=context["success_condition"],
        hardness_specific_note=context["hardness_specific_note"],
    )
    return COMPLIANCE_SYSTEM_PROMPT_TEMPLATE, user_prompt


if __name__ == "__main__":
    print(json.dumps(get_prompt_engineering_record(), indent=2, ensure_ascii=False))
