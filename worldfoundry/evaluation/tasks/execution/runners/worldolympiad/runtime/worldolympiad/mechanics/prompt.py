"""Active mechanics prompt engineering used by physical VLM judges.

This file only records prompt assets reachable from `physical/vlm_judge.py`.
Standalone mechanics prompts that are not called by `physical/` are excluded.
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
    "mechanics/vlm_judge.py:make_expectation_text",
    "problem/problem_set.py:MECHANICS_QUESTIONS",
]

EXCLUDED_LOCAL_PROMPT_SOURCES = [
    "mechanics/vlm_judge.py:build_prompts",
    "mechanics/vlm_judge.py:predict_expected_outcome",
    "mechanics/vlm_judge.py:predict_sub_expectations",
]

MECHANICS_RULE_CONTEXTS: List[Dict[str, str]] = [
    {
        "question_id": "gravity",
        "rule": "Do free-moving objects downward consistently with gravity?",
        "expected_behavior": "First judge whether objects are unsupported, airborne, falling, jumping, driving over uneven terrain, or staying grounded on a support surface. Unsupported objects should fall or arc downward under gravity; supported ground vehicles, people, and objects should remain in plausible contact with the ground unless a visible jump, ramp, collision, or lift explains vertical motion. Penalize floating, sinking through the ground, sudden vertical pops, or hovering without support.",
    },
    {
        "question_id": "buoyancy",
        "rule": "Do objects on or in a fluid behave consistently with buoyancy (floating items stay near the surface, sinking items submerge)?",
        "expected_behavior": "Floating objects should remain on/near the surface; dense objects should descend.",
    },
    {
        "question_id": "compression",
        "rule": "When objects or support surfaces are stressed, loaded, squeezed, or pressed, do they deform or remain rigid in a plausible manner?",
        "expected_behavior": "E.g., cans dent when crushed; soft materials compress smoothly under load; rigid vehicles/metal bodies should mostly keep their shape unless there is collision or heavy force. In robot pick-and-place or grasping scenes, gripping, pinching, pressing, or loading an object on a surface also counts as relevant stress, even if deformation is subtle. In vehicle, racing, or navigation scenes, tire-ground contact, suspension loading, dust/soil displacement, body rigidity during acceleration/turning, or lack of impossible warping can make stress/deformation relevant even without a crash. Deformation should start only after visible squeezing, support, load, contact, or other applied stress. A wooden stick without force will not suddenly deform, and the same object/person/vehicle should stay visually consistent instead of morphing into a different shape or identity.",
    },
    {
        "question_id": "impact",
        "rule": "Do contact, traction, collisions, impacts, and momentum changes produce reasonable motion transitions?",
        "expected_behavior": "Look for momentum transfer, bouncing, shattering, resting poses, contact with the ground, tire/foot contact, traction, braking, turning, or acceleration that matches visible forces and contacts. In robot manipulation scenes, putting an object down, bumping it into a table/container/another object, or releasing it into contact also counts as an impact/contact event. In vehicle, racing, sports, or navigation scenes, tire-ground contact, acceleration from rest, sliding, dust kick-up, braking, steering, direction changes, speed changes, near-collisions, or collisions are relevant contact/momentum events even when there is no crash. Abrupt deformation, direction change, or speed change should happen after visible contact/impact/control input rather than before. The moving objects/people/vehicles should remain temporally consistent instead of suddenly changing shape, count, scale, or identity across frames.",
    },
]

RELEVANCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are MechanicsFilter, decide if the video tests the given mechanics rule. "
    "Treat visible physical motion broadly: robotic or human manipulation, vehicles, sports, racing, navigation, object transport, ground contact, acceleration, braking, turning, sliding, dust kick-up, falling, jumping, and collisions can make mechanics rules related even when the effect is subtle. "
    "For gravity, any unsupported vertical motion, airborne object, jump/ramp, object staying grounded, or impossible floating/ground penetration is relevant. "
    "For compression/stress, visible grasping/pressing/loading, tire or foot contact under load, suspension/soil displacement, rigid body shape preservation, or stress on a deformable object is enough to mark related. "
    "For impact/contact/momentum, placement, collision with a table/container, tire-ground contact, object-ground contact, sliding, braking, acceleration from rest, steering, dust kick-up, abrupt direction or speed changes, or near-collision in vehicle/sports scenes is enough to mark related. "
    "A racing, driving, walking, running, robot navigation, or sports scene with visible motion over a support surface should usually be related to at least one mechanics rule. "
    "Use related=false only when the rule truly cannot be judged from the interaction shown, not merely because no obvious failure occurs. "
    "Return JSON: {\"related\": true|false, \"confidence\": 0-1, "
    "\"reason\": \"short\"}."
)

RELEVANCE_USER_PROMPT_TEMPLATE = """Video description: {video_prompt}
Rule: {rule}
Expectation: {expected_behavior}
Related means this rule can be judged from the interaction, support/contact relationship, or motion pattern, not only that a dramatic failure is already visible.
Before deciding, inspect the reference video for concrete cues such as objects/vehicles/people, ground or surface contact, acceleration, braking, turning, sliding, falling, floating, deformation, collision, dust/debris, and temporal continuity.
If the video contains visible moving objects whose motion can be checked against the rule, prefer related=true with a concise evidence-based reason.
For vehicle, racing, sports, walking/running, or navigation scenes, tire/foot/object contact with the support surface plus acceleration, steering, sliding, dust/debris, or speed change is enough evidence for impact/contact/momentum relevance even without a crash.
Is this rule relevant to the video?"""

COMPLIANCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are PhysicsJudge, an expert at evaluating mechanics in videos. "
    "Compare the generated video to a ground-truth reference for this rule.\n"
    "Rule: {rule}\n"
    "Expected behavior: {expected_behavior}\n"
    "When the rule involves interaction, check whether visible contact, support, ground contact, control input, or plausible force happens before any deformation, rebound, acceleration, braking, turn, slide, or abrupt speed change. "
    "Also watch for object/person/vehicle consistency across frames so identity, count, scale, and overall shape do not change implausibly. "
    "Compare against the reference for scene-level physical cues such as groundedness, contact timing, direction of motion, dust/splash/debris, and continuity; do not require frame-exact matching.\n"
    "Return strict JSON: {{\"compliant\": true|false, \"confidence\": 0-1, "
    "\"explanation\": \"3-5 sentences\", \"observations\": \"specific visual evidence about contacts, motion, support, and any artifacts\"}}."
)

COMPLIANCE_USER_PROMPT_TEMPLATE = """You will see two videos: generated candidate (to judge) and ground-truth reference.
Video description: {video_prompt}
Decide if the generated video follows the mechanics rule and aligns with the reference."""


def get_prompt_engineering_record() -> Dict[str, object]:
    return {
        "active_entrypoint": ACTIVE_ENTRYPOINT,
        "active_entrypoints": ACTIVE_ENTRYPOINTS,
        "active_local_sources": ACTIVE_LOCAL_SOURCES,
        "excluded_local_prompt_sources": EXCLUDED_LOCAL_PROMPT_SOURCES,
        "rule_contexts": MECHANICS_RULE_CONTEXTS,
        "relevance_prompt": {
            "system_template": RELEVANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": RELEVANCE_USER_PROMPT_TEMPLATE,
        },
        "compliance_prompt": {
            "system_template": COMPLIANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": COMPLIANCE_USER_PROMPT_TEMPLATE,
        },
    }


def get_rule_context(question_id: str) -> Dict[str, str]:
    for context in MECHANICS_RULE_CONTEXTS:
        if context["question_id"] == question_id:
            return context
    raise ValueError(f"Unknown mechanics question id: {question_id}")


def build_relevance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_rule_context(question_id)
    user_prompt = RELEVANCE_USER_PROMPT_TEMPLATE.format(
        video_prompt=video_prompt,
        rule=context["rule"],
        expected_behavior=context["expected_behavior"],
    )
    return RELEVANCE_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_compliance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_rule_context(question_id)
    system_prompt = COMPLIANCE_SYSTEM_PROMPT_TEMPLATE.format(
        rule=context["rule"],
        expected_behavior=context["expected_behavior"],
    )
    user_prompt = COMPLIANCE_USER_PROMPT_TEMPLATE.format(video_prompt=video_prompt)
    return system_prompt, user_prompt


if __name__ == "__main__":
    print(json.dumps(get_prompt_engineering_record(), indent=2, ensure_ascii=False))
