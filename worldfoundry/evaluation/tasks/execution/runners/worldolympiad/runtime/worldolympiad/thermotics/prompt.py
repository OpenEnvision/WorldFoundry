"""Active thermotics prompt engineering used by physical VLM judges.

This file only records prompt assets reachable from `physical/vlm_judge.py`.
Standalone thermotics prompts that are not called by `physical/` are excluded.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple


ACTIVE_ENTRYPOINTS = [
    "physical/vlm_judge.py",
    "physical/vlm_judge_prompt_engineering.py",
]
ACTIVE_ENTRYPOINT = ACTIVE_ENTRYPOINTS[0]

ACTIVE_LOCAL_SOURCES = [
    "thermotics/vlm_analysis.py:PHASE_TRANSITION_DESCRIPTIONS",
]

EXCLUDED_LOCAL_PROMPT_SOURCES = [
    "thermotics/vlm_analysis.py:create_vlm_prompt",
]

PHASE_TRANSITION_CONTEXTS: Dict[str, Dict[str, str]] = {
    "melting": {
        "name": "Melting (Solid -> Liquid)",
        "description": "A solid substance should gradually transition to liquid when heated above its melting point. Liquid should form and increase in volume as the solid decreases.",
        "expected_behavior": "The solid substance should decrease in size/volume as it transforms into liquid. The process should be gradual and continuous. Fragmentation (breaking into pieces) is common during melting.",
        "violations": "Solid increasing in size, liquid turning into solid, instantaneous disappearance without liquid formation, solid remaining unchanged despite heat. No liquid formation.",
    },
    "sublimation": {
        "name": "Sublimation (Solid -> Gas)",
        "description": "A solid should directly transition to gas without passing through the liquid phase.",
        "expected_behavior": "The solid should rapidly decrease or disappear while gas/vapor appears around it. No liquid phase should be visible. The process is typically faster than melting.",
        "violations": "Solid melting to liquid first, gas condensing to solid, solid remaining stable, liquid forming as intermediate phase.",
    },
    "vaporization": {
        "name": "Vaporization (Liquid -> Gas)",
        "description": "A liquid should transition to gas when heated (boiling) or over time (evaporation).",
        "expected_behavior": "The liquid should decrease in volume or completely disappear over time. Bubbles may form (boiling) or surface may gradually recede (evaporation). Gas/vapor may be visible.",
        "violations": "Liquid increasing in volume, gas condensing to liquid, liquid remaining constant despite heat/time, liquid freezing.",
    },
    "condensation": {
        "name": "Condensation (Gas -> Liquid)",
        "description": "A gas should transition to liquid when cooled below its condensation point.",
        "expected_behavior": "Liquid should form and increase in volume. Small droplets may appear and merge. Gas may become less visible as it condenses.",
        "violations": "Liquid evaporating, gas remaining as gas, liquid freezing directly to solid, no liquid formation.",
    },
    "deposition": {
        "name": "Deposition (Gas -> Solid)",
        "description": "A gas should directly transition to solid without passing through the liquid phase.",
        "expected_behavior": "Solid should form and increase in size/volume directly from gas. No liquid phase should be visible as an intermediate. Crystals or frost-like structures often form.",
        "violations": "Gas condensing to liquid first, solid melting, liquid freezing to solid, no solid formation.",
    },
    "freezing": {
        "name": "Freezing (Liquid -> Solid)",
        "description": "A liquid should transition to solid when cooled below its freezing point.",
        "expected_behavior": "The liquid should solidify and may expand slightly (e.g., ice) or contract. The surface may become rigid. The shape should become more defined and regular.",
        "violations": "Liquid remaining liquid, solid melting, liquid evaporating, no solidification occurring.",
    },
}

RELEVANCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are PhysicsFilter, an expert at spotting thermal phenomena in video.\n"
    "Decide if the described video contains enough evidence to evaluate this phase transition, even if the transition is subtle, partial, implied by context, or only visible through secondary cues. "
    "Relevant thermal cues include heat sources, flames, glowing objects, steam, vapor, smoke-like gas, fog, condensation droplets, frost/ice/crystals, bubbling, boiling, evaporation, wet/dry surface changes, melting/softening, shrinking solids, growing liquid pools, disappearing liquid, or temperature-driven state changes. "
    "Use related=true when the phase transition can be judged from visible material state over time or from clear prompt/video evidence; use related=false for purely mechanical motion with no heat, cold, gas, liquid, ice, vapor, flame, or state-change cue. "
    "Respond with strict JSON: {\"related\": true|false, \"confidence\": 0-1, "
    "\"reason\": \"short\"}."
)

RELEVANCE_USER_PROMPT_TEMPLATE = """Video description: {video_prompt}
Phase transition: {description}
Expected behavior: {expected_behavior}
Inspect the reference video for state-change cues: solid/liquid/gas presence, heat or cold source, steam/smoke/fog, droplets, frost/ice/crystals, bubbles, surface wetness, shrinking/growing material, or gradual disappearance/formation.
If any cue makes this transition plausible to evaluate, mark related=true and cite the cue. If the scene only shows unrelated mechanical motion, mark related=false.
Is this phase transition present?"""

COMPLIANCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are PhysicsJudge, comparing a generated video against a ground-truth "
    "reference for thermal physics.\n"
    "Phase transition: {name}\n"
    "Expected behavior: {expected_behavior}\n"
    "Common violations: {violations}\n"
    "Judge both direct transitions and secondary visual evidence such as steam, fog, droplets, bubbles, wet/dry changes, frost, crystals, softening, shrinking, liquid pooling, or disappearance. "
    "Compare temporal order and continuity against the reference: state changes should be gradual unless the reference shows a sudden event, and material identity should remain consistent instead of morphing into unrelated objects. "
    "Return strict JSON: {{\"compliant\": true|false, \"confidence\": 0-1, "
    "\"explanation\": \"3-5 sentences\", \"observations\": \"specific thermal/state-change evidence\"}}."
)

COMPLIANCE_USER_PROMPT_TEMPLATE = """You will see two videos:
1) Generated candidate video (the one to judge)
2) Ground-truth reference video
Video description: {video_prompt}
Assess whether the generated video follows the phase transition correctly and is consistent with the reference.
Mention concrete evidence about material state, heat/cold cues, visible vapor/liquid/solid, timing, and any artifacts."""


def get_prompt_engineering_record() -> Dict[str, object]:
    return {
        "active_entrypoint": ACTIVE_ENTRYPOINT,
        "active_entrypoints": ACTIVE_ENTRYPOINTS,
        "active_local_sources": ACTIVE_LOCAL_SOURCES,
        "excluded_local_prompt_sources": EXCLUDED_LOCAL_PROMPT_SOURCES,
        "phase_transition_contexts": PHASE_TRANSITION_CONTEXTS,
        "relevance_prompt": {
            "system_template": RELEVANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": RELEVANCE_USER_PROMPT_TEMPLATE,
        },
        "compliance_prompt": {
            "system_template": COMPLIANCE_SYSTEM_PROMPT_TEMPLATE,
            "user_template": COMPLIANCE_USER_PROMPT_TEMPLATE,
        },
    }


def get_phase_transition_context(question_id: str) -> Dict[str, str]:
    if question_id not in PHASE_TRANSITION_CONTEXTS:
        raise ValueError(f"Unknown thermotics question id: {question_id}")
    return PHASE_TRANSITION_CONTEXTS[question_id]


def build_relevance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_phase_transition_context(question_id)
    user_prompt = RELEVANCE_USER_PROMPT_TEMPLATE.format(
        video_prompt=video_prompt,
        description=context["description"],
        expected_behavior=context["expected_behavior"],
    )
    return RELEVANCE_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_compliance_prompt(question_id: str, video_prompt: str) -> Tuple[str, str]:
    context = get_phase_transition_context(question_id)
    system_prompt = COMPLIANCE_SYSTEM_PROMPT_TEMPLATE.format(
        name=context["name"],
        expected_behavior=context["expected_behavior"],
        violations=context["violations"],
    )
    user_prompt = COMPLIANCE_USER_PROMPT_TEMPLATE.format(video_prompt=video_prompt)
    return system_prompt, user_prompt


if __name__ == "__main__":
    print(json.dumps(get_prompt_engineering_record(), indent=2, ensure_ascii=False))
