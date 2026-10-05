"""HarnessEval skill contracts."""

from __future__ import annotations

from dataclasses import dataclass


FAMILIES = (
    "exploratory_transition",
    "intentional_transition",
    "physical_transition",
    "drift_resistance",
    "return_revisit_consistency",
    "offscreen_evolution",
)

CORE_SKILLS = {
    "exploratory_transition": ("viewpoint_trajectory_verifier",),
    "intentional_transition": ("intentional_change_verifier_vlm",),
    "physical_transition": ("physical_response_verifier_vlm",),
    "drift_resistance": ("drift_degradation_analyzer",),
    "return_revisit_consistency": ("return_consistency_verifier",),
    "offscreen_evolution": ("offscreen_evolution_verifier",),
    "persistence_sequence": (
        "drift_degradation_analyzer",
        "return_consistency_verifier",
        "offscreen_evolution_verifier",
    ),
}

SKILLS = (
    "render_quality_inspector",
    "motion_quality_inspector",
    "appearance_consistency_inspector",
    "physical_plausibility_inspector",
    "viewpoint_trajectory_verifier",
    "intentional_change_verifier_vlm",
    "physical_response_verifier_vlm",
    "physical_law_validator",
    "drift_degradation_analyzer",
    "return_consistency_verifier",
    "offscreen_evolution_verifier",
)

SKILL_SPECS = {
    "render_quality_inspector": {
        "default_role": "observation",
        "question": "Are the generated frames aesthetically and technically clean?",
    },
    "motion_quality_inspector": {
        "default_role": "observation",
        "question": "Is the generated motion present and temporally smooth?",
    },
    "appearance_consistency_inspector": {
        "default_role": "observation",
        "question": "Does visual appearance remain consistent through the rollout?",
    },
    "physical_plausibility_inspector": {
        "default_role": "observation",
        "question": "Is the rendered motion physically and causally plausible?",
    },
    "viewpoint_trajectory_verifier": {
        "default_role": "core",
        "question": (
            "Does the case explicitly require active camera, viewpoint, or navigation "
            "movement along a path or action sequence? Fixed-camera constraints, stable "
            "framing, visibility requirements, and incidental camera motion do not make "
            "this skill applicable."
        ),
    },
    "intentional_change_verifier_vlm": {
        "default_role": "core",
        "question": (
            "Is the case's primary target an explicitly specified intentional semantic or "
            "object-state transition while protected anchors remain intact? A causal response "
            "to a physical intervention, a navigation or viewpoint change, long-horizon drift, "
            "or offscreen evolution does not count as an intentional change."
        ),
    },
    "physical_response_verifier_vlm": {
        "default_role": "core",
        "question": (
            "Is the case's primary target the causal response and temporal dynamics produced "
            "by an explicit physical intervention or force? A directly specified intentional "
            "semantic or object-state edit does not count as a physical response."
        ),
    },
    "physical_law_validator": {
        "default_role": "diagnostic",
        "diagnostic_for": ("physical_transition",),
        "question": (
            "Does evaluating the causal physical response require a case-specific physical-law "
            "check? General visual plausibility and intentional semantic or object-state edits "
            "do not make this skill applicable."
        ),
    },
    "drift_degradation_analyzer": {
        "default_role": "core",
        "question": (
            "Is resistance to progressive visual, geometric, and semantic degradation an "
            "explicit target of a long-horizon rollout with many action turns or chunks? "
            "An ordinary short multi-step rollout does not make this skill applicable."
        ),
    },
    "return_consistency_verifier": {
        "default_role": "core",
        "question": "Does a returned view remain consistent after a non-static loop?",
    },
    "offscreen_evolution_verifier": {
        "default_role": "core",
        "question": "Does an offscreen process continue evolving in the same world?",
    },
}


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    score_schema: str
    plan_schema: str
    skills: tuple[str, ...]
    observation_skills: tuple[str, ...]


PROTOCOL = EvaluationProtocol(
    name="harnesseval",
    score_schema="harnesseval.skill_eval",
    plan_schema="harnesseval.skill_plan",
    skills=SKILLS,
    observation_skills=(
        "render_quality_inspector",
        "motion_quality_inspector",
        "appearance_consistency_inspector",
        "physical_plausibility_inspector",
    ),
)


def canonical_family(family: str) -> str:
    return "persistence_sequence" if family.startswith("persistence_sequence") else family


def get_protocol(name: str) -> EvaluationProtocol:
    if name in {"harnesseval", "default"}:
        return PROTOCOL
    raise ValueError(f"unknown evaluation protocol {name!r}; choose harnesseval")
