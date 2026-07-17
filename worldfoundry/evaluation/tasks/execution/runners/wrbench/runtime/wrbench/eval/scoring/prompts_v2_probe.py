"""Runtime V2 D3-D6 yes/no probe prompts.

This module is intentionally separate from the direct numeric Runtime V2 prompt.
The scoring signal is the next-token probability of answering Yes versus No for
each compact probe; continuous dimension scores are derived after the call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PROBE_CATALOG_VERSION = "runtime_v2_probe_catalog_v1"
DEFAULT_PROMPT_MODE = "runtime_v2_probe_logprob"
PROMPT_MODE_V3_MINIMAL_WORLD_STATE = "runtime_v2_probe_logprob_v3_minimal_world_state"
PROMPT_MODE_P2_EVIDENCE_BOUNDARY = "runtime_v2_probe_logprob_p2_evidence_boundary"
PROMPT_MODE_P3_ANCHOR_IDENTITY_LIGHT = "runtime_v2_probe_logprob_p3_anchor_identity_light"
PROMPT_MODE_P4_D5_LOCATION_STRICT = "runtime_v2_probe_logprob_p4_d5_location_strict"
PROMPT_MODE_P5_D5_RETURN_GATE_SOFT = "runtime_v2_probe_logprob_p5_d5_return_gate_soft"
PROMPT_MODE_P6_D5_WORLD_REGION_CONTINUITY = (
    "runtime_v2_probe_logprob_p6_d5_world_region_continuity"
)
PROMPT_MODE_P6C_D5_RETURN_POSITION_REJECT_RELIEF = (
    "runtime_v2_probe_logprob_p6c_d5_return_position_reject_relief"
)
PROMPT_MODE_P7_D3_RELATION_SPLIT = "runtime_v2_probe_logprob_p7_d3_relation_split"
PROMPT_MODE_P8_D4_SUBJECT_CAUSAL_RESULT = (
    "runtime_v2_probe_logprob_p8_d4_subject_causal_result"
)
PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED = (
    "runtime_v2_probe_logprob_p9_d4_p8_d5_p6_combined"
)
PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY = (
    "runtime_v2_probe_logprob_p10_shared_oov_judgeability"
)
PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY = (
    "runtime_v2_probe_logprob_p11_shared_gate_d6_state_continuity"
)
PROMPT_MODE_P12_QWEN3VL_EVIDENCE_CONDITIONED = (
    "runtime_v2_probe_logprob_p12_qwen3vl_evidence_conditioned"
)
PROMPT_MODE_P13_QWEN3VL_SUBQUESTION_CONDITIONED = (
    "runtime_v2_probe_logprob_p13_qwen3vl_subquestion_conditioned"
)
PROMPT_MODE_QWEN36_35B_P9_SCORE = (
    "runtime_v2_probe_logprob_qwen36_35b_p9_score"
)
PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE = (
    "runtime_v2_probe_logprob_qwen36_35b_shared_oov_gate"
)
SUPPORTED_PROMPT_MODES = {
    DEFAULT_PROMPT_MODE,
    PROMPT_MODE_V3_MINIMAL_WORLD_STATE,
    PROMPT_MODE_P2_EVIDENCE_BOUNDARY,
    PROMPT_MODE_P3_ANCHOR_IDENTITY_LIGHT,
    PROMPT_MODE_P4_D5_LOCATION_STRICT,
    PROMPT_MODE_P5_D5_RETURN_GATE_SOFT,
    PROMPT_MODE_P6_D5_WORLD_REGION_CONTINUITY,
    PROMPT_MODE_P6C_D5_RETURN_POSITION_REJECT_RELIEF,
    PROMPT_MODE_P7_D3_RELATION_SPLIT,
    PROMPT_MODE_P8_D4_SUBJECT_CAUSAL_RESULT,
    PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
    PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
    PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY,
    PROMPT_MODE_P12_QWEN3VL_EVIDENCE_CONDITIONED,
    PROMPT_MODE_P13_QWEN3VL_SUBQUESTION_CONDITIONED,
    PROMPT_MODE_QWEN36_35B_P9_SCORE,
    PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE,
}
CAMERA_MOTION_CONTEXT_KEY = "__camera_motion__"


@dataclass(frozen=True)
class RuntimeV2Probe:
    probe_id: str
    dimension: str
    role: str
    polarity: str
    question: str
    gate_kind: str | None = None


RUNTIME_V2_PROBE_CATALOG: tuple[RuntimeV2Probe, ...] = (
    RuntimeV2Probe(
        probe_id="D3_POSITION_RELATION",
        dimension="spatial_fidelity",
        role="score",
        polarity="positive",
        question=(
            "During visible, judgeable moments, is the prompt-critical subject "
            "or target in a plausible world-space position, relation, contact, "
            "placement, and path relative to stable scene anchors and the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D3_STATIC_ANCHORS",
        dimension="spatial_fidelity",
        role="score",
        polarity="positive",
        question=(
            "Do stable anchors such as walls, floor, furniture, doors, counters, "
            "or fixed background structure support the same spatial interpretation "
            "rather than contradicting the subject or target motion?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D3_NEG_WRONG_SPATIAL",
        dimension="spatial_fidelity",
        role="score",
        polarity="negative",
        question=(
            "Is there clear visible spatial counterevidence, such as the subject "
            "going to the wrong target, wrong side, impossible placement, frame-locked "
            "sliding, or a path that contradicts the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_ACTION_STATE",
        dimension="state_fidelity",
        role="score",
        polarity="positive",
        question=(
            "During visible, judgeable moments, does the subject or target actually "
            "perform the prompt-required action, pose, state, contact result, or "
            "state change rather than merely being described by the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_TEMPORAL_PROGRESS",
        dimension="state_fidelity",
        role="score",
        polarity="positive",
        question=(
            "Does the visible action or state evolve appropriately over time, or "
            "remain stably maintained when the prompt requires a static state?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_NEG_ABSENT_FREEZE",
        dimension="state_fidelity",
        role="score",
        polarity="negative",
        question=(
            "Is there clear visible state counterevidence, such as absent action, "
            "wrong behavior, freezing, looping, reset, regression, or an impossible "
            "pose/state transition?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_HIDDEN",
        dimension="spatial_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="hidden",
        question=(
            "Does the prompt-critical subject or target become fully out of view, "
            "fully occluded, or spatially unjudgeable for a relevant interval?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_RETURN",
        dimension="spatial_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="return",
        question=(
            "After that spatially hidden interval, does a later view show the "
            "expected world region where the subject or target should be spatially "
            "judgeable again?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_UNIDENTIFIABLE",
        dimension="spatial_reasoning",
        role="gate",
        polarity="negative",
        gate_kind="unidentifiable",
        question=(
            "After the later view returns, is the subject or target so unidentifiable "
            "that no spatial continuity judgment can be made at all?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_RETURN_POSITION",
        dimension="spatial_reasoning",
        role="score",
        polarity="positive",
        question=(
            "When the expected region returns, is the subject or target position "
            "spatially plausible relative to the last judgeable pre-hidden evidence, "
            "elapsed hidden time, the prompt, and stable scene anchors?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_NEG_VANISH_RESET",
        dimension="spatial_reasoning",
        role="score",
        polarity="negative",
        question=(
            "When the expected region returns, is there clear spatial failure such "
            "as the target being absent, reappearing at a wrong/impossible location, "
            "resetting to an earlier place, or making an impossible jump?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_HIDDEN",
        dimension="state_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="hidden",
        question=(
            "Does the prompt-critical action, pose, result, or state become hidden "
            "or state-unjudgeable for a relevant interval?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_RETURN",
        dimension="state_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="return",
        question=(
            "After that state-hidden interval, does later evidence make the action, "
            "pose, result, or state judgeable again?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_UNIDENTIFIABLE",
        dimension="state_reasoning",
        role="gate",
        polarity="negative",
        gate_kind="unidentifiable",
        question=(
            "After later evidence returns, is the subject or target so broken or "
            "unclear that no action/state continuity judgment can be made at all?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_HIDDEN_PROGRESS",
        dimension="state_reasoning",
        role="score",
        polarity="positive",
        question=(
            "When later evidence returns, does the action, pose, result, or state "
            "look like a plausible continuation through hidden time rather than a "
            "paused frame?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_NEG_FREEZE_RESET",
        dimension="state_reasoning",
        role="score",
        polarity="negative",
        question=(
            "When later evidence returns, is there clear hidden-time state failure "
            "such as freezing, reset, regression, vanished result, or an impossible "
            "skipped action/state change?"
        ),
    ),
)


SHARED_OOV_GATE_PROBES: tuple[RuntimeV2Probe, ...] = (
    RuntimeV2Probe(
        probe_id="OOV_EVER_UNJUDGEABLE",
        dimension="shared_oov_judgeability",
        role="gate",
        polarity="positive",
        gate_kind="ever_unjudgeable",
        question=(
            "Does the prompt-critical subject, target, action, result, or state "
            "ever become not judgeable enough to identify it and judge both its "
            "coarse position and action/state?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="OOV_RETURN_JUDGEABLE",
        dimension="shared_oov_judgeability",
        role="gate",
        polarity="positive",
        gate_kind="return_judgeable",
        question=(
            "After that unjudgeable interval, does it become judgeable again "
            "enough to compare spatial continuity and state/action continuity?"
        ),
    ),
)


V3_QUESTION_OVERRIDES: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "Does the visible evidence support a coherent 3D world-space state for "
        "the prompt-critical subject or target relative to stable scene anchors?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do static anchors such as floor, walls, furniture, counters, doors, or "
        "fixed background structure make the subject or target motion and final "
        "placement spatially plausible?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial contradiction, such as wrong target, "
        "impossible placement, frame-locked motion, an implausible jump, or a "
        "path that breaks the prompt-critical world state?"
    ),
    "D4_ACTION_STATE": (
        "Does the visible evidence support the prompt-critical action, pose, "
        "state, contact result, or final outcome over time?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Does the action or state progress, persist, or remain stable in the way "
        "the prompt requires across the sampled video?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state contradiction, such as absent action, wrong "
        "behavior, freeze, loop, reset, regression, or impossible state transition?"
    ),
    "D5_GATE_HIDDEN": (
        "Does camera motion, occlusion, or framing make the prompt-critical subject, "
        "target, or expected world region spatially hidden or unjudgeable for a "
        "relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does the subject, target, or expected world region "
        "become spatially inspectable again?"
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After the later view returns, is the subject or target so unidentifiable "
        "that no spatial continuity judgment can be made at all?"
    ),
    "D5_RETURN_POSITION": (
        "When the subject, target, or expected region is inspectable again, is its "
        "spatial state plausible given the earlier evidence, elapsed hidden time, "
        "static anchors, and the prompt?"
    ),
    "D5_NEG_VANISH_RESET": (
        "When the subject, target, or expected region is inspectable again, is "
        "there clear spatial failure such as absence, reset, wrong location, "
        "implausible jump, or inconsistent reappearance?"
    ),
    "D6_GATE_HIDDEN": (
        "Does the prompt-critical action, pose, result, or state become hidden or "
        "state-unjudgeable for a relevant interval?"
    ),
    "D6_GATE_RETURN": (
        "After that interval, does later evidence make the action, pose, result, "
        "or state inspectable again?"
    ),
    "D6_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject, target, action, or result "
        "so broken or unclear that no state-continuity judgment can be made at all?"
    ),
    "D6_HIDDEN_PROGRESS": (
        "When later evidence returns, does the action, pose, result, or state look "
        "like a plausible continuation through hidden time?"
    ),
    "D6_NEG_FREEZE_RESET": (
        "When later evidence returns, is there clear hidden-time state failure such "
        "as freeze, reset, regression, vanished result, or impossible skipped change?"
    ),
}


P2_QUESTION_OVERRIDES: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "Is there clear visible evidence that the prompt-critical subject or "
        "target has a plausible 3D position, relation, contact, placement, or "
        "path relative to static scene anchors and the prompt?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do static anchors such as floor, walls, furniture, doors, counters, or "
        "fixed background structure make the subject or target's motion and "
        "placement spatially consistent rather than ambiguous?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial counterevidence, such as the wrong target, "
        "wrong side, impossible placement, frame-locked sliding, implausible jump, "
        "or a path that contradicts the prompt-critical world state?"
    ),
    "D4_ACTION_STATE": (
        "Is the prompt-critical action, pose, contact, result, or state visibly "
        "performed or reached in the video, rather than only implied by the text?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Does visible evidence show the action or state progressing, completing, "
        "or staying maintained over time in the way the prompt requires?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state counterevidence, such as absent action, "
        "wrong behavior, freeze, loop, reset, regression, or impossible transition?"
    ),
    "D5_GATE_HIDDEN": (
        "Does camera motion, occlusion, or framing make the prompt-critical subject, "
        "target, or expected world region spatially unjudgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later video evidence show the subject, target, "
        "or expected world region well enough to judge spatial continuity?"
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After the later view returns, is the subject or target too broken, changed, "
        "or unidentifiable to make a spatial continuity judgment?"
    ),
    "D5_RETURN_POSITION": (
        "When the subject, target, or expected region becomes judgeable again, is "
        "its position plausible given earlier visible evidence, elapsed hidden time, "
        "static anchors, and the prompt-required destination or region?"
    ),
    "D5_NEG_VANISH_RESET": (
        "When the subject, target, or expected region becomes judgeable again, is "
        "there clear spatial failure: absent target, reset to an earlier place, "
        "wrong location, impossible jump, or inconsistent reappearance?"
    ),
    "D6_GATE_HIDDEN": (
        "Does the prompt-critical action, pose, result, or state become hidden or "
        "not judgeable for a relevant interval?"
    ),
    "D6_GATE_RETURN": (
        "After that interval, does later video evidence show the action, pose, "
        "result, or state well enough to judge state continuity?"
    ),
    "D6_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject, target, action, or result "
        "too broken or unclear to judge state continuity?"
    ),
    "D6_HIDDEN_PROGRESS": (
        "When later evidence returns, does the action, pose, result, or state look "
        "like it plausibly continued through hidden time rather than simply pausing?"
    ),
    "D6_NEG_FREEZE_RESET": (
        "When later evidence returns, is there clear hidden-time state failure such "
        "as freeze, reset, regression, vanished result, or impossible skipped change?"
    ),
}


P3_QUESTION_OVERRIDES: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "Does the visible evidence support a coherent 3D world-space state for "
        "the prompt-critical subject or target relative to stable scene anchors?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do static anchors such as floor, walls, furniture, doors, counters, or "
        "fixed background structure keep the subject or target motion spatially "
        "interpretable rather than misleading?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial counterevidence, such as the wrong target, "
        "wrong side, impossible placement, frame-locked sliding, implausible jump, "
        "or a path that contradicts the prompt-critical world state?"
    ),
    "D5_GATE_HIDDEN": (
        "Does the prompt-critical subject or target become out of view, occluded, "
        "or otherwise not directly judgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later evidence make the subject or target "
        "judgeable again?"
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject or target so changed or "
        "unclear that no spatial continuity judgment can be made at all?"
    ),
    "D5_RETURN_POSITION": (
        "When the subject, target, or expected region becomes judgeable again, is "
        "its position plausible relative to the last judgeable evidence, elapsed "
        "hidden time, static anchors, and the expected region or destination?"
    ),
    "D5_NEG_VANISH_RESET": (
        "When the subject, target, or expected region becomes judgeable again, is "
        "there visible counterevidence such as absence, wrong location, reset, "
        "an impossible jump, or inconsistent reappearance?"
    ),
    "D6_GATE_HIDDEN": (
        "Does the prompt-critical action, pose, result, or state become hidden or "
        "not directly judgeable for a relevant interval?"
    ),
    "D6_GATE_RETURN": (
        "After that interval, does later evidence make the action, pose, result, "
        "or state judgeable again?"
    ),
    "D6_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject, target, action, or result "
        "so broken or unclear that no state-continuity judgment can be made at all?"
    ),
    "D6_HIDDEN_PROGRESS": (
        "When later evidence returns, does the action, pose, result, or state look "
        "like a plausible continuation through hidden time rather than a paused frame?"
    ),
    "D6_NEG_FREEZE_RESET": (
        "When later evidence returns, is there visible counterevidence such as "
        "freeze, reset, regression, vanished result, or an impossible skipped change?"
    ),
}


P4_QUESTION_OVERRIDES: dict[str, str] = {
    "D5_GATE_HIDDEN": (
        "Does the prompt-critical subject or target become out of view, occluded, "
        "or otherwise spatially unjudgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later evidence show the expected world region "
        "or the same subject or target well enough to judge its location?"
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject or target too changed, broken, "
        "or unclear to match for a spatial continuity judgment?"
    ),
    "D5_RETURN_POSITION": (
        "When later evidence is judgeable, does the same prompt-critical subject "
        "or target occupy a prompt-required world position or expected region that "
        "is plausible relative to before-hidden evidence and static anchors? "
        "Visibility alone is insufficient."
    ),
    "D5_NEG_VANISH_RESET": (
        "When later evidence is judgeable, is there visible spatial counterevidence "
        "such as absence from the expected region, wrong location, reset, impossible "
        "jump, or inconsistent reappearance?"
    ),
}


P5_QUESTION_OVERRIDES: dict[str, str] = {
    "D5_GATE_HIDDEN": (
        "Does the prompt-critical subject or target become out of view, occluded, "
        "or otherwise spatially unjudgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later evidence make the same subject, target, "
        "or expected region visible enough for a spatial judgment? Do not require "
        "evidence that the location is correct."
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject or target too unclear to "
        "match for any spatial continuity judgment?"
    ),
    "D5_RETURN_POSITION": (
        "When later evidence is judgeable, is the same prompt-critical subject "
        "or target plausibly in the expected region or destination relative to "
        "before-hidden evidence and static anchors? Do not require exact pixel "
        "position or camera framing match."
    ),
    "D5_NEG_VANISH_RESET": (
        "When later evidence is judgeable, is there visible spatial counterevidence "
        "such as absence from the expected region, wrong location, reset, impossible "
        "jump, or inconsistent reappearance?"
    ),
}


P6_QUESTION_OVERRIDES: dict[str, str] = {
    "D5_GATE_HIDDEN": (
        "Does the prompt-critical subject or target become out of view, occluded, "
        "or otherwise spatially unjudgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later evidence make the same subject, target, "
        "or expected region visible enough for a spatial judgment? Do not require "
        "evidence that the location is correct."
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject or target too unclear to "
        "match for any spatial continuity judgment?"
    ),
    "D5_RETURN_POSITION": (
        "When later evidence is judgeable, does it support spatial continuity of "
        "the same prompt-critical subject or target in the expected world region "
        "relative to stable anchors and before-hidden evidence? Reappearance or "
        "visibility alone is insufficient, but exact pixel position or camera "
        "framing match is not required."
    ),
    "D5_NEG_VANISH_RESET": (
        "When later evidence is judgeable, is there visible spatial counterevidence "
        "such as absence from the expected region, wrong location, reset, impossible "
        "jump, or inconsistent reappearance?"
    ),
}


P6C_QUESTION_OVERRIDES: dict[str, str] = {
    **P6_QUESTION_OVERRIDES,
    "D5_RETURN_POSITION": (
        "When later evidence is judgeable, does it reasonably support continuity "
        "of the same prompt-critical subject or target in the expected world region, "
        "using stable anchors and before-hidden evidence? Do not require exact "
        "placement, identical camera framing, or complete trajectory visibility. "
        "Answer Yes when the coarse world-region relation is plausible and there "
        "is no visible spatial counterevidence; reappearance alone is still "
        "insufficient."
    ),
}


P7_QUESTION_OVERRIDES: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "Does visible evidence support the prompt-critical subject or target's "
        "coarse final world-space relation to the required target, support surface, "
        "container, destination, or surrounding environment?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do stable anchors support a coherent 3D scene for the subject or target's "
        "world motion, placement, contact, support, containment, or return toward "
        "the expected region?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial counterevidence, such as wrong target or "
        "side, missing required contact/support/containment, camera-dragged or "
        "frame-locked placement, impossible jump, or impossible path?"
    ),
}


P8_QUESTION_OVERRIDES: dict[str, str] = {
    "D4_ACTION_STATE": (
        "Does the prompt-critical primary subject and any causal target actually "
        "perform or reach the required action, pose, contact result, state change, "
        "or final state?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Is the required action/result completed and maintained over a stable "
        "ending segment, or is a required static state maintained without visible "
        "regression?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state counterevidence, such as only starting or "
        "suggesting the action, wrong behavior, freeze, loop, reset, regression, "
        "or wrong final state?"
    ),
}


P11_D6_QUESTION_OVERRIDES: dict[str, str] = {
    "D6_HIDDEN_PROGRESS": (
        "When later evidence is judgeable again, does the action, pose, result, "
        "or state look like it plausibly continued, completed, or stayed "
        "maintained through the unjudgeable interval, rather than freezing, "
        "resetting, regressing, or skipping impossibly?"
    ),
    "D6_NEG_FREEZE_RESET": (
        "When later evidence is judgeable again, is there clear state-continuity "
        "failure through the unjudgeable interval, such as freezing, resetting, "
        "regressing, losing a result/state, or skipping the required action/state "
        "impossibly?"
    ),
}


def validate_prompt_mode(prompt_mode: str) -> str:
    if prompt_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(
            f"unsupported prompt_mode: {prompt_mode}; "
            f"expected one of {sorted(SUPPORTED_PROMPT_MODES)}"
        )
    return prompt_mode


def question_for_probe(probe: RuntimeV2Probe, prompt_mode: str = DEFAULT_PROMPT_MODE) -> str:
    prompt_mode = validate_prompt_mode(prompt_mode)
    if prompt_mode == PROMPT_MODE_QWEN36_35B_P9_SCORE:
        prompt_mode = PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED
    elif prompt_mode == PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE:
        prompt_mode = PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY
    if prompt_mode in {
        PROMPT_MODE_P12_QWEN3VL_EVIDENCE_CONDITIONED,
        PROMPT_MODE_P13_QWEN3VL_SUBQUESTION_CONDITIONED,
    }:
        prompt_mode = PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY
    if prompt_mode == PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY:
        if probe.probe_id in P11_D6_QUESTION_OVERRIDES:
            return P11_D6_QUESTION_OVERRIDES[probe.probe_id]
        prompt_mode = PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY
    if prompt_mode == PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY:
        if probe.dimension == "state_fidelity":
            return P8_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
        if probe.dimension == "spatial_reasoning":
            return P6_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
        return probe.question
    if prompt_mode == PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED:
        if probe.dimension == "state_fidelity":
            return P8_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
        if probe.dimension == "spatial_reasoning":
            return P6_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
        return probe.question
    if prompt_mode == PROMPT_MODE_P8_D4_SUBJECT_CAUSAL_RESULT:
        return P8_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P7_D3_RELATION_SPLIT:
        return P7_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P6C_D5_RETURN_POSITION_REJECT_RELIEF:
        return P6C_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P6_D5_WORLD_REGION_CONTINUITY:
        return P6_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P5_D5_RETURN_GATE_SOFT:
        return P5_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P4_D5_LOCATION_STRICT:
        return P4_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P3_ANCHOR_IDENTITY_LIGHT:
        return P3_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_P2_EVIDENCE_BOUNDARY:
        return P2_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    if prompt_mode == PROMPT_MODE_V3_MINIMAL_WORLD_STATE:
        return V3_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    return probe.question


def active_probe_catalog(prompt_mode: str = DEFAULT_PROMPT_MODE) -> tuple[RuntimeV2Probe, ...]:
    prompt_mode = validate_prompt_mode(prompt_mode)
    if prompt_mode in {
        PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
        PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY,
        PROMPT_MODE_P12_QWEN3VL_EVIDENCE_CONDITIONED,
        PROMPT_MODE_P13_QWEN3VL_SUBQUESTION_CONDITIONED,
        PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE,
    }:
        score_probes = tuple(probe for probe in RUNTIME_V2_PROBE_CATALOG if probe.role == "score")
        return score_probes + SHARED_OOV_GATE_PROBES
    return RUNTIME_V2_PROBE_CATALOG


def probes_for_dimension(dimension: str) -> list[RuntimeV2Probe]:
    return [probe for probe in RUNTIME_V2_PROBE_CATALOG if probe.dimension == dimension]


def probe_by_id(probe_id: str) -> RuntimeV2Probe:
    for probe in RUNTIME_V2_PROBE_CATALOG:
        if probe.probe_id == probe_id:
            return probe
    raise KeyError(probe_id)


def format_task_context(task_context: dict[str, Any] | None) -> str:
    if not task_context:
        return "- none"
    lines: list[str] = []
    for key in sorted(task_context):
        value = task_context[key]
        if value is None:
            continue
        text = str(value).strip()
        if text:
            lines.append(f"- {key}: {text}")
    return "\n".join(lines) if lines else "- none"


def format_camera_motion_context(task_context: dict[str, Any] | None) -> str | None:
    if not task_context or CAMERA_MOTION_CONTEXT_KEY not in task_context:
        return None
    text = str(task_context.get(CAMERA_MOTION_CONTEXT_KEY) or "").strip()
    return text or None


def task_context_without_camera_motion(task_context: dict[str, Any] | None) -> dict[str, Any]:
    if not task_context:
        return {}
    return {key: value for key, value in task_context.items() if key != CAMERA_MOTION_CONTEXT_KEY}


def format_evidence_context(
    evidence_context: dict[str, Any] | None,
    *,
    evidence_context_mode: str | None = None,
) -> str:
    if not evidence_context:
        return ""
    mode = evidence_context_mode or "visibility_v1"
    label = "Qwen3-VL structured visibility evidence from a previous pass"
    if mode == "subquestion_v1":
        label = "Qwen3-VL structured subquestion evidence from a previous pass"
    compact_json = json.dumps(evidence_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"""
{label}:
{compact_json}

Use this evidence as a checklist, not as ground truth. Inspect the video again.
If the evidence conflicts with the video, answer from the video.
Wrong but visible/judgeable evidence is still judgeable; correctness is scored
by D5/D6 score probes, not by the gate.
"""


def build_runtime_v2_probe_prompt(
    *,
    world_state_prompt: str,
    video_id: str,
    probe: RuntimeV2Probe,
    task_context: dict[str, Any] | None = None,
    fps: str | None = None,
    frames_used: int | None = None,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    evidence_context: dict[str, Any] | None = None,
    evidence_context_mode: str | None = None,
) -> str:
    prompt_mode = validate_prompt_mode(prompt_mode)
    sampling_bits: list[str] = []
    if fps:
        sampling_bits.append(f"frames sampled at {fps} fps")
    if frames_used is not None:
        sampling_bits.append(f"approximately {frames_used} video frames")
    sampling_text = ", ".join(sampling_bits) if sampling_bits else "the frames provided by the runner"
    if (
        prompt_mode
        in {
            PROMPT_MODE_P8_D4_SUBJECT_CAUSAL_RESULT,
            PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
            PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
            PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY,
            PROMPT_MODE_QWEN36_35B_P9_SCORE,
            PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE,
        }
        and probe.dimension == "state_fidelity"
    ):
        scaffold = """Use this internal observation scaffold:
- Inspect the entire sampled video and maintain a simple 3D scene while watching it.
- For this D4 visible-state probe, focus on the prompt-critical primary subject and any object or region that causally determines the requested result.
- Ignore incidental moving objects unless they directly affect the prompt-required action, contact result, state change, or final state.
- Judge whether the required action/result is actually reached and then maintained when the prompt needs persistence.
- For prompts describing a static pose or state, check that the state remains stable instead of briefly appearing, resetting, looping, or regressing.
- Track the prompt-critical subject, target, result object, and expected region when they determine the required action or state.
- For negative probes, answer Yes only when visible state counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P7_D3_RELATION_SPLIT and probe.dimension == "spatial_fidelity":
        scaffold = """Use this internal observation scaffold:
- Inspect the entire sampled video and maintain a simple 3D scene while watching it.
- For this D3 visible-spatial probe, focus on the prompt-critical subject, target, support surface, container, destination, and surrounding environment.
- Use stable anchors to judge coarse world-space relation, contact, support, containment, placement, path, and whether a subject plausibly returns toward an expected region.
- Do not mechanically subtract camera motion from object motion; use the evolving 3D scene and static anchors to decide what is happening.
- Do not answer a strong Yes or strong No from ambiguous small motion, brief glimpses, or weak spatial cues.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P6C_D5_RETURN_POSITION_REJECT_RELIEF and probe.dimension == "spatial_reasoning":
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this D5 spatial-continuity probe, compare the last judgeable before-hidden evidence with later judgeable evidence.
- Track whether the same prompt-critical subject or target can be matched across the hidden interval.
- Gate probes only judge whether later evidence is inspectable enough for a spatial judgment; they do not judge whether the video is good or whether the location is correct.
- For return-position probes, judge plausible coarse world-region continuity relative to static anchors; do not require exact placement, identical framing, or complete trajectory visibility.
- Visible or identifiable again is not enough by itself; later evidence should support spatial continuity in the expected world region.
- Hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif (
        prompt_mode
        in {
            PROMPT_MODE_P6_D5_WORLD_REGION_CONTINUITY,
            PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
            PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
            PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY,
            PROMPT_MODE_QWEN36_35B_P9_SCORE,
            PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE,
        }
        and probe.dimension == "spatial_reasoning"
    ):
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this D5 spatial-continuity probe, compare the last judgeable before-hidden evidence with later judgeable evidence.
- Track whether the same prompt-critical subject, target, result object, or expected region can be matched across the hidden interval.
- Gate probes only judge whether later evidence is inspectable enough for a spatial judgment; they do not judge whether the video is good or whether the location is correct.
- For return-position probes, judge expected world-region continuity relative to static anchors, not exact pixel position or camera framing.
- Visible or identifiable again is not enough by itself; later evidence should support spatial continuity in the expected world region.
- Hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif (
        prompt_mode
        in {
            PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
            PROMPT_MODE_P11_SHARED_GATE_D6_STATE_CONTINUITY,
            PROMPT_MODE_QWEN36_35B_SHARED_OOV_GATE,
        }
        and probe.dimension == "shared_oov_judgeability"
    ):
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this shared OoV judgeability gate, decide only whether the prompt-critical evidence becomes judgeable, not whether it is correct.
- Judgeable means the subject, target, action, result, or state can be identified well enough to compare coarse position and action/state.
- Use before-hidden / after-return evidence; the hidden middle alone is not a failure.
- Wrong position, wrong state, reset, freeze, or failed action is still applicable when visible and judgeable; that should be scored by D5/D6, not marked N/A.
- Do not use a fixed visible-area percentage rule.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P5_D5_RETURN_GATE_SOFT and probe.dimension == "spatial_reasoning":
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this D5 spatial-continuity probe, compare the last judgeable before-hidden evidence with later judgeable evidence.
- Track whether the same prompt-critical subject or target can be matched across the hidden interval.
- Gate probes only judge whether later evidence is inspectable enough for a spatial judgment; they do not judge whether the video is good.
- For return-position probes, judge coarse expected-region continuity relative to static anchors, not exact pixel position or camera framing.
- Visible or identifiable again is not enough by itself; plausible expected-region continuity without visible counterevidence can support Yes.
- Hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P4_D5_LOCATION_STRICT and probe.dimension == "spatial_reasoning":
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this D5 spatial-continuity probe, compare the last judgeable before-hidden evidence with later judgeable evidence.
- Track whether the same prompt-critical subject or target can be matched across the hidden interval.
- Visible or identifiable again is not enough for a positive return-position answer; judge whether it is in the prompt-required world position or expected region.
- Hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P2_EVIDENCE_BOUNDARY:
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- Use static anchors to judge world-space motion; do not use a fixed camera-motion subtraction rule.
- Track only prompt-critical subjects, targets, actions, results, and expected regions.
- Answer Yes only for clear video evidence that supports this exact probe.
- Answer No when evidence is missing, too ambiguous, too brief, or contradicted by visible evidence.
- For hidden intervals, compare last judgeable evidence with later judgeable evidence; hidden time can imply continuation, but absence, reset, freeze, or impossible jumps are counterevidence.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_P3_ANCHOR_IDENTITY_LIGHT:
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- Use static anchors to separate camera/view change from the evolving world state, but do not apply a fixed camera-motion subtraction rule.
- Track prompt-critical subjects, targets, actions, results, and expected regions across time.
- Prefer weak-evidence restraint on D3 when anchors conflict, motion is brief, or the spatial cue is ambiguous.
- For D5 and D6, compare the last judgeable evidence with later judgeable evidence; hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode == PROMPT_MODE_V3_MINIMAL_WORLD_STATE:
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- Use static anchors to separate camera/view change from the evolving world state.
- Track prompt-critical subjects, targets, actions, results, and expected regions through time.
- For hidden or out-of-view intervals, infer plausible world continuation and judge the later evidence when it returns.
- Answer Yes only when video evidence supports the probe; answer No when required evidence is missing or clearly contradicted.
- Judge the video evidence, not the caption-like expectation."""
    else:
        scaffold = """Use this internal observation scaffold:
- Maintain a simple mental 3D scene.
- Use stable scene anchors such as walls, floor, furniture, doors, counters, and fixed background structure.
- Judge position, relation, contact, placement, plausible path, action, state, and result from visible evidence.
- Do not apply a fixed camera-motion subtraction rule.
- For hidden intervals, compare before hidden evidence with after return evidence. Hidden time can imply continuation; it should not be treated as a pause unless the video shows that.
- Answer from the video evidence, not from a caption-like expectation."""

    camera_motion_context = format_camera_motion_context(task_context)
    if camera_motion_context is None:
        task_context_block = f"""Task context from manifest, when available:
{format_task_context(task_context)}"""
        video_id_block = f"""Video id:
{video_id}
"""
    else:
        extra_task_context = task_context_without_camera_motion(task_context)
        extra_task_context_text = format_task_context(extra_task_context)
        extra_task_context_block = ""
        if extra_task_context_text.strip() not in {"", "{}", "- none"}:
            extra_task_context_block = f"""

Additional prompt/task context, when available:
{extra_task_context_text}"""
        task_context_block = f"""How camera move:
{camera_motion_context}{extra_task_context_block}"""
        video_id_block = ""

    return f"""You are a strict video consistency auditor for AI-generated mobile-camera videos.

Your job is to answer one binary probe. Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Inspect the entire provided sampled video before answering. You are seeing {sampling_text}, not necessarily every source frame.

{scaffold}

Text prompt used to generate the video:
{world_state_prompt}

{video_id_block}
{task_context_block}
{format_evidence_context(evidence_context, evidence_context_mode=evidence_context_mode)}

Probe id:
{probe.probe_id}

Probe dimension:
{probe.dimension}

Probe role:
{probe.role}

Probe polarity:
{probe.polarity}

Question:
{question_for_probe(probe, prompt_mode)}

Answer exactly one token: Yes or No."""
