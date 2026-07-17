#!/usr/bin/env python3
"""Local Qwen3-VL structured subject judgeability evidence runner.

This diagnostic route asks Qwen3-VL-Instruct for strict JSON evidence about
prompt-critical subject judgeability. It writes candidate evidence only; it
does not update canonical scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


try:
    from .runtime_common import scoring_video_path
except ImportError:
    from wrbench.eval.scoring.runtime_common import scoring_video_path


SUBJECT_SCHEMA_VERSION = "qwen3vl_subject_judgeability_v1"
OBJECT_SCHEMA_VERSION = "qwen3vl_object_judgeability_v2"
GUARDED_SCHEMA_VERSION = "qwen3vl_guarded_teacher_gate_v3"
SUBJECT_CLEAN_SCHEMA_VERSION = "qwen3vl_subject_judgeability_v2_clean"
GUARDED_CLEAN_SCHEMA_VERSION = "qwen3vl_guarded_teacher_gate_v4_clean"
SHARED_DIRECT3Q_CLEAN_SCHEMA_VERSION = "qwen3vl_shared_oov_direct3q_clean_v1"
VISIBLE_BOOL_CLEAN_SCHEMA_VERSION = "qwen3vl_oov_gate_v5_clean_visible_bool_v1"
OOV_GAP_BOOL_CLEAN_SCHEMA_VERSION = "qwen3vl_oov_gate_v6_clean_oov_gap_bool_v1"
OOV_GAP_SCAN_CLEAN_SCHEMA_VERSION = "qwen3vl_oov_gate_v7_clean_visibility_scan_v2"
OOV_GAP_TRIPLET_CLEAN_SCHEMA_VERSION = "qwen3vl_oov_gate_v8_clean_triplet_v1"
OOV_GAP_TRIPLET_SHEET_CLEAN_SCHEMA_VERSION = "qwen3vl_oov_gate_v11_clean_triplet_sheet_v3"
OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_SCHEMA_VERSION = (
    "qwen3vl_oov_gate_v5_clean_per_second_audit_bool_v1"
)
OOV_GAP_PER_SECOND_STRICT_COLLAPSE_SCHEMA_VERSION = (
    "qwen3vl_oov_gate_v6_clean_per_second_strict_collapse_v1"
)
OOV_SUBJECT_RESULT_INTEGRITY_SCHEMA_VERSION = (
    "qwen3vl_oov_gate_v8_subject_result_integrity_capped_v1"
)
SCHEMA_VERSION = SUBJECT_SCHEMA_VERSION
PROMPT_SCHEMA_SUBJECT = "subject_judgeability_v1"
PROMPT_SCHEMA_OBJECT = "object_judgeability_v2"
PROMPT_SCHEMA_GUARDED = "guarded_teacher_gate_v3"
PROMPT_SCHEMA_SUBJECT_CLEAN = "subject_judgeability_v2_clean"
PROMPT_SCHEMA_GUARDED_CLEAN = "guarded_teacher_gate_v4_clean"
PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN = "shared_oov_direct3q_clean_v1"
PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN = "visible_after_gap_bool_clean_v1"
PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN = "oov_gap_later_comparable_bool_clean_v1"
PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN = "oov_gap_visibility_scan_clean_v2"
PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN = "oov_gap_triplet_clean_v1"
PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN = "oov_gap_triplet_sheet_clean_v3"
PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN = "oov_gap_per_second_audit_bool_clean_v1"
PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE = "oov_gap_per_second_strict_collapse_v1"
PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY = "oov_subject_result_integrity_v1"
SUPPORTED_PROMPT_SCHEMAS = {
    PROMPT_SCHEMA_SUBJECT,
    PROMPT_SCHEMA_OBJECT,
    PROMPT_SCHEMA_GUARDED,
    PROMPT_SCHEMA_SUBJECT_CLEAN,
    PROMPT_SCHEMA_GUARDED_CLEAN,
    PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN,
    PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN,
    PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
    PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY,
}
DEFAULT_MODEL_PATH = ""
DEFAULT_FPS = "4"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_MAX_NEW_TOKENS = 768
RAW_FILENAME = "raw_qwen3vl_judgeability_evidence.jsonl"
PARSED_FILENAME = "parsed_qwen3vl_judgeability_evidence.jsonl"
EVIDENCE_FILENAME = "evidence.jsonl"

JUDGEABILITY_VALUES = {"yes", "no"}
DIRECT3Q_VALUES = {"yes", "no", "unclear"}
VISIBLE_BOOL_POSITIVE_REASONS = {"clear_later_evidence", "clear_absence_failure_or_change"}
VISIBLE_BOOL_NEGATIVE_REASONS = {
    "no_reference_before_gap",
    "no_meaningful_visibility_gap",
    "no_later_relevant_evidence",
    "later_too_unclear",
    "later_irrelevant",
    "request_not_visual",
    "uncertain",
}
VISIBLE_BOOL_REASON_CODES = VISIBLE_BOOL_POSITIVE_REASONS | VISIBLE_BOOL_NEGATIVE_REASONS
OOV_GAP_BOOL_POSITIVE_REASONS = {
    "clear_reappearance",
    "clear_comparable_region_or_result",
    "clear_absence_or_change",
}
OOV_GAP_BOOL_NEGATIVE_REASONS = {
    "no_meaningful_unjudgeable_interval",
    "continuous_visible_or_judgeable",
    "no_clear_before_reference",
    "no_later_comparable_evidence",
    "later_too_unclear",
    "later_irrelevant",
    "request_not_visual",
    "uncertain",
}
OOV_GAP_BOOL_REASON_CODES = OOV_GAP_BOOL_POSITIVE_REASONS | OOV_GAP_BOOL_NEGATIVE_REASONS
OOV_GAP_SCAN_VALUES = {"yes", "no", "unclear"}
OOV_GAP_TRIPLET_VALUES = {"yes", "no", "unclear"}
OOV_GAP_TRIPLET_LATER_TYPES = {
    "same_subject",
    "target_region",
    "absence",
    "change",
    "failure",
    "replacement",
    "unclear",
}
OOV_GAP_PER_SECOND_AUDIT_STATUSES = {
    "visible",
    "not_visible",
    "unrelated",
    "unclear",
}
OOV_GAP_PER_SECOND_AUDIT_POSITIVE_REASONS = {"yes_after_gap"}
OOV_GAP_PER_SECOND_AUDIT_NEGATIVE_REASONS = {
    "always_visible",
    "no_start",
    "no_gap",
    "no_later_visible",
    "later_unrelated",
    "too_unclear",
    "not_visual",
    "unclear",
}
OOV_GAP_PER_SECOND_AUDIT_REASON_CODES = (
    OOV_GAP_PER_SECOND_AUDIT_POSITIVE_REASONS | OOV_GAP_PER_SECOND_AUDIT_NEGATIVE_REASONS
)
OOV_GAP_PER_SECOND_STRICT_COLLAPSE_STATUSES = {
    "visible",
    "not_visible",
    "broken_scene",
    "collapsed",
    "unclear",
}
OOV_SUBJECT_RESULT_JUDGEABLE_VALUES = {"yes", "no", "unclear"}
OOV_SUBJECT_RESULT_SCENE_VALUES = {"coherent", "broken", "unclear"}
REQUIRED_KEYS = [
    "video_id",
    "main_subject",
    "per_second",
]


PROMPT_TEMPLATE = """You are a strict video judgeability auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the single main prompt-critical subject.
2. Inspect the video in temporal order, one whole second at a time.
3. For each second, decide whether the same main subject is judgeable.

judgeable = yes only when:
- the same main subject is visible and identifiable enough;
- its coarse position can be judged;
- its action or state can be judged.

judgeable = no when:
- the subject is out of frame, fully occluded, too blurred, too small, or unidentifiable;
- or only a fragment is visible and coarse position/action/state cannot be judged.

Do not score D5 or D6. Do not decide whether the video is correct. Do not explain beyond short visual notes.

Text prompt:
{world_state_prompt}

Video id:
{video_id}

Return this exact JSON schema:
{{
  "video_id": "<string>",
  "main_subject": "<short subject name>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short visual note>"
    }}
  ]
}}"""


CLEAN_PROMPT_TEMPLATE = """You are a strict video judgeability auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the single main prompt-critical subject.
2. Inspect the video in temporal order, one whole second at a time.
3. For each second, decide whether the same main subject is judgeable.

judgeable = yes only when:
- the same main subject is visible and identifiable enough;
- its coarse position can be judged;
- its action or state can be judged.

judgeable = no when:
- the subject is out of frame, fully occluded, too blurred, too small, or unidentifiable;
- or only a fragment is visible and coarse position/action/state cannot be judged.

Do not score D5 or D6. Do not decide whether the video is correct. Do not explain beyond short visual notes.

Text prompt:
{world_state_prompt}

Return this exact JSON schema:
{{
  "main_subject": "<short subject name>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short visual note>"
    }}
  ]
}}"""


OBJECT_PROMPT_TEMPLATE = """You are a strict object-centric video judgeability auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the single prompt-critical object, subject, target, result object, or expected region that best determines whether D5/D6 can be judged.
2. Inspect the video in temporal order, one whole second at a time.
3. For each second, decide whether that prompt-critical object is judgeable enough to compare its coarse position and action/state.
4. Write only short visual evidence. Do not score correctness and do not explain your reasoning.

object_judgeable = yes when:
- the same prompt-critical object or target/result evidence is visible and identifiable enough;
- its coarse position or spatial relation can be judged for D5;
- its action, state, result, or state evidence can be judged for D6.

object_judgeable = no when:
- there is no relevant OoV/unjudgeable interval;
- there is no return after the interval;
- the returned object/result evidence is out of frame, fully occluded, too blurred, too small, or unidentifiable;
- or the returned evidence is too unclear to compare coarse position/action/state.

Gate rules:
- Gate only decides whether D5/D6 are applicable, not whether the video is correct.
- Wrong but visible/judgeable returned evidence is applicable and should not be marked N/A.
- Use N/A only for no_initial_judgeable, no_oov, no_return, unjudgeable_return, or unclear evidence.

Text prompt:
{world_state_prompt}

Video id:
{video_id}

Return this exact JSON schema:
{{
  "video_id": "<string>",
  "prompt_critical_object": "<short object/subject/target/result name>",
  "expected_motion_or_state": "<short phrase>",
  "per_second": [
    {{
      "sec": <integer second>,
      "object_judgeable": "<yes|no>",
      "observed_motion_or_state": "<short visual note>"
    }}
  ],
  "before_hidden_evidence": "<short|null>",
  "after_return_evidence": "<short|null>",
  "gate": {{
    "d5_applicable": <true|false>,
    "d6_applicable": <true|false>,
    "na_reason": "<no_initial_judgeable|no_oov|no_return|unjudgeable_return|unclear|null>"
  }}
}}"""


GUARDED_PROMPT_TEMPLATE = """You are a strict D5/D6 judgeability gate auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the prompt-critical subject, object, target, result evidence, or region needed to judge the prompt.
2. Identify static anchors that help compare before/after positions or state.
3. Inspect the video in temporal order, one whole second at a time.
4. Decide only whether D5/D6 are judgeable/applicable. Do not decide whether the video is correct.

Critical gate rules:
- Wrong but visible and identifiable returned evidence is applicable.
- A visible but irrelevant object is not enough; evidence must be prompt-critical.
- If the object/person/state returns in a wrong place or wrong state but can be visually judged, D5/D6 are applicable.
- Mark N/A only for no usable returned evidence, no OoV interval, no return after OoV, unidentifiable evidence, or completely unjudgeable evidence.
- Do not use N/A for incorrect-but-visible outcomes.

Text prompt:
{world_state_prompt}

Video id:
{video_id}

Return this exact JSON schema:
{{
  "video_id": "<string>",
  "prompt_critical_evidence": "<short subject/object/target/result name>",
  "expected_motion_or_state": "<short phrase>",
  "static_anchors": ["<short anchor>", "..."],
  "before_hidden_evidence": "<short|null>",
  "after_return_evidence": "<short|null>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short prompt-critical visual note>"
    }}
  ],
  "gate": {{
    "d5_applicable": <true|false>,
    "d6_applicable": <true|false>,
    "na_reason": "<no_usable_returned_evidence|no_oov|no_return|unidentifiable|completely_unjudgeable|unclear|null>"
  }},
  "audit": {{
    "wrong_but_judgeable": <true|false>,
    "anchor_available": <true|false>,
    "after_return_available": <true|false>
  }}
}}"""


GUARDED_CLEAN_PROMPT_TEMPLATE = """You are a strict D5/D6 judgeability gate auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the prompt-critical subject, object, target, result evidence, or region needed to judge the prompt.
2. Identify static anchors that help compare before/after positions or state.
3. Inspect the video in temporal order, one whole second at a time.
4. Decide only whether D5/D6 are judgeable/applicable. Do not decide whether the video is correct.

Critical gate rules:
- Wrong but visible and identifiable returned evidence is applicable.
- A visible but irrelevant object is not enough; evidence must be prompt-critical.
- If the object/person/state returns in a wrong place or wrong state but can be visually judged, D5/D6 are applicable.
- Mark N/A only for no usable returned evidence, no OoV interval, no return after OoV, unidentifiable evidence, or completely unjudgeable evidence.
- Do not use N/A for incorrect-but-visible outcomes.

Text prompt:
{world_state_prompt}

Return this exact JSON schema:
{{
  "prompt_critical_evidence": "<short subject/object/target/result name>",
  "expected_motion_or_state": "<short phrase>",
  "static_anchors": ["<short anchor>", "..."],
  "before_hidden_evidence": "<short|null>",
  "after_return_evidence": "<short|null>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short prompt-critical visual note>"
    }}
  ],
  "gate": {{
    "d5_applicable": <true|false>,
    "d6_applicable": <true|false>,
    "na_reason": "<no_usable_returned_evidence|no_oov|no_return|unidentifiable|completely_unjudgeable|unclear|null>"
  }},
  "audit": {{
    "wrong_but_judgeable": <true|false>,
    "anchor_available": <true|false>,
    "after_return_available": <true|false>
  }}
}}"""


SHARED_DIRECT3Q_CLEAN_PROMPT_TEMPLATE = """You are a strict video judgeability auditor.

Return only valid JSON. Do not return markdown.

Use the text prompt only to identify the relevant subject, object, target, result, action/state, or expected region. Answer from the video evidence.

Your job is to answer three judgeability questions about the video.

Question 1:
Does the prompt-critical visual evidence become out of view, occluded, hidden by camera motion, heavily cropped, too blurred, too small, or otherwise unclear such that the prompt-critical evidence cannot be judged for concrete position or action/state during that interval?

Question 2:
After that interval, is later video evidence clear enough to judge the prompt-critical subject, object, result, expected region, support, container, or stable anchor for concrete position, absence, action, state, result, reset, vanish, or failure?

Question 3:
If later evidence returns, is it still too unclear or unidentifiable to make any concrete position/action/state judgment?

Rules:
- Do not decide whether the video is correct.
- Wrong but clearly visible returned evidence is still judgeable.
- A failed action, wrong position, reset, vanished result, or absent object is still judgeable if the later view is clear enough to judge it.
- Do not require proof that it is the exact same physical instance; use later clear evidence of the relevant subject, object, or region when it is enough to judge the prompt-critical state.
- Set oov_applicable true only when there is a relevant visibility gap and later evidence returns clear enough to judge.
- Set oov_applicable false when there is no initial clear evidence, no relevant visibility gap, no later return, or the later evidence is still too unclear to judge.

Text prompt:
{world_state_prompt}

Return this exact JSON schema:
{{
  "prompt_critical_evidence": "<short subject/object/result/region>",
  "gap_observed": "<yes|no|unclear>",
  "after_return_clear_enough": "<yes|no|unclear>",
  "too_unclear_after_return": "<yes|no|unclear>",
  "before_gap_evidence": "<short visual evidence|null>",
  "after_return_evidence": "<short visual evidence|null>",
  "oov_applicable": <true|false>,
  "na_reason": "<no_initial_clear_evidence|no_visibility_gap|no_later_return|return_too_unclear|unclear|null>",
  "short_reason": "<one short visual reason>"
}}"""


VISIBLE_BOOL_CLEAN_PROMPT_TEMPLATE = """You are deciding whether a video has enough visible evidence for a human to evaluate the text request after a visibility interruption.

Use only the video and the text request. Use the text request only to identify what visual evidence to inspect. The text request itself is not evidence that the event, result, gap, return, absence, or failure occurred. Do not use metadata, filenames, model names, dataset names, or outside knowledge.

Text request:
{world_state_prompt}

Answer true only when all of these are satisfied:
- before the visibility gap, some prompt-relevant reference is visible enough to orient the request. This can be the relevant entity, target region, support surface, expected result region, action context, or stable visual context;
- then the relevant evidence becomes not clearly judgeable for a meaningful interval. It does not need to fully disappear. Camera motion, cropping, partial out-of-frame view, occlusion, blur, being too small, cut/discontinuity, or unstable visual evidence can count;
- later, the video shows clear enough relevant evidence to judge the requested visual outcome.

Later clear evidence can include:
- the relevant entity or region again;
- the expected result clearly absent or vanished;
- a reset, failed action, wrong state, wrong position, or non-completion;
- a clear identity switch or replacement, only if it is tied to the requested role, region, action, object, or result.

Do not judge whether the requested outcome is correct. A wrong outcome, failed action, missing object, absence, reset, or prompt-tied identity switch is still judgeable if it is visible clearly enough.

Answer false only when the video lacks enough relevant visible evidence: no prompt-relevant reference before the gap, no meaningful visibility gap, no later relevant evidence, later evidence too unclear, later evidence irrelevant, or the request is not visually judgeable from the video.

Use "no_meaningful_visibility_gap" only when the relevant evidence remains continuously visible and judgeable, or when there is no real prompt-relevant hidden/unjudgeable interval. Do not answer true for ordinary continuously visible evidence with no such gap.

Use "later_irrelevant" when the later visible person/object/region is not in the requested role, target region, support surface, action, object, or result. Do not call an unrelated later person or object an identity switch.

Do not use "uncertain" merely because correctness, exact continuity, exact identity, or exact trajectory is uncertain. Use "uncertain" only when the visible evidence itself is too ambiguous to decide whether there is after-gap judgeable evidence.

The brief_reason must mention the visible after-gap evidence, absence, failure, reset, wrong state, or relevant region. Do not restate only the text request.

Return exactly one JSON object. Do not include markdown.

{{
  "has_visible_after_gap_evidence_to_judge": <true|false>,
  "reason_code": "<clear_later_evidence|clear_absence_failure_or_change|no_reference_before_gap|no_meaningful_visibility_gap|no_later_relevant_evidence|later_too_unclear|later_irrelevant|request_not_visual|uncertain>",
  "brief_reason": "<one short sentence based only on visible evidence>"
}}"""


OOV_GAP_BOOL_CLEAN_PROMPT_TEMPLATE = """You are judging whether this single video contains a request-relevant out-of-view / out-of-observation pattern.

Use only the video and the text request. Use the text request only to identify what subject, object, action, state, target area, support surface, or expected result should be inspected. The text request itself is not evidence that anything disappeared, returned, changed, or failed. Do not use metadata, filenames, model names, dataset names, or outside knowledge.

Text request:
{world_state_prompt}

Answer true only if all of these are satisfied:

1. Before the interval, the request-relevant subject, object, action, state, target area, support surface, or expected result area is visible enough to establish a visual reference.

2. Then, for a meaningful interval, the request-relevant evidence becomes not visually judgeable from the video pixels. This can happen because it leaves the frame, is hidden, occluded, heavily cropped, too blurred, too small, blocked by camera motion, or the relevant target/result area is no longer visible enough to judge.

3. Later, the video provides clear comparable visual evidence. This can be the same subject/object reappearing, the relevant target/result area becoming visible again, or clear visual evidence of absence, wrong state, reset, failure, replacement, or changed result.

Answer false if any of these apply:

- The request-relevant subject/object/action/state remains continuously visible enough to judge.
- The video only shows ordinary action progress, such as someone walking, sitting, placing an object, carrying an item, or interacting with a target while remaining judgeable.
- Later frames are clearer, closer, or more complete, but there was no meaningful interval where the request-relevant evidence became unjudgeable.
- Camera motion, scene motion, or wording in the text request suggests a gap, but the video pixels do not show a real unjudgeable interval.
- The later visible person/object/region is unrelated to the requested role, target area, action, object, support surface, or result.
- The later evidence is only blurry, ghosted, too small, unstable, or identity-ambiguous, so the requested subject/state/result cannot be identified from pixels.
- There is no clear before reference or no later comparable evidence.

Important:
- Later clear visibility is necessary for a true answer, but it is not sufficient.
- Do not require the exact same physical object to reappear if the relevant target/result area clearly shows absence, change, failure, reset, or a comparable result.
- Do not reject a video merely because some frames are blurry or ghosted; reject only if the later comparable evidence itself is not identifiable from pixels.
- If uncertain whether a meaningful unjudgeable interval exists, answer false.
- If uncertain whether later evidence is comparable to the request, answer false.

Return exactly one JSON object. Do not include markdown.

{{
  "has_oov_gap_with_later_comparable_evidence": <true|false>,
  "reason_code": "<clear_reappearance|clear_comparable_region_or_result|clear_absence_or_change|no_meaningful_unjudgeable_interval|continuous_visible_or_judgeable|no_clear_before_reference|no_later_comparable_evidence|later_too_unclear|later_irrelevant|request_not_visual|uncertain>",
  "brief_reason": "<one short sentence naming what became unjudgeable and what later evidence is comparable, or why no such pattern exists>"
}}"""


OOV_GAP_SCAN_CLEAN_PROMPT_TEMPLATE = """You are a strict temporal visibility auditor for a single video.

Use only the video and the text request. Use the text request only to identify the prompt-critical subject, object, action, state, target area, support surface, expected result, or comparable region to inspect. The text request itself is not evidence that anything disappeared, returned, changed, failed, or stayed visible. Do not use metadata, filenames, model names, dataset names, or outside knowledge.

Text request:
{world_state_prompt}

Task:
1. Identify the prompt-critical visual evidence needed to judge the request.
2. Fill visibility_scan before deciding final_oov_applicable.
3. Scan the video chronologically. Return exactly 8 checkpoints in visibility_scan, covering the beginning, several middle portions, and the end. Include any checkpoint where the evidence is at the frame edge, cropped, absent, hidden, blurred, or replaced.
4. For each checkpoint, judge only that local moment. Do not infer visibility from earlier or later moments.

judgeable = yes when:
- the prompt-critical subject/object/action/state is visible enough to judge; or
- the relevant target/result/support/region is clearly visible enough to judge absence, change, wrong state, reset, failure, or comparable result.
- Wrong, absent, replaced, reset, failed, or changed later evidence is still judgeable when the visible pixels are clear enough to judge it.

judgeable = no when:
- the prompt-critical evidence is off-screen, hidden, occluded, heavily cropped, too blurred, too small, blocked by camera motion, or otherwise not visually judgeable at that checkpoint.
- only a tiny edge fragment or cropped sliver is visible and its position/action/state cannot be judged.

judgeable = unclear when:
- you cannot tell whether the prompt-critical evidence is judgeable at that checkpoint.

Final rule:
- final_oov_applicable should be true only when visibility_scan shows a judgeable reference first, then a later not-judgeable or unclear interval, and then later judgeable comparable evidence.
- final_oov_applicable should be false when the evidence remains judgeable across the beginning, middle, and later portions, or when no later comparable judgeable evidence appears.
- Do not answer false merely because the evidence is visible in the first and last parts; inspect the middle checkpoints.
- Do not mark later evidence not-judgeable merely because the requested action failed, the original subject is absent, or a replacement/wrong state appears. If that later visible evidence is clear enough to judge the request result, mark it yes.
- Do not decide whether the requested outcome is correct. Wrong but clearly judgeable later evidence is still applicable.

Return exactly one JSON object. Do not include markdown.

{{
  "prompt_critical_evidence": "<short subject/object/action/state/target/result/region>",
  "visibility_scan": [
    {{
      "step": <integer chronological index starting at 0>,
      "phase": "<early|middle|later>",
      "judgeable": "<yes|no|unclear>",
      "note": "<short visual note for this checkpoint>"
    }}
  ],
  "final_oov_applicable": <true|false>,
  "brief_reason": "<one short sentence explaining the temporal pattern>"
}}"""


OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_PROMPT_TEMPLATE = """You are checking one video.

Use only the video and the text request. Use the text request only to decide what visual thing matters: the relevant person, animal, object, action, state, target area, support surface, or final result. The text request itself is not evidence that anything happened.

Text request:
{world_state_prompt}

Judge only whether the relevant visual thing can be seen well enough to judge. Do not judge whether the requested event is correct.

Inspect the sampled video in chronological order. For each whole second you can inspect, write one per_second row.

status values:
- visible: the relevant visual thing is clear enough to judge in this second. This also includes a clearly visible target/result area that lets you judge absence, vanished result, failed action, reset, replacement, wrong position, wrong state, or wrong final state.
- not_visible: the relevant visual thing and the relevant target/result area are not clear enough to judge in this second because they are hidden, outside the frame, too cropped, too small, too blurred, blocked, distorted, or missing without a clear relevant area.
- unrelated: something is visible, but it is not tied to the requested person, object, action, state, target area, support surface, or result.
- unclear: the second is ambiguous.

When more than one status seems possible:
- Use visible when the person/object/action/result itself is clear, or when the target/result area clearly shows absence, failure, reset, replacement, wrong position, wrong state, or wrong final state.
- Use not_visible when neither the relevant visual thing nor a useful target/result area is clear enough to judge.
- Use unrelated only when the visible content is not connected to the request.

Do not mark visible merely because generic background, floor, wall, furniture, or a support surface is visible. A support surface counts only when it helps judge the requested thing in that second.

Judge each second locally. Do not infer visibility from earlier or later seconds.

The per_second list is audit evidence. The final boolean should reflect the whole video. Do not set false only because the exact boundary second of the hidden or unclear interval is hard to locate, as long as the video visibly contains a meaningful request-related interval that is not clear enough to judge, followed by clear judgeable evidence.

Set can_judge_after_gap = true when:
1. before the unclear/hidden interval, there is at least one visible second;
2. then there is a request-related interval with at least one not_visible second. A single unclear second alone is not enough;
3. later, there is a visible second that a human can use to judge the request.

Set can_judge_after_gap = false for ordinary continuous visibility with no request-related not_visible interval, for later visible evidence unrelated to the request, or when the video is too unclear to judge.

Do not set false merely because correctness, exact continuity, exact identity, or exact trajectory is uncertain.
A wrong outcome, failed action, missing object, visible absence, reset, replacement, or wrong final state is still judgeable if it is clearly visible and tied to the request.

Return exactly one JSON object. Do not include markdown.

{{
  "look_for": "<short description of the relevant person/object/action/state/target area/support surface/final result>",
  "per_second": [
    {{
      "sec": <integer second>,
      "status": "<visible|not_visible|unrelated|unclear>",
      "note": "<short visual evidence>"
    }}
  ],
  "can_judge_after_gap": <true|false>,
  "reason_code": "<yes_after_gap|always_visible|no_start|no_gap|no_later_visible|later_unrelated|too_unclear|not_visual|unclear>",
  "brief_reason": "<one short sentence using seconds from per_second>"
}}"""


OOV_GAP_PER_SECOND_STRICT_COLLAPSE_PROMPT_TEMPLATE = """You are checking one video.

Use only the video and the text request. Use the text request only to decide what visual thing matters: the requested person, animal, object, action, state, target area, support surface, or final result. The text request itself is not evidence that anything happened.

Text request:
{world_state_prompt}

Your job is to describe what can actually be judged from the video, one whole second at a time.

For each whole second you can inspect, write one per_second row with exactly one status.

status values:
- visible: the requested person/object/action/result is actually in frame and clear enough to judge in this second. For placement or interaction tasks, a target/result area counts as visible only when the relevant object or final placed/result state is also visible enough to judge.
- not_visible: the requested person/object/action/result is outside the frame, hidden, blocked, cropped out, too small, or absent in this second. A background target area alone, such as an empty pallet, table, brick stack, counter, chair, sofa, bed, bucket, or floor, is not visible evidence.
- collapsed: the video content breaks down in this second: severe ghosting, warped or melted people/objects, duplicated bodies, smeared foreground, unstable generated content, or scene collapse that prevents judging the requested thing.
- unclear: ambiguous, but not severe enough to call collapsed.

Important distinctions:
- If only the background or support surface is visible, use not_visible.
- If only a tiny edge fragment is visible and the requested person/object/action cannot be judged, use not_visible.
- If a relevant object is still visible enough to judge the requested result, use visible even if the actor is partly outside the frame.
- If the scene visibly breaks or turns into a ghosted/warped generated artifact, use collapsed, not not_visible.
- Judge each second locally. Do not infer visibility from earlier or later seconds.

Return exactly one JSON object. Do not include markdown.

{{
  "look_for": "<short description of the requested person/object/action/result to inspect>",
  "per_second": [
    {{
      "sec": <integer second>,
      "status": "<visible|not_visible|collapsed|unclear>",
      "note": "<short visual evidence>"
    }}
  ],
  "brief_reason": "<one short sentence summarizing the visible/not_visible/collapsed pattern>"
}}"""


OOV_SUBJECT_RESULT_INTEGRITY_PROMPT_TEMPLATE = """You are checking one video.

Use only the video and the text request. Use the text request only to decide what visual subject, result evidence, and scene quality matter. The text request itself is not evidence that anything happened.

Text request:
{world_state_prompt}

Your job is to describe what can actually be judged from the video, one whole second at a time.

Step 1: Choose one main_subject. Prefer the person, animal, or moving object that must remain visually followable to judge the request.
Step 2: Describe result_evidence. This is the target object, final placed object, target area, support surface, or visible result that can help judge the request after the main subject is not visible.
Step 3: For each whole second, fill all three lists.

The video may be shown to you as sampled frames. Do not treat frame count as seconds. Use at most 8 chronological rows in each list. All three lists must use the same sec values, starting at 0 and moving through the visible beginning, middle, and end of the video.

subject_per_second:
- yes: the same main_subject is in frame, identifiable, and clear enough to judge position, action, or state.
- no: the main_subject is outside the frame, hidden, blocked, cropped out, too small, too blurred, or not identifiable.
- unclear: ambiguous.

result_evidence_per_second:
- yes: relevant result evidence is visible and clear enough to judge the request, such as the placed object, target area with the relevant object/result, visible absence/failure/reset, or final state.
- no: the relevant result evidence is not visible or not clear enough. A generic background or empty support surface alone is no.
- unclear: ambiguous.

scene_integrity_per_second:
- coherent: the frame is visually coherent enough to judge the subject or result if they are visible.
- broken: the generated video breaks visually: severe ghosting, melted or duplicated bodies/objects, incompatible scene jump, impossible geometry, or severe smear/noise that prevents judging the request.
- unclear: ambiguous.

Important:
- Judge the same main_subject locally each second. Do not mark the subject yes because only the target area or support surface is visible.
- If the subject is gone but the relevant placed object or final result is clear, subject can be no while result_evidence is yes.
- If the video itself breaks, mark scene_integrity broken even if something resembling the subject remains.
- Do not decide whether the requested outcome is correct. Only report what is judgeable from pixels.

Return exactly one JSON object. Do not include markdown.

{{
  "main_subject": "<short subject name>",
  "result_evidence": "<short description of result/target evidence>",
  "subject_per_second": [
    {{"sec": <integer second>, "judgeable": "<yes|no|unclear>", "note": "<short visual evidence>"}}
  ],
  "result_evidence_per_second": [
    {{"sec": <integer second>, "judgeable": "<yes|no|unclear>", "note": "<short visual evidence>"}}
  ],
  "scene_integrity_per_second": [
    {{"sec": <integer second>, "status": "<coherent|broken|unclear>", "note": "<short visual evidence>"}}
  ],
  "brief_reason": "<one short sentence summarizing the subject/result/scene pattern>"
}}"""


OOV_GAP_TRIPLET_CLEAN_PROMPT_TEMPLATE = """You are a strict visual evidence gate for a single video.

Use only the video and the text request. Use the text request only to identify the prompt-critical subject, object, action, state, target area, support surface, expected result, or comparable region to inspect. The text request itself is not evidence that anything disappeared, returned, changed, failed, or stayed visible. Do not use metadata, filenames, model names, dataset names, or outside knowledge.

Text request:
{world_state_prompt}

Fill the three evidence fields before deciding final_oov_applicable. Your job is not to score whether the requested event is correct. Your job is only to decide whether there is enough before/gap/later visual evidence for the event to be judged.

Field 1: before_reference
- present = yes when the prompt-critical subject/object/target/result region is visible enough early in the video to establish a visual reference.

Field 2: middle_unjudgeable_gap
- present = yes when, after the before_reference, the prompt-critical evidence becomes not visually judgeable for a meaningful middle interval.
- Count off-screen, hidden, occluded, heavily cropped, tiny edge fragments, too blurred, camera-motion blocked, or otherwise unidentifiable evidence as a gap.
- present = no only when the prompt-critical evidence remains judgeable through the middle portion.

Field 3: later_comparable_evidence
- present = yes when later video pixels provide clear comparable evidence after the gap.
- Do not require the exact same subject or object to reappear.
- Later comparable evidence can be the same subject/object, the target/result/support region, visible absence, change, failure, replacement, reset, wrong state, or wrong result.
- present = no only when there is no later clear comparable visual evidence after the gap.

Important:
- Do not answer false merely because the subject is visible in the first and last parts; the middle gap still matters.
- Do not mark later evidence absent merely because the requested action failed or the original subject is gone. If later pixels clearly show absence, change, failure, replacement, reset, wrong state, or the target/result region, mark later_comparable_evidence present = yes and choose the matching type.
- final_oov_applicable should be true only when before_reference, middle_unjudgeable_gap, and later_comparable_evidence are all present = yes.

Return exactly one JSON object. Do not include markdown.

{{
  "prompt_critical_evidence": "<short subject/object/action/state/target/result/region>",
  "before_reference": {{
    "present": "<yes|no|unclear>",
    "note": "<short visual evidence>"
  }},
  "middle_unjudgeable_gap": {{
    "present": "<yes|no|unclear>",
    "note": "<short visual evidence>"
  }},
  "later_comparable_evidence": {{
    "present": "<yes|no|unclear>",
    "type": "<same_subject|target_region|absence|change|failure|replacement|unclear>",
    "note": "<short visual evidence>"
  }},
  "final_oov_applicable": <true|false>,
  "brief_reason": "<one short sentence explaining the before/gap/later evidence>"
}}"""


OOV_GAP_TRIPLET_SHEET_CLEAN_PROMPT_TEMPLATE = """You are checking a contact sheet made from one video.

Use the contact sheet as visual evidence. Use the text request only to decide what to look for: the person, animal, object, action, state, target area, support surface, or final result that matters for the request.

The sheet shows sampled frames in chronological order. Each frame has a label such as idx0, idx8, idx16, or similar. Refer to those labels in your notes. Judge only the frames that are shown.

Text request:
{world_state_prompt}

The decision is about whether the sampled frames show a three-part visual pattern, not whether the event is correct.

Answer final_applicable = true only when all three fields below are present = yes:

First scan the sampled frames one by one in chronological order. For every labeled frame you can read, write one frame_visibility entry.
- visible = yes only when the relevant visual evidence is identifiable in that frame.
- visible = no when it is absent, fully hidden, outside the frame, too cropped, too small, too blurred, or too distorted to identify.
- visible = unclear when the frame is ambiguous.
- Do not summarize a range as visible if any frame inside the range is not identifiable.

Field 1: early_reference
- present = yes when an early labeled frame makes the relevant visual evidence clear enough to know what to track.

Field 2: middle_cannot_follow
- present = yes when any non-early frame after early_reference shows that the relevant visual evidence cannot be visually followed. This includes visible = no or visible = unclear in frame_visibility after an early yes.
- present = no only when frame_visibility has no later no/unclear frame for the relevant visual evidence.
- present = no when the relevant person or object naturally leaves after the requested event is already visible and understandable.

Field 3: later_comparison
- present = yes when a later labeled frame gives clear comparable visual evidence after middle_cannot_follow. This may show the same subject/object, the target or result area, a visible absence, a changed state, a failed result, a replacement, or a wrong final state.
- present = no when the later frames are too unclear or unrelated to compare with the early reference.

If the full three-part pattern is uncertain, set final_applicable = false.

Return exactly one JSON object. Do not include markdown.

{{
  "critical_evidence": "<short subject/object/action/state/target/result/region>",
  "frame_visibility": [
    {{
      "label": "<frame label such as idx0>",
      "visible": "<yes|no|unclear>",
      "note": "<short visual evidence>"
    }}
  ],
  "early_reference": {{
    "present": "<yes|no|unclear>",
    "note": "<short visual evidence with frame label>"
  }},
  "middle_cannot_follow": {{
    "present": "<yes|no|unclear>",
    "note": "<short visual evidence with frame label>"
  }},
  "later_comparison": {{
    "present": "<yes|no|unclear>",
    "type": "<same_subject|target_region|absence|change|failure|replacement|unclear>",
    "note": "<short visual evidence with frame label>"
  }},
  "final_applicable": <true|false>,
  "brief_reason": "<one short sentence explaining the three visual fields>"
}}"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_shard(video_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def select_manifest_shard(manifest: list[dict[str, Any]], *, num_shards: int, shard_id: int) -> list[dict[str, Any]]:
    return [
        item
        for item in manifest
        if item.get("video_id") and stable_shard(str(item["video_id"]), num_shards) == shard_id
    ]


def attach_outer_request_metadata(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row_index, item in enumerate(manifest):
        item_copy = dict(item)
        video_id = str(item_copy.get("video_id") or "")
        item_copy["row_index"] = int(item_copy.get("row_index", row_index))
        if not item_copy.get("request_id") and video_id:
            digest = hashlib.sha256(f"{row_index}:{video_id}".encode("utf-8")).hexdigest()[:16]
            item_copy["request_id"] = f"qwen3vl-{row_index:06d}-{digest}"
        annotated.append(item_copy)
    return annotated


def id_map(rows: list[dict[str, Any]], *, label: str = "rows") -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.get("video_id") is None:
            continue
        video_id = str(row.get("video_id"))
        if video_id in mapped:
            duplicates.append(video_id)
        mapped[video_id] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate video_id values: {sorted(set(duplicates))[:10]}")
    return mapped


def build_judgeability_prompt(
    *,
    world_state_prompt: str,
    video_id: str,
    prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
) -> str:
    if prompt_schema == PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY:
        return OOV_SUBJECT_RESULT_INTEGRITY_PROMPT_TEMPLATE.format(
            world_state_prompt=world_state_prompt
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE:
        return OOV_GAP_PER_SECOND_STRICT_COLLAPSE_PROMPT_TEMPLATE.format(
            world_state_prompt=world_state_prompt
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN:
        return OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_PROMPT_TEMPLATE.format(
            world_state_prompt=world_state_prompt
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN:
        return OOV_GAP_TRIPLET_SHEET_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN:
        return OOV_GAP_TRIPLET_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN:
        return OOV_GAP_SCAN_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN:
        return OOV_GAP_BOOL_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN:
        return VISIBLE_BOOL_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN:
        return SHARED_DIRECT3Q_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_GUARDED_CLEAN:
        return GUARDED_CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_SUBJECT_CLEAN:
        return CLEAN_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt)
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return GUARDED_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt, video_id=video_id)
    if prompt_schema == PROMPT_SCHEMA_OBJECT:
        return OBJECT_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt, video_id=video_id)
    if prompt_schema != PROMPT_SCHEMA_SUBJECT:
        raise ValueError(f"unsupported prompt_schema: {prompt_schema}")
    return PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt, video_id=video_id)


def schema_version_for_prompt_schema(prompt_schema: str) -> str:
    if prompt_schema == PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY:
        return OOV_SUBJECT_RESULT_INTEGRITY_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE:
        return OOV_GAP_PER_SECOND_STRICT_COLLAPSE_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN:
        return OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN:
        return OOV_GAP_TRIPLET_SHEET_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN:
        return OOV_GAP_TRIPLET_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN:
        return OOV_GAP_SCAN_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN:
        return OOV_GAP_BOOL_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN:
        return VISIBLE_BOOL_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN:
        return SHARED_DIRECT3Q_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_SUBJECT_CLEAN:
        return SUBJECT_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_GUARDED_CLEAN:
        return GUARDED_CLEAN_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_OBJECT:
        return OBJECT_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return GUARDED_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_SUBJECT:
        return SUBJECT_SCHEMA_VERSION
    raise ValueError(f"unsupported prompt_schema: {prompt_schema}")


def clean_schema_for_prompt_schema(prompt_schema: str) -> bool:
    return prompt_schema in {
        PROMPT_SCHEMA_SUBJECT_CLEAN,
        PROMPT_SCHEMA_GUARDED_CLEAN,
        PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN,
        PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN,
        PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
        PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY,
    }


def build_visibility_prompt(*, world_state_prompt: str, video_id: str) -> str:
    """Compatibility wrapper for older local tests and callers."""
    return build_judgeability_prompt(world_state_prompt=world_state_prompt, video_id=video_id)


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""
    candidate = _strip_json_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no valid JSON object found in model output")


def _require_type(errors: list[str], payload: dict[str, Any], key: str, expected_type: type | tuple[type, ...]) -> None:
    if key not in payload:
        errors.append(f"missing key: {key}")
    elif not isinstance(payload[key], expected_type):
        errors.append(f"{key} must be {expected_type}")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped and stripped.lower() != "null" else None
    return str(value).strip() or None


def _bool_or_error(errors: list[str], value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
    errors.append(f"{key} must be boolean")
    return False


def _direct3q_value_or_error(errors: list[str], value: Any, key: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DIRECT3Q_VALUES:
        return normalized
    errors.append(f"{key} invalid: {value!r}")
    return "unclear"


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
    return None


def _derive_temporal_oov_gate(
    normalized_rows: list[dict[str, Any]],
    *,
    judgeable_key: str,
) -> dict[str, Any]:
    """Derive the hard D5/D6 applicability gate from sampled visibility rows."""
    first_gap_index: int | None = None
    saw_prior_judgeable = False
    for index, entry in enumerate(normalized_rows):
        judgeable = str(entry.get(judgeable_key) or "")
        if judgeable == "yes":
            saw_prior_judgeable = True
        elif judgeable == "no" and saw_prior_judgeable:
            first_gap_index = index
            break

    if first_gap_index is not None:
        gap_row = normalized_rows[first_gap_index]
        start_sec = float(gap_row.get("sec", first_gap_index))
        end_sec = start_sec + 1.0
        for entry in normalized_rows[first_gap_index + 1 :]:
            if str(entry.get(judgeable_key) or "") == "no":
                end_sec = float(entry.get("sec", end_sec)) + 1.0
            else:
                break
        oov_interval = {"start_sec": start_sec, "end_sec": end_sec, "status": "present"}
        return_judgeable = any(
            str(entry.get(judgeable_key) or "") == "yes"
            for entry in normalized_rows[first_gap_index + 1 :]
        )
        na_reason = None if return_judgeable else "no_return"
    else:
        oov_interval = {"start_sec": None, "end_sec": None, "status": "absent"}
        return_judgeable = False
        if any(str(entry.get(judgeable_key) or "") == "no" for entry in normalized_rows):
            na_reason = "no_initial_judgeable"
        else:
            na_reason = "no_oov"

    return {
        "first_gap_index": first_gap_index,
        "oov_interval": oov_interval,
        "return_judgeable": bool(return_judgeable),
        "applicable": na_reason is None,
        "na_reason": na_reason,
    }


def _derive_visibility_scan_oov_gate(normalized_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_reference_index: int | None = None
    for index, entry in enumerate(normalized_rows):
        if str(entry.get("judgeable") or "") == "yes":
            first_reference_index = index
            break

    if first_reference_index is None:
        return {
            "first_reference_index": None,
            "first_gap_index": None,
            "return_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
            "return_judgeable": False,
            "applicable": False,
            "na_reason": "no_initial_judgeable",
        }

    first_gap_index: int | None = None
    for index in range(first_reference_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("judgeable") or "") != "yes":
            first_gap_index = index
            break

    if first_gap_index is None:
        return {
            "first_reference_index": first_reference_index,
            "first_gap_index": None,
            "return_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
            "return_judgeable": False,
            "applicable": False,
            "na_reason": "no_oov",
        }

    gap_end_index = first_gap_index
    for index in range(first_gap_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("judgeable") or "") != "yes":
            gap_end_index = index
            continue
        break

    return_index: int | None = None
    for index in range(gap_end_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("judgeable") or "") == "yes":
            return_index = index
            break

    start_step = float(normalized_rows[first_gap_index].get("step", first_gap_index))
    end_step = float(normalized_rows[gap_end_index].get("step", gap_end_index)) + 1.0
    oov_interval = {"start_sec": start_step, "end_sec": end_step, "status": "present"}
    return_judgeable = return_index is not None
    return {
        "first_reference_index": first_reference_index,
        "first_gap_index": first_gap_index,
        "return_index": return_index,
        "oov_interval": oov_interval,
        "return_judgeable": return_judgeable,
        "applicable": return_judgeable,
        "na_reason": None if return_judgeable else "no_later_comparable_evidence",
    }


def _derive_direct3q_oov_gate(
    *,
    gap_observed: str,
    after_return_clear_enough: str,
    too_unclear_after_return: str,
    before_gap_evidence: str | None,
    after_return_evidence: str | None,
) -> dict[str, Any]:
    if (
        gap_observed == "yes"
        and after_return_clear_enough == "yes"
        and too_unclear_after_return != "yes"
    ):
        return {"applicable": True, "na_reason": None}
    if gap_observed == "no":
        if before_gap_evidence is None:
            return {"applicable": False, "na_reason": "no_initial_clear_evidence"}
        return {"applicable": False, "na_reason": "no_visibility_gap"}
    if gap_observed == "unclear":
        return {"applicable": False, "na_reason": "unclear"}
    if after_return_clear_enough == "no":
        if after_return_evidence is None:
            return {"applicable": False, "na_reason": "no_later_return"}
        return {"applicable": False, "na_reason": "return_too_unclear"}
    if after_return_clear_enough == "unclear":
        return {"applicable": False, "na_reason": "unclear"}
    if too_unclear_after_return == "yes":
        return {"applicable": False, "na_reason": "return_too_unclear"}
    return {"applicable": False, "na_reason": "unclear"}


def add_direct3q_gate_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    shared_app = bool(parsed["oov_applicable"])
    shared_reason = None if shared_app else str(parsed.get("na_reason") or "unclear")
    gap_observed = str(parsed.get("gap_observed") or "unclear")
    after_return_clear_enough = str(parsed.get("after_return_clear_enough") or "unclear")
    too_unclear_after_return = str(parsed.get("too_unclear_after_return") or "unclear")
    oov_status = "present" if gap_observed == "yes" else ("absent" if gap_observed == "no" else "unclear")
    return_judgeable = after_return_clear_enough == "yes" and too_unclear_after_return != "yes"
    notes = [
        str(parsed.get("before_gap_evidence") or "").strip(),
        str(parsed.get("after_return_evidence") or "").strip(),
        str(parsed.get("short_reason") or "").strip(),
    ]
    parsed.update(
        {
            "main_subject": parsed["prompt_critical_evidence"],
            "segments": [],
            "subject_judgeable": bool(return_judgeable),
            "target_judgeable": bool(return_judgeable),
            "action_state_judgeable": bool(return_judgeable),
            "oov_interval": {"start_sec": None, "end_sec": None, "status": oov_status},
            "return_judgeable": bool(return_judgeable),
            "notes_short": "; ".join(note for note in notes if note)[:360],
            "shared_oov_applicable": shared_app,
            "shared_oov_na_reason": shared_reason,
            "evidence_shared_oov_applicable": shared_app,
            "evidence_shared_oov_na_reason": shared_reason,
            "evidence_subject_judgeable": bool(return_judgeable),
            "evidence_target_judgeable": bool(return_judgeable),
            "evidence_action_state_judgeable": bool(return_judgeable),
            "evidence_oov_interval_status": oov_status,
            "evidence_return_judgeable": bool(return_judgeable),
            "evidence_d5_applicable": shared_app,
            "evidence_d6_applicable": shared_app,
            "evidence_d5_na_reason": None if shared_app else shared_reason,
            "evidence_d6_na_reason": None if shared_app else shared_reason,
        }
    )
    return parsed


def add_evidence_first_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    timeline = parsed.get("per_second") or []
    segments: list[dict[str, Any]] = []
    for index, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            continue
        sec = float(entry.get("sec", index))
        visible_subject = str(entry.get("judgeable") or "no").strip()
        note = str(entry.get("note") or "").strip()
        segments.append(
            {
                "start_sec": sec,
                "end_sec": sec + 1.0,
                "visible_subject": visible_subject,
                "note": note,
            }
        )

    normalized_rows = parsed.get("per_second") or []
    temporal_gate = _derive_temporal_oov_gate(normalized_rows, judgeable_key="judgeable")
    first_gap_index = temporal_gate["first_gap_index"]
    oov_interval = temporal_gate["oov_interval"]
    return_judgeable = bool(temporal_gate["return_judgeable"])
    na_reason = temporal_gate["na_reason"]
    evidence_applicable = bool(temporal_gate["applicable"])

    parsed.update(
        {
            "segments": segments,
            "subject_judgeable": bool(return_judgeable or first_gap_index is None),
            "target_judgeable": bool(return_judgeable or first_gap_index is None),
            "action_state_judgeable": bool(return_judgeable or first_gap_index is None),
            "oov_interval": oov_interval,
            "return_judgeable": bool(return_judgeable),
            "notes_short": "; ".join(
                str(entry.get("note") or "").strip()
                for entry in normalized_rows[:3]
                if str(entry.get("note") or "").strip()
            ),
            "shared_oov_applicable": bool(evidence_applicable),
            "shared_oov_na_reason": na_reason,
            "evidence_shared_oov_applicable": bool(evidence_applicable),
            "evidence_shared_oov_na_reason": na_reason,
            "evidence_subject_judgeable": bool(return_judgeable or first_gap_index is None),
            "evidence_target_judgeable": bool(return_judgeable or first_gap_index is None),
            "evidence_action_state_judgeable": bool(return_judgeable or first_gap_index is None),
            "evidence_oov_interval_status": oov_interval["status"],
            "evidence_return_judgeable": bool(return_judgeable),
            "evidence_d5_applicable": bool(evidence_applicable),
            "evidence_d6_applicable": bool(evidence_applicable),
            "evidence_d5_na_reason": None if evidence_applicable else na_reason or "unclear",
            "evidence_d6_na_reason": None if evidence_applicable else na_reason or "unclear",
        }
    )
    return parsed


def add_object_evidence_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    timeline = parsed.get("per_second") or []
    segments: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            continue
        sec = int(entry.get("sec", index))
        judgeable = str(entry.get("object_judgeable") or entry.get("judgeable") or "no").strip()
        note = str(entry.get("observed_motion_or_state") or entry.get("note") or "").strip()
        row = {
            "sec": sec,
            "object_judgeable": judgeable,
            "observed_motion_or_state": note,
            "judgeable": judgeable,
            "note": note,
        }
        normalized_rows.append(row)
        segments.append(
            {
                "start_sec": float(sec),
                "end_sec": float(sec) + 1.0,
                "visible_subject": judgeable,
                "object_judgeable": judgeable,
                "note": note,
            }
        )

    temporal_gate = _derive_temporal_oov_gate(normalized_rows, judgeable_key="object_judgeable")
    oov_interval = temporal_gate["oov_interval"]
    return_judgeable = bool(temporal_gate["return_judgeable"])

    gate = parsed.get("gate") or {}
    model_d5_app = bool(gate.get("d5_applicable"))
    model_d6_app = bool(gate.get("d6_applicable"))
    model_na_reason = _string_or_none(gate.get("na_reason"))
    if not temporal_gate["applicable"]:
        d5_app = False
        d6_app = False
        shared_app = False
        shared_reason = str(temporal_gate["na_reason"] or "unclear")
    else:
        d5_app = model_d5_app
        d6_app = model_d6_app
        shared_app = bool(d5_app and d6_app)
        shared_reason = None if shared_app else model_na_reason or "unclear"

    parsed.update(
        {
            "per_second": normalized_rows,
            "main_subject": parsed.get("prompt_critical_object"),
            "segments": segments,
            "subject_judgeable": bool(return_judgeable),
            "target_judgeable": bool(return_judgeable),
            "action_state_judgeable": bool(return_judgeable),
            "oov_interval": oov_interval,
            "return_judgeable": bool(return_judgeable),
            "notes_short": "; ".join(
                str(entry.get("observed_motion_or_state") or "").strip()
                for entry in normalized_rows[:3]
                if str(entry.get("observed_motion_or_state") or "").strip()
            ),
            "shared_oov_applicable": shared_app,
            "shared_oov_na_reason": shared_reason,
            "evidence_shared_oov_applicable": shared_app,
            "evidence_shared_oov_na_reason": shared_reason,
            "evidence_subject_judgeable": bool(return_judgeable),
            "evidence_target_judgeable": bool(return_judgeable),
            "evidence_action_state_judgeable": bool(return_judgeable),
            "evidence_oov_interval_status": oov_interval["status"],
            "evidence_return_judgeable": bool(return_judgeable),
            "evidence_d5_applicable": d5_app,
            "evidence_d6_applicable": d6_app,
            "evidence_d5_na_reason": None if d5_app else shared_reason,
            "evidence_d6_na_reason": None if d6_app else shared_reason,
        }
    )
    return parsed


def add_guarded_teacher_gate_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    timeline = parsed.get("per_second") or []
    segments: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            continue
        sec = int(entry.get("sec", index))
        judgeable = str(entry.get("judgeable") or "no").strip()
        note = str(entry.get("note") or "").strip()
        row = {
            "sec": sec,
            "judgeable": judgeable,
            "note": note,
        }
        normalized_rows.append(row)
        segments.append(
            {
                "start_sec": float(sec),
                "end_sec": float(sec) + 1.0,
                "visible_subject": judgeable,
                "prompt_critical_judgeable": judgeable,
                "note": note,
            }
        )

    temporal_gate = _derive_temporal_oov_gate(normalized_rows, judgeable_key="judgeable")
    oov_interval = temporal_gate["oov_interval"]
    return_judgeable = bool(temporal_gate["return_judgeable"])

    gate = parsed.get("gate") or {}
    model_d5_app = bool(gate.get("d5_applicable"))
    model_d6_app = bool(gate.get("d6_applicable"))
    model_na_reason = _string_or_none(gate.get("na_reason"))
    if not temporal_gate["applicable"]:
        d5_app = False
        d6_app = False
        shared_app = False
        shared_reason = str(temporal_gate["na_reason"] or "unclear")
    else:
        d5_app = model_d5_app
        d6_app = model_d6_app
        shared_app = bool(d5_app and d6_app)
        shared_reason = None if shared_app else model_na_reason or "unclear"
    audit = parsed.get("audit") if isinstance(parsed.get("audit"), dict) else {}
    anchor_available = _optional_bool(audit.get("anchor_available"))

    parsed.update(
        {
            "per_second": normalized_rows,
            "main_subject": parsed.get("prompt_critical_evidence"),
            "segments": segments,
            "subject_judgeable": bool(return_judgeable),
            "target_judgeable": bool(anchor_available if anchor_available is not None else return_judgeable),
            "action_state_judgeable": bool(return_judgeable),
            "oov_interval": oov_interval,
            "return_judgeable": bool(return_judgeable),
            "notes_short": "; ".join(
                str(entry.get("note") or "").strip()
                for entry in normalized_rows[:3]
                if str(entry.get("note") or "").strip()
            ),
            "shared_oov_applicable": shared_app,
            "shared_oov_na_reason": shared_reason,
            "evidence_shared_oov_applicable": shared_app,
            "evidence_shared_oov_na_reason": shared_reason,
            "evidence_subject_judgeable": bool(return_judgeable),
            "evidence_target_judgeable": bool(anchor_available if anchor_available is not None else return_judgeable),
            "evidence_action_state_judgeable": bool(return_judgeable),
            "evidence_oov_interval_status": oov_interval["status"],
            "evidence_return_judgeable": bool(return_judgeable),
            "evidence_d5_applicable": d5_app,
            "evidence_d6_applicable": d6_app,
            "evidence_d5_na_reason": None if d5_app else shared_reason,
            "evidence_d6_na_reason": None if d6_app else shared_reason,
        }
    )
    return parsed


def validate_judgeability_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
    clean_schema: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [key for key in REQUIRED_KEYS if not (clean_schema and key == "video_id")]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")
    if clean_schema:
        payload = dict(payload)
        payload["video_id"] = expected_video_id
    else:
        if (
            payload.get("video_id") != expected_video_id
            and allow_video_id_repair
            and isinstance(payload.get("video_id"), str)
        ):
            payload = dict(payload)
            payload["video_id"] = expected_video_id
        if payload.get("video_id") != expected_video_id:
            errors.append(
                f"video_id mismatch: expected {expected_video_id!r}, got {payload.get('video_id')!r}"
            )

    _require_type(errors, payload, "main_subject", str)
    _require_type(errors, payload, "per_second", list)

    normalized_per_second: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("per_second") or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        for key in ("sec", "judgeable", "note"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        judgeable = str(entry.get("judgeable") or "").strip()
        if judgeable not in JUDGEABILITY_VALUES:
            errors.append(f"per_second[{index}].judgeable invalid: {entry.get('judgeable')!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_per_second.append(
            {
                "sec": sec,
                "judgeable": judgeable,
                "note": str(note).strip()[:160],
            }
        )
    if errors:
        raise ValueError("; ".join(errors))

    parsed = {
        "video_id": payload.get("video_id"),
        "main_subject": str(payload.get("main_subject") or "").strip(),
        "per_second": sorted(normalized_per_second, key=lambda row: row["sec"]),
    }
    parsed.update(
        {
            "schema_version": SUBJECT_CLEAN_SCHEMA_VERSION if clean_schema else SCHEMA_VERSION,
        }
    )
    return add_evidence_first_fields(parsed)


def validate_object_judgeability_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "video_id",
        "prompt_critical_object",
        "expected_motion_or_state",
        "per_second",
        "gate",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")
    if (
        payload.get("video_id") != expected_video_id
        and allow_video_id_repair
        and isinstance(payload.get("video_id"), str)
    ):
        payload = dict(payload)
        payload["video_id"] = expected_video_id
    if payload.get("video_id") != expected_video_id:
        errors.append(
            f"video_id mismatch: expected {expected_video_id!r}, got {payload.get('video_id')!r}"
        )

    _require_type(errors, payload, "prompt_critical_object", str)
    _require_type(errors, payload, "expected_motion_or_state", str)
    _require_type(errors, payload, "per_second", list)
    _require_type(errors, payload, "gate", dict)

    normalized_per_second: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("per_second") or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        for key in ("sec", "object_judgeable", "observed_motion_or_state"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        object_judgeable = str(entry.get("object_judgeable") or "").strip()
        if object_judgeable not in JUDGEABILITY_VALUES:
            errors.append(
                f"per_second[{index}].object_judgeable invalid: {entry.get('object_judgeable')!r}"
            )
        note = entry.get("observed_motion_or_state")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].observed_motion_or_state must be string")
            note = "" if note is None else str(note)
        normalized_per_second.append(
            {
                "sec": sec,
                "object_judgeable": object_judgeable,
                "observed_motion_or_state": str(note).strip()[:160],
            }
        )

    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    d5_app = _bool_or_error(errors, gate.get("d5_applicable"), "gate.d5_applicable")
    d6_app = _bool_or_error(errors, gate.get("d6_applicable"), "gate.d6_applicable")
    na_reason = _string_or_none(gate.get("na_reason"))
    allowed_na_reasons = {
        "no_initial_judgeable",
        "no_oov",
        "no_return",
        "unjudgeable_return",
        "parse_error",
        "unclear",
    }
    if (not d5_app or not d6_app) and na_reason not in allowed_na_reasons:
        errors.append(f"gate.na_reason invalid: {gate.get('na_reason')!r}")

    if errors:
        raise ValueError("; ".join(errors))

    parsed = {
        "video_id": payload.get("video_id"),
        "prompt_critical_object": str(payload.get("prompt_critical_object") or "").strip(),
        "expected_motion_or_state": str(payload.get("expected_motion_or_state") or "").strip()[:240],
        "per_second": sorted(normalized_per_second, key=lambda row: row["sec"]),
        "before_hidden_evidence": _string_or_none(payload.get("before_hidden_evidence")),
        "after_return_evidence": _string_or_none(payload.get("after_return_evidence")),
        "gate": {
            "d5_applicable": d5_app,
            "d6_applicable": d6_app,
            "na_reason": None if d5_app and d6_app else na_reason,
        },
        "schema_version": OBJECT_SCHEMA_VERSION,
    }
    return add_object_evidence_fields(parsed)


def validate_guarded_teacher_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
    clean_schema: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "video_id",
        "prompt_critical_evidence",
        "expected_motion_or_state",
        "static_anchors",
        "before_hidden_evidence",
        "after_return_evidence",
        "per_second",
        "gate",
    ]
    if clean_schema:
        required_keys = [key for key in required_keys if key != "video_id"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")
    if clean_schema:
        payload = dict(payload)
        payload["video_id"] = expected_video_id
    else:
        if (
            payload.get("video_id") != expected_video_id
            and allow_video_id_repair
            and isinstance(payload.get("video_id"), str)
        ):
            payload = dict(payload)
            payload["video_id"] = expected_video_id
        if payload.get("video_id") != expected_video_id:
            errors.append(
                f"video_id mismatch: expected {expected_video_id!r}, got {payload.get('video_id')!r}"
            )

    _require_type(errors, payload, "prompt_critical_evidence", str)
    _require_type(errors, payload, "expected_motion_or_state", str)
    _require_type(errors, payload, "static_anchors", list)
    _require_type(errors, payload, "per_second", list)
    _require_type(errors, payload, "gate", dict)

    anchors: list[str] = []
    if isinstance(payload.get("static_anchors"), list):
        for index, value in enumerate(payload["static_anchors"]):
            if not isinstance(value, str):
                errors.append(f"static_anchors[{index}] must be string")
                continue
            stripped = value.strip()
            if stripped:
                anchors.append(stripped[:120])

    normalized_per_second: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("per_second") or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        for key in ("sec", "judgeable", "note"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        judgeable = str(entry.get("judgeable") or "").strip()
        if judgeable not in JUDGEABILITY_VALUES:
            errors.append(f"per_second[{index}].judgeable invalid: {entry.get('judgeable')!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_per_second.append(
            {
                "sec": sec,
                "judgeable": judgeable,
                "note": str(note).strip()[:180],
            }
        )

    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    d5_app = _bool_or_error(errors, gate.get("d5_applicable"), "gate.d5_applicable")
    d6_app = _bool_or_error(errors, gate.get("d6_applicable"), "gate.d6_applicable")
    na_reason = _string_or_none(gate.get("na_reason"))
    allowed_na_reasons = {
        "no_usable_returned_evidence",
        "no_oov",
        "no_return",
        "no_initial_judgeable",
        "unidentifiable",
        "unjudgeable_return",
        "completely_unjudgeable",
        "parse_error",
        "unclear",
    }
    if (not d5_app or not d6_app) and na_reason not in allowed_na_reasons:
        errors.append(f"gate.na_reason invalid: {gate.get('na_reason')!r}")

    audit_payload = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
    audit = {
        "wrong_but_judgeable": _optional_bool(audit_payload.get("wrong_but_judgeable")),
        "anchor_available": _optional_bool(audit_payload.get("anchor_available")),
        "after_return_available": _optional_bool(audit_payload.get("after_return_available")),
    }

    if errors:
        raise ValueError("; ".join(errors))

    parsed = {
        "video_id": payload.get("video_id"),
        "prompt_critical_evidence": str(payload.get("prompt_critical_evidence") or "").strip(),
        "expected_motion_or_state": str(payload.get("expected_motion_or_state") or "").strip()[:240],
        "static_anchors": anchors,
        "before_hidden_evidence": _string_or_none(payload.get("before_hidden_evidence")),
        "after_return_evidence": _string_or_none(payload.get("after_return_evidence")),
        "per_second": sorted(normalized_per_second, key=lambda row: row["sec"]),
        "gate": {
            "d5_applicable": d5_app,
            "d6_applicable": d6_app,
            "na_reason": None if d5_app and d6_app else na_reason,
        },
        "audit": audit,
        "schema_version": GUARDED_CLEAN_SCHEMA_VERSION if clean_schema else GUARDED_SCHEMA_VERSION,
    }
    return add_guarded_teacher_gate_fields(parsed)


def validate_shared_direct3q_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "prompt_critical_evidence",
        "gap_observed",
        "after_return_clear_enough",
        "too_unclear_after_return",
        "before_gap_evidence",
        "after_return_evidence",
        "oov_applicable",
        "na_reason",
        "short_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, "prompt_critical_evidence", str)
    _require_type(errors, payload, "short_reason", str)
    gap_observed = _direct3q_value_or_error(errors, payload.get("gap_observed"), "gap_observed")
    after_return_clear_enough = _direct3q_value_or_error(
        errors, payload.get("after_return_clear_enough"), "after_return_clear_enough"
    )
    too_unclear_after_return = _direct3q_value_or_error(
        errors, payload.get("too_unclear_after_return"), "too_unclear_after_return"
    )
    model_oov_app = _bool_or_error(errors, payload.get("oov_applicable"), "oov_applicable")
    model_na_reason = _string_or_none(payload.get("na_reason"))
    allowed_na_reasons = {
        "no_initial_clear_evidence",
        "no_visibility_gap",
        "no_later_return",
        "return_too_unclear",
        "parse_error",
        "unclear",
    }
    if model_na_reason is not None and model_na_reason not in allowed_na_reasons:
        errors.append(f"na_reason invalid: {payload.get('na_reason')!r}")

    before_gap_evidence = _string_or_none(payload.get("before_gap_evidence"))
    after_return_evidence = _string_or_none(payload.get("after_return_evidence"))
    derived_gate = _derive_direct3q_oov_gate(
        gap_observed=gap_observed,
        after_return_clear_enough=after_return_clear_enough,
        too_unclear_after_return=too_unclear_after_return,
        before_gap_evidence=before_gap_evidence,
        after_return_evidence=after_return_evidence,
    )

    if errors:
        raise ValueError("; ".join(errors))

    derived_app = bool(derived_gate["applicable"])
    derived_reason = _string_or_none(derived_gate["na_reason"])
    parsed = {
        "video_id": expected_video_id,
        "prompt_critical_evidence": str(payload.get("prompt_critical_evidence") or "").strip(),
        "gap_observed": gap_observed,
        "after_return_clear_enough": after_return_clear_enough,
        "too_unclear_after_return": too_unclear_after_return,
        "before_gap_evidence": before_gap_evidence,
        "after_return_evidence": after_return_evidence,
        "oov_applicable": derived_app,
        "na_reason": None if derived_app else derived_reason or "unclear",
        "short_reason": str(payload.get("short_reason") or "").strip()[:240],
        "model_oov_applicable": model_oov_app,
        "model_na_reason": model_na_reason,
        "gate_consistent_with_truth_table": bool(
            model_oov_app == derived_app
            and ((derived_app and model_na_reason is None) or (not derived_app and model_na_reason == derived_reason))
        ),
        "gate": {
            "oov_applicable": derived_app,
            "na_reason": None if derived_app else derived_reason or "unclear",
        },
        "schema_version": SHARED_DIRECT3Q_CLEAN_SCHEMA_VERSION,
    }
    return add_direct3q_gate_fields(parsed)


def _brief_reason_mentions_after_gap_evidence(reason: str) -> bool:
    normalized = reason.strip().lower()
    if not normalized:
        return False
    evidence_terms = {
        "after",
        "later",
        "visible",
        "again",
        "absent",
        "absence",
        "vanish",
        "vanished",
        "failure",
        "failed",
        "reset",
        "wrong",
        "region",
        "area",
        "gap",
        "reappears",
    }
    return any(term in normalized for term in evidence_terms)


def add_visible_bool_gate_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    shared_app = bool(parsed["oov_applicable"])
    shared_reason = None if shared_app else str(parsed.get("na_reason") or "uncertain")
    parsed.update(
        {
            "main_subject": "",
            "segments": [],
            "subject_judgeable": shared_app,
            "target_judgeable": shared_app,
            "action_state_judgeable": shared_app,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "unclear"},
            "return_judgeable": shared_app,
            "notes_short": str(parsed.get("brief_reason") or "").strip()[:360],
            "shared_oov_applicable": shared_app,
            "shared_oov_na_reason": shared_reason,
            "evidence_shared_oov_applicable": shared_app,
            "evidence_shared_oov_na_reason": shared_reason,
            "evidence_subject_judgeable": shared_app,
            "evidence_target_judgeable": shared_app,
            "evidence_action_state_judgeable": shared_app,
            "evidence_oov_interval_status": "unclear",
            "evidence_return_judgeable": shared_app,
            "evidence_d5_applicable": shared_app,
            "evidence_d6_applicable": shared_app,
            "evidence_d5_na_reason": shared_reason,
            "evidence_d6_na_reason": shared_reason,
        }
    )
    return parsed


def validate_visible_bool_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "has_visible_after_gap_evidence_to_judge",
        "reason_code",
        "brief_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    raw_app = _bool_or_error(
        errors,
        payload.get("has_visible_after_gap_evidence_to_judge"),
        "has_visible_after_gap_evidence_to_judge",
    )
    _require_type(errors, payload, "reason_code", str)
    _require_type(errors, payload, "brief_reason", str)
    reason_code = str(payload.get("reason_code") or "").strip()
    if reason_code not in VISIBLE_BOOL_REASON_CODES:
        errors.append(f"reason_code invalid: {payload.get('reason_code')!r}")

    if errors:
        raise ValueError("; ".join(errors))

    brief_reason = str(payload.get("brief_reason") or "").strip()[:240]
    reason_boolean_conflict = bool(
        (raw_app and reason_code in VISIBLE_BOOL_NEGATIVE_REASONS)
        or ((not raw_app) and reason_code in VISIBLE_BOOL_POSITIVE_REASONS)
    )
    brief_reason_missing_after_gap_evidence = bool(
        raw_app and not _brief_reason_mentions_after_gap_evidence(brief_reason)
    )
    parsed = {
        "video_id": expected_video_id,
        "has_visible_after_gap_evidence_to_judge": raw_app,
        "reason_code": reason_code,
        "brief_reason": brief_reason,
        "oov_applicable": raw_app,
        "na_reason": None if raw_app else reason_code,
        "reason_boolean_conflict": reason_boolean_conflict,
        "brief_reason_missing_after_gap_evidence": brief_reason_missing_after_gap_evidence,
        "gate": {
            "oov_applicable": raw_app,
            "na_reason": None if raw_app else reason_code,
        },
        "schema_version": VISIBLE_BOOL_CLEAN_SCHEMA_VERSION,
    }
    return add_visible_bool_gate_fields(parsed)


def validate_oov_gap_bool_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "has_oov_gap_with_later_comparable_evidence",
        "reason_code",
        "brief_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    raw_app = _bool_or_error(
        errors,
        payload.get("has_oov_gap_with_later_comparable_evidence"),
        "has_oov_gap_with_later_comparable_evidence",
    )
    _require_type(errors, payload, "reason_code", str)
    _require_type(errors, payload, "brief_reason", str)
    reason_code = str(payload.get("reason_code") or "").strip()
    if reason_code not in OOV_GAP_BOOL_REASON_CODES:
        errors.append(f"reason_code invalid: {payload.get('reason_code')!r}")

    if errors:
        raise ValueError("; ".join(errors))

    brief_reason = str(payload.get("brief_reason") or "").strip()[:240]
    reason_boolean_conflict = bool(
        (raw_app and reason_code in OOV_GAP_BOOL_NEGATIVE_REASONS)
        or ((not raw_app) and reason_code in OOV_GAP_BOOL_POSITIVE_REASONS)
    )
    parsed = {
        "video_id": expected_video_id,
        "has_oov_gap_with_later_comparable_evidence": raw_app,
        "reason_code": reason_code,
        "brief_reason": brief_reason,
        "oov_applicable": raw_app,
        "na_reason": None if raw_app else reason_code,
        "reason_boolean_conflict": reason_boolean_conflict,
        "gate": {
            "oov_applicable": raw_app,
            "na_reason": None if raw_app else reason_code,
        },
        "schema_version": OOV_GAP_BOOL_CLEAN_SCHEMA_VERSION,
    }
    return add_visible_bool_gate_fields(parsed)


def validate_oov_gap_scan_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "prompt_critical_evidence",
        "visibility_scan",
        "final_oov_applicable",
        "brief_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, "prompt_critical_evidence", str)
    _require_type(errors, payload, "visibility_scan", list)
    model_oov_app = _bool_or_error(errors, payload.get("final_oov_applicable"), "final_oov_applicable")
    _require_type(errors, payload, "brief_reason", str)

    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("visibility_scan") or []):
        if not isinstance(entry, dict):
            errors.append(f"visibility_scan[{index}] must be object")
            continue
        for key in ("step", "phase", "judgeable", "note"):
            if key not in entry:
                errors.append(f"visibility_scan[{index}] missing {key}")
        try:
            step = int(entry.get("step"))
        except (TypeError, ValueError):
            errors.append(f"visibility_scan[{index}].step must be integer")
            step = index
        phase = entry.get("phase")
        if not isinstance(phase, str):
            errors.append(f"visibility_scan[{index}].phase must be string")
            phase = "" if phase is None else str(phase)
        judgeable = str(entry.get("judgeable") or "").strip()
        if judgeable not in OOV_GAP_SCAN_VALUES:
            errors.append(f"visibility_scan[{index}].judgeable invalid: {entry.get('judgeable')!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"visibility_scan[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_rows.append(
            {
                "step": step,
                "phase": str(phase).strip()[:40],
                "judgeable": judgeable,
                "note": str(note).strip()[:180],
            }
        )

    if not normalized_rows:
        errors.append("visibility_scan must not be empty")

    if errors:
        raise ValueError("; ".join(errors))

    normalized_rows = sorted(normalized_rows, key=lambda row: row["step"])
    derived_gate = _derive_visibility_scan_oov_gate(normalized_rows)
    derived_app = bool(derived_gate["applicable"])
    derived_reason = _string_or_none(derived_gate["na_reason"])
    shared_reason = None if derived_app else derived_reason or "unclear"
    return_judgeable = bool(derived_gate["return_judgeable"])
    oov_interval = derived_gate["oov_interval"]
    segments = [
        {
            "start_sec": float(row.get("step", index)),
            "end_sec": float(row.get("step", index)) + 1.0,
            "visible_subject": row.get("judgeable"),
            "prompt_critical_judgeable": row.get("judgeable"),
            "note": row.get("note"),
        }
        for index, row in enumerate(normalized_rows)
    ]

    parsed = {
        "video_id": expected_video_id,
        "prompt_critical_evidence": str(payload.get("prompt_critical_evidence") or "").strip()[:160],
        "visibility_scan": normalized_rows,
        "final_oov_applicable": model_oov_app,
        "model_oov_applicable": model_oov_app,
        "brief_reason": str(payload.get("brief_reason") or "").strip()[:240],
        "oov_applicable": derived_app,
        "na_reason": shared_reason,
        "gate_consistent_with_visibility_scan": bool(model_oov_app == derived_app),
        "gate": {
            "oov_applicable": derived_app,
            "na_reason": shared_reason,
        },
        "schema_version": OOV_GAP_SCAN_CLEAN_SCHEMA_VERSION,
        "main_subject": str(payload.get("prompt_critical_evidence") or "").strip()[:160],
        "segments": segments,
        "subject_judgeable": return_judgeable,
        "target_judgeable": return_judgeable,
        "action_state_judgeable": return_judgeable,
        "oov_interval": oov_interval,
        "return_judgeable": return_judgeable,
        "notes_short": "; ".join(row["note"] for row in normalized_rows[:3] if row["note"]),
        "shared_oov_applicable": derived_app,
        "shared_oov_na_reason": shared_reason,
        "evidence_shared_oov_applicable": derived_app,
        "evidence_shared_oov_na_reason": shared_reason,
        "evidence_subject_judgeable": return_judgeable,
        "evidence_target_judgeable": return_judgeable,
        "evidence_action_state_judgeable": return_judgeable,
        "evidence_oov_interval_status": oov_interval["status"],
        "evidence_return_judgeable": return_judgeable,
        "evidence_d5_applicable": derived_app,
        "evidence_d6_applicable": derived_app,
        "evidence_d5_na_reason": shared_reason,
        "evidence_d6_na_reason": shared_reason,
    }
    return parsed


def _derive_per_second_audit_pattern(normalized_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_reference_index: int | None = None
    for index, row in enumerate(normalized_rows):
        if str(row.get("status") or "") == "visible":
            first_reference_index = index
            break

    if first_reference_index is None:
        return {
            "applicable": False,
            "na_reason": "no_start",
            "first_reference_index": None,
            "first_unjudgeable_index": None,
            "later_clear_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    first_unjudgeable_index: int | None = None
    for index in range(first_reference_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("status") or "") == "not_visible":
            first_unjudgeable_index = index
            break

    if first_unjudgeable_index is None:
        return {
            "applicable": False,
            "na_reason": "no_gap",
            "first_reference_index": first_reference_index,
            "first_unjudgeable_index": None,
            "later_clear_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    start_sec = float(normalized_rows[first_unjudgeable_index].get("sec", first_unjudgeable_index))
    end_sec = start_sec + 1.0
    for row in normalized_rows[first_unjudgeable_index + 1 :]:
        if str(row.get("status") or "") == "not_visible":
            end_sec = float(row.get("sec", end_sec)) + 1.0
        else:
            break
    oov_interval = {"start_sec": start_sec, "end_sec": end_sec, "status": "present"}

    later_clear_index: int | None = None
    for index in range(first_unjudgeable_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("status") or "") == "visible":
            later_clear_index = index
            break

    if later_clear_index is None:
        return {
            "applicable": False,
            "na_reason": "no_later_visible",
            "first_reference_index": first_reference_index,
            "first_unjudgeable_index": first_unjudgeable_index,
            "later_clear_index": None,
            "oov_interval": oov_interval,
        }

    return {
        "applicable": True,
        "na_reason": None,
        "first_reference_index": first_reference_index,
        "first_unjudgeable_index": first_unjudgeable_index,
        "later_clear_index": later_clear_index,
        "oov_interval": oov_interval,
    }


def _derive_per_second_strict_collapse_pattern(normalized_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_reference_index: int | None = None
    for index, row in enumerate(normalized_rows):
        if str(row.get("status") or "") == "visible":
            first_reference_index = index
            break

    if first_reference_index is None:
        return {
            "applicable": False,
            "na_reason": "no_start",
            "positive_reason": None,
            "first_reference_index": None,
            "first_problem_index": None,
            "later_visible_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    first_problem_index: int | None = None
    first_problem_status: str | None = None
    for index in range(first_reference_index + 1, len(normalized_rows)):
        status = str(normalized_rows[index].get("status") or "")
        if status in {"not_visible", "collapsed", "broken_scene"}:
            first_problem_index = index
            first_problem_status = status
            break

    if first_problem_index is None:
        return {
            "applicable": False,
            "na_reason": "no_gap",
            "positive_reason": None,
            "first_reference_index": first_reference_index,
            "first_problem_index": None,
            "later_visible_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    start_sec = float(normalized_rows[first_problem_index].get("sec", first_problem_index))
    end_sec = start_sec + 1.0
    for row in normalized_rows[first_problem_index + 1 :]:
        if str(row.get("status") or "") == first_problem_status:
            end_sec = float(row.get("sec", end_sec)) + 1.0
        else:
            break
    oov_interval = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "status": "present",
        "problem_status": first_problem_status,
    }

    if first_problem_status in {"collapsed", "broken_scene"}:
        return {
            "applicable": True,
            "na_reason": None,
            "positive_reason": (
                "broken_scene_after_visible"
                if first_problem_status == "broken_scene"
                else "collapse_after_visible"
            ),
            "first_reference_index": first_reference_index,
            "first_problem_index": first_problem_index,
            "later_visible_index": None,
            "oov_interval": oov_interval,
        }

    later_visible_index: int | None = None
    for index in range(first_problem_index + 1, len(normalized_rows)):
        if str(normalized_rows[index].get("status") or "") == "visible":
            later_visible_index = index
            break

    if later_visible_index is None:
        return {
            "applicable": False,
            "na_reason": "no_later_visible",
            "positive_reason": None,
            "first_reference_index": first_reference_index,
            "first_problem_index": first_problem_index,
            "later_visible_index": None,
            "oov_interval": oov_interval,
        }

    return {
        "applicable": True,
        "na_reason": None,
        "positive_reason": "visible_after_gap",
        "first_reference_index": first_reference_index,
        "first_problem_index": first_problem_index,
        "later_visible_index": later_visible_index,
        "oov_interval": oov_interval,
    }


def _normalize_subject_result_rows(
    raw_rows: Any,
    *,
    value_key: str,
    allowed_values: set[str],
    errors: list[str],
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        errors.append(f"{field_name} must be list")
        return []

    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            errors.append(f"{field_name}[{index}] must be object")
            continue
        for key in ("sec", value_key, "note"):
            if key not in entry:
                errors.append(f"{field_name}[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"{field_name}[{index}].sec must be integer")
            sec = index
        value = str(entry.get(value_key) or "").strip()
        if value not in allowed_values:
            errors.append(f"{field_name}[{index}].{value_key} invalid: {entry.get(value_key)!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"{field_name}[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_rows.append(
            {
                "sec": sec,
                value_key: value,
                "note": str(note).strip()[:180],
            }
        )
    return sorted(normalized_rows, key=lambda row: row["sec"])


def _contiguous_interval_from_index(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    value_key: str,
    matching_values: set[str],
) -> dict[str, Any]:
    start_sec = float(rows[start_index].get("sec", start_index))
    end_sec = start_sec + 1.0
    for row in rows[start_index + 1 :]:
        if str(row.get(value_key) or "") in matching_values:
            end_sec = float(row.get("sec", end_sec)) + 1.0
            continue
        break
    return {"start_sec": start_sec, "end_sec": end_sec, "status": "present"}


def _derive_subject_result_integrity_pattern(
    *,
    subject_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_subject_index: int | None = None
    for index, row in enumerate(subject_rows):
        if str(row.get("judgeable") or "") == "yes":
            first_subject_index = index
            break

    if first_subject_index is None:
        return {
            "applicable": False,
            "na_reason": "no_start",
            "positive_reason": None,
            "first_subject_index": None,
            "first_gap_index": None,
            "later_subject_index": None,
            "later_result_index": None,
            "first_broken_scene_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    first_subject_sec = int(subject_rows[first_subject_index].get("sec", first_subject_index))
    first_broken_scene_index: int | None = None
    for index, row in enumerate(scene_rows):
        if int(row.get("sec", index)) <= first_subject_sec:
            continue
        if str(row.get("status") or "") == "broken":
            first_broken_scene_index = index
            break

    if first_broken_scene_index is not None:
        return {
            "applicable": True,
            "na_reason": None,
            "positive_reason": "scene_break_after_subject_visible",
            "first_subject_index": first_subject_index,
            "first_gap_index": None,
            "later_subject_index": None,
            "later_result_index": None,
            "first_broken_scene_index": first_broken_scene_index,
            "oov_interval": _contiguous_interval_from_index(
                scene_rows,
                start_index=first_broken_scene_index,
                value_key="status",
                matching_values={"broken"},
            ),
        }

    first_gap_index: int | None = None
    for index in range(first_subject_index + 1, len(subject_rows)):
        if str(subject_rows[index].get("judgeable") or "") == "no":
            first_gap_index = index
            break

    if first_gap_index is None:
        return {
            "applicable": False,
            "na_reason": "no_subject_gap",
            "positive_reason": None,
            "first_subject_index": first_subject_index,
            "first_gap_index": None,
            "later_subject_index": None,
            "later_result_index": None,
            "first_broken_scene_index": None,
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
        }

    first_gap_sec = int(subject_rows[first_gap_index].get("sec", first_gap_index))
    later_subject_index: int | None = None
    for index in range(first_gap_index + 1, len(subject_rows)):
        if str(subject_rows[index].get("judgeable") or "") == "yes":
            later_subject_index = index
            break

    if later_subject_index is not None:
        return {
            "applicable": True,
            "na_reason": None,
            "positive_reason": "subject_return_after_gap",
            "first_subject_index": first_subject_index,
            "first_gap_index": first_gap_index,
            "later_subject_index": later_subject_index,
            "later_result_index": None,
            "first_broken_scene_index": None,
            "oov_interval": _contiguous_interval_from_index(
                subject_rows,
                start_index=first_gap_index,
                value_key="judgeable",
                matching_values={"no"},
            ),
        }

    later_result_index: int | None = None
    for index, row in enumerate(result_rows):
        if int(row.get("sec", index)) <= first_gap_sec:
            continue
        if str(row.get("judgeable") or "") == "yes":
            later_result_index = index
            break

    if later_result_index is not None:
        return {
            "applicable": True,
            "na_reason": None,
            "positive_reason": "result_evidence_after_subject_gap",
            "first_subject_index": first_subject_index,
            "first_gap_index": first_gap_index,
            "later_subject_index": None,
            "later_result_index": later_result_index,
            "first_broken_scene_index": None,
            "oov_interval": _contiguous_interval_from_index(
                subject_rows,
                start_index=first_gap_index,
                value_key="judgeable",
                matching_values={"no"},
            ),
        }

    return {
        "applicable": False,
        "na_reason": "no_later_subject_or_result",
        "positive_reason": None,
        "first_subject_index": first_subject_index,
        "first_gap_index": first_gap_index,
        "later_subject_index": None,
        "later_result_index": None,
        "first_broken_scene_index": None,
        "oov_interval": {"start_sec": None, "end_sec": None, "status": "unclear"},
    }


def _per_second_audit_conflict_types(
    *,
    raw_app: bool,
    reason_code: str,
    normalized_rows: list[dict[str, Any]],
    derived_pattern: dict[str, Any],
) -> list[str]:
    conflicts: list[str] = []
    secs = [int(row["sec"]) for row in normalized_rows]
    unique_secs = sorted(set(secs))
    if len(unique_secs) != len(secs) or unique_secs != list(range(unique_secs[0], unique_secs[-1] + 1)):
        conflicts.append("missing_or_duplicate_seconds")

    if raw_app and derived_pattern.get("first_unjudgeable_index") is None:
        conflicts.append("raw_true_scan_no_unjudgeable")
    if raw_app and derived_pattern.get("first_unjudgeable_index") is not None and derived_pattern.get("later_clear_index") is None:
        conflicts.append("raw_true_scan_no_later_clear")
        later_rows = normalized_rows[int(derived_pattern["first_unjudgeable_index"]) + 1 :]
        if any(str(row.get("status") or "") == "unrelated" for row in later_rows):
            conflicts.append("raw_true_later_irrelevant")
    if (not raw_app) and bool(derived_pattern.get("applicable")):
        conflicts.append("raw_false_scan_has_clear_unjudgeable_clear")
    if raw_app and reason_code in OOV_GAP_PER_SECOND_AUDIT_NEGATIVE_REASONS:
        conflicts.append("raw_true_reason_negative")
    if (not raw_app) and reason_code in OOV_GAP_PER_SECOND_AUDIT_POSITIVE_REASONS:
        conflicts.append("raw_false_reason_positive")
    return conflicts


def validate_oov_gap_per_second_audit_bool_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    tracked_key = "look_for" if "look_for" in payload else "tracked_visual"
    final_key = (
        "can_judge_after_gap"
        if "can_judge_after_gap" in payload
        else "has_after_interruption_judgeable_evidence"
    )
    required_keys = [
        tracked_key,
        "per_second",
        final_key,
        "reason_code",
        "brief_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, tracked_key, str)
    _require_type(errors, payload, "per_second", list)
    raw_app = _bool_or_error(
        errors,
        payload.get(final_key),
        final_key,
    )
    _require_type(errors, payload, "reason_code", str)
    _require_type(errors, payload, "brief_reason", str)
    reason_code = str(payload.get("reason_code") or "").strip()
    if reason_code not in OOV_GAP_PER_SECOND_AUDIT_REASON_CODES:
        errors.append(f"reason_code invalid: {payload.get('reason_code')!r}")

    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("per_second") or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        status_key = "status" if "status" in entry else "evidence_status"
        for key in ("sec", status_key, "note"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        status = str(entry.get(status_key) or "").strip()
        status_aliases = {
            "clear_relevant_evidence": "visible",
            "clear_relevant_region_or_absence": "visible",
            "unjudgeable": "not_visible",
            "irrelevant_visible": "unrelated",
        }
        status = status_aliases.get(status, status)
        if status not in OOV_GAP_PER_SECOND_AUDIT_STATUSES:
            errors.append(f"per_second[{index}].status invalid: {entry.get(status_key)!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_rows.append(
            {
                "sec": sec,
                "status": status,
                "note": str(note).strip()[:180],
            }
        )

    if not normalized_rows:
        errors.append("per_second must not be empty")
    if errors:
        raise ValueError("; ".join(errors))

    normalized_rows = sorted(normalized_rows, key=lambda row: row["sec"])
    derived_pattern = _derive_per_second_audit_pattern(normalized_rows)
    conflicts = _per_second_audit_conflict_types(
        raw_app=raw_app,
        reason_code=reason_code,
        normalized_rows=normalized_rows,
        derived_pattern=derived_pattern,
    )
    shared_reason = None if raw_app else reason_code
    oov_interval = derived_pattern["oov_interval"] if raw_app else {"start_sec": None, "end_sec": None, "status": "unclear"}
    segments = [
        {
            "start_sec": float(row.get("sec", index)),
            "end_sec": float(row.get("sec", index)) + 1.0,
            "visible_subject": (
                "yes"
                if row["status"] == "visible"
                else "no"
                if row["status"] == "not_visible"
                else "unclear"
            ),
            "prompt_critical_judgeable": (
                "yes"
                if row["status"] == "visible"
                else "no"
                if row["status"] == "not_visible"
                else "unclear"
            ),
            "note": row["note"],
        }
        for index, row in enumerate(normalized_rows)
    ]

    parsed = {
        "video_id": expected_video_id,
        "look_for": str(payload.get(tracked_key) or "").strip()[:160],
        "tracked_visual": str(payload.get(tracked_key) or "").strip()[:160],
        "main_subject": str(payload.get(tracked_key) or "").strip()[:160],
        "per_second": normalized_rows,
        "can_judge_after_gap": raw_app,
        "has_after_interruption_judgeable_evidence": raw_app,
        "final_oov_applicable": raw_app,
        "model_oov_applicable": raw_app,
        "reason_code": reason_code,
        "brief_reason": str(payload.get("brief_reason") or "").strip()[:240],
        "scan_derived_pattern": bool(derived_pattern["applicable"]),
        "scan_derived_na_reason": _string_or_none(derived_pattern["na_reason"]),
        "gate_consistent_with_scan": bool(raw_app == bool(derived_pattern["applicable"])),
        "conflict_types": conflicts,
        "reason_boolean_conflict": bool(
            (raw_app and reason_code in OOV_GAP_PER_SECOND_AUDIT_NEGATIVE_REASONS)
            or ((not raw_app) and reason_code in OOV_GAP_PER_SECOND_AUDIT_POSITIVE_REASONS)
        ),
        "oov_applicable": raw_app,
        "na_reason": shared_reason,
        "gate": {
            "oov_applicable": raw_app,
            "na_reason": shared_reason,
        },
        "schema_version": OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_SCHEMA_VERSION,
        "segments": segments,
        "subject_judgeable": raw_app,
        "target_judgeable": raw_app,
        "action_state_judgeable": raw_app,
        "oov_interval": oov_interval,
        "return_judgeable": raw_app,
        "notes_short": "; ".join(row["note"] for row in normalized_rows[:3] if row["note"]),
        "shared_oov_applicable": raw_app,
        "shared_oov_na_reason": shared_reason,
        "evidence_shared_oov_applicable": raw_app,
        "evidence_shared_oov_na_reason": shared_reason,
        "evidence_subject_judgeable": raw_app,
        "evidence_target_judgeable": raw_app,
        "evidence_action_state_judgeable": raw_app,
        "evidence_oov_interval_status": oov_interval["status"],
        "evidence_return_judgeable": raw_app,
        "evidence_d5_applicable": raw_app,
        "evidence_d6_applicable": raw_app,
        "evidence_d5_na_reason": shared_reason,
        "evidence_d6_na_reason": shared_reason,
    }
    return parsed


def validate_oov_gap_per_second_strict_collapse_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = ["look_for", "per_second", "brief_reason"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, "look_for", str)
    _require_type(errors, payload, "per_second", list)
    _require_type(errors, payload, "brief_reason", str)

    normalized_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("per_second") or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        for key in ("sec", "status", "note"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        status = str(entry.get("status") or "").strip()
        if status not in OOV_GAP_PER_SECOND_STRICT_COLLAPSE_STATUSES:
            errors.append(f"per_second[{index}].status invalid: {entry.get('status')!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized_rows.append(
            {
                "sec": sec,
                "status": status,
                "note": str(note).strip()[:180],
            }
        )

    if not normalized_rows:
        errors.append("per_second must not be empty")
    if errors:
        raise ValueError("; ".join(errors))

    normalized_rows = sorted(normalized_rows, key=lambda row: row["sec"])
    derived_pattern = _derive_per_second_strict_collapse_pattern(normalized_rows)
    derived_app = bool(derived_pattern["applicable"])
    shared_reason = None if derived_app else _string_or_none(derived_pattern["na_reason"])
    positive_reason = _string_or_none(derived_pattern.get("positive_reason"))
    oov_interval = (
        derived_pattern["oov_interval"]
        if derived_app
        else {"start_sec": None, "end_sec": None, "status": "unclear"}
    )
    segments = [
        {
            "start_sec": float(row.get("sec", index)),
            "end_sec": float(row.get("sec", index)) + 1.0,
            "visible_subject": (
                "yes"
                if row["status"] == "visible"
                else "no"
                if row["status"] in {"not_visible", "collapsed", "broken_scene"}
                else "unclear"
            ),
            "prompt_critical_judgeable": (
                "yes"
                if row["status"] == "visible"
                else "no"
                if row["status"] in {"not_visible", "collapsed", "broken_scene"}
                else "unclear"
            ),
            "note": row["note"],
        }
        for index, row in enumerate(normalized_rows)
    ]
    collapse_present = any(row["status"] in {"collapsed", "broken_scene"} for row in normalized_rows)
    broken_scene_present = any(row["status"] == "broken_scene" for row in normalized_rows)

    parsed = {
        "video_id": expected_video_id,
        "look_for": str(payload.get("look_for") or "").strip()[:160],
        "tracked_visual": str(payload.get("look_for") or "").strip()[:160],
        "main_subject": str(payload.get("look_for") or "").strip()[:160],
        "per_second": normalized_rows,
        "can_judge_after_gap": derived_app,
        "has_after_interruption_judgeable_evidence": derived_app,
        "final_oov_applicable": derived_app,
        "model_oov_applicable": derived_app,
        "reason_code": positive_reason or shared_reason or "unclear",
        "brief_reason": str(payload.get("brief_reason") or "").strip()[:240],
        "scan_derived_pattern": derived_app,
        "scan_derived_na_reason": shared_reason,
        "scan_derived_positive_reason": positive_reason,
        "collapse_present": collapse_present,
        "broken_scene_present": broken_scene_present,
        "gate_consistent_with_scan": True,
        "conflict_types": [],
        "reason_boolean_conflict": False,
        "oov_applicable": derived_app,
        "na_reason": shared_reason,
        "gate": {
            "oov_applicable": derived_app,
            "na_reason": shared_reason,
        },
        "schema_version": OOV_GAP_PER_SECOND_STRICT_COLLAPSE_SCHEMA_VERSION,
        "segments": segments,
        "subject_judgeable": derived_app,
        "target_judgeable": derived_app,
        "action_state_judgeable": derived_app,
        "oov_interval": oov_interval,
        "return_judgeable": derived_app,
        "notes_short": "; ".join(row["note"] for row in normalized_rows[:3] if row["note"]),
        "shared_oov_applicable": derived_app,
        "shared_oov_na_reason": shared_reason,
        "evidence_shared_oov_applicable": derived_app,
        "evidence_shared_oov_na_reason": shared_reason,
        "evidence_subject_judgeable": derived_app,
        "evidence_target_judgeable": derived_app,
        "evidence_action_state_judgeable": derived_app,
        "evidence_oov_interval_status": oov_interval["status"],
        "evidence_return_judgeable": derived_app,
        "evidence_d5_applicable": derived_app,
        "evidence_d6_applicable": derived_app,
        "evidence_d5_na_reason": shared_reason,
        "evidence_d6_na_reason": shared_reason,
    }
    return parsed


def validate_oov_subject_result_integrity_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "main_subject",
        "result_evidence",
        "subject_per_second",
        "result_evidence_per_second",
        "scene_integrity_per_second",
        "brief_reason",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, "main_subject", str)
    _require_type(errors, payload, "result_evidence", str)
    _require_type(errors, payload, "brief_reason", str)
    subject_rows = _normalize_subject_result_rows(
        payload.get("subject_per_second"),
        value_key="judgeable",
        allowed_values=OOV_SUBJECT_RESULT_JUDGEABLE_VALUES,
        errors=errors,
        field_name="subject_per_second",
    )
    result_rows = _normalize_subject_result_rows(
        payload.get("result_evidence_per_second"),
        value_key="judgeable",
        allowed_values=OOV_SUBJECT_RESULT_JUDGEABLE_VALUES,
        errors=errors,
        field_name="result_evidence_per_second",
    )
    scene_rows = _normalize_subject_result_rows(
        payload.get("scene_integrity_per_second"),
        value_key="status",
        allowed_values=OOV_SUBJECT_RESULT_SCENE_VALUES,
        errors=errors,
        field_name="scene_integrity_per_second",
    )
    if not subject_rows:
        errors.append("subject_per_second must not be empty")
    if not result_rows:
        errors.append("result_evidence_per_second must not be empty")
    if not scene_rows:
        errors.append("scene_integrity_per_second must not be empty")
    if errors:
        raise ValueError("; ".join(errors))

    derived_pattern = _derive_subject_result_integrity_pattern(
        subject_rows=subject_rows,
        result_rows=result_rows,
        scene_rows=scene_rows,
    )
    derived_app = bool(derived_pattern["applicable"])
    shared_reason = None if derived_app else str(derived_pattern.get("na_reason") or "unclear")
    positive_reason = _string_or_none(derived_pattern.get("positive_reason"))
    oov_interval = (
        derived_pattern["oov_interval"]
        if derived_app
        else {"start_sec": None, "end_sec": None, "status": "unclear"}
    )
    segments = [
        {
            "start_sec": float(row.get("sec", index)),
            "end_sec": float(row.get("sec", index)) + 1.0,
            "visible_subject": (
                "yes"
                if row["judgeable"] == "yes"
                else "no"
                if row["judgeable"] == "no"
                else "unclear"
            ),
            "prompt_critical_judgeable": (
                "yes"
                if row["judgeable"] == "yes"
                else "no"
                if row["judgeable"] == "no"
                else "unclear"
            ),
            "note": row["note"],
        }
        for index, row in enumerate(subject_rows)
    ]
    scene_break_present = any(row["status"] == "broken" for row in scene_rows)
    return_judgeable = bool(
        positive_reason in {"subject_return_after_gap", "result_evidence_after_subject_gap"}
    )

    parsed = {
        "video_id": expected_video_id,
        "main_subject": str(payload.get("main_subject") or "").strip()[:160],
        "result_evidence": str(payload.get("result_evidence") or "").strip()[:180],
        "subject_per_second": subject_rows,
        "result_evidence_per_second": result_rows,
        "scene_integrity_per_second": scene_rows,
        "per_second": subject_rows,
        "brief_reason": str(payload.get("brief_reason") or "").strip()[:240],
        "can_judge_after_gap": derived_app,
        "has_after_interruption_judgeable_evidence": derived_app,
        "final_oov_applicable": derived_app,
        "model_oov_applicable": derived_app,
        "reason_code": positive_reason or shared_reason or "unclear",
        "scan_derived_pattern": derived_app,
        "scan_derived_na_reason": shared_reason,
        "scan_derived_positive_reason": positive_reason,
        "scene_break_present": scene_break_present,
        "gate_consistent_with_scan": True,
        "conflict_types": [],
        "reason_boolean_conflict": False,
        "oov_applicable": derived_app,
        "na_reason": shared_reason,
        "gate": {
            "oov_applicable": derived_app,
            "na_reason": shared_reason,
        },
        "schema_version": OOV_SUBJECT_RESULT_INTEGRITY_SCHEMA_VERSION,
        "segments": segments,
        "subject_judgeable": derived_app,
        "target_judgeable": derived_app,
        "action_state_judgeable": derived_app,
        "oov_interval": oov_interval,
        "return_judgeable": return_judgeable,
        "notes_short": "; ".join(
            note
            for note in (
                *(row["note"] for row in subject_rows[:2]),
                *(row["note"] for row in result_rows[:2]),
                *(row["note"] for row in scene_rows[:1]),
            )
            if note
        )[:360],
        "shared_oov_applicable": derived_app,
        "shared_oov_na_reason": shared_reason,
        "evidence_shared_oov_applicable": derived_app,
        "evidence_shared_oov_na_reason": shared_reason,
        "evidence_subject_judgeable": derived_app,
        "evidence_target_judgeable": derived_app,
        "evidence_action_state_judgeable": derived_app,
        "evidence_oov_interval_status": oov_interval["status"],
        "evidence_return_judgeable": return_judgeable,
        "evidence_d5_applicable": derived_app,
        "evidence_d6_applicable": derived_app,
        "evidence_d5_na_reason": shared_reason,
        "evidence_d6_na_reason": shared_reason,
    }
    return parsed


def _triplet_present_or_error(errors: list[str], value: Any, field: str) -> str:
    present = str(value or "").strip()
    if present not in OOV_GAP_TRIPLET_VALUES:
        errors.append(f"{field}.present invalid: {value!r}")
        return "unclear"
    return present


def _normalize_frame_visibility(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_items = payload.get("frame_visibility")
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw in raw_items[:32]:
        if not isinstance(raw, dict):
            continue
        visible = str(raw.get("visible") or "").strip()
        if visible not in OOV_GAP_TRIPLET_VALUES:
            visible = "unclear"
        normalized.append(
            {
                "label": str(raw.get("label") or "").strip()[:40],
                "visible": visible,
                "note": str(raw.get("note") or "").strip()[:160],
            }
        )
    return normalized


def _derive_triplet_oov_gate(
    *,
    before_present: str,
    gap_present: str,
    later_present: str,
) -> dict[str, Any]:
    if before_present != "yes":
        return {
            "applicable": False,
            "na_reason": "no_initial_judgeable" if before_present == "no" else "unclear",
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
            "return_judgeable": False,
        }
    if gap_present != "yes":
        return {
            "applicable": False,
            "na_reason": "no_oov" if gap_present == "no" else "unclear",
            "oov_interval": {"start_sec": None, "end_sec": None, "status": "absent"},
            "return_judgeable": False,
        }
    oov_interval = {"start_sec": 1.0, "end_sec": 2.0, "status": "present"}
    if later_present != "yes":
        return {
            "applicable": False,
            "na_reason": "no_later_comparable_evidence" if later_present == "no" else "unclear",
            "oov_interval": oov_interval,
            "return_judgeable": False,
        }
    return {
        "applicable": True,
        "na_reason": None,
        "oov_interval": oov_interval,
        "return_judgeable": True,
    }


def validate_oov_gap_triplet_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    schema_version: str = OOV_GAP_TRIPLET_CLEAN_SCHEMA_VERSION,
) -> dict[str, Any]:
    errors: list[str] = []
    critical_key = "critical_evidence" if "critical_evidence" in payload else "prompt_critical_evidence"
    before_key = "early_reference" if "early_reference" in payload else "before_reference"
    gap_key = "middle_cannot_follow" if "middle_cannot_follow" in payload else "middle_unjudgeable_gap"
    later_key = "later_comparison" if "later_comparison" in payload else "later_comparable_evidence"
    final_key = "final_applicable" if "final_applicable" in payload else "final_oov_applicable"
    required_keys = [critical_key, before_key, gap_key, later_key, final_key, "brief_reason"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    _require_type(errors, payload, critical_key, str)
    _require_type(errors, payload, before_key, dict)
    _require_type(errors, payload, gap_key, dict)
    _require_type(errors, payload, later_key, dict)
    model_oov_app = _bool_or_error(errors, payload.get(final_key), final_key)
    _require_type(errors, payload, "brief_reason", str)

    before_payload = payload.get(before_key) if isinstance(payload.get(before_key), dict) else {}
    gap_payload = payload.get(gap_key) if isinstance(payload.get(gap_key), dict) else {}
    later_payload = payload.get(later_key) if isinstance(payload.get(later_key), dict) else {}
    before_present = _triplet_present_or_error(errors, before_payload.get("present"), "before_reference")
    gap_present = _triplet_present_or_error(errors, gap_payload.get("present"), "middle_unjudgeable_gap")
    later_present = _triplet_present_or_error(errors, later_payload.get("present"), "later_comparable_evidence")
    later_type = str(later_payload.get("type") or "unclear").strip()
    if later_type not in OOV_GAP_TRIPLET_LATER_TYPES:
        errors.append(f"later_comparable_evidence.type invalid: {later_payload.get('type')!r}")
        later_type = "unclear"

    for field_name, field_payload in (
        ("before_reference", before_payload),
        ("middle_unjudgeable_gap", gap_payload),
        ("later_comparable_evidence", later_payload),
    ):
        if not isinstance(field_payload.get("note"), str):
            errors.append(f"{field_name}.note must be string")

    if errors:
        raise ValueError("; ".join(errors))

    derived_gate = _derive_triplet_oov_gate(
        before_present=before_present,
        gap_present=gap_present,
        later_present=later_present,
    )
    derived_app = bool(derived_gate["applicable"])
    shared_reason = None if derived_app else str(derived_gate.get("na_reason") or "unclear")
    oov_interval = derived_gate["oov_interval"]
    return_judgeable = bool(derived_gate["return_judgeable"])
    before = {
        "present": before_present,
        "note": str(before_payload.get("note") or "").strip()[:180],
    }
    gap = {
        "present": gap_present,
        "note": str(gap_payload.get("note") or "").strip()[:180],
    }
    later = {
        "present": later_present,
        "type": later_type,
        "note": str(later_payload.get("note") or "").strip()[:180],
    }
    frame_visibility = _normalize_frame_visibility(payload)
    segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "visible_subject": before_present,
            "prompt_critical_judgeable": before_present,
            "note": before["note"],
        },
        {
            "start_sec": 1.0,
            "end_sec": 2.0,
            "visible_subject": "no" if gap_present == "yes" else "yes",
            "prompt_critical_judgeable": "no" if gap_present == "yes" else gap_present,
            "note": gap["note"],
        },
        {
            "start_sec": 2.0,
            "end_sec": 3.0,
            "visible_subject": later_present,
            "prompt_critical_judgeable": later_present,
            "note": later["note"],
        },
    ]
    parsed = {
        "video_id": expected_video_id,
        "prompt_critical_evidence": str(payload.get(critical_key) or "").strip()[:160],
        "before_reference": before,
        "middle_unjudgeable_gap": gap,
        "later_comparable_evidence": later,
        "final_applicable": model_oov_app,
        "final_oov_applicable": model_oov_app,
        "model_oov_applicable": model_oov_app,
        "brief_reason": str(payload.get("brief_reason") or "").strip()[:240],
        "oov_applicable": derived_app,
        "na_reason": shared_reason,
        "gate_consistent_with_triplet": bool(model_oov_app == derived_app),
        "gate": {
            "oov_applicable": derived_app,
            "na_reason": shared_reason,
        },
        "schema_version": schema_version,
        "main_subject": str(payload.get(critical_key) or "").strip()[:160],
        "frame_visibility": frame_visibility,
        "segments": segments,
        "subject_judgeable": return_judgeable,
        "target_judgeable": return_judgeable,
        "action_state_judgeable": return_judgeable,
        "oov_interval": oov_interval,
        "return_judgeable": return_judgeable,
        "notes_short": "; ".join(note for note in (before["note"], gap["note"], later["note"]) if note),
        "shared_oov_applicable": derived_app,
        "shared_oov_na_reason": shared_reason,
        "evidence_shared_oov_applicable": derived_app,
        "evidence_shared_oov_na_reason": shared_reason,
        "evidence_subject_judgeable": return_judgeable,
        "evidence_target_judgeable": return_judgeable,
        "evidence_action_state_judgeable": return_judgeable,
        "evidence_oov_interval_status": oov_interval["status"],
        "evidence_return_judgeable": return_judgeable,
        "evidence_d5_applicable": derived_app,
        "evidence_d6_applicable": derived_app,
        "evidence_d5_na_reason": shared_reason,
        "evidence_d6_na_reason": shared_reason,
    }
    return parsed


def parse_judgeability_response(
    raw_text: str,
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
    prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    if prompt_schema == PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY:
        return validate_oov_subject_result_integrity_payload(
            payload,
            expected_video_id=expected_video_id,
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE:
        return validate_oov_gap_per_second_strict_collapse_payload(
            payload,
            expected_video_id=expected_video_id,
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN:
        return validate_oov_gap_per_second_audit_bool_payload(
            payload,
            expected_video_id=expected_video_id,
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN:
        return validate_oov_gap_triplet_gate_payload(
            payload,
            expected_video_id=expected_video_id,
            schema_version=OOV_GAP_TRIPLET_SHEET_CLEAN_SCHEMA_VERSION,
        )
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN:
        return validate_oov_gap_triplet_gate_payload(payload, expected_video_id=expected_video_id)
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN:
        return validate_oov_gap_scan_gate_payload(payload, expected_video_id=expected_video_id)
    if prompt_schema == PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN:
        return validate_oov_gap_bool_gate_payload(payload, expected_video_id=expected_video_id)
    if prompt_schema == PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN:
        return validate_visible_bool_gate_payload(payload, expected_video_id=expected_video_id)
    if prompt_schema == PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN:
        return validate_shared_direct3q_gate_payload(payload, expected_video_id=expected_video_id)
    if prompt_schema == PROMPT_SCHEMA_GUARDED_CLEAN:
        return validate_guarded_teacher_gate_payload(
            payload,
            expected_video_id=expected_video_id,
            allow_video_id_repair=allow_video_id_repair,
            clean_schema=True,
        )
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return validate_guarded_teacher_gate_payload(
            payload,
            expected_video_id=expected_video_id,
            allow_video_id_repair=allow_video_id_repair,
        )
    if prompt_schema == PROMPT_SCHEMA_OBJECT:
        return validate_object_judgeability_payload(
            payload,
            expected_video_id=expected_video_id,
            allow_video_id_repair=allow_video_id_repair,
        )
    if prompt_schema not in {PROMPT_SCHEMA_SUBJECT, PROMPT_SCHEMA_SUBJECT_CLEAN}:
        raise ValueError(f"unsupported prompt_schema: {prompt_schema}")
    return validate_judgeability_payload(
        payload,
        expected_video_id=expected_video_id,
        allow_video_id_repair=allow_video_id_repair,
        clean_schema=prompt_schema == PROMPT_SCHEMA_SUBJECT_CLEAN,
    )


def validate_visibility_payload(payload: dict[str, Any], *, expected_video_id: str) -> dict[str, Any]:
    """Compatibility wrapper for the former visibility-named parser."""
    return validate_judgeability_payload(payload, expected_video_id=expected_video_id)


def parse_visibility_response(raw_text: str, *, expected_video_id: str) -> dict[str, Any]:
    """Compatibility wrapper for the former visibility-named parser."""
    return parse_judgeability_response(raw_text, expected_video_id=expected_video_id)


def derive_frames_used_from_processor_inputs(inputs: Any, processor: Any) -> int:
    try:
        grid = inputs.get("video_grid_thw") if hasattr(inputs, "get") else inputs["video_grid_thw"]
    except Exception:
        return 0
    if grid is None:
        return 0
    try:
        temporal_grid = grid[0, 0]
    except Exception:
        try:
            temporal_grid = grid[0][0]
        except Exception:
            return 0
    try:
        temporal_grid_value = int(temporal_grid.item())
    except Exception:
        try:
            temporal_grid_value = int(temporal_grid)
        except Exception:
            return 0
    temporal_patch_size = getattr(getattr(processor, "video_processor", None), "temporal_patch_size", 1)
    try:
        patch_size = int(temporal_patch_size)
    except Exception:
        patch_size = 1
    return max(0, temporal_grid_value * max(1, patch_size))


def _version_or_unavailable(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _cuda_build_or_unavailable() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "unavailable"
    return str(getattr(torch.version, "cuda", None) or "unavailable")


def model_config_model_type(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        config = load_json(config_path)
    except Exception:
        return None
    value = config.get("model_type") if isinstance(config, dict) else None
    return str(value) if value is not None else None


class LocalQwen3VLVideoEvidenceRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        fps: str,
        dtype: str,
        attn_implementation: str,
        local_rank: int,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
    ) -> None:
        self.model_path = model_path
        self.fps = fps
        self.max_new_tokens = int(max_new_tokens)
        if prompt_schema not in SUPPORTED_PROMPT_SCHEMAS:
            raise ValueError(f"unsupported prompt_schema: {prompt_schema}")
        self.prompt_schema = prompt_schema
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        dtype_obj = {"bfloat16": torch.bfloat16}[dtype]
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        video_backend = os.environ["WORLD_STATE_VIDEO_BACKEND"].strip()
        if not video_backend:
            raise RuntimeError("WORLD_STATE_VIDEO_BACKEND is required")
        video_processor = getattr(self.processor, "video_processor", None)
        if video_backend and video_processor is not None:
            from transformers.video_utils import load_video  # type: ignore

            def fetch_videos_with_backend(video_url_or_urls, sample_indices_fn=None):
                if isinstance(video_url_or_urls, list):
                    return list(
                        zip(
                            *[
                                fetch_videos_with_backend(x, sample_indices_fn=sample_indices_fn)
                                for x in video_url_or_urls
                            ]
                        )
                    )
                return load_video(
                    video_url_or_urls,
                    backend=video_backend,
                    sample_indices_fn=sample_indices_fn,
                )

            video_processor.fetch_videos = fetch_videos_with_backend

        def load_with(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=dtype_obj,
                    device_map={"": local_rank},
                    attn_implementation=attn_implementation,
                )
            except TypeError:
                return model_cls.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=dtype_obj,
                    device_map={"": local_rank},
                    attn_implementation=attn_implementation,
                )

        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            self.model = load_with(Qwen3VLForConditionalGeneration)
        except Exception:
            self.model = load_with(AutoModelForImageTextToText)
        self.model.eval()

    def run_one(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        import torch  # type: ignore

        video_id = str(item["video_id"])
        request_id = str(item.get("request_id") or "")
        row_index = item.get("row_index")
        video_path = scoring_video_path(item)
        image_path = str(item.get("image_path") or item.get("sheet_path") or "")
        media_type = "image" if image_path else "video"
        media_path = image_path or video_path
        world_state_prompt = str(item.get("world_state_prompt") or item.get("prompt_text") or "")
        prompt = build_judgeability_prompt(
            world_state_prompt=world_state_prompt,
            video_id=video_id,
            prompt_schema=self.prompt_schema,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: media_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        processor_kwargs: dict[str, Any] = {}
        if media_type == "video" and self.fps != "full":
            processor_kwargs["videos_kwargs"] = {"fps": float(self.fps)}
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs or None,
        )
        frames_used = derive_frames_used_from_processor_inputs(inputs, self.processor)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_len:]
        raw_text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        raw_row = {
            "video_id": video_id,
            "request_id": request_id,
            "row_index": row_index,
            "path": media_path,
            "video_path": video_path,
            "image_path": image_path,
            "media_type": media_type,
            "world_state_prompt": world_state_prompt,
            "sampling_fps": self.fps,
            "frames_used": int(frames_used),
            "raw_response": raw_text,
            "schema_version": schema_version_for_prompt_schema(self.prompt_schema),
            "prompt_schema": self.prompt_schema,
        }
        try:
            parsed_row = parse_judgeability_response(
                raw_text,
                expected_video_id=video_id,
                allow_video_id_repair=True,
                prompt_schema=self.prompt_schema,
            )
            parsed_row.update(
                {
                    "request_id": request_id,
                    "row_index": row_index,
                    "path": media_path,
                    "video_path": video_path,
                    "image_path": image_path,
                    "media_type": media_type,
                    "world_state_prompt": world_state_prompt,
                    "sampling_fps": self.fps,
                    "frames_used": int(frames_used),
                }
            )
            raw_row["parse_status"] = "ok"
            return raw_row, parsed_row
        except Exception as exc:
            raw_row["parse_status"] = "error"
            raw_row["parse_error"] = f"{type(exc).__name__}: {exc}"
            return raw_row, None


def merge_sharded_outputs(
    *,
    output_dir: Path,
    manifest: list[dict[str, Any]],
    require_complete: bool = True,
) -> dict[str, Any]:
    raw_chunk_paths = sorted(output_dir.glob("raw_qwen3vl_judgeability_evidence_shard_*.jsonl"))
    parsed_chunk_paths = sorted(output_dir.glob("parsed_qwen3vl_judgeability_evidence_shard_*.jsonl"))
    if not raw_chunk_paths and (output_dir / RAW_FILENAME).exists():
        raw_chunk_paths = [output_dir / RAW_FILENAME]
    if not parsed_chunk_paths and (output_dir / PARSED_FILENAME).exists():
        parsed_chunk_paths = [output_dir / PARSED_FILENAME]

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    for path in raw_chunk_paths:
        raw_rows.extend(load_jsonl(path))
    for path in parsed_chunk_paths:
        parsed_rows.extend(load_jsonl(path))

    raw_by_id = id_map(raw_rows, label="raw evidence rows")
    parsed_by_id = id_map(parsed_rows, label="parsed evidence rows")
    manifest_ids = [str(item.get("video_id")) for item in manifest if item.get("video_id")]
    manifest_id_set = set(manifest_ids)
    missing_raw = [video_id for video_id in manifest_ids if video_id not in raw_by_id]
    missing_parsed = [video_id for video_id in manifest_ids if video_id not in parsed_by_id]
    unused_raw = sorted(video_id for video_id in raw_by_id if video_id not in manifest_id_set)
    unused_parsed = sorted(video_id for video_id in parsed_by_id if video_id not in manifest_id_set)
    if require_complete and missing_raw:
        raise RuntimeError(f"cannot merge incomplete evidence output: {len(missing_raw)} manifest videos are missing")

    ordered_raw = [raw_by_id[video_id] for video_id in manifest_ids if video_id in raw_by_id]
    ordered_parsed = [parsed_by_id[video_id] for video_id in manifest_ids if video_id in parsed_by_id]
    write_jsonl(output_dir / RAW_FILENAME, ordered_raw)
    write_jsonl(output_dir / PARSED_FILENAME, ordered_parsed)
    write_jsonl(output_dir / EVIDENCE_FILENAME, ordered_parsed)
    summary = {
        "schema_version": (
            str(ordered_parsed[0].get("schema_version"))
            if ordered_parsed and ordered_parsed[0].get("schema_version")
            else SCHEMA_VERSION
        ),
        "merge_status": "complete" if not missing_raw else "incomplete",
        "manifest_records": len(manifest_ids),
        "raw_records_written": len(ordered_raw),
        "parsed_records_written": len(ordered_parsed),
        "evidence_records_written": len(ordered_parsed),
        "parse_rate": round(len(ordered_parsed) / len(ordered_raw), 6) if ordered_raw else 0.0,
        "missing_video_ids": missing_raw,
        "missing_parsed_video_ids": missing_parsed,
        "unused_raw_video_ids": unused_raw,
        "unused_parsed_video_ids": unused_parsed,
        "request_mapping": [
            {
                "video_id": str(item.get("video_id")),
                "request_id": item.get("request_id"),
                "row_index": item.get("row_index"),
                "raw_present": str(item.get("video_id")) in raw_by_id,
                "parsed_present": str(item.get("video_id")) in parsed_by_id,
            }
            for item in manifest
            if item.get("video_id")
        ],
        "raw_chunk_files": [str(path) for path in raw_chunk_paths],
        "parsed_chunk_files": [str(path) for path in parsed_chunk_paths],
    }
    write_json(output_dir / "merge_summary_qwen3vl_judgeability_evidence.json", summary)
    return summary


def build_run_config(
    *,
    args: argparse.Namespace,
    raw_records_written: int,
    parsed_records_written: int,
    records_expected: int,
    max_frames_observed: int,
    local_rank: int,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version_for_prompt_schema(args.prompt_schema),
        "prompt_schema": args.prompt_schema,
        "fps": str(args.fps),
        "max_frames_observed": int(max_frames_observed),
        "model_path": str(args.model_path),
        "model_config_model_type": model_config_model_type(args.model_path),
        "python_path": sys.executable,
        "transformers_version": _version_or_unavailable("transformers"),
        "torch_version": _version_or_unavailable("torch"),
        "cuda_build": _cuda_build_or_unavailable(),
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": int(args.max_new_tokens),
        "video_sampling_policy": (
            "processor_default_not_all_source_frames"
            if str(args.fps) == "full"
            else "processor_kwargs_fps"
        ),
        "device_map": {"": int(local_rank)},
        "local_rank": int(local_rank),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "input_manifest_path": str(args.manifest_path),
        "input_manifest_sha256": sha256_file(args.manifest_path),
        "records_expected": int(records_expected),
        "raw_records_written": int(raw_records_written),
        "parsed_records_written": int(parsed_records_written),
        "parse_rate": round(parsed_records_written / raw_records_written, 6) if raw_records_written else 0.0,
        "skip_existing": bool(args.skip_existing),
    }


def stage_result_text(*, config: dict[str, Any]) -> str:
    parse_rate = float(config.get("parse_rate") or 0.0)
    records_expected = int(config.get("records_expected") or 0)
    raw_records_written = int(config.get("raw_records_written") or 0)
    decision = "pass" if raw_records_written >= records_expected and parse_rate >= 0.95 else "retry_once"
    prompt_schema = str(config.get("prompt_schema") or PROMPT_SCHEMA_SUBJECT)
    mapping_note = (
        "- Human N/A diagnostic mapping is parsed from the direct P9-style three-question gate.\n"
        if prompt_schema == PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN
        else "- Human N/A diagnostic mapping is post-processed from yes -> no -> yes transitions.\n"
    )
    return (
        f"# Stage {config.get('schema_version') or SCHEMA_VERSION}\n\n"
        f"- scorer: qwen3vl_video_evidence/{prompt_schema}\n"
        f"- decision: {decision}\n\n"
        "## metrics\n"
        f"- records_expected: {records_expected}\n"
        f"- raw_records_written: {raw_records_written}\n"
        f"- parsed_records_written: {config.get('parsed_records_written')}\n"
        f"- parse_rate: {parse_rate:.6f}\n"
        f"- max_frames_observed: {config.get('max_frames_observed')}\n\n"
        "## notes\n"
        "- Candidate subject judgeability evidence generated. This does not overwrite canonical V7.\n"
        f"{mapping_note}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen3-VL structured subject judgeability evidence runner")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Local Qwen3-VL model directory (set via wrbench.runtime.json eval.scorers.qwen3vl_model).")
    parser.add_argument("--fps", required=True)
    parser.add_argument("--max-videos", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--allow-incomplete-merge", action="store_true")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--prompt-schema", choices=sorted(SUPPORTED_PROMPT_SCHEMAS), required=True)
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.max_videos < 0:
        parser.error("--max-videos must be >= 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.dtype != DEFAULT_DTYPE:
        parser.error("--dtype must be bfloat16 for this route")
    if args.attn_implementation != DEFAULT_ATTN_IMPLEMENTATION:
        parser.error("--attn-implementation must be flash_attention_2 for this route")
    if str(args.fps) != "full":
        try:
            fps_value = float(args.fps)
        except ValueError:
            parser.error("--fps must be a positive number or full")
        if fps_value <= 0:
            parser.error("--fps must be positive or full")
        args.fps = str(int(fps_value)) if fps_value.is_integer() else str(fps_value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_rank = int(args.local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("--manifest-path must contain a JSON list")
    manifest = attach_outer_request_metadata(manifest)

    selected_manifest = manifest[: args.max_videos] if args.max_videos else manifest
    if args.merge_only:
        summary = merge_sharded_outputs(
            output_dir=args.output_dir,
            manifest=selected_manifest,
            require_complete=not args.allow_incomplete_merge,
        )
        config = build_run_config(
            args=args,
            raw_records_written=int(summary["raw_records_written"]),
            parsed_records_written=int(summary["parsed_records_written"]),
            records_expected=int(summary["manifest_records"]),
            max_frames_observed=0,
            local_rank=local_rank,
        )
        write_json(args.output_dir / "run_config.json", config)
        (args.output_dir / "stage_result.md").write_text(stage_result_text(config=config), encoding="utf-8")
        return 0

    shard_items = select_manifest_shard(manifest, num_shards=args.num_shards, shard_id=args.shard_id)
    if args.max_videos:
        shard_items = shard_items[: args.max_videos]

    raw_path = args.output_dir / (RAW_FILENAME if args.num_shards == 1 else f"raw_qwen3vl_judgeability_evidence_shard_{args.shard_id}.jsonl")
    parsed_path = args.output_dir / (PARSED_FILENAME if args.num_shards == 1 else f"parsed_qwen3vl_judgeability_evidence_shard_{args.shard_id}.jsonl")
    existing_raw = load_jsonl(raw_path) if args.skip_existing else []
    existing_parsed = load_jsonl(parsed_path) if args.skip_existing else []
    existing_raw_by_id = id_map(existing_raw, label="existing raw rows")

    runner = LocalQwen3VLVideoEvidenceRunner(
        model_path=args.model_path,
        fps=str(args.fps),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_rank=local_rank,
        max_new_tokens=args.max_new_tokens,
        prompt_schema=args.prompt_schema,
    )

    new_raw: list[dict[str, Any]] = []
    new_parsed: list[dict[str, Any]] = []
    for item in shard_items:
        video_id = str(item["video_id"])
        if args.skip_existing and video_id in existing_raw_by_id:
            continue
        raw_row, parsed_row = runner.run_one(item)
        new_raw.append(raw_row)
        if parsed_row is not None:
            new_parsed.append(parsed_row)

    raw_by_id = id_map(existing_raw + new_raw, label="raw rows")
    parsed_by_id = id_map(existing_parsed + new_parsed, label="parsed rows")
    ordered_raw = [raw_by_id[str(item["video_id"])] for item in shard_items if str(item.get("video_id")) in raw_by_id]
    ordered_parsed = [parsed_by_id[str(item["video_id"])] for item in shard_items if str(item.get("video_id")) in parsed_by_id]
    write_jsonl(raw_path, ordered_raw)
    write_jsonl(parsed_path, ordered_parsed)
    if args.num_shards == 1:
        write_jsonl(args.output_dir / EVIDENCE_FILENAME, ordered_parsed)

    max_frames_observed = max([int(row.get("frames_used") or 0) for row in ordered_raw] or [0])
    config = build_run_config(
        args=args,
        raw_records_written=len(ordered_raw),
        parsed_records_written=len(ordered_parsed),
        records_expected=len(shard_items),
        max_frames_observed=max_frames_observed,
        local_rank=local_rank,
    )
    write_json(
        args.output_dir / ("run_config.json" if args.num_shards == 1 else f"run_config_shard_{args.shard_id}.json"),
        config,
    )
    (args.output_dir / ("stage_result.md" if args.num_shards == 1 else f"stage_result_shard_{args.shard_id}.md")).write_text(
        stage_result_text(config=config),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
