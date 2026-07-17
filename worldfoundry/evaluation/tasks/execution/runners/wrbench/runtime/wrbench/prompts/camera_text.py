"""Camera prompt text generation — stdlib only.

Provides natural-language camera clauses for TI2V prompts and API-model prompt
assembly (copy-optimized vs Hailuo command injection).
"""

from __future__ import annotations

import re
from typing import Any

_SUBJECT_PRONOUN_RE = re.compile(r"\b(He|She|It|he|she|it)\b")
_VERB_SPLIT_RE = re.compile(
    r"\s+(?:stands|walks|runs|jumps|sits|remains|places|tips|knocks|folds|carries|moves|"
    r"approaches|rests|kneels|eats|drinks|reads|writes|opens|closes|lifts|drops|pulls|"
    r"pushes|turns|looks|reaches|holds|picks|puts|leans|falls|lies|sleeps|wakes|waits|"
    r"enters|exits|leaves|arrives|departs)\b",
    re.I,
)


CAMERA_CLAUSES: dict[str, str] = {
    "none": "",
    "static": "The camera remains fixed in place throughout, with no movement.",
    "yaw_LR": (
        "The camera slowly moves left until {pronoun} is out of frame, "
        "revealing more {offscreen_area}. "
        "The camera moves right back to the original framing. "
        "Single continuous shot, no cuts."
    ),
    "yaw_RL": (
        "The camera slowly moves right until {pronoun} is out of frame, "
        "revealing more {offscreen_area}. "
        "The camera moves left back to the original framing. "
        "Single continuous shot, no cuts."
    ),
}

API_CAMERA_MOTIONS = ("static", "yaw_LR", "yaw_RL")
COPY_OPTIMIZED_CAMERA_PROMPT = "copy_optimized_camera_prompt"
ADD_HAILUO_CAMERA_COMMANDS = "add_hailuo_camera_commands_to_no_camera_prompt"
SUPPORTED_PROMPT_BUILD_METHODS = {
    COPY_OPTIMIZED_CAMERA_PROMPT,
    ADD_HAILUO_CAMERA_COMMANDS,
}
CAMERA_MOTION_DESCRIPTIONS = {
    "static": "keep the camera still",
    "yaw_LR": "pan left, then pan right back",
    "yaw_RL": "pan right, then pan left back",
}

MODEL_PROMPT_BUILD_METHODS = {
    "hailuo-02": ADD_HAILUO_CAMERA_COMMANDS,
    "hailuo-2.3": ADD_HAILUO_CAMERA_COMMANDS,
    "hailuo-2.3-fast": ADD_HAILUO_CAMERA_COMMANDS,
    "kling-v2-6": COPY_OPTIMIZED_CAMERA_PROMPT,
    "kling-v3": COPY_OPTIMIZED_CAMERA_PROMPT,
    "kling-v3-omni": COPY_OPTIMIZED_CAMERA_PROMPT,
    "luma-ray2": COPY_OPTIMIZED_CAMERA_PROMPT,
    "pika-2.2": COPY_OPTIMIZED_CAMERA_PROMPT,
    "pixverse-v4": COPY_OPTIMIZED_CAMERA_PROMPT,
    "pixverse-v4.5": COPY_OPTIMIZED_CAMERA_PROMPT,
    "pixverse-v5": COPY_OPTIMIZED_CAMERA_PROMPT,
    "runway-gen4-turbo": COPY_OPTIMIZED_CAMERA_PROMPT,
    "runway-gen4.5": COPY_OPTIMIZED_CAMERA_PROMPT,
    "seedance-2.0": COPY_OPTIMIZED_CAMERA_PROMPT,
    "sora-2": COPY_OPTIMIZED_CAMERA_PROMPT,
    "sora-2-pro": COPY_OPTIMIZED_CAMERA_PROMPT,
    "veo-3.1-fast": COPY_OPTIMIZED_CAMERA_PROMPT,
    "veo-3.1-lite": COPY_OPTIMIZED_CAMERA_PROMPT,
    "vidu-q2": COPY_OPTIMIZED_CAMERA_PROMPT,
    "vidu-q2-pro": COPY_OPTIMIZED_CAMERA_PROMPT,
    "vidu-q2-pro-fast": COPY_OPTIMIZED_CAMERA_PROMPT,
    "viduq2-pro-fast": COPY_OPTIMIZED_CAMERA_PROMPT,
}

_HAILUO_CAMERA_CLAUSES = {
    "static": "[Static shot]",
    "yaw_LR": "[Pan left]",
    "yaw_RL": "[Pan right]",
}

_HAILUO_RETURN_CLAUSES = {
    "yaw_LR": "then [Pan right] back to the original framing.",
    "yaw_RL": "then [Pan left] back to the original framing.",
}

_HAILUO_FIRST_FRAME_PRESERVATION = (
    "Preserve the first-frame scene layout, subject identity, and object placement."
)

_SOURCE_GAPS = ("none", *API_CAMERA_MOTIONS)
_REQUIRED_SOURCE_FIELDS = ("variant_id", "family_id", "reasoning_tier", "oov_gap", "ti2v_prompt")

# Map wrbench preset names to camera gap keys used in CAMERA_CLAUSES.
PRESET_TO_GAP = {
    "static": "static",
    "yaw_LR": "yaw_LR",
    "yaw_RL": "yaw_RL",
    "pan_LR": "yaw_LR",  # pan presets use yaw-style NL for API copy path
    "pan_RL": "yaw_RL",
}


def extract_subject_phrase(
    world_state_prompt: str,
    known_subjects: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Extract an explicit subject noun phrase from a world-state sentence."""
    text = str(world_state_prompt or "").strip()
    if not text:
        raise ValueError("world_state_prompt is empty")
    if known_subjects:
        for subject in sorted(known_subjects, key=len, reverse=True):
            if text.startswith(subject):
                return subject
    split = _VERB_SPLIT_RE.split(text, maxsplit=1)
    if split and split[0].strip():
        return split[0].strip().rstrip(",")
    return text.split(".", 1)[0].strip()


def anchor_subject_in_prompt(prompt: str, subject_phrase: str) -> str:
    """Replace the first subject pronoun with an explicit subject noun phrase."""
    phrase = str(subject_phrase or "").strip()
    if not phrase:
        raise ValueError("subject_phrase is empty")

    def _replace(match: re.Match[str]) -> str:
        start = match.start()
        prefix = prompt[:start]
        stripped = prefix.rstrip()
        if not stripped:
            return phrase
        if stripped.endswith((",", ";", ":")):
            return phrase[0].lower() + phrase[1:]
        if stripped.endswith((".", "?", "!")):
            return phrase
        return phrase[0].lower() + phrase[1:]

    updated, count = _SUBJECT_PRONOUN_RE.subn(_replace, prompt, count=1)
    if count == 0:
        return prompt
    return updated


def assemble_ti2v_prompt(
    scene_start: str,
    event: str,
    pronoun: str,
    offscreen_area: str,
    oov_gap: str,
    *,
    subject_phrase: str | None = None,
) -> str:
    """Assemble the unified WRBench video prompt for a specific camera gap."""
    parts = [scene_start.strip()]
    if event and event.strip():
        parts.append(event.strip())
    if oov_gap not in CAMERA_CLAUSES:
        raise ValueError(f"Unknown oov_gap={oov_gap!r}; expected one of {list(CAMERA_CLAUSES)}")
    camera_template = CAMERA_CLAUSES[oov_gap]
    if camera_template:
        parts.append(camera_template.format(pronoun=pronoun, offscreen_area=offscreen_area))
    prompt = " ".join(parts)
    if subject_phrase:
        prompt = anchor_subject_in_prompt(prompt, subject_phrase)
    return prompt


def camera_clause(
    gap: str,
    *,
    pronoun: str = "they",
    offscreen_area: str = "empty floor space",
) -> str:
    """Return the natural-language camera clause for a gap key."""
    if gap not in CAMERA_CLAUSES:
        raise ValueError(f"Unknown gap {gap!r}; expected one of {list(CAMERA_CLAUSES)}")
    template = CAMERA_CLAUSES[gap]
    if not template:
        return ""
    return template.format(pronoun=pronoun, offscreen_area=offscreen_area)


def preset_camera_text(
    preset_name: str,
    *,
    pronoun: str = "they",
    offscreen_area: str = "empty floor space",
) -> str:
    """Map a wrbench preset name to its natural-language camera clause."""
    gap = PRESET_TO_GAP.get(preset_name)
    if gap is None:
        raise KeyError(f"Unknown preset {preset_name!r}; no camera text mapping")
    return camera_clause(gap, pronoun=pronoun, offscreen_area=offscreen_area)


def _canonical_model(model: str) -> str:
    value = str(model or "").strip().lower()
    if not value:
        raise ValueError("API prompt preview model is required")
    if value == "default":
        raise ValueError("API prompt preview model must be explicit, not default")
    return value


def _require_camera_motion(camera_motion: str) -> str:
    value = str(camera_motion).strip()
    if value not in API_CAMERA_MOTIONS:
        raise ValueError(f"Unsupported API prompt preview camera_motion: {camera_motion}")
    return value


def _tier_key(item: dict[str, Any]) -> str:
    tier = str(item.get("reasoning_tier") or "").strip()
    divergence = item.get("divergence_id")
    if divergence:
        return f"{tier}_{divergence}"
    return tier


def _require_source_row(row: dict[str, Any]) -> dict[str, Any]:
    for field in _REQUIRED_SOURCE_FIELDS:
        if field not in row or str(row[field]).strip() == "":
            raise ValueError(f"Source row missing {field}: {row.get('variant_id')}")
    gap = str(row["oov_gap"]).strip()
    if gap not in _SOURCE_GAPS:
        raise ValueError(f"Unsupported source oov_gap for API prompt preview: {gap}")
    family_id = str(row["family_id"]).strip()
    tier = _tier_key(row)
    expected_variant_id = f"{family_id}__{tier}__{gap}"
    if str(row["variant_id"]).strip() != expected_variant_id:
        raise ValueError(
            f"Source row variant_id mismatch: expected {expected_variant_id}, got {row['variant_id']}"
        )
    return row


def _split_prompt_final_sentence(prompt: str) -> tuple[str, str]:
    text = prompt.strip()
    if not text:
        raise ValueError("Prompt is empty")
    if not text.endswith("."):
        return text, ""
    prefix = text[:-1]
    idx = prefix.rfind(". ")
    if idx < 0:
        return text, ""
    return text[: idx + 1].strip(), text[idx + 2 :].strip()


def get_model_prompt_build_method(model: str) -> str:
    """Return the prompt build method for an exact configured model id."""
    canonical = _canonical_model(model)
    if canonical not in MODEL_PROMPT_BUILD_METHODS:
        raise ValueError(f"API prompt preview model is not configured: {model}")
    build_method = str(MODEL_PROMPT_BUILD_METHODS[canonical] or "").strip()
    if not build_method:
        raise ValueError(f"API prompt preview model has no prompt build method: {model}")
    if build_method not in SUPPORTED_PROMPT_BUILD_METHODS:
        raise ValueError(
            f"API prompt preview model has unsupported prompt build method: {model} -> {build_method}"
        )
    return build_method


def source_camera_motion_for_preview(model: str, camera_motion: str) -> str:
    """Return the required source row camera motion for a preview request."""
    camera_motion = _require_camera_motion(camera_motion)
    build_method = get_model_prompt_build_method(model)
    if build_method == ADD_HAILUO_CAMERA_COMMANDS:
        return "none"
    if build_method == COPY_OPTIMIZED_CAMERA_PROMPT:
        return camera_motion
    raise ValueError(f"Unsupported API prompt preview build method: {build_method}")


def _format_hailuo_prompt(base_prompt: str, camera_motion: str) -> str:
    base = base_prompt.strip()
    if camera_motion == "static":
        return "\n".join([base, _HAILUO_FIRST_FRAME_PRESERVATION, _HAILUO_CAMERA_CLAUSES[camera_motion]])
    scene_prompt, motion_prompt = _split_prompt_final_sentence(base)
    parts = [
        scene_prompt,
        _HAILUO_FIRST_FRAME_PRESERVATION,
        _HAILUO_CAMERA_CLAUSES[camera_motion],
        motion_prompt,
        _HAILUO_RETURN_CLAUSES[camera_motion],
        "Single continuous shot, no cuts.",
    ]
    return "\n".join(part for part in parts if part)


def build_prompt_to_send(source_prompt_text: str, camera_motion: str, *, model: str) -> str:
    """Build the exact prompt text to send for a model and camera motion."""
    camera_motion = _require_camera_motion(camera_motion)
    prompt = str(source_prompt_text).strip()
    if not prompt:
        raise ValueError("Source prompt text is empty")
    build_method = get_model_prompt_build_method(model)
    if build_method == COPY_OPTIMIZED_CAMERA_PROMPT:
        return prompt
    if build_method == ADD_HAILUO_CAMERA_COMMANDS:
        return _format_hailuo_prompt(prompt, camera_motion)
    raise ValueError(f"Unsupported API prompt preview build method: {build_method}")


def build_api_prompt_preview_row(
    source_row: dict[str, Any],
    *,
    model: str,
    camera_motion: str,
) -> dict[str, Any]:
    """Build one public API prompt preview row from a source prompt row."""
    camera_motion = _require_camera_motion(camera_motion)
    source_row = _require_source_row(source_row)
    build_method = get_model_prompt_build_method(model)
    expected_source_motion = source_camera_motion_for_preview(model, camera_motion)
    source_camera_motion = str(source_row["oov_gap"]).strip()
    if source_camera_motion != expected_source_motion:
        raise ValueError(
            "API prompt preview source row mismatch for "
            f"{model}/{camera_motion}: expected oov_gap={expected_source_motion}, "
            f"got {source_camera_motion} ({source_row['variant_id']})"
        )

    family_id = str(source_row["family_id"]).strip()
    tier = _tier_key(source_row)
    source_prompt_text = str(source_row["ti2v_prompt"]).strip()
    prompt_to_send = build_prompt_to_send(source_prompt_text, camera_motion, model=model)
    return {
        "preview_id": f"{family_id}__{tier}__{_canonical_model(model)}__{camera_motion}",
        "model": _canonical_model(model),
        "family_id": family_id,
        "reasoning_tier": tier,
        "camera_motion": camera_motion,
        "camera_motion_description": CAMERA_MOTION_DESCRIPTIONS[camera_motion],
        "source_variant_id": str(source_row["variant_id"]).strip(),
        "source_camera_motion": source_camera_motion,
        "prompt_build_method": build_method,
        "source_prompt_text": source_prompt_text,
        "prompt_to_send": prompt_to_send,
    }


def build_api_prompt_preview_rows(
    source_rows: list[dict[str, Any]],
    *,
    model: str,
    camera_motions: tuple[str, ...] = API_CAMERA_MOTIONS,
) -> list[dict[str, Any]]:
    """Build public preview rows for each source variant and requested camera motion."""
    for camera_motion in camera_motions:
        _require_camera_motion(camera_motion)
    build_method = get_model_prompt_build_method(model)
    checked_rows = [_require_source_row(row) for row in source_rows]

    if build_method == COPY_OPTIMIZED_CAMERA_PROMPT:
        expected_groups: set[tuple[str, str]] = set()
        camera_rows_by_group: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for row in checked_rows:
            gap = str(row["oov_gap"]).strip()
            if gap == "none" or gap in camera_motions:
                family_id = str(row["family_id"]).strip()
                tier = _tier_key(row)
                group_key = (family_id, tier)
                expected_groups.add(group_key)
                if gap in camera_motions:
                    camera_rows_by_group.setdefault(group_key, {})[gap] = row
        if not expected_groups:
            raise ValueError(f"No prompt variants found for {model}")
        missing_by_group = {
            group_key: [
                camera_motion
                for camera_motion in camera_motions
                if camera_motion not in camera_rows_by_group.get(group_key, {})
            ]
            for group_key in sorted(expected_groups)
        }
        missing_by_group = {group_key: missing for group_key, missing in missing_by_group.items() if missing}
        if missing_by_group:
            details = "; ".join(
                f"{family_id}/{tier}: {missing}" for (family_id, tier), missing in missing_by_group.items()
            )
            raise ValueError(f"Missing existing camera prompt variants for {model}: {details}")
        rows: list[dict[str, Any]] = []
        for row in checked_rows:
            if row["oov_gap"] in camera_motions:
                rows.append(build_api_prompt_preview_row(row, model=model, camera_motion=str(row["oov_gap"])))
        return rows

    if build_method == ADD_HAILUO_CAMERA_COMMANDS:
        rows = []
        for row in checked_rows:
            if row["oov_gap"] != "none":
                raise ValueError(f"Hailuo prompt preview rows require only oov_gap=none rows: {row['variant_id']}")
            for camera_motion in camera_motions:
                rows.append(build_api_prompt_preview_row(row, model=model, camera_motion=camera_motion))
        if not rows:
            raise ValueError(f"No content-only prompt variants found for {model}; expected oov_gap=none")
        return rows

    raise ValueError(f"Unsupported API prompt preview build method: {build_method}")
