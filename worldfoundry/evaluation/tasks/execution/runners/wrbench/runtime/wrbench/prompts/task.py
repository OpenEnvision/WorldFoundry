"""Task / TI2V prompt generation — LLM and deterministic paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from wrbench.datasets import prompt_templates_dir
from wrbench.prompts.camera_text import CAMERA_CLAUSES, assemble_ti2v_prompt


def load_ti2v_system_prompt() -> str:
    return (prompt_templates_dir() / "ti2v_system_prompt.txt").read_text(encoding="utf-8").strip()


def build_ti2v_user_message(tier_variants: list[dict[str, Any]], family: dict[str, Any]) -> str:
    v = tier_variants[0]
    t2i_scene = family.get("t2i_scene")
    if not t2i_scene:
        raise ValueError(f"family[{v['family_id']}] missing t2i_scene")
    first_frame_guidance = family.get("first_frame_guidance")
    if not isinstance(first_frame_guidance, dict):
        raise ValueError(f"family[{v['family_id']}] missing first_frame_guidance dict")
    texture = first_frame_guidance.get("texture")
    if not texture:
        raise ValueError(f"family[{v['family_id']}] missing first_frame_guidance.texture")
    payload = {
        "family_id": v["family_id"],
        "reasoning_tier": v["reasoning_tier"],
        "divergence_id": v.get("divergence_id"),
        "event_delta": v["event_delta"],
        "world_state_prompt": v["world_state_prompt"],
        "family": {
            "scene": family["scene"],
            "primary_object": family["primary_object"],
            "supporting_objects": family["supporting_objects"],
            "t2i_scene": t2i_scene,
            "first_frame_texture": texture,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def generate_ti2v_content_llm(
    tier_variants: list[dict[str, Any]],
    family: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    provider: str | None = None,
    api_key: str | None = None,
    llm_call: Any | None = None,
) -> dict[str, str]:
    """Call LLM to produce shared creative fields for a tier group."""
    system_prompt = load_ti2v_system_prompt()
    user_message = build_ti2v_user_message(tier_variants, family)

    if llm_call is not None:
        result = llm_call(system_prompt, user_message)
    else:
        from wrbench.prompts.llm import call_llm_json

        result = call_llm_json(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            temperature=temperature,
            provider=provider,
            api_key=api_key,
        )

    for key in ("scene_start", "event", "pronoun", "offscreen_area"):
        if key not in result:
            raise ValueError(f"LLM response missing key: {key!r} — got: {result}")
    return {
        "scene_start": str(result["scene_start"]).strip(),
        "event": str(result["event"]).strip(),
        "pronoun": str(result["pronoun"]).strip(),
        "offscreen_area": str(result["offscreen_area"]).strip(),
    }


def generate_ti2v_variants_llm(
    tier_variants: list[dict[str, Any]],
    family: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    provider: str | None = None,
    api_key: str | None = None,
    llm_call: Any | None = None,
) -> list[dict[str, Any]]:
    """Fill ``ti2v_prompt`` on each variant in *tier_variants* via LLM content."""
    content = generate_ti2v_content_llm(
        tier_variants,
        family,
        model=model,
        temperature=temperature,
        provider=provider,
        api_key=api_key,
        llm_call=llm_call,
    )
    out = []
    for variant in tier_variants:
        row = dict(variant)
        gap = str(row["oov_gap"])
        row["ti2v_prompt"] = assemble_ti2v_prompt(
            content["scene_start"],
            content["event"],
            content["pronoun"],
            content["offscreen_area"],
            gap,
        )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Deterministic Natural-25 style path (no LLM)
# ---------------------------------------------------------------------------

OOV_GAPS = list(CAMERA_CLAUSES.keys())

TIER_DEFS = [
    {"tier": "T0", "event_key": None, "event_delta": "none", "divergence_id": None},
    {"tier": "T1", "event_key": "t1", "event_delta": "spatial", "divergence_id": None},
    {"tier": "T2", "event_key": "div_a_state_only", "event_delta": "state_only", "divergence_id": "div_a"},
    {"tier": "T2", "event_key": "div_b", "event_delta": "full", "divergence_id": "div_b"},
]

KNOWN_SUBJECTS = sorted(
    [
        "The adult person",
        "The office worker",
        "The cafe customer",
        "The cat",
        "The dog",
        "The goat",
        "The child",
        "The guest",
        "The customer",
        "The waiter",
        "The student",
        "The reader",
        "The patient",
        "The passenger",
        "The worker",
    ],
    key=len,
    reverse=True,
)

ANIMAL_KEYWORDS = {"cat", "dog", "goat", "bird", "rabbit", "horse"}


def detect_pronoun(t2i_scene: str, primary_name: str) -> str:
    if any(w in primary_name.lower() for w in ANIMAL_KEYWORDS):
        return "it"
    lower = t2i_scene.lower()
    if re.search(r"\bwoman\b|\bshe\b|\bher\b|\bfemale\b", lower):
        return "she"
    return "he"


def build_scene_start(t2i_scene: str, family: dict[str, Any]) -> str:
    interactor = next(o for o in family["supporting_objects"] if o["role"] == "interactor")
    anchors = [o for o in family["supporting_objects"] if o["role"] == "anchor"]

    facing_pos = t2i_scene.find(" facing ")
    if facing_pos > 0:
        comma_pos = t2i_scene.rfind(",", 0, facing_pos)
        primary_clause = t2i_scene[:comma_pos] if comma_pos > 0 else t2i_scene.split(".")[0]
    else:
        primary_clause = t2i_scene.split(".")[0]

    primary_clause = primary_clause.replace(" on the left third of the frame", " on the left")
    primary_clause = re.sub(
        r",\s+with\s+(?:a\s+)?(?:relaxed|calm|curious|composed|focused|professional|warm|cheerful|content|pleasant|alert)"
        r"(?:[,\s]+(?:and\s+)?(?:\w+\s+)*?)?(?:posture|expression)",
        "",
        primary_clause,
    )
    primary_clause = re.sub(
        r",\s+with\s+(?:his|her|its)\s+(?:hands?|ears?)\s+(?:in\s+(?:his|her|its)\s+pockets?|perked\s+up)",
        "",
        primary_clause,
    )
    primary_clause = re.sub(r"\bsits\s+on\s+the\s+left\b", "stands on the left", primary_clause)
    primary_clause = re.sub(r"\s{2,}", " ", primary_clause).strip().rstrip(",")

    int_name = interactor["name"]
    int_cap = int_name[0].upper() + int_name[1:]
    anchor_str = " and ".join(a["name"] for a in anchors)

    parts = [f"{primary_clause}. {int_cap} sits far on the right"]
    if anchor_str:
        parts.append(f", with {anchor_str} in the background")
    parts.append(".")
    return "".join(parts)


def replace_subject(event_text: str, pronoun: str) -> str:
    for subj in KNOWN_SUBJECTS:
        if event_text.startswith(subj):
            return pronoun.capitalize() + event_text[len(subj) :]
    return event_text


def make_variant_id(family_id: str, tier: str, oov_gap: str, divergence_id: str | None) -> str:
    tier_part = f"{tier}_{divergence_id}" if divergence_id else tier
    return f"{family_id}__{tier_part}__{oov_gap}"


def generate_variants_deterministic(
    candidates: list[dict[str, Any]],
    families: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build variant records deterministically from candidates + families."""
    variants: list[dict[str, Any]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        family = families.get(cid)
        if not family:
            continue
        t2i_scene = family.get("t2i_scene", "")
        if not t2i_scene:
            continue

        pronoun = detect_pronoun(t2i_scene, family["primary_object"]["name"])
        offscreen_area = cand["offscreen_area"]
        scene_start = build_scene_start(t2i_scene, family)
        events = cand["events"]

        for td in TIER_DEFS:
            if td["event_key"] is None:
                event = f"{pronoun.capitalize()} remains perfectly still in place."
                world_state = events["t0"]
            else:
                raw = events[td["event_key"]]
                if isinstance(raw, dict):
                    event = raw["middle_sentence"]
                    world_state = raw["world_state"]
                else:
                    raw_event = replace_subject(raw, pronoun)
                    event = f"Immediately, {raw_event[0].lower()}{raw_event[1:]}"
                    world_state = raw

            for gap in OOV_GAPS:
                variants.append(
                    {
                        "variant_id": make_variant_id(cid, td["tier"], gap, td["divergence_id"]),
                        "family_id": cid,
                        "reasoning_tier": td["tier"],
                        "oov_gap": gap,
                        "event_delta": td["event_delta"],
                        "divergence_id": td["divergence_id"],
                        "world_state_prompt": world_state,
                        "expected_state": world_state,
                        "ti2v_prompt": assemble_ti2v_prompt(
                            scene_start,
                            event,
                            pronoun,
                            offscreen_area,
                            gap,
                        ),
                    }
                )
    return variants


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
