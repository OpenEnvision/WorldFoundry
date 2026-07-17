"""Bundled WRBench dataset paths shipped with the WRBench package."""

from __future__ import annotations

import csv
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

PROMPT_PROFILE_TI2V_ACTIVE = "ti2v_active"
PROMPT_PROFILE_T2V_LAYOUT_ANCHOR = "t2v_layout_anchor"
NATURAL25_PROMPT_PROFILES: tuple[str, ...] = (
    PROMPT_PROFILE_TI2V_ACTIVE,
    PROMPT_PROFILE_T2V_LAYOUT_ANCHOR,
)


def _package_data_dir() -> Path:
    """Resolve WorldFoundry benchmark assets, with standalone WRBench fallback."""
    configured = os.environ.get("WORLDFOUNDRY_WRBENCH_ASSETS_ROOT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"WORLDFOUNDRY_WRBENCH_ASSETS_ROOT does not exist: {path}")

    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name != "worldfoundry":
            continue
        path = parent / "data" / "benchmarks" / "assets" / "wrbench"
        if path.is_dir():
            return path

    standalone = Path(str(resources.files("wrbench") / "data"))
    if standalone.is_dir():
        return standalone
    raise FileNotFoundError(
        "WRBench assets were not found under WorldFoundry benchmark assets; "
        "set WORLDFOUNDRY_WRBENCH_ASSETS_ROOT"
    )


def data_dir() -> Path:
    return _package_data_dir()


def model_configs_dir() -> Path:
    """Return the data-backed WRBench camera-control model contracts."""
    return data_dir() / "model_configs"


def prompt_templates_dir() -> Path:
    """Return the data-backed WRBench prompt template directory."""
    return data_dir() / "prompts" / "templates"


def natural25_dir() -> Path:
    return data_dir() / "natural25"


def natural25_families_path() -> Path:
    return natural25_dir() / "families.jsonl"


def natural25_scene_events_path() -> Path:
    return natural25_dir() / "scene_events_25x4.csv"


def natural25_variants_path() -> Path:
    return natural25_dir() / "variants.jsonl"


def natural25_t2v_layout_anchors_path() -> Path:
    return natural25_dir() / "t2v_layout_anchors.jsonl"


def natural25_t2v_event_tails_path() -> Path:
    return natural25_dir() / "t2v_event_tails.jsonl"


def natural25_prompt_profile_path(profile_id: str) -> Path:
    return natural25_dir() / "prompt_profiles" / f"{profile_id}.json"


def natural25_t2v_rotation_stress_camera_scope_path() -> Path:
    return natural25_dir() / "camera_scopes" / "t2v_rotation_stress_30_60.json"


def natural25_legacy_variants_path() -> Path:
    return natural25_dir() / "variants.legacy_pronoun_20260620.jsonl"


def _require_nonempty(value: Any, *, field_name: str, row_id: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{row_id!r} missing {field_name}")
    return text


def _ti2v_prompt(variant: dict[str, Any]) -> str:
    return _require_nonempty(
        variant.get("ti2v_prompt"),
        field_name="ti2v_prompt",
        row_id=str(variant.get("variant_id")),
    )


def _strip_sentence_period(value: Any, *, field_name: str, row_id: str) -> str:
    return _require_nonempty(value, field_name=field_name, row_id=row_id).rstrip(".").strip()


def load_natural25_t2v_layout_anchors(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load reviewed Natural-25 layout anchors for T2V prompt-only models."""
    anchor_path = Path(path) if path is not None else natural25_t2v_layout_anchors_path()
    anchors: dict[str, dict[str, Any]] = {}
    expected_keys = {"family_id", "scene", "subject", "interactor", "open_surface", "anchors"}
    for index, row in enumerate(load_jsonl(anchor_path)):
        keys = set(row)
        if keys != expected_keys:
            raise ValueError(f"T2V layout anchor row {index} keys must be {sorted(expected_keys)}, got {sorted(keys)}")
        family_id = _require_nonempty(row.get("family_id"), field_name="family_id", row_id=f"anchor row {index}")
        if family_id in anchors:
            raise ValueError(f"duplicate T2V layout anchor family_id {family_id!r}")
        for field_name in ("scene", "subject", "interactor", "open_surface"):
            _require_nonempty(row.get(field_name), field_name=field_name, row_id=family_id)
        anchor_list = row.get("anchors")
        if not isinstance(anchor_list, list) or not anchor_list:
            raise ValueError(f"T2V layout anchor {family_id!r} anchors must be a non-empty list")
        if not all(isinstance(item, str) and item.strip() for item in anchor_list):
            raise ValueError(f"T2V layout anchor {family_id!r} anchors must be non-empty strings")
        anchors[family_id] = dict(row)
    return anchors


def load_natural25_t2v_event_tails(path: str | Path | None = None) -> dict[str, str]:
    """Load reviewed T2V event tails keyed by Natural-25 ``variant_id``."""
    tails_path = Path(path) if path is not None else natural25_t2v_event_tails_path()
    tails: dict[str, str] = {}
    expected_keys = {"variant_id", "t2v_event_tail"}
    for index, row in enumerate(load_jsonl(tails_path)):
        keys = set(row)
        if keys != expected_keys:
            raise ValueError(f"T2V event tail row {index} keys must be {sorted(expected_keys)}, got {sorted(keys)}")
        variant_id = _require_nonempty(row.get("variant_id"), field_name="variant_id", row_id=f"event tail row {index}")
        if variant_id in tails:
            raise ValueError(f"duplicate T2V event tail variant_id {variant_id!r}")
        tails[variant_id] = _require_nonempty(row.get("t2v_event_tail"), field_name="t2v_event_tail", row_id=variant_id)
    return tails


def _join_anchors(anchor_names: list[str]) -> str:
    names = [name.strip() for name in anchor_names if name.strip()]
    if len(names) == 1:
        text = names[0]
    else:
        text = " and ".join(names[:2])
    return text[:1].upper() + text[1:]


def _build_t2v_layout_anchor_prompt(
    variant: dict[str, Any],
    *,
    layout_anchors: dict[str, dict[str, Any]],
    event_tails: dict[str, str],
) -> str:
    variant_id = str(variant.get("variant_id"))
    oov_gap = _require_nonempty(variant.get("oov_gap"), field_name="oov_gap", row_id=variant_id)
    if oov_gap != "none":
        raise ValueError(f"{variant_id!r} t2v_layout_anchor requires oov_gap 'none', got {oov_gap!r}")
    family_id = _require_nonempty(variant.get("family_id"), field_name="family_id", row_id=variant_id)
    anchor = layout_anchors.get(family_id)
    if anchor is None:
        raise ValueError(f"{variant_id!r} missing T2V layout anchor for family_id {family_id!r}")

    scene = _strip_sentence_period(anchor["scene"], field_name="scene", row_id=family_id)
    subject = _strip_sentence_period(anchor["subject"], field_name="subject", row_id=family_id)
    interactor = _strip_sentence_period(anchor["interactor"], field_name="interactor", row_id=family_id)
    open_surface = _strip_sentence_period(anchor["open_surface"], field_name="open_surface", row_id=family_id)
    anchor_text = _join_anchors(list(anchor["anchors"]))
    tail = event_tails.get(variant_id)
    if tail is None:
        raise ValueError(f"{variant_id!r} missing T2V event tail")
    return (
        f"In {scene}, {subject}; {interactor}. "
        f"The layout includes {open_surface}. {anchor_text}. "
        f"{tail}"
    )


def resolve_variant_prompt(
    variant: dict[str, Any],
    *,
    prompt_profile: str = PROMPT_PROFILE_TI2V_ACTIVE,
    layout_anchors: dict[str, dict[str, Any]] | None = None,
    t2v_event_tails: dict[str, str] | None = None,
) -> str:
    """Return a Natural-25 generation prompt under an explicit prompt profile."""
    if prompt_profile == PROMPT_PROFILE_TI2V_ACTIVE:
        return _ti2v_prompt(variant)
    if prompt_profile == PROMPT_PROFILE_T2V_LAYOUT_ANCHOR:
        if layout_anchors is None:
            layout_anchors = load_natural25_t2v_layout_anchors()
        if t2v_event_tails is None:
            t2v_event_tails = load_natural25_t2v_event_tails()
        return _build_t2v_layout_anchor_prompt(
            variant,
            layout_anchors=layout_anchors,
            event_tails=t2v_event_tails,
        )
    raise ValueError(f"unsupported Natural-25 prompt_profile {prompt_profile!r}")


def resolve_variant_prompt_profile(prompt_profile: str) -> str:
    """Validate and normalize a Natural-25 prompt profile id."""
    profile = str(prompt_profile or "").strip()
    if profile not in NATURAL25_PROMPT_PROFILES:
        raise ValueError(f"unsupported Natural-25 prompt_profile {prompt_profile!r}")
    return profile


def natural25_first_frames_dir() -> Path:
    return natural25_dir() / "first_frames"


def natural25_first_frames_manifest_path() -> Path:
    return natural25_dir() / "first_frames_manifest.json"


def natural25_first_frame_path(family_id: str) -> Path:
    return natural25_first_frames_dir() / f"{family_id}.png"


def published_results_dir() -> Path:
    return data_dir() / "results"


def published_results_csv() -> Path:
    return published_results_dir() / "wrbench_23model_results.csv"


def published_results_json() -> Path:
    return published_results_dir() / "wrbench_23model_results.json"


def published_t2v_results_json() -> Path:
    return published_results_dir() / "wrbench_t2v_results.json"


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_natural25_families() -> dict[str, dict[str, Any]]:
    """Load Natural-25 family records keyed by ``family_id``."""
    return {row["family_id"]: row for row in load_jsonl(natural25_families_path())}


def build_natural25_candidates(*, offscreen_area: str = "empty floor space") -> list[dict[str, Any]]:
    """Build deterministic task candidates from ``scene_events_25x4.csv``."""
    path = natural25_scene_events_path()
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candidates.append(
                {
                    "candidate_id": row["family_id"],
                    "offscreen_area": offscreen_area,
                    "events": {
                        "t0": row["event_T0"],
                        "t1": row["event_T1"],
                        "div_a_state_only": row["event_T2_div_a"],
                        "div_b": row["event_T2_div_b"],
                    },
                }
            )
    return candidates
