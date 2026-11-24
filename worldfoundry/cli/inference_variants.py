# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Inference variant registry and resolution shared by the TUI, command builders, and catalog.

The variant tables predate the model catalog and keep the shell inference IDs used by
``scripts/inference/run_infer.sh``; running this module translates that script's flags.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from worldfoundry.runtime.inference_catalog import get_model_inference_spec, model_inference_spec
from worldfoundry.runtime.inference_specs.lingbot_world import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
)

# ── Model variant registry ──────────────────────────────────────────
# Maps model family IDs to their ordered inference variant tuples.
# Each variant maps to an ``INFER_VARIANT_GROUPS`` category and an
# ``INFER_VARIANT_LABELS`` display label.
INFER_MODEL_VARIANTS = {
    "open-magvit2": (
        "open-magvit2-b",
        "open-magvit2-l",
        "open-magvit2-xl",
    ),
    "show-o": (
        "show-o-256",
        "show-o-512",
    ),
    "depth-anything-v1": (
        "depth-anything-v1-small",
        "depth-anything-v1-base",
        "depth-anything-v1-large",
    ),
    "depth-anything-v2": (
        "depth-anything-v2-small",
        "depth-anything-v2-base",
        "depth-anything-v2-large",
        "depth-anything-v2-giant",
    ),
    "depth-anything-v3": (
        "da3-small",
        "da3-base",
        "da3-large",
        "da3-large-1.1",
        "da3-giant",
        "da3-giant-1.1",
        "da3metric-large",
        "da3mono-large",
        "da3nested-giant-large",
        "da3nested-giant-large-1.1",
    ),
    "moge": (
        "moge-v1-vitl",
        "moge-v2-vits-normal",
        "moge-v2-vitb-normal",
        "moge-v2-vitl-normal",
        "moge-v2-vitl",
    ),
    "vggt": ("vggt-1b", "vggt-omega"),
    "cut3r": ("cut3r",),
    "gen3c": ("gen3c-cosmos-7b",),
    "pi3": ("pi3",),
    "pi3x": ("pi3x",),
    "loger": ("loger", "loger-star"),
    "flash-world": ("flash-world",),
    "wan2.1": (
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b-480p",
        "wan2.1-t2v-14b-720p",
        "wan2.1-i2v-14b-480p",
        "wan2.1-i2v-14b-720p",
    ),
    "wan2.2": ("wan2.2-ti2v-5b",),
    "matrix-game-2": (
        "matrix-game-2-universal",
        "matrix-game-2-gta-drive",
        "matrix-game-2-templerun",
    ),
    "matrix-game-3": ("matrix-game-3",),
    "yume-1p5": (
        "yume-1p5-i2v",
        "yume-1p5-t2v",
        "yume-1p5-v2v",
    ),
    LINGBOT_WORLD_MODEL_ID: (
        LINGBOT_VARIANT_BASE_CAM,
        LINGBOT_VARIANT_BASE_ACT_PREVIEW,
        LINGBOT_VARIANT_FAST,
    ),
    "cogvideox": (
        "cogvideox-2b-t2v",
        "cogvideox-5b-t2v",
        "cogvideox-5b-i2v",
        "cogvideox-1.5-5b-t2v",
        "cogvideox-1.5-5b-i2v",
    ),
    "dynamicrafter": ("dynamicrafter-512-i2v", "dynamicrafter-1024-i2v"),
    "ltx-video": ("ltx-video-i2v",),
    "motionctrl": ("motionctrl",),
    "animatediff": (
        "animatediff-v2",
        "animatediff-v3",
        "animatediff-v15",
    ),
    "zeroscope": (
        "zeroscope-576w",
        "zeroscope-xl",
    ),
    "sana-video": (
        "sana-video-2b-480p",
        "sana-video-2b-720p",
        "longsana-video-2b-480p",
    ),
    "cosmos-predict2.5": (
        "cosmos-predict2.5-2b",
        "cosmos-predict2.5-14b",
    ),
    "infinite-world": ("infinite-world",),
    "neoverse": ("neoverse",),
    "worldcam": ("worldcam",),
    "recammaster": ("recammaster",),
    "astra": ("astra",),
    "hunyuan": (
        "hunyuan-gamecraft",
        "hunyuan-mirror",
        "hunyuan-worldplay",
        "hy-world-2.0",
    ),
    "lyra": (
        "lyra1",
        "lyra2",
    ),
    "wow": ("wow",),
    "worldfm": ("worldfm",),
    "inspatio-world": ("inspatio-world",),
    "longvie2": ("longvie2",),
    "fantasy-world": (
        "fantasy-world-wan21",
        "fantasy-world-wan22",
    ),
}


# Short display labels for every inference variant ID.
INFER_VARIANT_LABELS = {
    "open-magvit2-b": "b",
    "open-magvit2-l": "l",
    "open-magvit2-xl": "xl",
    "show-o-256": "256",
    "show-o-512": "512",
    "depth-anything-v1-small": "small",
    "depth-anything-v1-base": "base",
    "depth-anything-v1-large": "large",
    "depth-anything-v2-small": "small",
    "depth-anything-v2-base": "base",
    "depth-anything-v2-large": "large",
    "depth-anything-v2-giant": "giant",
    "da3-small": "small",
    "da3-base": "base",
    "da3-large": "large",
    "da3-large-1.1": "large-1.1",
    "da3-giant": "giant",
    "da3-giant-1.1": "giant-1.1",
    "da3metric-large": "metric-large",
    "da3mono-large": "mono-large",
    "da3nested-giant-large": "nested-giant-large",
    "da3nested-giant-large-1.1": "nested-giant-large-1.1",
    "moge-v1-vitl": "v1-vitl",
    "moge-v2-vits-normal": "v2-vits-normal",
    "moge-v2-vitb-normal": "v2-vitb-normal",
    "moge-v2-vitl-normal": "v2-vitl-normal",
    "moge-v2-vitl": "v2-vitl",
    "vggt": "1b",
    "vggt-1b": "1b",
    "vggt-omega": "omega",
    "cut3r": "default",
    "gen3c": "default",
    "gen3c-cosmos-7b": "cosmos-7b",
    "pi3": "default",
    "pi3x": "default",
    "loger": "base",
    "loger-star": "star",
    "flash-world": "default",
    "wan2.2": "ti2v-5b",
    "wan2.1-t2v-1.3b": "t2v-1.3b",
    "wan2.1-t2v-14b-480p": "t2v-14b-480p",
    "wan2.1-t2v-14b-720p": "t2v-14b-720p",
    "wan2.1-i2v-14b-480p": "i2v-14b-480p",
    "wan2.1-i2v-14b-720p": "i2v-14b-720p",
    "wan2.2-ti2v-5b": "ti2v-5b",
    "matrix-game-2": "universal",
    "matrix-game-2-universal": "universal",
    "matrix-game-2-gta-drive": "gta-drive",
    "matrix-game-2-templerun": "templerun",
    "matrix-game-3": "3.0",
    "yume-1p5": "5b-720p",
    "yume-1p5-i2v": "i2v",
    "yume-1p5-t2v": "t2v",
    "yume-1p5-v2v": "v2v",
    LINGBOT_WORLD_MODEL_ID: "base-cam",
    LINGBOT_VARIANT_BASE_CAM: "base-cam",
    LINGBOT_VARIANT_BASE_ACT_PREVIEW: "base-act-preview",
    LINGBOT_VARIANT_FAST: "fast",
    "lingbot-world-base-cam": "base-cam",
    "lingbot-world-base-act-preview": "base-act-preview",
    "lingbot-world-fast": "fast",
    "cogvideox-2b-t2v": "2b-t2v",
    "cogvideox-5b-t2v": "5b-t2v",
    "cogvideox-5b-i2v": "5b-i2v",
    "cogvideox-1.5-5b-t2v": "1.5-5b-t2v",
    "cogvideox-1.5-5b-i2v": "1.5-5b-i2v",
    "dynamicrafter-512-i2v": "512-i2v",
    "dynamicrafter-1024-i2v": "1024-i2v",
    "ltx-video-i2v": "i2v",
    "motionctrl": "both",
    "animatediff-v2": "v2",
    "animatediff-v3": "v3",
    "animatediff-v15": "v15",
    "zeroscope-576w": "576w",
    "zeroscope-xl": "xl",
    "sana-video-2b-480p": "480p",
    "sana-video-2b-720p": "720p",
    "longsana-video-2b-480p": "longsana-480p",
    "cosmos-predict2.5-2b": "2b",
    "cosmos-predict2.5-14b": "14b",
    "infinite-world": "default",
    "neoverse": "default",
    "worldcam": "default",
    "recammaster": "default",
    "astra": "default",
    "hunyuan-gamecraft": "gamecraft",
    "hunyuan-mirror": "mirror",
    "hunyuan-worldplay": "worldplay",
    "hy-world-2.0": "hy-world-2.0",
    "lyra1": "Lyra 1",
    "lyra2": "Lyra 2",
    "wow": "default",
    "worldfm": "default",
    "inspatio-world": "default",
    "longvie2": "default",
    "fantasy-world-wan21": "wan21",
    "fantasy-world-wan22": "wan22",
}


# Reverse mapping from variant ID to model family ID, built from INFER_MODEL_VARIANTS.
INFER_VARIANT_TO_MODEL = {
    variant: model_id for model_id, variants in INFER_MODEL_VARIANTS.items() for variant in variants
}


INFER_VARIANT_TO_MODEL.update(
    {
        "lingbot-world-base-cam": LINGBOT_WORLD_MODEL_ID,
        "lingbot-world-base-act-preview": LINGBOT_WORLD_MODEL_ID,
        "lingbot-world-fast": LINGBOT_WORLD_MODEL_ID,
    }
)


# Group-category mapping for every inference variant (e.g. "depth", "world-model").
INFER_VARIANT_GROUPS = {
    "open-magvit2-b": "image-gen",
    "open-magvit2-l": "image-gen",
    "open-magvit2-xl": "image-gen",
    "show-o-256": "image-gen",
    "show-o-512": "image-gen",
    "depth-anything-v1-small": "depth",
    "depth-anything-v1-base": "depth",
    "depth-anything-v1-large": "depth",
    "depth-anything-v2-small": "depth",
    "depth-anything-v2-base": "depth",
    "depth-anything-v2-large": "depth",
    "depth-anything-v2-giant": "depth",
    "da3-small": "depth",
    "da3-base": "depth",
    "da3-large": "depth",
    "da3-large-1.1": "depth",
    "da3-giant": "depth",
    "da3-giant-1.1": "depth",
    "da3metric-large": "depth",
    "da3mono-large": "depth",
    "da3nested-giant-large": "depth",
    "da3nested-giant-large-1.1": "depth",
    "moge-v1-vitl": "depth",
    "moge-v2-vits-normal": "depth",
    "moge-v2-vitb-normal": "depth",
    "moge-v2-vitl-normal": "depth",
    "moge-v2-vitl": "depth",
    "vggt": "3d-scene",
    "vggt-1b": "3d-scene",
    "vggt-omega": "3d-scene",
    "cut3r": "3d-scene",
    "gen3c": "3d-scene",
    "gen3c-cosmos-7b": "3d-scene",
    "pi3": "3d-scene",
    "pi3x": "3d-scene",
    "loger": "3d-scene",
    "loger-star": "3d-scene",
    "flash-world": "3d-scene",
    "wan2.1-t2v-1.3b": "interactive-video",
    "wan2.1-t2v-14b-480p": "interactive-video",
    "wan2.1-t2v-14b-720p": "interactive-video",
    "wan2.1-i2v-14b-480p": "interactive-video",
    "wan2.1-i2v-14b-720p": "interactive-video",
    "wan2.2": "interactive-video",
    "wan2.2-ti2v-5b": "interactive-video",
    "matrix-game-1": "world-model",
    "matrix-game-2": "navigation-video",
    "matrix-game-2-universal": "navigation-video",
    "matrix-game-2-gta-drive": "navigation-video",
    "matrix-game-2-templerun": "navigation-video",
    "matrix-game-3": "navigation-video",
    "yume-1p5": "navigation-video",
    "yume-1p5-i2v": "navigation-video",
    "yume-1p5-t2v": "navigation-video",
    "yume-1p5-v2v": "navigation-video",
    "lingbot-world": "navigation-video",
    LINGBOT_VARIANT_BASE_CAM: "navigation-video",
    LINGBOT_VARIANT_BASE_ACT_PREVIEW: "navigation-video",
    LINGBOT_VARIANT_FAST: "navigation-video",
    "lingbot-world-base-cam": "navigation-video",
    "lingbot-world-base-act-preview": "navigation-video",
    "lingbot-world-fast": "navigation-video",
    "cogvideox-2b-t2v": "video-open",
    "cogvideox-5b-t2v": "video-open",
    "cogvideox-5b-i2v": "video-open",
    "cogvideox-1.5-5b-t2v": "video-open",
    "cogvideox-1.5-5b-i2v": "video-open",
    "dynamicrafter-512-i2v": "video-open",
    "dynamicrafter-1024-i2v": "video-open",
    "ltx-video-i2v": "video-open",
    "motionctrl": "video-open",
    "animatediff-v2": "video-open",
    "animatediff-v3": "video-open",
    "animatediff-v15": "video-open",
    "zeroscope-576w": "video-open",
    "zeroscope-xl": "video-open",
    "sana-video-2b-480p": "video-open",
    "sana-video-2b-720p": "video-open",
    "longsana-video-2b-480p": "video-open",
    "cosmos-predict2.5-2b": "world-model",
    "cosmos-predict2.5-14b": "world-model",
    "infinite-world": "world-model",
    "neoverse": "world-model",
    "worldcam": "world-model",
    "recammaster": "world-model",
    "astra": "world-model",
    "hunyuan-gamecraft": "world-model",
    "hunyuan-mirror": "world-model",
    "hunyuan-worldplay": "world-model",
    "hy-world-2.0": "world-model",
    "lyra1": "world-model",
    "lyra2": "world-model",
    "wow": "world-model",
    "worldfm": "world-model",
    "inspatio-world": "world-model",
    "longvie2": "world-model",
    "fantasy-world-wan21": "world-model",
    "fantasy-world-wan22": "world-model",
}


INFER_MODEL_GROUPS = INFER_VARIANT_GROUPS


"""Alias for :data:`INFER_VARIANT_GROUPS` — maps variant IDs to group categories."""


# Mapping from model family ID to its group category, derived from the first variant.
INFER_MODEL_FAMILY_GROUPS = {
    model_id: INFER_VARIANT_GROUPS[variants[0]] for model_id, variants in INFER_MODEL_VARIANTS.items()
}


# Preferred default variant for model families that have multiple choices.
INFER_MODEL_DEFAULT_VARIANTS = {
    "open-magvit2": "open-magvit2-xl",
    "show-o": "show-o-512",
    "moge": "moge-v2-vitl-normal",
    "wan2.1": "wan2.1-i2v-14b-480p",
    "hunyuan": "hunyuan-worldplay",
    "lyra": "lyra2",
}


# The shell inference registry predates the model catalog and uses a handful of
# different punctuation conventions.  Keep those IDs as command aliases, but
# present the catalog's canonical family ID in the TUI so one model is not
# rendered twice (for example ``flash-world`` and ``flashworld``).
SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID = {
    "cosmos-predict2.5": "cosmos-predict-2.5",
    "fantasy-world": "fantasyworld",
    "flash-world": "flashworld",
    "longvie2": "longvie-2",
}


TUI_MODEL_ID_TO_SCRIPT_INFER_FAMILY = {
    tui_model_id: script_family for script_family, tui_model_id in SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.items()
}


# Studio exposes these implementation variants as separate create-job entries.
# In the TUI model browser they belong under the corresponding family selector.
STUDIO_MODEL_TO_TUI_MODEL_ID = {
    "fantasyworld-wan21": "fantasyworld",
    "fantasyworld-wan22": "fantasyworld",
}


# Canonical task labels per inference group category.
INFER_GROUP_TASKS = {
    "image-gen": ("text-to-image", "image-generation"),
    "depth": ("depth-estimation",),
    "3d-scene": ("3d-scene", "reconstruction"),
    "video-open": ("text-to-video", "image-to-video"),
    "interactive-video": ("image-to-video", "interactive-video"),
    "navigation-video": ("navigation-video", "world-model"),
    "world-model": ("world-model", "image-to-video", "interactive-video"),
}


def infer_variant_options(model_id: str) -> tuple[tuple[str, str], ...]:
    """Return display label and exact infer variant options for a model family."""

    family_id = _script_infer_family_id(model_id)
    variants = _ordered_script_variants(family_id)
    if variants:
        options: list[tuple[str, str]] = []
        seen_targets: set[tuple[str, str]] = set()
        for variant in variants:
            try:
                target_model, target_variant, target_label = _studio_runtime_target_for_script_variant(
                    model_id,
                    variant,
                )
            except ValueError:
                continue
            target = (target_model, target_variant)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            exact_target = _studio_entries_by_model_id().get(variant)
            label = INFER_VARIANT_LABELS.get(variant, variant) if exact_target is not None else target_label
            options.append((label, variant))
        return tuple(options)
    core_spec = _studio_inference_spec(model_id)
    if core_spec is not None:
        return tuple((variant.label, variant.variant_id) for variant in core_spec.variants)
    return ()


def infer_variant_label(variant_id: str) -> str:
    """Return a short display label for *variant_id*, falling back to the raw ID."""
    return INFER_VARIANT_LABELS.get(variant_id, variant_id)


def resolve_infer_model_variant(model_id: str, ckpt_type: str | None = None) -> str:
    """Resolve *ckpt_type* to a concrete inference variant ID for *model_id*.

    For script-based models, maps the label or ID to an entry in
    :data:`INFER_MODEL_VARIANTS`. For studio models, delegates to the
    studio inference spec.

    Raises:
        ValueError: When *ckpt_type* is not a valid variant for the model family.
    """
    family_id = _script_infer_family_id(model_id)
    if (
        model_id in INFER_VARIANT_GROUPS
        and family_id in INFER_MODEL_VARIANTS
        and model_id not in INFER_MODEL_VARIANTS
        and (ckpt_type is None or not str(ckpt_type).strip())
    ):
        return model_id

    variants = _ordered_script_variants(family_id)
    if variants is not None:
        if ckpt_type is None or not str(ckpt_type).strip():
            return variants[0]

        requested = str(ckpt_type).strip()
        requested_key = _option_key(requested)
        for variant in variants:
            label = INFER_VARIANT_LABELS.get(variant, variant)
            if requested == variant or requested_key in {_option_key(variant), _option_key(label)}:
                return variant
        valid = ", ".join(INFER_VARIANT_LABELS.get(variant, variant) for variant in variants)
        raise ValueError(f"unknown checkpoint type {ckpt_type!r} for {family_id!r}; choose one of: {valid}")

    core_spec = _studio_inference_spec(model_id)
    if core_spec is not None:
        requested = ckpt_type
        if (requested is None or not str(requested).strip()) and model_id != family_id:
            requested = model_id
        try:
            return core_spec.variant(requested).variant_id
        except ValueError as exc:
            valid = ", ".join(variant.label for variant in core_spec.variants)
            raise ValueError(
                f"unknown checkpoint type {ckpt_type!r} for {family_id!r}; choose one of: {valid}"
            ) from exc

    raise ValueError(f"model infer script is not registered for {model_id!r}")


def _option_key(value: str) -> str:
    """Normalize a variant label or ID for case-insensitive, hyphen-normalized matching."""
    return value.strip().casefold().replace("_", "-")


def _script_infer_family_id(model_id: str) -> str:
    """Resolve a variant *model_id* to its script-infer family ID using :data:`INFER_VARIANT_TO_MODEL`."""
    return INFER_VARIANT_TO_MODEL.get(
        model_id,
        TUI_MODEL_ID_TO_SCRIPT_INFER_FAMILY.get(model_id, model_id),
    )


def is_script_infer_model_id(model_id: str) -> bool:
    """Return whether *model_id* has a registered shell-based inference script."""
    return _script_infer_family_id(model_id) in INFER_MODEL_VARIANTS


def _ordered_script_variants(family_id: str) -> tuple[str, ...] | None:
    """Return the ordered variant tuple for *family_id*, placing the default variant first.

    Returns ``None`` when *family_id* is not registered in :data:`INFER_MODEL_VARIANTS`.
    """
    variants = INFER_MODEL_VARIANTS.get(family_id)
    if variants is None:
        return None
    default_variant = INFER_MODEL_DEFAULT_VARIANTS.get(family_id, variants[0])
    if default_variant not in variants:
        return variants
    return (default_variant, *(variant for variant in variants if variant != default_variant))


def _workload_from_template(template_id: str) -> str:
    """Map a Studio template ID to a short workload label (e.g. ``"i2v"``, ``"world"``)."""
    if template_id == "conditioned-video":
        return "i2v"
    if template_id == "text-video":
        return "t2v"
    if template_id == "scene-3d":
        return "3d"
    if template_id == "depth-geometry":
        return "geometry"
    if template_id in {"embodied-policy", "visual-action"}:
        return "action"
    if template_id == "hosted-api":
        return "api"
    return "world"


def _infer_group_from_workload(workload: str) -> str:
    """Map a workload label to an infer variant group (e.g. ``"i2v"`` → ``"video-open"``)."""
    if workload == "geometry":
        return "depth"
    if workload == "3d":
        return "3d-scene"
    if workload in {"t2v", "i2v"}:
        return "video-open"
    if workload in {"action", "api"}:
        return workload
    return "world-model"


@lru_cache(maxsize=1)
def _studio_entries_by_model_id() -> dict[str, Any]:
    """Index studio catalog entries by ``model_id`` and aliases, cached across calls."""
    try:
        from worldfoundry.studio.inference.listing import _studio_catalog

        entries: dict[str, Any] = {}
        for entry in _studio_catalog():
            entries.setdefault(entry.model_id, entry)
            for alias in getattr(entry, "aliases", ()) or ():
                entries.setdefault(str(alias), entry)
        return entries
    except Exception:
        return {}


def _unique_studio_entries() -> tuple[Any, ...]:
    """Return deduplicated studio catalog entries, one per unique ``model_id``."""
    seen: set[str] = set()
    entries: list[Any] = []
    for entry in _studio_entries_by_model_id().values():
        if entry.model_id in seen:
            continue
        seen.add(entry.model_id)
        entries.append(entry)
    return tuple(entries)


def _tui_model_id_for_studio_entry(entry: Any) -> str:
    """Return the family row that should own a Studio catalog entry in the TUI."""
    explicit = STUDIO_MODEL_TO_TUI_MODEL_ID.get(entry.model_id)
    if explicit is not None:
        return explicit
    candidate_ids = _dedupe_text((entry.model_id, *(getattr(entry, "aliases", ()) or ())))
    for candidate_id in candidate_ids:
        script_family = _script_infer_family_id(candidate_id)
        if script_family in INFER_MODEL_VARIANTS:
            return SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.get(script_family, script_family)
    return entry.model_id


def _studio_entry_for_model(model_id: str) -> Any | None:
    """Return the studio catalog entry for *model_id*, falling back to its family ID."""
    entries = _studio_entries_by_model_id()
    direct_entry = entries.get(model_id)
    if direct_entry is not None:
        return direct_entry
    family_id = INFER_VARIANT_TO_MODEL.get(model_id, model_id)
    return entries.get(family_id)


def _studio_runtime_target_for_script_variant(model_id: str, variant_id: str) -> tuple[str, str, str]:
    """Resolve a legacy shell variant to one concrete Studio model contract.

    Several old shell variants now share one Studio runtime.  Returning the
    concrete model and contract variant lets the TUI deduplicate those choices
    and avoids forwarding obsolete variant IDs to ``workspace_job``.
    """
    entries = _studio_entries_by_model_id()
    family_id = _script_infer_family_id(model_id)
    entry = next(
        (entries[candidate] for candidate in _dedupe_text((variant_id, model_id, family_id)) if candidate in entries),
        None,
    )
    if entry is None:
        raise ValueError(f"no Studio runtime is registered for {model_id!r} variant {variant_id!r}")
    spec = _studio_inference_spec(entry.model_id)
    if spec is None or not spec.variants:
        raise ValueError(f"Studio runtime {entry.model_id!r} has no inference contract")

    requested_key = _option_key(variant_id)
    selected = next(
        (
            variant
            for variant in spec.variants
            if requested_key in {_option_key(variant.variant_id), _option_key(variant.label)}
        ),
        None,
    )
    if selected is None:
        selected = next(
            (variant for variant in spec.variants if requested_key.endswith(_option_key(variant.variant_id))),
            None,
        )
    if selected is None:
        selected = spec.variant(None)
    return entry.model_id, selected.variant_id, selected.label


def _studio_inference_spec(model_id: str):
    """Build an :class:`InferenceSpec` for *model_id*, merging studio catalog metadata with core defaults."""
    entry = _studio_entry_for_model(model_id)
    if entry is None:
        return get_model_inference_spec(model_id)
    from worldfoundry.studio.inference.catalog import template_id_hint

    template_id = template_id_hint(entry)
    return model_inference_spec(
        model_family_id=entry.model_id,
        display_name=entry.display_name,
        default_model_ref=entry.default_model_ref,
        default_load_kwargs=entry.default_load_kwargs,
        default_call_kwargs=entry.default_call_kwargs,
        supports_stream=entry.supports_stream,
        workload_type=_workload_from_template(template_id),
        supported_call_params=(*entry.call_params, *entry.stream_params),
    )


def is_studio_infer_model_id(model_id: str) -> bool:
    """Return whether *model_id* has a studio-runtime inference backend (not a shell script)."""
    if is_script_infer_model_id(model_id):
        return False
    return _studio_entry_for_model(model_id) is not None


def _dedupe_text(values: Iterable[str | None]) -> tuple[str, ...]:
    """Filter out empty and duplicate strings from *values*, preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def infer_model_variant_ids(model_id: str) -> tuple[str, ...]:
    """Return the variant IDs for *model_id* (extracted from :func:`infer_variant_options`)."""
    options = infer_variant_options(model_id)
    return tuple(value for _label, value in options)


def exec_run_infer_sh(argv: list[str] | None = None) -> None:
    """Translate ``run_infer.sh`` flags and exec the shared Studio infer pipeline."""
    import argparse

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--category")
    parser.add_argument("--model")
    parser.add_argument("--ckpt-type", "--weight-type", dest="ckpt_type")
    args, unknown = parser.parse_known_args(raw_argv)

    infer_command = [sys.executable, "-m", "worldfoundry.studio.workspace_job", "infer"]
    if not args.category or not args.model:
        os.execvp(sys.executable, infer_command + raw_argv)

    model_key = str(args.model)
    script_family = _script_infer_family_id(model_key)
    command_model = script_family if script_family in INFER_MODEL_VARIANTS else model_key
    try:
        resolved_variant = resolve_infer_model_variant(model_key, args.ckpt_type)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc

    infer_command.extend(["--model-id", command_model])
    if resolved_variant:
        infer_command.extend(["--variant-id", resolved_variant])
    infer_command.extend(unknown)
    os.execvp(sys.executable, infer_command)


if __name__ == "__main__":
    exec_run_infer_sh()
