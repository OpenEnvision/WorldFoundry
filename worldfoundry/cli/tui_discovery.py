"""Catalog discovery, model-variant resolution, and shell-command builders for the TUI.

This module is the data backbone of the WorldFoundry TUI. It defines the
model/benchmark catalog dataclasses, resolves inference variant
IDs, and builds the shell command tuples that the TUI displays and executes.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.core.inference import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    get_model_inference_spec,
    model_inference_spec,
)
from worldfoundry.evaluation.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_RUNTIME_ENVIRONMENTS_ROOT,
    MODEL_RUNTIME_PROFILES_ROOT,
    MODEL_ZOO_DIR,
    load_manifest,
    load_manifest_collection,
)
from worldfoundry.runtime.env import resolve_cache_dir

# ── Default directories ──────────────────────────────────────────
DEFAULT_MODEL_ZOO_DIR = MODEL_ZOO_DIR
DEFAULT_BENCHMARK_ZOO_DIR = BENCHMARK_ZOO_DIR
DEFAULT_RUNTIME_PROFILES_PATH = MODEL_RUNTIME_PROFILES_ROOT
DEFAULT_CONDA_ENVS_PATH = MODEL_RUNTIME_ENVIRONMENTS_ROOT
DEFAULT_INFERENCE_EVIDENCE_PATH = MODEL_ZOO_DIR.parent / "validation" / "inference_evidence.yaml"
# NOTE: These defaults mirror the global paths from :mod:`worldfoundry.evaluation.utils`.
TUI_ACTIONS = ("infer", "eval", "ui")
"""Available TUI action modes."""

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
        "hunyuan-world-voyager",
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
    "hunyuan-world-voyager": "world-voyager",
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
    "hunyuan-world-voyager": "world-model",
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
# Camera preset label/value pairs for the Astra world model.
ASTRA_CAM_TYPE_OPTIONS = (
    ("1 Forward", "1"),
    ("2 Rotate left", "2"),
    ("3 Rotate right", "3"),
    ("4 Forward left", "4"),
    ("5 Forward right", "5"),
    ("6 S curve", "6"),
    ("7 Left then right", "7"),
)
# Canonical ordering of inference control field IDs for consistent TUI layout.
INFER_CONTROL_ORDER = (
    "prompt",
    "negative_prompt",
    "input",
    "input_dir",
    "video",
    "trajectory_file",
    "task",
    "mode",
    "resize_mode",
    "size",
    "frames",
    "steps",
    "frames_per_generation",
    "guidance_scale",
    "seed",
    "fps",
    "dtype",
    "max_sequence_length",
    "cam_type",
    "interactions",
    "output_formats",
    "trajectory",
    "angle",
    "distance",
    "orbit_radius",
    "zoom_ratio",
    "alpha_threshold",
    "static_scene",
    "low_vram",
    "disable_lora",
    "vis_rendering",
    "offload_t5",
    "offload_transformer_during_vae",
    "offload_vae",
    "output_path",
)


@dataclass(frozen=True)
class InferControlSpec:
    """Specification for a single inference control field shown in the TUI.

    Attributes:
        field_id: Canonical field identifier (e.g. ``"prompt"``, ``"seed"``).
        label: Display label for the field.
        placeholder: Placeholder text shown in the input widget.
        required: Whether the field must be filled before running.
        help: Help text for the field (not currently rendered in the TUI).
    """

    field_id: str
    label: str
    placeholder: str = ""
    required: bool = False
    help: str = ""


# Default InferControlSpec instances keyed by field ID, used before variant overrides.
_INFER_CONTROL_DEFAULT_SPECS = {
    "prompt": InferControlSpec("prompt", "Prompt", "Describe the scene or action..."),
    "negative_prompt": InferControlSpec("negative_prompt", "Negative prompt", "Optional negative prompt"),
    "input": InferControlSpec("input", "Input path", "image/video path"),
    "input_dir": InferControlSpec("input_dir", "Input directory", "directory or config path"),
    "video": InferControlSpec("video", "Video path", "optional video path"),
    "trajectory_file": InferControlSpec("trajectory_file", "Trajectory file", "custom trajectory JSON"),
    "task": InferControlSpec("task", "Task", "task override"),
    "mode": InferControlSpec("mode", "Mode", "mode override"),
    "resize_mode": InferControlSpec("resize_mode", "Resize mode", "center_crop"),
    "size": InferControlSpec("size", "Size", "832*480"),
    "frames": InferControlSpec("frames", "Frames", "frame count"),
    "steps": InferControlSpec("steps", "Steps", "sampling steps"),
    "frames_per_generation": InferControlSpec("frames_per_generation", "Frame chunk", "8"),
    "guidance_scale": InferControlSpec("guidance_scale", "Guidance scale", "CFG / guidance"),
    "seed": InferControlSpec("seed", "Seed", "random seed"),
    "fps": InferControlSpec("fps", "FPS", "output fps"),
    "dtype": InferControlSpec("dtype", "Dtype", "float16 / bfloat16"),
    "max_sequence_length": InferControlSpec("max_sequence_length", "Max sequence", "text encoder length"),
    "cam_type": InferControlSpec("cam_type", "Camera type", ""),
    "interactions": InferControlSpec("interactions", "Actions", "forward,camera_l,..."),
    "output_formats": InferControlSpec("output_formats", "Output formats", "video,spz,ply"),
    "trajectory": InferControlSpec("trajectory", "Trajectory", "tilt_up / move_right"),
    "angle": InferControlSpec("angle", "Angle", "15"),
    "distance": InferControlSpec("distance", "Distance", "0.1"),
    "orbit_radius": InferControlSpec("orbit_radius", "Orbit radius", "1.0"),
    "zoom_ratio": InferControlSpec("zoom_ratio", "Zoom ratio", "1.0"),
    "alpha_threshold": InferControlSpec("alpha_threshold", "Alpha threshold", "1.0"),
    "static_scene": InferControlSpec("static_scene", "Static scene", "1 / 0"),
    "low_vram": InferControlSpec("low_vram", "Low VRAM", "1 / 0"),
    "disable_lora": InferControlSpec("disable_lora", "Disable LoRA", "1 / 0"),
    "vis_rendering": InferControlSpec("vis_rendering", "Vis rendering", "1 / 0"),
    "offload_t5": InferControlSpec("offload_t5", "Offload T5", "1 / 0"),
    "offload_transformer_during_vae": InferControlSpec(
        "offload_transformer_during_vae",
        "Offload transformer",
        "1 / 0",
    ),
    "offload_vae": InferControlSpec("offload_vae", "Offload VAE", "1 / 0"),
    "output_path": InferControlSpec("output_path", "Artifact path", "optional exact artifact path"),
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


def infer_control_specs(model_id: str, ckpt_type: str | None = None) -> tuple[InferControlSpec, ...]:
    """Return the ordered :class:`InferControlSpec` list for the resolved variant.

    Applies variant-specific label/placeholder overrides from
    ``_infer_control_text_overrides``.
    """
    fields = infer_control_fields(model_id, ckpt_type)
    try:
        variant = resolve_infer_model_variant(model_id, ckpt_type)
    except ValueError:
        variant = str(ckpt_type or model_id)
    overrides = _infer_control_text_overrides(variant)
    specs: list[InferControlSpec] = []
    for field_id in fields:
        spec = _INFER_CONTROL_DEFAULT_SPECS.get(
            field_id, InferControlSpec(field_id, field_id.replace("_", " ").title())
        )
        override = overrides.get(field_id)
        if override:
            spec = replace(spec, **override)
        specs.append(spec)
    return tuple(specs)


def infer_control_fields(model_id: str, ckpt_type: str | None = None) -> tuple[str, ...]:
    """Return the set of inference control field IDs applicable to the resolved variant.

    Each model variant exposes a different subset of control fields. The returned
    tuple is sorted according to :data:`INFER_CONTROL_ORDER` for consistent TUI layout.

    NOTE: When no specific variant match is found, falls back to ``{"input"}``.
    """
    # ── Studio-runtime models use spec-driven fields ──
    if not is_script_infer_model_id(model_id):
        studio_fields = _control_fields_from_studio_spec(model_id)
        if studio_fields:
            return studio_fields
    variant = resolve_infer_model_variant(model_id, ckpt_type)
    fields: set[str]
    video_advanced_fields = {"negative_prompt", "guidance_scale", "dtype", "max_sequence_length"}
    if variant.startswith("open-magvit2"):
        fields = {"prompt", "steps", "guidance_scale", "output_path"}
    elif variant.startswith("show-o"):
        fields = {"prompt", "size", "steps", "guidance_scale", "output_path"}
    elif INFER_VARIANT_GROUPS.get(variant) == "depth":
        fields = {"input"}
    elif variant in {"vggt-1b", "vggt-omega", "cut3r"}:
        fields = {"input", "input_dir", "task"}
    elif variant == "gen3c-cosmos-7b":
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "trajectory"}
    elif variant in {"pi3", "pi3x", "loger", "loger-star"}:
        fields = {"input", "video", "task", "mode", "interactions", "fps"}
    elif variant == "flash-world":
        fields = {
            "input_dir",
            "input",
            "prompt",
            "size",
            "frames",
            "fps",
            "interactions",
            "output_formats",
            "offload_t5",
            "offload_transformer_during_vae",
            "offload_vae",
        }
    elif variant.startswith("wan2.1-t2v"):
        fields = {"prompt", "size", "frames", "steps", "seed", "fps", "output_path"}
    elif variant.startswith("wan2.1-i2v"):
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "output_path"}
    elif variant == "wan2.2-ti2v-5b":
        fields = {"input", "prompt", "mode", "size", "frames", "steps", "seed", "output_path"}
    elif variant.startswith("matrix-game-2"):
        fields = {"input", "mode", "frames", "seed", "fps", "interactions", "output_path"}
    elif variant == "matrix-game-3":
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "yume-1p5-t2v":
        fields = {"prompt", "task", "size", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "yume-1p5-v2v":
        fields = {"video", "prompt", "task", "size", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "yume-1p5-i2v":
        fields = {"input", "prompt", "task", "size", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant in {"lingbot-world-v2", "lingbot-world-infinity"}:
        fields = {"input", "input_dir", "prompt", "frames", "seed", "output_path"}
    elif variant in {
        LINGBOT_VARIANT_BASE_CAM,
        LINGBOT_VARIANT_BASE_ACT_PREVIEW,
        LINGBOT_VARIANT_FAST,
    } or variant.startswith("lingbot-world"):
        fields = {"input", "input_dir", "prompt", "mode", "frames", "steps", "seed", "interactions", "output_path"}
    elif variant in {
        "cogvideox-2b-t2v",
        "cogvideox-5b-t2v",
        "cogvideox-1.5-5b-t2v",
    }:
        fields = {"prompt", "size", "frames", "steps", "seed", "fps", "output_path"} | video_advanced_fields
    elif variant in {"zeroscope-576w", "zeroscope-xl"}:
        fields = {"prompt", "size", "frames", "steps", "seed", "fps", "output_path"}
    elif variant in {"sana-video-2b-480p", "sana-video-2b-720p", "longsana-video-2b-480p"}:
        fields = {"prompt", "frames", "steps", "guidance_scale", "seed", "fps", "output_path"}
    elif variant in {"cogvideox-5b-i2v", "cogvideox-1.5-5b-i2v"}:
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "output_path"} | video_advanced_fields
    elif variant in {"dynamicrafter-512-i2v", "dynamicrafter-1024-i2v", "ltx-video-i2v"}:
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "output_path"}
    elif variant == "motionctrl":
        fields = {"prompt", "size", "steps", "seed", "fps", "interactions", "guidance_scale", "output_path"}
    elif variant.startswith("animatediff"):
        fields = {
            "prompt",
            "negative_prompt",
            "size",
            "frames",
            "steps",
            "seed",
            "fps",
            "guidance_scale",
            "output_path",
        }
    elif variant.startswith("cosmos-predict2.5"):
        fields = {"input", "prompt", "mode", "size", "frames", "steps", "seed", "fps", "output_path"}
    elif variant == "astra":
        fields = {"input", "prompt", "frames", "frames_per_generation", "fps", "cam_type", "output_path"}
    elif variant == "infinite-world":
        fields = {"input", "prompt", "frames", "seed", "fps", "interactions", "output_path"}
    elif variant == "lyra2":
        fields = {"input", "prompt", "size", "steps", "guidance_scale", "seed", "fps", "interactions", "output_path"}
    elif variant == "hunyuan-world-voyager":
        fields = {"input", "input_dir", "prompt", "frames", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "neoverse":
        fields = {
            "input",
            "video",
            "prompt",
            "negative_prompt",
            "trajectory",
            "trajectory_file",
            "mode",
            "resize_mode",
            "size",
            "frames",
            "steps",
            "guidance_scale",
            "seed",
            "fps",
            "angle",
            "distance",
            "orbit_radius",
            "zoom_ratio",
            "alpha_threshold",
            "static_scene",
            "low_vram",
            "disable_lora",
            "vis_rendering",
            "output_path",
        }
    elif variant == "worldcam":
        fields = {"input", "prompt", "size", "steps", "seed", "fps", "output_path"}
    elif variant == "wow":
        fields = {"input", "prompt", "frames", "steps", "seed", "fps", "output_path"}
    elif variant == "recammaster":
        fields = {"video", "prompt", "size", "frames", "fps", "trajectory", "output_path"}
    elif variant == "hunyuan-gamecraft":
        fields = {"input", "prompt", "size", "frames", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "hunyuan-worldplay":
        fields = {"input", "prompt", "mode", "frames", "steps", "seed", "fps", "interactions", "output_path"}
    elif variant == "hunyuan-mirror":
        fields = {"input", "input_dir", "output_path"}
    elif variant == "hy-world-2.0":
        fields = {"input", "input_dir", "task", "output_path"}
    elif variant == "lyra1":
        fields = {
            "input",
            "prompt",
            "mode",
            "size",
            "frames",
            "steps",
            "guidance_scale",
            "seed",
            "fps",
            "interactions",
            "output_path",
        }
    elif variant == "worldfm":
        fields = {"input", "fps", "output_path"}
    elif variant == "inspatio-world":
        fields = {"video", "prompt", "trajectory", "trajectory_file", "fps", "output_path"}
    elif variant == "longvie2":
        fields = {"input", "video", "prompt", "seed", "fps", "output_path"}
    elif variant.startswith("fantasy-world"):
        fields = {"input", "prompt", "frames", "steps", "fps", "output_path"}
    else:
        # NOTE: Unknown variant — minimal default controls
        fields = {"input"}
    # ── Sort fields according to the canonical order ──
    return tuple(field for field in INFER_CONTROL_ORDER if field in fields)


def _infer_control_text_overrides(variant: str) -> dict[str, dict[str, str]]:
    """Return variant-specific label and placeholder overrides for inference control fields."""
    if variant.startswith(("depth-anything", "da3", "moge")):
        return {
            "input": {
                "label": "Input image",
                "placeholder": "path/to/image.png",
            }
        }
    if variant in {"vggt-1b", "vggt-omega", "cut3r"}:
        return {
            "input": {
                "label": "Input image or folder",
                "placeholder": "image path; use Input directory for sequences",
            },
            "input_dir": {
                "label": "Input directory",
                "placeholder": "image sequence directory",
            },
            "task": {
                "label": "Export task",
                "placeholder": "official / two_stage_3dgs",
            },
        }
    if variant == "flash-world":
        return {
            "input_dir": {
                "label": "JSON config",
                "placeholder": "FlashWorld examples/1.json",
            },
            "input": {
                "label": "Reference image",
                "placeholder": "optional image override",
            },
            "output_formats": {
                "placeholder": "video,spz,ply",
            },
        }
    if variant in {"lingbot-world-v2", "lingbot-world-infinity"}:
        return {
            "input": {
                "label": "Reference image",
                "placeholder": "path/to/image.jpg",
            },
            "input_dir": {
                "label": "Action directory",
                "placeholder": "directory with poses.npy and intrinsics.npy",
            },
        }
    if variant.startswith("lingbot-world") or variant in {
        LINGBOT_VARIANT_BASE_CAM,
        LINGBOT_VARIANT_BASE_ACT_PREVIEW,
        LINGBOT_VARIANT_FAST,
    }:
        return {
            "input": {
                "label": "Reference image",
                "placeholder": "examples/05/image.jpg",
            },
            "input_dir": {
                "label": "Action/example directory",
                "placeholder": "examples/05",
            },
            "mode": {
                "label": "Control mode",
                "placeholder": "cam / act",
            },
        }
    if variant in {"recammaster", "inspatio-world"}:
        return {
            "video": {
                "label": "Input video",
                "placeholder": "path/to/video.mp4",
            },
            "trajectory": {
                "label": "Camera trajectory",
                "placeholder": "100,100,0,0,30" if variant == "recammaster" else "trajectory preset",
            },
        }
    if variant == "neoverse":
        return {
            "input": {"label": "Reference image"},
            "video": {"label": "Input video"},
            "mode": {
                "label": "Trajectory mode",
                "placeholder": "relative / global",
            },
            "interactions": {
                "label": "Camera action",
            },
            "trajectory": {
                "placeholder": "tilt_up / move_right / orbit",
            },
            "trajectory_file": {
                "placeholder": "custom trajectory JSON",
            },
        }
    if variant == "astra":
        return {
            "input": {
                "label": "Input image",
                "placeholder": "path/to/image.png",
            },
            "cam_type": {
                "label": "Camera preset",
            },
            "frames_per_generation": {
                "placeholder": "8",
            },
        }
    if variant.startswith("matrix-game"):
        return {
            "input": {
                "label": "Reference image",
            },
            "interactions": {
                "label": "Game actions",
                "placeholder": "forward,left,right",
            },
        }
    if variant.startswith("yume-1p5"):
        return {
            "interactions": {
                "label": "Camera actions",
            },
            "task": {
                "label": "Generation task",
            },
        }
    if variant == "worldfm":
        return {
            "input": {
                "label": "Panorama image",
                "placeholder": "path/to/panorama.png",
            }
        }
    if variant == "hunyuan-mirror":
        return {
            "input": {
                "label": "Input path",
                "placeholder": "image/video/directory path",
            },
            "input_dir": {
                "label": "Input directory",
                "placeholder": "image sequence directory",
            },
        }
    if variant == "hunyuan-world-voyager":
        return {
            "input": {
                "label": "Reference image",
                "placeholder": "ref_image.png",
            },
            "input_dir": {
                "label": "Condition directory",
                "placeholder": "examples/case1",
            },
            "interactions": {
                "label": "Camera path",
            },
        }
    if variant == "hy-world-2.0":
        return {
            "input": {
                "label": "Input path",
                "placeholder": "image directory or image path",
            },
            "input_dir": {
                "label": "Input directory",
                "placeholder": "image sequence directory",
            },
            "task": {
                "label": "Generation task",
                "placeholder": "worldrecon / panorama",
            },
        }
    if variant == "worldcam":
        return {
            "input": {
                "label": "Conditioning video",
                "placeholder": "path/to/video.mp4",
            }
        }
    if variant == "wan2.2-ti2v-5b":
        return {
            "input": {
                "label": "Input image",
                "placeholder": "path/to/image.png",
            },
            "mode": {
                "label": "Generation mode",
                "placeholder": "ti2v-5B",
            },
        }
    if variant.startswith("wan2.1-i2v") or variant in {"wan2.2-ti2v-5b", "gen3c-cosmos-7b"}:
        return {
            "input": {
                "label": "Input image",
                "placeholder": "path/to/image.png",
            }
        }
    if variant.endswith("-i2v") or "-i2v" in variant:
        return {
            "input": {
                "label": "Input image",
                "placeholder": "path/to/image.png",
            }
        }
    if variant.endswith("-v2v") or "-v2v" in variant:
        return {
            "video": {
                "label": "Input video",
                "placeholder": "path/to/video.mp4",
            }
        }
    return {}


def resolve_infer_model_variant(model_id: str, ckpt_type: str | None = None) -> str:
    """Resolve *ckpt_type* to a concrete inference variant ID for *model_id*.

    For script-based models, maps the label or ID to an entry in
    :data:`INFER_MODEL_VARIANTS`. For studio models, delegates to the
    studio inference spec.

    Raises:
        ValueError: When *ckpt_type* is not a valid variant for the model family.
    """
    family_id = _script_infer_family_id(model_id)
    if model_id in INFER_VARIANT_GROUPS and family_id in INFER_MODEL_VARIANTS and model_id not in INFER_MODEL_VARIANTS:
        if ckpt_type is None or not str(ckpt_type).strip():
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
        from worldfoundry.studio.studio_catalog import _studio_catalog

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
    try:
        from worldfoundry.studio.studio_catalog import _template_id_hint
    except Exception:
        template_id = ""
    else:
        template_id = _template_id_hint(entry)
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


def _normalise_infer_field(value: str) -> str:
    """Normalize an inference field ID for matching: strip, lowercase, replace hyphens with underscores."""
    return str(value or "").strip().lower().replace("-", "_")


def _control_fields_from_studio_spec(model_id: str) -> tuple[str, ...]:
    """Extract inference control field IDs from the studio inference spec's task inputs."""
    spec = _studio_inference_spec(model_id)
    entry = _studio_entry_for_model(model_id)
    if spec is None or entry is None:
        return ()
    try:
        task = spec.task(None)
    except ValueError:
        return ()
    fields: set[str] = set()
    for field in task.inputs:
        field_id = _normalise_infer_field(field.field_id)
        target = _normalise_infer_field(field.target)
        if target == "prompt" or field_id == "prompt":
            fields.add("prompt")
        elif field_id in {"negative_prompt", "negative"}:
            fields.add("negative_prompt")
        elif target == "input_path" or field_id in {
            "input",
            "input_path",
            "image",
            "images",
            "image_path",
            "video",
            "videos",
            "video_path",
            "data_path",
        }:
            fields.add("video" if field_id in {"video", "videos", "video_path"} else "input")
        elif field_id in {"frames", "num_frames", "video_length"}:
            fields.add("frames")
        elif field_id == "fps":
            fields.add("fps")
        elif field_id in {"height", "width", "size", "user_height", "user_width", "resize_h", "resize_w"}:
            fields.add("size")
        elif field_id in {"steps", "num_inference_steps", "sampling_steps", "infer_steps", "num_steps"}:
            fields.add("steps")
        elif field_id in {"guidance", "guidance_scale", "cfg_scale", "scale"}:
            fields.add("guidance_scale")
        elif field_id == "seed":
            fields.add("seed")
        elif field_id in {"interactions", "interaction", "interaction_signal", "action"}:
            fields.add("interactions")
        elif field_id == "output_path":
            fields.add("output_path")
        elif field_id in INFER_CONTROL_ORDER:
            fields.add(field_id)
    return tuple(field for field in INFER_CONTROL_ORDER if field in fields)


def is_studio_infer_model_id(model_id: str) -> bool:
    """Return whether *model_id* has a studio-runtime inference backend (not a shell script)."""
    if is_script_infer_model_id(model_id):
        return False
    return _studio_entry_for_model(model_id) is not None


# ── Catalog row dataclasses ─────────────────────────────────────────
# NOTE: Each frozen dataclass represents a single row in the TUI's catalog
# tables (models, benchmarks, runtime profiles).


@dataclass(frozen=True)
class ModelCatalogRow:
    """Row representation of a model zoo entry for the TUI catalog tables.

    Attributes:
        model_id: Unique model family identifier.
        name: Human-readable display name.
        tasks: Supported task labels (e.g. ``"text-to-video"``).
        provider: Optional model provider / source label.
        source_status: Repository availability status.
        integration_status: Integration readiness level.
        runner_status: Runner verification status.
        runner_kind: Runner type — ``"infer_script"`` or ``"studio_runtime"``.
        runnable_variants: Variant IDs that have runnable infer scripts.
        runtime_profile: Optional runtime profile identifier.
        min_vram_gb: Minimum VRAM requirement in GB, or ``None``.
        hf_repo_ids: HuggingFace repo identifiers for weights.
        official_repo_url: Official repository URL, or ``None``.
        requires_auth: Whether HuggingFace gated access is required.
        notes: Free-form notes tuple.
    """

    model_id: str
    name: str
    tasks: tuple[str, ...]
    provider: str | None
    source_status: str
    integration_status: str
    runner_status: str
    runner_kind: str
    runnable_variants: tuple[str, ...]
    runtime_profile: str | None
    min_vram_gb: float | None
    hf_repo_ids: tuple[str, ...]
    official_repo_url: str | None
    requires_auth: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "tasks": list(self.tasks),
            "provider": self.provider,
            "source_status": self.source_status,
            "integration_status": self.integration_status,
            "runner_status": self.runner_status,
            "runner_kind": self.runner_kind,
            "runnable_variants": list(self.runnable_variants),
            "runtime_profile": self.runtime_profile,
            "min_vram_gb": self.min_vram_gb,
            "hf_repo_ids": list(self.hf_repo_ids),
            "official_repo_url": self.official_repo_url,
            "requires_auth": self.requires_auth,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkCatalogRow:
    """Row representation of a benchmark zoo entry for the TUI catalog tables.

    Attributes:
        benchmark_id: Unique benchmark identifier.
        name: Human-readable display name.
        domains: Evaluation domain labels (e.g. ``"dynamic_photorealistic"``).
        modalities: Supported modality labels.
        tags: Additional tagging metadata.
        source_status: Dataset availability status.
        integration_status: Integration readiness level.
        verification_status: Verification / correctness status.
        maturity: Maturity level label.
        official_benchmark_verified: Whether the official benchmark was verified.
        integration_evidence: Whether integration evidence is available.
        leaderboard_valid: Whether the benchmark supports leaderboard submission.
        runner_target: Target runner identifier, or ``None``.
        install_profile: Conda / pip install profile, or ``None``.
        metrics: Metric identifiers supported by this benchmark.
        hf_dataset_ids: HuggingFace dataset identifiers.
        official_repo_url: Official repository URL, or ``None``.
        paper_url: Paper URL, or ``None``.
        requires_auth: Whether gated dataset access is required.
        notes: Free-form notes tuple.
    """

    benchmark_id: str
    name: str
    domains: tuple[str, ...]
    modalities: tuple[str, ...]
    tags: tuple[str, ...]
    source_status: str
    integration_status: str
    verification_status: str
    maturity: str
    official_benchmark_verified: bool
    integration_evidence: bool
    leaderboard_valid: bool
    runner_target: str | None
    install_profile: str | None
    metrics: tuple[str, ...]
    hf_dataset_ids: tuple[str, ...]
    official_repo_url: str | None
    paper_url: str | None
    requires_auth: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "domains": list(self.domains),
            "modalities": list(self.modalities),
            "tags": list(self.tags),
            "source_status": self.source_status,
            "integration_status": self.integration_status,
            "verification_status": self.verification_status,
            "maturity": self.maturity,
            "official_benchmark_verified": self.official_benchmark_verified,
            "integration_evidence": self.integration_evidence,
            "leaderboard_valid": self.leaderboard_valid,
            "runner_target": self.runner_target,
            "install_profile": self.install_profile,
            "metrics": list(self.metrics),
            "hf_dataset_ids": list(self.hf_dataset_ids),
            "official_repo_url": self.official_repo_url,
            "paper_url": self.paper_url,
            "requires_auth": self.requires_auth,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RuntimeProfileRow:
    """Row representation of a runtime profile entry for the TUI catalog.

    Attributes:
        profile_id: Profile identifier (typically matches a model ID).
        task_family: Task family the profile covers, or ``None``.
        artifact_kind: Kind of artifact produced, or ``None``.
        backend_stage: Backend stage label.
        integration_status: Integration readiness level.
        runtime_status: Runtime / execution readiness level.
        conda_env_name: Conda environment name, or ``None``.
        conda_env_prefix: Conda environment prefix path, or ``None``.
        conda_driver_status: Conda driver availability status, or ``None``.
        conda_env_exists: Whether the Conda environment directory exists.
        conda_python_exists: Whether ``python`` exists inside the Conda environment.
        validation_imports: Import paths validated for the environment.
        notes: Free-form notes tuple.
    """

    profile_id: str
    task_family: str | None
    artifact_kind: str | None
    backend_stage: str
    integration_status: str
    runtime_status: str
    conda_env_name: str | None
    conda_env_prefix: str | None
    conda_driver_status: str | None
    conda_env_exists: bool
    conda_python_exists: bool
    validation_imports: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "task_family": self.task_family,
            "artifact_kind": self.artifact_kind,
            "backend_stage": self.backend_stage,
            "integration_status": self.integration_status,
            "runtime_status": self.runtime_status,
            "conda_env_name": self.conda_env_name,
            "conda_env_prefix": self.conda_env_prefix,
            "conda_driver_status": self.conda_driver_status,
            "conda_env_exists": self.conda_env_exists,
            "conda_python_exists": self.conda_python_exists,
            "validation_imports": list(self.validation_imports),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TuiCatalog:
    """Aggregated catalog container for all TUI data.

    Attributes:
        models: Ordered model catalog rows.
        benchmarks: Ordered benchmark catalog rows.
        runtime_profiles: Ordered runtime profile rows.
        conda_status: Mapping of Conda availability diagnostics, or ``None``.
    """

    models: tuple[ModelCatalogRow, ...]
    benchmarks: tuple[BenchmarkCatalogRow, ...]
    runtime_profiles: tuple[RuntimeProfileRow, ...] = ()
    conda_status: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "models": [row.to_dict() for row in self.models],
            "benchmarks": [row.to_dict() for row in self.benchmarks],
            "runtime_profiles": [row.to_dict() for row in self.runtime_profiles],
            "conda_status": dict(self.conda_status or {}),
            "summary": {
                "models": len(self.models),
                "benchmarks": len(self.benchmarks),
                "runtime_profiles": len(self.runtime_profiles),
                "runnable_models": sum(1 for row in self.models if row.runner_kind == "runnable_runner"),
                "integrated_benchmarks": sum(1 for row in self.benchmarks if row.integration_status == "integrated"),
                "leaderboard_ready_benchmarks": sum(1 for row in self.benchmarks if row.leaderboard_valid),
                "conda_envs_existing": int((self.conda_status or {}).get("envs_existing", 0)),
            },
        }


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


def _model_tasks(entry) -> tuple[str, ...]:
    """Collect deduplicated task labels from a model entry and its variants."""
    variant_tasks = [variant.task for variant in entry.variants if variant.task]
    return _dedupe_text((*entry.tasks, *variant_tasks))


def _runnable_variant_ids(entry) -> tuple[str, ...]:
    """Return variant IDs that have runnable runner entries."""
    return _dedupe_text(variant.variant_id for variant in entry.variants if variant.is_runnable_runner_entry)


def _model_runtime_profile(entry) -> str | None:
    """Return the first non-empty runtime profile identifier from an entry or its variants."""
    if entry.runtime_profile:
        return entry.runtime_profile
    for variant in entry.variants:
        if variant.runtime_profile:
            return variant.runtime_profile
    return None


def _model_min_vram(entry) -> float | None:
    """Return the minimum VRAM requirement across the entry and its variants, or ``None``."""
    values = [entry.min_vram_gb, *(variant.min_vram_gb for variant in entry.variants)]
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return min(concrete)


def _model_to_row(entry) -> ModelCatalogRow:
    """Convert a model zoo registry entry to a :class:`ModelCatalogRow`."""
    return ModelCatalogRow(
        model_id=entry.model_id,
        name=entry.name or entry.model_id,
        tasks=_model_tasks(entry),
        provider=entry.provider,
        source_status=entry.source_status,
        integration_status=entry.integration_status,
        runner_status=entry.verification_status,
        runner_kind=entry.runner_entry_kind,
        runnable_variants=_runnable_variant_ids(entry),
        runtime_profile=_model_runtime_profile(entry),
        min_vram_gb=_model_min_vram(entry),
        hf_repo_ids=entry.hf_repo_ids,
        official_repo_url=entry.official_repo_url,
        requires_auth=entry.requires_auth,
        notes=entry.notes,
    )


def _normalised_evidence_id(value: object) -> str:
    """Normalize model identifiers for evidence matching across punctuation aliases."""
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


@lru_cache(maxsize=1)
def _verified_inference_evidence_ids() -> frozenset[str]:
    """Load model IDs backed by recorded real-inference evidence."""
    try:
        payload = load_manifest(DEFAULT_INFERENCE_EVIDENCE_PATH)
    except (OSError, ValueError):
        return frozenset()
    verified = {
        _normalised_evidence_id(model_id)
        for model_id in payload.get("verified_methods") or ()
        if _normalised_evidence_id(model_id)
    }
    for model_id, item in (payload.get("methods") or {}).items():
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").strip().casefold()
        if status not in {"verified", "pass", "passed", "complete", "completed"}:
            continue
        verified.add(_normalised_evidence_id(model_id))
        verified.update(
            _normalised_evidence_id(alias) for alias in item.get("aliases") or () if _normalised_evidence_id(alias)
        )
    return frozenset(verified)


def _inference_runner_status(
    model_id: str,
    variants: Sequence[str] = (),
    aliases: Sequence[str] = (),
) -> str:
    """Distinguish recorded real inference from a structurally valid contract."""
    verified = _verified_inference_evidence_ids()
    if any(_normalised_evidence_id(value) in verified for value in (model_id, *aliases)):
        return "verified"
    variant_matches = [_normalised_evidence_id(variant) in verified for variant in variants]
    if variant_matches and all(variant_matches):
        return "verified"
    if any(variant_matches):
        return "partially_verified"
    return "contract_ready"


def _infer_virtual_model_row(model_id: str, group: str) -> ModelCatalogRow:
    """Build a :class:`ModelCatalogRow` for a script-infer model not present in the zoo registry."""
    return ModelCatalogRow(
        model_id=model_id,
        name=model_id,
        tasks=INFER_GROUP_TASKS.get(group, (group,)),
        provider=None,
        source_status="in_tree",
        integration_status="integrated",
        runner_status=_inference_runner_status(model_id, INFER_MODEL_VARIANTS.get(model_id, ())),
        runner_kind="infer_script",
        runnable_variants=INFER_MODEL_VARIANTS.get(model_id, ()),
        runtime_profile=None,
        min_vram_gb=None,
        hf_repo_ids=(),
        official_repo_url=None,
        requires_auth=False,
        notes=(f"scripts/inference/run_infer.sh --category {group}",),
    )


def _merge_script_infer_row(existing: ModelCatalogRow | None, script_row: ModelCatalogRow) -> ModelCatalogRow:
    """Merge a script-infer row into an existing model row, overlaying infer-specific metadata."""
    if existing is None:
        return script_row
    return replace(
        existing,
        integration_status="integrated",
        runner_status=script_row.runner_status,
        tasks=_dedupe_text((*script_row.tasks, *existing.tasks)),
        runnable_variants=script_row.runnable_variants,
        runtime_profile=existing.runtime_profile or script_row.runtime_profile,
        min_vram_gb=existing.min_vram_gb if existing.min_vram_gb is not None else script_row.min_vram_gb,
        hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *script_row.hf_repo_ids)),
        notes=_dedupe_text((*existing.notes, *script_row.notes)),
    )


def _studio_infer_model_row(entry: Any) -> ModelCatalogRow:
    """Build a :class:`ModelCatalogRow` from a studio catalog entry with runtime metadata."""
    try:
        from worldfoundry.studio.studio_catalog import _template_id_hint
    except Exception:
        template_id = ""
    else:
        template_id = _template_id_hint(entry)
    workload = _workload_from_template(template_id)
    group = _infer_group_from_workload(workload)
    spec = _studio_inference_spec(entry.model_id)
    task_names = tuple(task.label or task.task_id for task in spec.tasks) if spec is not None else ()
    variants = tuple(variant.variant_id for variant in spec.variants) if spec is not None else ("default",)
    return ModelCatalogRow(
        model_id=entry.model_id,
        name=entry.display_name or entry.model_id,
        tasks=_dedupe_text((*INFER_GROUP_TASKS.get(group, (group,)), *task_names)),
        provider=None,
        source_status="in_tree",
        integration_status="integrated",
        runner_status=_inference_runner_status(
            entry.model_id,
            variants,
            tuple(getattr(entry, "aliases", ()) or ()),
        ),
        runner_kind="studio_runtime",
        runnable_variants=variants,
        runtime_profile=None,
        min_vram_gb=None,
        hf_repo_ids=(entry.default_model_ref,) if entry.default_model_ref and "/" in entry.default_model_ref else (),
        official_repo_url=None,
        requires_auth=False,
        notes=("studio_runtime", "worldfoundry.studio.workspace_job infer"),
    )


def _row_has_studio_runtime(row: ModelCatalogRow) -> bool:
    """Return whether *row* carries a ``studio_runtime`` runner kind or note."""
    return row.runner_kind == "studio_runtime" or any("studio_runtime" in note for note in row.notes)


def is_infer_model_row(row: ModelCatalogRow) -> bool:
    """Return whether *row* supports inference (script-based or studio-runtime)."""
    script_family = _script_infer_family_id(row.model_id)
    if script_family in INFER_MODEL_VARIANTS:
        tui_family = SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.get(script_family, script_family)
        return row.model_id == tui_family and bool(infer_variant_options(row.model_id))
    return _row_has_studio_runtime(row)


def infer_model_variant_ids(model_id: str) -> tuple[str, ...]:
    """Return the variant IDs for *model_id* (extracted from :func:`infer_variant_options`)."""
    options = infer_variant_options(model_id)
    return tuple(value for _label, value in options)


def _merge_studio_infer_row(existing: ModelCatalogRow | None, studio_row: ModelCatalogRow) -> ModelCatalogRow:
    """Merge a studio-runtime row into an existing model row, enriching metadata where applicable."""
    if existing is None:
        return studio_row
    if is_script_infer_model_id(existing.model_id):
        return replace(
            existing,
            integration_status="integrated",
            runner_status=existing.runner_status,
            tasks=_dedupe_text((*existing.tasks, *studio_row.tasks)),
            hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *studio_row.hf_repo_ids)),
            notes=_dedupe_text((*existing.notes, "workspace_job infer available")),
        )
    return replace(
        existing,
        integration_status="integrated",
        runner_status=studio_row.runner_status,
        runner_kind="studio_runtime",
        tasks=_dedupe_text((*existing.tasks, *studio_row.tasks)),
        runnable_variants=studio_row.runnable_variants,
        hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *studio_row.hf_repo_ids)),
        notes=_dedupe_text((*existing.notes, *studio_row.notes)),
    )


def _benchmark_to_row(entry) -> BenchmarkCatalogRow:
    """Convert a benchmark zoo registry entry to a :class:`BenchmarkCatalogRow`."""
    return BenchmarkCatalogRow(
        benchmark_id=entry.benchmark_id,
        name=entry.name or entry.benchmark_id,
        domains=entry.domains,
        modalities=entry.modalities,
        tags=entry.tags,
        source_status=entry.source_status,
        integration_status=entry.integration_status,
        verification_status=entry.verification_status,
        maturity=entry.maturity or "planned",
        official_benchmark_verified=entry.official_benchmark_verified,
        integration_evidence=entry.integration_evidence,
        leaderboard_valid=entry.leaderboard_valid,
        runner_target=entry.runner_target,
        install_profile=entry.install_profile,
        metrics=tuple(metric.metric_id for metric in entry.metrics),
        hf_dataset_ids=entry.hf_dataset_ids,
        official_repo_url=entry.official_repo_url,
        paper_url=entry.paper_url,
        requires_auth=entry.requires_auth,
        notes=entry.notes,
    )


def _read_manifest_mapping(path: Path, *, item_key: str) -> dict[str, Any]:
    """Read a YAML manifest file or collection directory into a mapping.

    Raises:
        TypeError: When the parsed payload is not a ``dict``.
    """
    if not path.exists():
        return {}
    payload = load_manifest_collection(path, item_key=item_key) if path.is_dir() else load_manifest(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return payload


def _runtime_profile_path(runtime_profile_dir: str | Path | None) -> Path:
    """Resolve the runtime profile YAML path from an override directory or the default location."""
    if runtime_profile_dir is None:
        return DEFAULT_RUNTIME_PROFILES_PATH
    path = Path(runtime_profile_dir)
    if path.is_file() or (path.is_dir() and (path / "_manifest.yaml").is_file()):
        return path
    return path / DEFAULT_RUNTIME_PROFILES_PATH.name


def _conda_env_path(runtime_profile_dir: str | Path | None) -> Path:
    """Resolve the Conda env YAML path from an override directory or the default location."""
    if runtime_profile_dir is None:
        return DEFAULT_CONDA_ENVS_PATH
    path = Path(runtime_profile_dir)
    if path.is_file() or (path.is_dir() and (path / "_manifest.yaml").is_file()):
        return path
    return path / DEFAULT_CONDA_ENVS_PATH.name


def _conda_env_root(payload: Mapping[str, Any], env_root: str | Path | None = None) -> Path:
    """Resolve the Conda envs root directory from manifest defaults, env overrides, or the fallback cache."""
    configured = None if env_root is None else str(env_root)
    defaults = payload.get("defaults")
    if configured in {None, ""} and isinstance(defaults, Mapping):
        configured = str(defaults.get("env_root") or "")
    resolved = _expand_configured_path(configured)
    return resolved if resolved is not None else resolve_cache_dir() / "conda_envs"


def _expand_configured_path(value: object | None) -> Path | None:
    """Expand environment variables and ``~`` in a configured path string, returning ``None`` for unresolved values."""
    if value is None:
        return None
    text = os.path.expandvars(str(value)).strip()
    if not text or "$" in text:
        return None
    return Path(text).expanduser()


def _conda_specs_by_model(
    path: Path,
    *,
    conda_envs_root: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, object]]:
    """Load Conda env specs per model from the manifest, computing existence and Python checks.

    Returns:
        A tuple of ``(specs_by_model_id, env_root, conda_status_mapping)``.
    """
    payload = _read_manifest_mapping(path, item_key="envs")
    env_root = _conda_env_root(payload, conda_envs_root)
    specs: dict[str, dict[str, Any]] = {}
    for item in payload.get("envs") or ():
        if not isinstance(item, Mapping) or not item.get("model_id"):
            continue
        spec = dict(item)
        env_name = str(spec.get("env_name") or spec["model_id"])
        spec_env_root = _expand_configured_path(spec.get("env_root")) or env_root
        env_prefix = spec_env_root / env_name
        spec["env_prefix"] = str(env_prefix)
        spec["env_exists"] = env_prefix.exists()
        spec["python_exists"] = (env_prefix / "bin" / "python").is_file() or (env_prefix / "python.exe").is_file()
        specs[str(spec["model_id"])] = spec

    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    conda_status: dict[str, object] = {
        "manifest_path": str(path),
        "manifest_found": path.exists(),
        "conda_executable": conda_executable,
        "conda_available": conda_executable is not None,
        "active_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "active_prefix": os.environ.get("CONDA_PREFIX"),
        "env_root": str(env_root),
        "env_specs": len(specs),
        "envs_existing": sum(1 for spec in specs.values() if spec.get("env_exists") is True),
        "envs_with_python": sum(1 for spec in specs.values() if spec.get("python_exists") is True),
    }
    return specs, env_root, conda_status


def _profile_to_row(item: Mapping[str, Any], conda_specs: Mapping[str, Mapping[str, Any]]) -> RuntimeProfileRow | None:
    """Convert a runtime profile manifest item to a :class:`RuntimeProfileRow`, merging Conda spec data."""
    profile_id = item.get("id")
    if not profile_id:
        return None
    spec = conda_specs.get(str(profile_id), {})
    return RuntimeProfileRow(
        profile_id=str(profile_id),
        task_family=None if item.get("task_family") is None else str(item.get("task_family")),
        artifact_kind=None if item.get("artifact_kind") is None else str(item.get("artifact_kind")),
        backend_stage=str(item.get("backend_stage") or "unknown"),
        integration_status=str(item.get("integration_status") or "unknown"),
        runtime_status=str(item.get("runtime_status") or "unknown"),
        conda_env_name=None if spec.get("env_name") is None else str(spec.get("env_name")),
        conda_env_prefix=None if spec.get("env_prefix") is None else str(spec.get("env_prefix")),
        conda_driver_status=None if spec.get("driver_status") is None else str(spec.get("driver_status")),
        conda_env_exists=spec.get("env_exists") is True,
        conda_python_exists=spec.get("python_exists") is True,
        validation_imports=_dedupe_text(str(value) for value in spec.get("validation_imports") or ()),
        notes=_dedupe_text(str(value) for value in item.get("notes") or ()),
    )


def load_runtime_profile_rows(
    *,
    runtime_profile_dir: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
) -> tuple[tuple[RuntimeProfileRow, ...], dict[str, object]]:
    """Load runtime profile rows and Conda status diagnostics from YAML manifests.

    Args:
        runtime_profile_dir: Override for the runtime profile directory.
        conda_envs_root: Override for the Conda envs root directory.

    Returns:
        A tuple of (sorted runtime profile rows, Conda status mapping).
    """
    profiles_payload = _read_manifest_mapping(_runtime_profile_path(runtime_profile_dir), item_key="profiles")
    conda_specs, _env_root, conda_status = _conda_specs_by_model(
        _conda_env_path(runtime_profile_dir),
        conda_envs_root=conda_envs_root,
    )
    rows = [
        row
        for item in profiles_payload.get("profiles") or ()
        if isinstance(item, Mapping)
        for row in (_profile_to_row(item, conda_specs),)
        if row is not None
    ]
    return tuple(sorted(rows, key=lambda row: row.profile_id)), conda_status


def load_tui_catalog(
    *,
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    runtime_profile_dir: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
) -> TuiCatalog:
    """Load and merge all model, benchmark, and runtime-profile catalog data.

    Merges registry entries with script-infer and studio-runtime rows to produce
    a unified :class:`TuiCatalog` ready for TUI rendering.

    Args:
        model_manifest_dir: Override for the model zoo manifest directory.
        benchmark_manifest_dir: Override for the benchmark zoo manifest directory.
        runtime_profile_dir: Override for the runtime profile directory.
        conda_envs_root: Override for the Conda envs root directory.

    Returns:
        A fully populated :class:`TuiCatalog`.
    """
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

    # ── Load registries ──
    model_registry = load_model_zoo_registry(model_manifest_dir or DEFAULT_MODEL_ZOO_DIR)
    benchmark_registry = load_benchmark_zoo_registry(benchmark_manifest_dir or DEFAULT_BENCHMARK_ZOO_DIR)
    runtime_profiles, conda_status = load_runtime_profile_rows(
        runtime_profile_dir=runtime_profile_dir,
        conda_envs_root=conda_envs_root,
    )
    # ── Build model rows and merge script-infer / studio-runtime entries ──
    model_rows = {row.model_id: row for row in (_model_to_row(entry) for entry in model_registry)}
    # ── Merge script-infer virtual rows ──
    for model_id, group in INFER_MODEL_FAMILY_GROUPS.items():
        script_row = _infer_virtual_model_row(model_id, group)
        tui_model_id = SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.get(model_id, model_id)
        script_row = replace(script_row, model_id=tui_model_id)
        model_rows[tui_model_id] = _merge_script_infer_row(model_rows.get(tui_model_id), script_row)
    # ── Merge studio-runtime rows ──
    for entry in _unique_studio_entries():
        tui_model_id = _tui_model_id_for_studio_entry(entry)
        studio_row = replace(_studio_infer_model_row(entry), model_id=tui_model_id)
        model_rows[tui_model_id] = _merge_studio_infer_row(model_rows.get(tui_model_id), studio_row)
    # ── Sort and build the final catalog ──
    models = tuple(sorted(model_rows.values(), key=lambda row: row.model_id))
    benchmarks = tuple(
        sorted((_benchmark_to_row(entry) for entry in benchmark_registry), key=lambda row: row.benchmark_id)
    )
    return TuiCatalog(
        models=models,
        benchmarks=benchmarks,
        runtime_profiles=runtime_profiles,
        conda_status=conda_status,
    )


# ── Shell-command builders ──────────────────────────────────────────


def build_model_benchmark_command(
    *,
    model_id: str,
    benchmark_id: str,
    output_dir: str | Path,
    mode: str = "official-run",
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    model_variant: str | None = None,
    requests_path: str | Path | None = None,
    task_name: str | None = None,
    generated_artifact_dir: str | Path | None = None,
    output_artifact: str | None = None,
    metrics: Sequence[str] = (),
    json_output: bool = False,
) -> tuple[str, ...]:
    """Build a ``worldfoundry run`` shell command tuple for evaluation.

    Args:
        model_id: Model family identifier.
        benchmark_id: Benchmark identifier.
        output_dir: Output directory path.
        mode: Run mode label (default ``"official-run"``).
        model_manifest_dir: Optional model manifest directory override.
        benchmark_manifest_dir: Optional benchmark manifest directory override.
        model_variant: Optional model variant override.
        requests_path: Optional requests JSON path override.
        task_name: Optional task name override.
        generated_artifact_dir: Optional generated artifact directory override.
        output_artifact: Optional output artifact override.
        metrics: Metric identifiers to evaluate.
        json_output: Whether to add ``--json`` flag.

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.
    """
    if generated_artifact_dir is not None:
        command = [
            *_worldfoundry_cli_prefix(),
            "score",
            "--benchmark",
            str(benchmark_id),
            "--artifacts",
            str(generated_artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
        _append_optional(command, "--mode", mode)
        _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
        if json_output:
            command.append("--json")
        return tuple(command)

    command = [
        *_worldfoundry_cli_prefix(),
        "run",
        "--model",
        str(model_id),
        "--benchmark",
        str(benchmark_id),
        "--output-dir",
        str(output_dir),
    ]
    _append_optional(command, "--mode", mode)
    _append_optional_path(command, "--model-manifest-dir", model_manifest_dir)
    _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
    _append_optional(command, "--model-variant", model_variant)
    _append_optional_path(command, "--requests-path", requests_path)
    _append_optional(command, "--task-name", task_name)
    _append_optional_path(command, "--generated-artifact-dir", generated_artifact_dir)
    _append_optional(command, "--output-artifact", output_artifact)
    for metric in metrics:
        command.extend(["--metric", str(metric)])
    if json_output:
        command.append("--json")
    return tuple(command)


def _build_workspace_infer_command(
    *,
    model_id: str,
    variant_id: str,
    output_dir: str | Path,
    ckpt_root: str | Path | None,
    conda_envs_root: str | Path | None,
    gpu: str | None,
    values: Sequence[tuple[str, object | None]],
) -> tuple[str, ...]:
    """Build one validated ``workspace_job infer`` command."""
    command = [
        sys.executable,
        "-m",
        "worldfoundry.studio.workspace_job",
        "infer",
        "--model-id",
        model_id,
        "--variant-id",
        variant_id,
        "--output-dir",
        str(output_dir),
    ]
    for flag, value in values:
        _append_optional(command, flag, value)

    env_values: list[str] = []
    if ckpt_root is not None:
        env_values.append(f"WORLDFOUNDRY_CKPT_DIR={Path(ckpt_root).expanduser()}")
    if conda_envs_root is not None:
        env_root = Path(conda_envs_root).expanduser()
        env_values.extend(
            (
                f"WORLDFOUNDRY_CONDA_ENVS_ROOT={env_root}",
                f"WORLDFOUNDRY_CONDA_ENV_ROOT={env_root}",
            )
        )
    if gpu:
        env_values.append(f"CUDA_VISIBLE_DEVICES={gpu}")
    if env_values:
        command = ["env", *env_values, *command]
    return tuple(command)


def build_model_infer_command(
    *,
    model_id: str,
    output_dir: str | Path,
    ckpt_type: str | None = None,
    ckpt_root: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    input_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    video_path: str | Path | None = None,
    trajectory_file: str | Path | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    task: str | None = None,
    mode: str | None = None,
    resize_mode: str | None = None,
    size: str | None = None,
    frames: str | int | None = None,
    steps: str | int | None = None,
    frames_per_generation: str | int | None = None,
    guidance_scale: str | int | float | None = None,
    seed: str | int | None = None,
    fps: str | int | None = None,
    dtype: str | None = None,
    max_sequence_length: str | int | None = None,
    cam_type: str | int | None = None,
    interactions: str | None = None,
    output_formats: str | None = None,
    trajectory: str | None = None,
    angle: str | int | float | None = None,
    distance: str | int | float | None = None,
    orbit_radius: str | int | float | None = None,
    zoom_ratio: str | int | float | None = None,
    alpha_threshold: str | int | float | None = None,
    static_scene: object | None = None,
    low_vram: object | None = None,
    disable_lora: object | None = None,
    vis_rendering: object | None = None,
    offload_t5: object | None = None,
    offload_transformer_during_vae: object | None = None,
    offload_vae: object | None = None,
    output_path: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
    gpu: str | None = None,
) -> tuple[str, ...]:
    """Build a shell command tuple for model inference.

    Resolves legacy model aliases to a concrete Studio inference contract and
    applies variant-specific defaults (e.g. Astra camera type and Neoverse
    trajectory mode).

    Args:
        model_id: Model family identifier.
        output_dir: Output directory path.
        ckpt_type: Checkpoint variant / function selector.
        ckpt_root: Checkpoint root directory override.
        ckpt_path: Exact checkpoint path or HuggingFace repo override.
        input_path: Input image/video path override.
        input_dir: Input directory override (FlashWorld JSON, etc.).
        video_path: Input video path override.
        trajectory_file: Custom camera trajectory JSON override.
        prompt: Text prompt override.
        negative_prompt: Negative prompt override.
        task: Task mode override.
        mode: Generation mode override.
        resize_mode: Resize mode override.
        size: Resolution override (e.g. ``"832*480"``).
        frames: Frame count override.
        steps: Sampling step override.
        frames_per_generation: Astra frame chunk override.
        guidance_scale: CFG / guidance scale override.
        seed: Random seed override.
        fps: Output FPS override.
        dtype: Runtime dtype override.
        max_sequence_length: Text encoder length override.
        cam_type: Astra camera preset override.
        interactions: Interaction / action list override.
        output_formats: FlashWorld output formats override.
        trajectory: Camera trajectory preset override.
        angle: Camera angle override.
        distance: Camera distance override.
        orbit_radius: Camera orbit radius override.
        zoom_ratio: Camera zoom ratio override.
        alpha_threshold: Alpha threshold override.
        static_scene: Static scene flag.
        low_vram: Low-VRAM mode flag.
        disable_lora: Disable LoRA flag.
        vis_rendering: Rendering visualization flag.
        offload_t5: T5 offload flag.
        offload_transformer_during_vae: Transformer offload during VAE flag.
        offload_vae: VAE offload flag.
        output_path: Exact output artifact path override.
        conda_envs_root: Conda envs root directory override.
        gpu: CUDA_VISIBLE_DEVICES override.

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.

    Raises:
        ValueError: When the model has no registered infer script or variant group.
    """
    resolved_variant = resolve_infer_model_variant(model_id, ckpt_type)
    script_family = _script_infer_family_id(model_id)
    script_infer = script_family in INFER_MODEL_VARIANTS
    if not script_infer and not is_studio_infer_model_id(model_id):
        raise ValueError(f"no inference runtime is registered for {model_id!r}")
    # NOTE: Apply variant-specific defaults before building the command
    if resolved_variant == "astra":
        frames = frames or 24
        frames_per_generation = frames_per_generation or 8
        fps = fps or 20
        cam_type = cam_type or 4
    if resolved_variant == "flash-world":
        fps = fps or 15
        output_formats = output_formats or "video,spz,ply"
        if input_dir is None:
            frames = frames or 24
    if resolved_variant == "neoverse":
        # NOTE: Neoverse defaults — trajectory ``tilt_up``, mode ``relative``, 81 frames
        if not trajectory and not trajectory_file and not interactions:
            trajectory = "tilt_up"
        mode = mode or "relative"
        resize_mode = resize_mode or "center_crop"
        size = size or "560*336"
        frames = frames or 81
        disable_lora_enabled = _truthy(disable_lora)
        steps = steps or (50 if disable_lora_enabled else 4)
        guidance_scale = guidance_scale or (5.0 if disable_lora_enabled else 1.0)
        seed = seed or 42
        fps = fps or 16
        alpha_threshold = alpha_threshold or 1.0
    if resolved_variant == "recammaster" and not trajectory:
        trajectory = "100,100,0,0,30"
    if script_infer:
        command_model, command_variant, _label = _studio_runtime_target_for_script_variant(
            model_id,
            resolved_variant,
        )
    else:
        command_model, command_variant = model_id, resolved_variant
    return _build_workspace_infer_command(
        model_id=command_model,
        variant_id=command_variant,
        output_dir=output_dir,
        ckpt_root=ckpt_root,
        conda_envs_root=conda_envs_root,
        gpu=gpu,
        values=(
            ("--model-ref", ckpt_path),
            ("--input-path", input_path),
            ("--input-dir", input_dir),
            ("--video-path", video_path),
            ("--trajectory-file", trajectory_file),
            ("--prompt", prompt),
            ("--negative-prompt", negative_prompt),
            ("--interactions", interactions),
            ("--task", task),
            ("--mode", mode),
            ("--resize-mode", resize_mode),
            ("--size", size),
            ("--frames", frames),
            ("--steps", steps),
            ("--frames-per-generation", frames_per_generation),
            ("--guidance-scale", guidance_scale),
            ("--seed", seed),
            ("--fps", fps),
            ("--dtype", dtype),
            ("--max-sequence-length", max_sequence_length),
            ("--cam-type", cam_type),
            ("--output-formats", output_formats),
            ("--trajectory", trajectory),
            ("--angle", angle),
            ("--distance", distance),
            ("--orbit-radius", orbit_radius),
            ("--zoom-ratio", zoom_ratio),
            ("--alpha-threshold", alpha_threshold),
            ("--static-scene", static_scene),
            ("--low-vram", low_vram),
            ("--disable-lora", disable_lora),
            ("--vis-rendering", vis_rendering),
            ("--offload-t5", offload_t5),
            ("--offload-transformer-during-vae", offload_transformer_during_vae),
            ("--offload-vae", offload_vae),
            ("--output-path", output_path),
        ),
    )


def build_studio_command(*, host: str = "127.0.0.1", port: int = 7860) -> tuple[str, ...]:
    """Build a shell command tuple that launches the local Studio web UI."""
    return (
        sys.executable,
        "-m",
        "worldfoundry.studio.cli",
        "--host",
        host,
        "--port",
        str(port),
    )


def build_suite_command(
    *,
    output_dir: str | Path,
    mode: str = "official-run",
    model_ids: Sequence[str] = (),
    benchmark_ids: Sequence[str] = (),
    suite_ids: Sequence[str] = (),
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    plan_only: bool = True,
) -> tuple[str, ...]:
    """Build a ``worldfoundry run`` shell command tuple for a model × benchmark suite.

    Args:
        output_dir: Output directory path for the suite plan / results.
        mode: Run mode label (default ``"official-run"``).
        model_ids: Model identifiers to include in the suite.
        benchmark_ids: Benchmark identifiers to include in the suite.
        suite_ids: Suite identifiers (alternative to explicit model/benchmark lists).
        model_manifest_dir: Optional model manifest directory override.
        benchmark_manifest_dir: Optional benchmark manifest directory override.
        plan_only: Whether to write a plan only (default ``True``).

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.
    """
    command = [
        *_worldfoundry_cli_prefix(),
        "run",
        "--output-dir",
        str(output_dir),
    ]
    _append_optional(command, "--mode", mode)
    _append_optional_path(command, "--model-manifest-dir", model_manifest_dir)
    _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
    for model_id in model_ids:
        command.extend(["--model", str(model_id)])
    for benchmark_id in benchmark_ids:
        command.extend(["--benchmark", str(benchmark_id)])
    for suite_id in suite_ids:
        command.extend(["--suite", str(suite_id)])
    if plan_only:
        command.append("--plan-only")
    return tuple(command)


def _worldfoundry_cli_prefix() -> tuple[str, str, str]:
    """Return the ``python -m worldfoundry.cli`` prefix for shell command tuples."""
    return (sys.executable, "-m", "worldfoundry.cli")


def format_shell_command(command: Sequence[str]) -> str:
    """Join a command tuple into a single shell-escaped string using :func:`shlex.join`."""
    return shlex.join([str(item) for item in command])


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    """Append *flag* and *value* to *command* when *value* is not ``None``."""
    if value is not None:
        command.extend([flag, str(value)])


def _append_optional_path(command: list[str], flag: str, value: str | Path | None) -> None:
    """Append *flag* and *value* to *command* when *value* is not ``None``, converting to a string."""
    if value is not None:
        command.extend([flag, str(value)])


def _truthy(value: object | None) -> bool:
    """Return whether *value* is truthy — treating ``"1"``/``"true"``/``"yes"``/``"on"`` as true."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _append_optional_flag(command: list[str], flag: str, value: object | None) -> None:
    """Append a boolean *flag* to *command* when *value* is truthy per :func:`_truthy`."""
    if _truthy(value):
        command.append(flag)


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
