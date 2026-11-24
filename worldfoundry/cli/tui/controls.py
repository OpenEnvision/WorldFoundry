# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Inference control fields shown by the TUI for each model variant."""

from __future__ import annotations

from dataclasses import dataclass, replace

from worldfoundry.cli.inference_variants import (
    INFER_VARIANT_GROUPS,
    _studio_entry_for_model,
    _studio_inference_spec,
    is_script_infer_model_id,
    resolve_infer_model_variant,
)
from worldfoundry.runtime.inference_specs.lingbot_world import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
)

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
    "call_json",
    "load_json",
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
    "call_json": InferControlSpec("call_json", "Inference options (JSON)", '{"controls": []}'),
    "load_json": InferControlSpec("load_json", "Model assets / loading (JSON)", '{"source_root": "/path/to/source"}'),
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
        studio_fields = _control_fields_from_studio_spec(model_id, ckpt_type)
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
                "placeholder": "worldrecon",
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


def _normalise_infer_field(value: str) -> str:
    """Normalize an inference field ID for matching: strip, lowercase, replace hyphens with underscores."""
    return str(value or "").strip().lower().replace("-", "_")


def _control_fields_from_studio_spec(model_id: str, variant_id: str | None = None) -> tuple[str, ...]:
    """Extract inference control field IDs from the studio inference spec's task inputs."""
    spec = _studio_inference_spec(model_id)
    entry = _studio_entry_for_model(model_id)
    if spec is None or entry is None:
        return ()
    try:
        from worldfoundry.runtime.interactive_inference_catalog import interactive_task_for_variant

        task = interactive_task_for_variant(spec, spec.variant(variant_id), spec.task(None))
    except ValueError:
        return ()
    fields: set[str] = {"call_json", "load_json", "output_path"}
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
