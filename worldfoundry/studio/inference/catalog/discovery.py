"""Catalog discovery: AST pipeline scanning, canonical runtime entries, entry building, and lookup."""

from __future__ import annotations

import ast
import copy
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.runtime.interactive_inference_catalog import studio_contract_override
from worldfoundry.studio.inference.paths import studio_hfd_cache_roots

from . import resolve
from .entry import CatalogEntry, _AstPipelineInfo
from .models import CURATED_OVERRIDES
from .params import (
    COGVIDEOX_STUDIO_PARENT_ID,
    DISPATCH_LOAD_PARAMS,
    OFFICIAL_ACTION_CALL_PARAMS,
    OFFICIAL_ACTION_LOAD_PARAMS,
    _cogvideox_aliases,
    _cogvideox_folded_studio_model_ids,
    _cogvideox_studio_parent_id,
    _cogvideox_variant_id,
    _cogvideox_variant_specs,
    _cogvideox_workspace_variants,
    _native_diffusion_video_call_params,
    _native_diffusion_video_load_params,
    _normalize_model_id,
    _official_video_call_params,
    _official_video_load_params,
    _sana_folded_studio_model_ids,
    _t2v_turbo_base_ref,
    _t2v_turbo_default_load_kwargs,
    _wan_2p1_i2v_default_ref,
    _wan_2p1_t2v_default_ref,
)
from .resolve import _CATALOG_CACHE_ROOTS, PIPELINES_ROOT, _CatalogPathSnapshot

ACTION_SYNTHESIS_ROOT = Path(__file__).resolve().parents[3] / "synthesis" / "action_generation"
NESTED_PIPELINE_DIRS: tuple[str, ...] = ()
ABSTRACT_RUNTIME_MODEL_IDS: frozenset[str] = frozenset(
    {
        "official-policy",
        "official-video",
        "three-d-four-d-runtime",
        "world-model-runtime",
    }
)


def _catalog_id_key(model_id: str) -> str:
    return _normalize_model_id(model_id).replace("-", "_")


STUDIO_HIDDEN_CATALOG_MODEL_IDS: frozenset[str] = (
    frozenset(
        {
            # Internal shared operator/contract surface. Concrete priors such as metric3d-prior,
            # unidepth-v2-prior, dap, and video-depth-anything-prior are the user-facing entries.
            "geometry-prior",
            # The complete Wan2.2 camera/base/LoRA/MoGe assets pass CPU schema validation,
            # but the official dual-A14B GPU memory strategy still needs a real artifact run.
            "fantasyworld-wan22",
            # Lyra-2 alias; keep lyra-1 / lyra-2 as the canonical Create Job entries.
            "lyra",
            # Runtime-environment alias for ltx-2.3-i2v. Keeping it as a separate
            # catalog row duplicates the same pipeline and invents a non-existent
            # ``Ltx23I2VPipeline`` class name during canonical-entry synthesis.
            "ltx2_3_i2v",
            # Hidden official weights; keep provenance in the model catalog but do not
            # expose a Studio infer row that cannot return a real artifact.
            "omnivinci",
            "emu3.5",
            "qwen2.5-omni",
            "spatial-ladder",
            "spatial-reasoner",
            # Image / Sprint / ControlNet checkpoints are variants on the Sana card.
            # CogVideoX 2B T2V and 5B I2V share the CogVideoX card with 5B T2V.
        }
    )
    | _sana_folded_studio_model_ids()
    | _cogvideox_folded_studio_model_ids()
)

# These Workspace matrix ids intentionally retain underscores while their AST
# pipeline ids are filename-derived hyphenated slugs. Resolve them through the
# fast single-entry path so a direct job does not scan and stat the entire
# checkpoint catalog before importing the requested pipeline.
FAST_CANONICAL_AST_ALIASES: Mapping[str, str] = {
    "videocrafter1_i2v": "videocrafter1-i2v",
    "videocrafter1_t2v": "videocrafter1-t2v",
    "videocrafter2_t2v": "videocrafter2-t2v",
}


def _source_root() -> Path:
    src_root = resolve._project_root() / "src"
    return src_root if (src_root / "worldfoundry").is_dir() else resolve._project_root()


def _slug_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("pipeline_"):
        stem = stem[len("pipeline_") :]
    return stem.replace("_", "-")


def _display_name_from_slug(slug: str) -> str:
    tokens = slug.replace(".", "-").split("-")
    pretty = []
    for token in tokens:
        if not token or token.lower() == "prior":
            continue
        if token.isupper():
            pretty.append(token)
        elif token.lower() in {"i2v", "t2v", "ti2v", "i2av", "t2av", "api", "vmem", "vggt"} or token.lower() in {
            "pi3",
            "gen3c",
            "wow",
            "cut3r",
            "loger",
        }:
            pretty.append(token.upper())
        elif token.lower().startswith("2p") and len(token) > 2:
            pretty.append(token.replace("p", "."))
        else:
            pretty.append(token.capitalize())
    return " ".join(pretty)


def _summary_from_category(category: str, family: str, supports_stream: bool) -> str:
    stream_hint = " Supports multi-turn continuation." if supports_stream else ""
    if category == "Embodied Action":
        return f"{family} pipeline for policy, action, and rollout artifacts.{stream_hint}"
    if category == "Visual Action":
        return f"{family} pipeline for latent-action or video-action artifacts.{stream_hint}"
    if category == "Video-to-Video":
        return (
            f"{family} pipeline for video-to-video rerendering, trajectory control, and camera-path edits.{stream_hint}"
        )
    if category == "3D Scene":
        return f"{family} pipeline for reconstruction, camera control, and spatial outputs.{stream_hint}"
    if category == "Depth / Geometry":
        return f"{family} pipeline for depth or geometry extraction.{stream_hint}"
    if category == "Remote API":
        return f"{family} pipeline backed by a hosted API endpoint.{stream_hint}"
    return f"{family} pipeline for world-model generation and interaction.{stream_hint}"


def _category_from_family(family: str, class_name: str, call_params: Sequence[str]) -> str:
    if family in {
        "embodied_action",
        "openvla",
        "openpi",
        "giga_brain_0",
        "dreamzero",
        "gr00t",
        "starvla",
        "lingbot_va",
        "octo",
        "rt1",
        "diffusion_policy",
        "act",
        "roboflamingo",
    }:
        return "Embodied Action"
    if family in {"lapa"}:
        return "Visual Action"
    if family in {
        "vggt",
        "vggt_omega",
        "cut3r",
        "pi3",
        "infinite_vggt",
        "lagernvs",
        "pixelsplat",
        "lingbot_map",
    }:
        return "3D Scene"
    if family in {
        "depth_anything",
        "worldfm",
        "geometry_prior",
    } or family.startswith("geometry_prior"):
        return "Depth / Geometry"
    if family in {"worldlabs", "worldlabs_marble_1p1", "runway", "luma", "veo", "sora", "minimax"}:
        return "Remote API"
    if "api_init" in class_name.lower():
        return "Remote API"
    if "output_dir" in call_params and "task_type" in call_params and family in {"hunyuan_world"}:
        return "3D Scene"
    return "Video Generation"


def _runtime_kind_from_family(model_id: str, family: str) -> str:
    if model_id in {"vggt", "vggt-omega", "cut3r"}:
        return "two_stage_3dgs"
    if model_id in {"pi3", "loger"}:
        return "pointcloud_nav"
    if model_id == "worldfm":
        return "worldfm"
    return "default"


def _default_backend(supports_api_init: bool, supports_from_pretrained: bool) -> str:
    if supports_from_pretrained:
        return "from_pretrained"
    if supports_api_init:
        return "api_init"
    return "auto"


def _append_dispatch_load_params(load_params: Sequence[str]) -> tuple[str, ...]:
    values = [str(item) for item in load_params]
    for item in DISPATCH_LOAD_PARAMS:
        if item not in values:
            values.append(item)
    return tuple(values)


def _resolve_extra_variants(value: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw_variant in resolve._resolve_override_value(value) or ():
        variant = dict(raw_variant)
        for key in ("load_kwargs", "call_kwargs"):
            if key in variant:
                variant[key] = dict(resolve._resolve_override_value(variant[key]) or {})
        checkpoints = []
        for raw_checkpoint in variant.get("checkpoints", ()) or ():
            checkpoint = dict(raw_checkpoint)
            if "uri" in checkpoint:
                checkpoint["uri"] = str(resolve._resolve_override_value(checkpoint["uri"]) or "")
            checkpoints.append(checkpoint)
        variant["checkpoints"] = tuple(checkpoints)
        rows.append(variant)
    return tuple(rows)


def _model_ref_env_key(model_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    return f"WORLDFOUNDRY_STUDIO_MODEL_REF_{normalized}"


def _resolve_default_model_ref(model_id: str, fallback: str) -> str:
    model_override = os.getenv(_model_ref_env_key(model_id), "").strip()
    if model_override:
        return model_override
    global_override = (
        os.getenv("WORLDFOUNDRY_STUDIO_MODEL_REF", "").strip()
        or os.getenv("WORLDFOUNDRY_STUDIO_FIXED_MODEL_REF", "").strip()
    )
    if global_override:
        return global_override
    return fallback


def _vchitect2_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Vchitect--Vchitect-2.0-2B",
        "Vchitect-2.0-2B",
        fallback="Vchitect/Vchitect-2.0-2B",
    )


def _sana_studio_parent_id(model_id: str) -> str | None:
    """Return ``sana`` when ``model_id`` is a folded image-family checkpoint."""

    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_ALIASES

    key = _normalize_model_id(model_id)
    folded = _sana_folded_studio_model_ids()
    if key in folded:
        return "sana"
    target = SANA_ALIASES.get(key)
    if target in folded:
        return "sana"
    return None


def _runtime_module_path(model_name: str) -> str:
    family = model_name.split("_", maxsplit=1)[0]
    return f"worldfoundry.pipelines.{family}.pipeline_{model_name}"


def _canonical_runtime_entries(model_id: str | None = None) -> Iterable[CatalogEntry]:
    """Resolve canonical defaults, optionally for just one id or alias."""
    runtime_specs = {
        "allegro_ti2v": {
            "generation_type": "i2v",
            "aliases": ("allegro", "allegro-ti2v"),
            "default_prompt": "The car drives along the road",
            "default_input_path": str(
                resolve._data_path("test_cases", "stable_virtual_camera", "basic", "blue-car.jpg")
            ),
            "load_params": (
                *DISPATCH_LOAD_PARAMS,
                "model_path",
                "required_components",
                "device",
                "lazy",
                "guidance_scale",
                "num_sampling_steps",
                "seed",
            ),
            "default_load_kwargs": {
                "guidance_scale": 8,
                "num_sampling_steps": 100,
                "seed": 1427329220,
            },
            "default_call_kwargs": {"fps": 15},
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "Allegro-TI2V",
                "rhymes-ai--Allegro-TI2V",
                fallback="rhymes-ai/Allegro-TI2V",
            ),
        },
        **_cogvideox_variant_specs(),
        "dynamicrafter_1024_i2v": {
            "generation_type": "i2v",
            "aliases": ("dynamicrafter-1024-i2v",),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "Doubiiu--DynamiCrafter_1024/model.ckpt",
                "DynamiCrafter_1024/model.ckpt",
                "AliVideo--DynamiCrafter_1024/model.ckpt",
            ),
        },
        "dynamicrafter_512_i2v": {
            "generation_type": "i2v",
            "aliases": ("dynamicrafter", "dynamicrafter-512-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "Doubiiu--DynamiCrafter_512/model.ckpt",
                "DynamiCrafter_512/model.ckpt",
                "AliVideo--DynamiCrafter_512/model.ckpt",
            ),
        },
        "easyanimate_i2v": {
            "generation_type": "i2v",
            "aliases": ("easyanimate-i2v", "easyanimate"),
            "default_prompt": (
                "A young man holding a bright sparkler gently turns his head and smiles as sparks scatter "
                "through the night air, with natural motion, stable facial details, and a steady camera."
            ),
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "hfd/alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
                "alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
                fallback="alibaba-pai/EasyAnimateV5.1-7b-zh-InP",
            ),
        },
        "gen_3_i2v": {
            "generation_type": "i2v",
            "aliases": ("gen-3-i2v", "gen3-i2v", "gen-3"),
            "default_prompt": "A gentle camera move around the subject with natural motion and stable lighting.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        },
        "ltx_video_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx-video-i2v", "ltx-video"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "LTX-Video",
                "Lightricks--LTX-Video",
                fallback="Lightricks/LTX-Video",
            ),
        },
        "ltx2_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx2-i2v", "ltx2", "ltx-2", "ltx-2-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "num_frames": 121,
                "fps": 24,
                "height": 512,
                "width": 768,
                "num_inference_steps": 11,
                "guidance_scale": 1.0,
                "seed": 171198,
            },
            "call_params": (
                "prompt",
                "images",
                "output_path",
                "return_dict",
                "fps",
                "num_frames",
                "num_inference_steps",
                "height",
                "width",
                "guidance_scale",
                "seed",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "lazy",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "offload_mode",
                "quantization",
            ),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "LTX-2/ltx-2-19b-distilled.safetensors",
                "Lightricks--LTX-2/ltx-2-19b-distilled.safetensors",
                fallback="Lightricks/LTX-2",
            ),
        },
        "ltx2_3_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx-2.x", "ltx2.3", "ltx-2.3", "ltx2.3-i2v", "ltx-2.3-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "num_frames": 121,
                "fps": 24,
                "height": 512,
                "width": 768,
                "num_inference_steps": 11,
                "guidance_scale": 1.0,
                "seed": 0,
            },
            "call_params": (
                "prompt",
                "images",
                "output_path",
                "return_dict",
                "fps",
                "num_frames",
                "num_inference_steps",
                "height",
                "width",
                "guidance_scale",
                "seed",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "lazy",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "offload_mode",
                "quantization",
            ),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
                "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
                fallback="Lightricks/LTX-2.3",
            ),
        },
        "minimax_i2v": {
            "generation_type": "i2v",
            "aliases": ("minimax-i2v",),
            "default_prompt": "A gentle camera move around the subject with natural motion and stable lighting.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        },
        "t2v_turbo_t2v": {
            "generation_type": "t2v",
            "aliases": ("t2v-turbo-t2v", "t2v-turbo"),
            "default_prompt": "An astronaut riding a horse.",
            "default_model_ref": _t2v_turbo_base_ref,
            "default_load_kwargs": _t2v_turbo_default_load_kwargs,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "base_checkpoint",
                "model_ckpt",
                "lora_checkpoint",
                "lora_path",
                "torch_dtype",
                "weight_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 320,
                "width": 512,
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 8,
                "guidance_scale": 7.5,
                "seed": 0,
                "lcm_origin_steps": 200,
            },
        },
        "vchitect_2_t2v": {
            "generation_type": "t2v",
            "aliases": ("vchitect-2-t2v", "vchitect2-t2v", "vchitect-2"),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": _vchitect2_default_ref,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "torch_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 432,
                "width": 768,
                "num_frames": 40,
                "num_inference_steps": 100,
                "guidance_scale": 7.5,
                "fps": 8,
                "seed": 0,
            },
        },
        "videocrafter1_i2v": {
            "generation_type": "i2v",
            "aliases": ("videocrafter1-i2v",),
            "default_prompt": "horses are walking on the grassland",
            "default_input_path": str(resolve._data_path("test_cases", "videocrafter", "i2v_prompts", "horse.png")),
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "VideoCrafter--Image2Video-512/model.ckpt",
                "VideoCrafter/videocrafter_i2v_512_v1.ckpt",
                "video_models/videocrafter_i2v_512_v1.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 50,
                "height": 320,
                "width": 512,
                "seed": 123,
            },
        },
        "videocrafter1_t2v": {
            "generation_type": "t2v",
            "aliases": ("videocrafter1-t2v",),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "VideoCrafter--Text2Video-1024/model.ckpt",
                "VideoCrafter/videocrafter_t2v_1024_v1.ckpt",
                "video_models/videocrafter_t2v_1024_v1.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 50,
                "height": 576,
                "width": 1024,
                "seed": 123,
            },
        },
        "videocrafter2_t2v": {
            "generation_type": "t2v",
            "aliases": ("videocrafter", "videocrafter2-t2v"),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "VideoCrafter--VideoCrafter2/model.ckpt",
                "VideoCrafter/videocrafter_t2v_512_v2.ckpt",
                "video_models/videocrafter_t2v_512_v2.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 28,
                "num_inference_steps": 50,
                "height": 320,
                "width": 512,
                "seed": 123,
            },
        },
        "wan2.1_i2v": {
            "generation_type": "i2v",
            "ast_slug": "wan2.1-i2v-14b-480p",
            "display_name": "Wan2.1 I2V",
            "aliases": ("wan2.1-i2v", "wan2p1-i2v", "wan2-1-i2v"),
            "default_model_ref": _wan_2p1_i2v_default_ref,
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 40,
                "shift": 3.0,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "call_params": _native_diffusion_video_call_params(),
            "load_params": _native_diffusion_video_load_params(),
        },
        "wan2.1_t2v": {
            "generation_type": "t2v",
            "ast_slug": "wan2.1-t2v-1.3b",
            "display_name": "Wan2.1 T2V",
            "aliases": ("wan2.1", "wan2.1-t2v", "wan2p1-t2v", "wan2-1-t2v"),
            "default_model_ref": _wan_2p1_t2v_default_ref,
            "default_prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "default_call_kwargs": {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 50,
                "shift": 8.0,
                "guidance_scale": 6.0,
                "seed": 42,
            },
            "call_params": _native_diffusion_video_call_params(),
            "load_params": _native_diffusion_video_load_params(),
        },
    }
    runtime_specs[COGVIDEOX_STUDIO_PARENT_ID] = {
        **runtime_specs[COGVIDEOX_STUDIO_PARENT_ID],
        "display_name": "CogVideoX",
        "summary": ("CogVideoX text-to-video and image-to-video family. Select 2B T2V, 5B T2V, or 5B I2V."),
        "aliases": _cogvideox_aliases(),
        "extra_variants": _cogvideox_workspace_variants,
        "tags": ("video", "t2v", "i2v"),
    }

    if model_id is not None:
        requested = _catalog_id_key(model_id)
        exact_specs = {name: spec for name, spec in runtime_specs.items() if _catalog_id_key(name) == requested}
        runtime_specs = exact_specs or {
            name: spec
            for name, spec in runtime_specs.items()
            if any(_catalog_id_key(candidate) == requested for candidate in spec.get("aliases", ()))
        }
        if not runtime_specs:
            return

    ast_by_slug: dict[str, _AstPipelineInfo] = {}
    for info in _discover_ast_pipelines():
        ast_by_slug.setdefault(info.model_id.replace("_", "-"), info)

    for canonical_name, spec in runtime_specs.items():
        slug = canonical_name
        slug_key = str(spec.get("ast_slug") or canonical_name.replace("_", "-"))
        matched = ast_by_slug.get(slug_key)
        module_path = matched.module_path if matched is not None else _runtime_module_path(canonical_name)
        class_name = matched.class_name if matched is not None else _runtime_class_name(canonical_name)
        family = matched.family if matched is not None else canonical_name.split("_", maxsplit=1)[0]
        display_name = str(spec.get("display_name") or _display_name_from_slug(slug.replace("_", "-")))
        generation_type = spec.get("generation_type", "i2v")
        summary = str(spec.get("summary") or f"Independent {generation_type.upper()} video pipeline for this model.")
        if "call_params" in spec:
            call_params = tuple(resolve._resolve_override_value(spec.get("call_params")) or ())
        else:
            matched_call_params = tuple(matched.call_params) if matched is not None else ()
            call_params = tuple(dict.fromkeys((*matched_call_params, *_official_video_call_params())))
        if "stream_params" in spec:
            stream_params = tuple(resolve._resolve_override_value(spec.get("stream_params")) or ())
        else:
            matched_stream_params = tuple(matched.stream_params) if matched is not None else ()
            stream_params = tuple(dict.fromkeys((*matched_stream_params, *_official_video_call_params())))
        default_call_kwargs = dict(resolve._resolve_override_value(spec.get("default_call_kwargs", {"fps": 16})) or {})
        if canonical_name.startswith("videocrafter"):
            call_params = (
                "prompt",
                "output_path",
                "fps",
                "return_dict",
                "frames",
                "num_frames",
                "frame_num",
                "ddim_steps",
                "num_inference_steps",
                "steps",
                "sample_steps",
                "infer_steps",
                "sampling_steps",
                "ddim_eta",
                "height",
                "width",
                "seed",
                "unconditional_guidance_scale",
            )
            if generation_type == "i2v":
                call_params = ("prompt", "images", *call_params[1:])
            stream_params = call_params
            default_call_kwargs.update(
                {
                    "ddim_eta": 1.0,
                    "unconditional_guidance_scale": 12.0,
                }
            )
        yield CatalogEntry(
            model_id=slug,
            display_name=display_name,
            module_path=module_path,
            class_name=class_name,
            family=family,
            category="Video Generation",
            summary=summary,
            call_params=call_params,
            stream_params=stream_params,
            load_params=tuple(
                resolve._resolve_override_value(
                    spec.get(
                        "load_params",
                        matched.load_params
                        if matched is not None and matched.load_params
                        else ("model_path", "required_components", "device", "lazy"),
                    )
                )
                or ()
            ),
            supports_stream=matched.supports_stream if matched is not None else True,
            supports_from_pretrained=True,
            supports_api_init=matched.supports_api_init if matched is not None else False,
            runtime_kind="default",
            default_backend="from_pretrained",
            default_model_ref=_resolve_default_model_ref(
                slug, str(resolve._resolve_override_value(spec.get("default_model_ref", "")) or "")
            ),
            default_prompt=str(resolve._resolve_override_value(spec.get("default_prompt", "")) or ""),
            default_input_path=str(resolve._resolve_override_value(spec.get("default_input_path", "")) or ""),
            default_task_type=str(resolve._resolve_override_value(spec.get("default_task_type", "")) or ""),
            default_load_kwargs=dict(resolve._resolve_override_value(spec.get("default_load_kwargs", {})) or {}),
            default_interactions=(),
            default_call_kwargs=default_call_kwargs,
            extra_variants=_resolve_extra_variants(spec.get("extra_variants", ())),
            aliases=tuple(spec.get("aliases", [])),
            tags=tuple(spec.get("tags") or ("video", generation_type)),
            notes="Integrated runtime modules are expected under worldfoundry.pipelines; pass only WorldFoundry checkpoint/cache paths or HF repo ids as model refs.",
        )


def _runtime_class_name(model_name: str) -> str:
    tokens = model_name.replace(".", "_").split("_")
    pretty_tokens = []
    for token in tokens:
        if token in {"i2v", "t2v", "ti2v"}:
            pretty_tokens.append(token.upper())
        else:
            pretty_tokens.append(token.capitalize())
    return "".join(pretty_tokens) + "Pipeline"


def _parse_method_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    args: list[str] = []
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        if arg.arg in {"self", "cls"}:
            continue
        args.append(arg.arg)
    return tuple(dict.fromkeys(args))


def _resolve_class_methods(
    class_nodes: dict[str, ast.ClassDef],
    class_name: str,
    cache: dict[str, dict[str, tuple[str, ...]]],
    seen: frozenset[str] = frozenset(),
) -> dict[str, tuple[str, ...]]:
    """Collect local inherited method signatures for AST-only catalog discovery."""

    if class_name in cache:
        return dict(cache[class_name])
    if class_name in seen or class_name not in class_nodes:
        return {}
    class_node = class_nodes[class_name]
    methods: dict[str, tuple[str, ...]] = {}
    next_seen = seen | {class_name}
    for base in class_node.bases:
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        methods.update(_resolve_class_methods(class_nodes, base_name, cache, next_seen))
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = _parse_method_args(item)
    cache[class_name] = methods
    return dict(methods)


def _class_inherits_any(
    class_nodes: Mapping[str, ast.ClassDef],
    class_name: str,
    expected_bases: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Resolve local inheritance far enough to recognize shared infra bases.

    Catalog discovery deliberately parses pipeline modules without importing
    heavyweight model dependencies.  Model-specific native pipelines commonly
    inherit their implementation from a base imported from another module, and
    variants then inherit from that local model class.  Looking only at methods
    declared in the leaf class therefore produced empty Workspace contracts.
    """

    if class_name in seen:
        return False
    node = class_nodes.get(class_name)
    if node is None:
        return class_name in expected_bases
    next_seen = seen | {class_name}
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if base_name in expected_bases:
            return True
        if base_name in class_nodes and _class_inherits_any(
            class_nodes,
            base_name,
            expected_bases,
            next_seen,
        ):
            return True
    return False


def _class_string_constant(node: ast.ClassDef, name: str) -> str | None:
    for item in node.body:
        if isinstance(item, ast.Assign):
            targets = item.targets
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            return item.value.value
    return None


def _class_bool_constant(node: ast.ClassDef, name: str) -> bool | None:
    for item in node.body:
        if isinstance(item, ast.Assign):
            targets = item.targets
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool):
            return item.value.value
    return None


def _pipeline_file_candidates() -> tuple[Path, ...]:
    """Return Studio-visible pipeline files without descending into vendored runtimes.

    Args:
        None.
    """
    candidates: list[Path] = []
    for family_dir in sorted(path for path in PIPELINES_ROOT.iterdir() if path.is_dir()):
        candidates.extend(
            sorted(path for path in family_dir.iterdir() if path.name.startswith("pipeline_") and path.suffix == ".py")
        )
    for relative_dir in NESTED_PIPELINE_DIRS:
        nested_dir = PIPELINES_ROOT / relative_dir
        if nested_dir.is_dir():
            candidates.extend(
                sorted(
                    path for path in nested_dir.iterdir() if path.name.startswith("pipeline_") and path.suffix == ".py"
                )
            )
    return tuple(dict.fromkeys(candidates))


def _action_synthesis_file_candidates() -> tuple[Path, ...]:
    """Return first-class action synthesis files used by Studio catalog."""

    if not ACTION_SYNTHESIS_ROOT.is_dir():
        return ()
    return tuple(
        sorted(
            path for path in ACTION_SYNTHESIS_ROOT.glob("*/*_synthesis.py") if path.name != "base_action_synthesis.py"
        )
    )


def _literal_keyword(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        try:
            return ast.literal_eval(keyword.value)
        except Exception:
            return None
    return None


def _discover_component_pipelines() -> tuple[_AstPipelineInfo, ...]:
    path = PIPELINES_ROOT / "component_pipelines.py"
    if not path.is_file():
        return ()
    rel_path = path.relative_to(_source_root())
    module_path = ".".join(rel_path.with_suffix("").parts)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    discovered: list[_AstPipelineInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "_component_pipeline_class":
            continue
        if not node.value.args:
            continue
        try:
            class_name = str(ast.literal_eval(node.value.args[0]))
        except Exception:
            continue
        model_id = _literal_keyword(node.value, "model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        generation_type = _literal_keyword(node.value, "generation_type")
        family = str(generation_type or model_id).replace("-", "_")
        discovered.append(
            _AstPipelineInfo(
                model_id=_normalize_model_id(model_id),
                display_name=_display_name_from_slug(model_id),
                module_path=module_path,
                class_name=class_name,
                family=family,
                call_params=("prompt", "images", "video", "interactions", "output_path", "fps"),
                stream_params=("prompt", "images", "video", "interactions", "output_path", "fps"),
                load_params=("model_path", "required_components", "device", "model_id"),
                supports_stream=True,
                supports_from_pretrained=True,
                supports_api_init=False,
                base_names=("ComponentPipeline", "PipelineABC"),
            )
        )
    return tuple(discovered)


def _discover_action_syntheses() -> tuple[_AstPipelineInfo, ...]:
    discovered: list[_AstPipelineInfo] = []
    for path in _action_synthesis_file_candidates():
        rel_path = path.relative_to(_source_root())
        module_path = ".".join(rel_path.with_suffix("").parts)
        family = path.parent.name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        method_cache: dict[str, dict[str, tuple[str, ...]]] = {}

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Synthesis"):
                continue
            base_names = tuple(
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "") for base in node.bases
            )
            model_id = _class_string_constant(node, "MODEL_ID")
            inherits_action_synthesis = any(
                base_name == "ActionModelSynthesis" or base_name.endswith("Synthesis") for base_name in base_names
            )
            if not model_id or not inherits_action_synthesis:
                continue
            methods = _resolve_class_methods(class_nodes, node.name, method_cache)

            slug = _normalize_model_id(model_id)
            discovered.append(
                _AstPipelineInfo(
                    model_id=slug,
                    display_name=_display_name_from_slug(slug),
                    module_path=module_path,
                    class_name=node.name,
                    family=family,
                    call_params=methods.get(
                        "predict",
                        methods.get("__call__", OFFICIAL_ACTION_CALL_PARAMS),
                    ),
                    stream_params=methods.get("stream", ()),
                    load_params=methods.get("from_pretrained", OFFICIAL_ACTION_LOAD_PARAMS),
                    supports_stream="stream" in methods,
                    supports_from_pretrained=True,
                    supports_api_init="api_init" in methods,
                    base_names=base_names,
                )
            )
    return tuple(discovered)


@lru_cache(maxsize=1)
def _discover_ast_pipelines() -> tuple[_AstPipelineInfo, ...]:
    discovered: list[_AstPipelineInfo] = []
    for path in _pipeline_file_candidates():
        rel_path = path.relative_to(_source_root())
        module_path = ".".join(rel_path.with_suffix("").parts)
        family = path.parent.name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        method_cache: dict[str, dict[str, tuple[str, ...]]] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue
            if not node.name.endswith("Pipeline"):
                continue
            if _class_bool_constant(node, "ABSTRACT_PIPELINE") is True:
                continue
            methods = _resolve_class_methods(class_nodes, node.name, method_cache)

            slug = _normalize_model_id(_class_string_constant(node, "MODEL_ID") or _slug_from_filename(path))
            display_name = _display_name_from_slug(slug)
            base_names = tuple(
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "") for base in node.bases
            )
            inherits_official_video = "OfficialVideoPipeline" in base_names
            inherits_native_diffusion = _class_inherits_any(
                class_nodes,
                node.name,
                frozenset({"NativeVisualDiffusionPipeline", "NativeTextToVideoPipeline"}),
            )
            inherited_official_call = _official_video_call_params() if inherits_official_video else ()
            inherited_official_load = _official_video_load_params() if inherits_official_video else ()
            inherited_native_call = _native_diffusion_video_call_params() if inherits_native_diffusion else ()
            inherited_native_load = _native_diffusion_video_load_params() if inherits_native_diffusion else ()
            discovered.append(
                _AstPipelineInfo(
                    model_id=slug,
                    display_name=display_name,
                    module_path=module_path,
                    class_name=node.name,
                    family=family,
                    call_params=methods.get("__call__", inherited_native_call or inherited_official_call),
                    stream_params=methods.get("stream", inherited_native_call or inherited_official_call),
                    load_params=methods.get(
                        "from_pretrained",
                        methods.get("api_init", inherited_native_load or inherited_official_load),
                    ),
                    supports_stream="stream" in methods or inherits_official_video or inherits_native_diffusion,
                    supports_from_pretrained=(
                        "from_pretrained" in methods or inherits_official_video or inherits_native_diffusion
                    ),
                    supports_api_init="api_init" in methods,
                    base_names=base_names,
                )
            )
    return tuple(discovered)


@lru_cache(maxsize=1)
def _discover_catalog_infos() -> tuple[_AstPipelineInfo, ...]:
    return (*_discover_ast_pipelines(), *_discover_component_pipelines(), *_discover_action_syntheses())


@lru_cache(maxsize=1)
def _catalog_model_names() -> dict[str, str]:
    """Return the model catalog's display name for each catalog model id."""
    from worldfoundry.zoo.catalog import load_model_zoo_registry

    return {entry.model_id: entry.name for entry in load_model_zoo_registry().list() if entry.name}


def _build_entry(info: _AstPipelineInfo) -> CatalogEntry:
    override = CURATED_OVERRIDES.get(info.model_id) or CURATED_OVERRIDES.get(_catalog_id_key(info.model_id), {})
    override = {**override, **studio_contract_override(info.model_id)}
    inferred_category = _category_from_family(info.family, info.class_name, info.call_params)
    if (
        inferred_category == "Video Generation"
        and info.module_path.startswith("worldfoundry.synthesis.action_generation.")
        and info.class_name.endswith("Synthesis")
    ):
        inferred_category = "Embodied Action"
    category = override.get(
        "category",
        inferred_category,
    )
    display_name = override.get("display_name") or _catalog_model_names().get(info.model_id) or info.display_name
    runtime_kind = override.get("runtime_kind", _runtime_kind_from_family(info.model_id, info.family))
    supports_from_pretrained = bool(override.get("supports_from_pretrained", info.supports_from_pretrained))
    supports_api_init = bool(override.get("supports_api_init", info.supports_api_init))
    default_backend = override.get(
        "default_backend",
        _default_backend(supports_api_init, supports_from_pretrained),
    )
    supports_stream = bool(override.get("supports_stream", info.supports_stream))
    summary = override.get(
        "summary",
        _summary_from_category(category, display_name, supports_stream),
    )
    if category in {"Video Generation", "Remote API"}:
        inferred_prompt = "A slow exploratory camera move through a believable scene with stable geometry and lighting."
    elif category in {"Embodied Action", "Visual Action"}:
        inferred_prompt = "Follow the instruction and predict the next executable action chunk."
    else:
        inferred_prompt = "Reconstruct the scene and surface geometry with stable camera motion."
    default_prompt = override.get("default_prompt", inferred_prompt)
    default_model_ref = resolve._resolve_override_value(override.get("default_model_ref", ""))
    default_load_kwargs = resolve._resolve_override_value(override.get("default_load_kwargs", {})) or {}
    extra_variants = _resolve_extra_variants(override.get("extra_variants", ()))
    call_params = tuple(resolve._resolve_override_value(override.get("call_params", info.call_params)) or ())
    profile_call_defaults = {
        key: copy.deepcopy(value)
        for key, value in resolve._runtime_profile_call_defaults(info.model_id)
        if key in call_params
    }
    default_call_kwargs = {
        **profile_call_defaults,
        **(resolve._resolve_override_value(override.get("default_call_kwargs", {})) or {}),
    }
    input_params = tuple(resolve._resolve_override_value(override.get("input_params", ())) or ())
    stream_params = tuple(resolve._resolve_override_value(override.get("stream_params", info.stream_params)) or ())
    load_params = tuple(resolve._resolve_override_value(override.get("load_params", info.load_params)) or ())
    if category in {"Embodied Action", "Visual Action"}:
        load_params = _append_dispatch_load_params(load_params)
    return CatalogEntry(
        model_id=info.model_id,
        display_name=display_name,
        module_path=info.module_path,
        class_name=info.class_name,
        family=info.family,
        category=category,
        summary=summary,
        call_params=call_params,
        input_params=input_params,
        stream_params=stream_params,
        load_params=load_params,
        supports_stream=supports_stream,
        supports_from_pretrained=supports_from_pretrained,
        supports_api_init=supports_api_init,
        runtime_kind=runtime_kind,
        default_backend=default_backend,
        default_model_ref=_resolve_default_model_ref(
            info.model_id,
            default_model_ref,
        ),
        default_endpoint=override.get("default_endpoint", ""),
        default_prompt=default_prompt,
        default_input_path=str(resolve._resolve_override_value(override.get("default_input_path", "")) or ""),
        default_task_type=override.get("default_task_type", ""),
        default_interactions=tuple(override.get("default_interactions", ())),
        default_load_kwargs=dict(default_load_kwargs),
        default_call_kwargs=dict(default_call_kwargs),
        extra_variants=extra_variants,
        suggested_task_types=tuple(override.get("suggested_task_types", ())),
        aliases=resolve._coerce_aliases(*override.get("aliases", ())),
        tags=tuple(override.get("tags", ())),
        env_hints=tuple(override.get("env_hints", ())),
        notes=override.get("notes", ""),
    )


@lru_cache(maxsize=1)
def discover_catalog() -> tuple[CatalogEntry, ...]:
    """Resolve shared checkpoint roots once per catalog build."""
    roots = studio_hfd_cache_roots(project_roots=resolve._project_root_candidates())
    children: dict[Path, frozenset[str]] = {}
    for root in roots:
        try:
            children[root] = frozenset(path.name for path in root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            children[root] = frozenset()
        except OSError:
            # Fall back to individual checks when a root cannot be enumerated.
            continue
    token = _CATALOG_CACHE_ROOTS.set(_CatalogPathSnapshot(roots, children))
    try:
        return _discover_catalog_entries()
    finally:
        _CATALOG_CACHE_ROOTS.reset(token)


def _discover_catalog_entries() -> tuple[CatalogEntry, ...]:
    by_module: dict[tuple[str, str], CatalogEntry] = {}
    for info in _discover_catalog_infos():
        entry = _build_entry(info)
        if entry.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if entry.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        if any(existing.model_id == entry.model_id for existing in by_module.values()):
            continue
        by_module[(entry.module_path, entry.class_name)] = entry

    for entry in _canonical_runtime_entries():
        if entry.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if entry.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        key = (entry.module_path, entry.class_name)
        previous = by_module.get(key)
        if previous is not None:
            merged_aliases = resolve._coerce_aliases(
                *previous.aliases, *entry.aliases, previous.model_id.replace("_", "-")
            )
            entry = CatalogEntry(
                model_id=entry.model_id,
                display_name=entry.display_name,
                module_path=entry.module_path,
                class_name=entry.class_name,
                family=entry.family,
                category=entry.category,
                summary=entry.summary,
                call_params=entry.call_params,
                input_params=entry.input_params,
                stream_params=entry.stream_params,
                load_params=entry.load_params,
                supports_stream=entry.supports_stream,
                supports_from_pretrained=entry.supports_from_pretrained,
                supports_api_init=entry.supports_api_init,
                runtime_kind=entry.runtime_kind,
                default_backend=entry.default_backend,
                default_model_ref=entry.default_model_ref,
                default_endpoint=entry.default_endpoint,
                default_prompt=entry.default_prompt,
                default_input_path=entry.default_input_path,
                default_task_type=entry.default_task_type,
                default_interactions=entry.default_interactions,
                default_load_kwargs=entry.default_load_kwargs,
                default_call_kwargs=entry.default_call_kwargs,
                extra_variants=entry.extra_variants or previous.extra_variants,
                suggested_task_types=entry.suggested_task_types,
                aliases=merged_aliases,
                tags=entry.tags,
                env_hints=entry.env_hints,
                notes=entry.notes or previous.notes,
            )
        by_module[key] = entry

    entries = list(by_module.values())
    entries.sort(key=lambda item: (item.category, item.display_name.lower(), item.model_id))
    return tuple(entries)


def _find_exact_entry_without_full_catalog(model_id: str) -> CatalogEntry | None:
    """Build one canonical or AST entry without resolving unrelated checkpoints.

    Canonical runtime defaults take precedence, as in the full catalog. Other
    aliases fall back to ``discover_catalog()`` for curated entry merging.
    """

    normalized = _normalize_model_id(model_id)
    if not normalized:
        return None
    parent_id = _sana_studio_parent_id(normalized) or _cogvideox_studio_parent_id(normalized)
    for entry in _canonical_runtime_entries(parent_id or normalized):
        if entry.model_id not in ABSTRACT_RUNTIME_MODEL_IDS | STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            return entry
    ast_model_id = FAST_CANONICAL_AST_ALIASES.get(parent_id or normalized, parent_id or normalized)
    for info in _discover_catalog_infos():
        if info.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if info.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        if _catalog_id_key(info.model_id) == _catalog_id_key(ast_model_id):
            entry = _build_entry(info)
            if parent_id is not None:
                entry = replace(
                    entry,
                    model_id=parent_id,
                    aliases=resolve._coerce_aliases(*entry.aliases, normalized, info.model_id),
                )
            elif normalized != ast_model_id:
                entry = replace(
                    entry,
                    model_id=normalized,
                    aliases=resolve._coerce_aliases(*entry.aliases, ast_model_id),
                )
            return entry
    return None


def find_runtime_entry(model_id: str) -> CatalogEntry:
    """Resolve the concrete pipeline entry, ignoring Studio family folding."""

    requested = _cogvideox_variant_id(model_id) or _catalog_id_key(model_id)
    if not requested:
        raise KeyError(f"Unknown Studio model id: {model_id}")
    canonical = next(iter(_canonical_runtime_entries(requested)), None)
    if canonical is not None:
        return canonical
    ast_model_id = FAST_CANONICAL_AST_ALIASES.get(requested, requested)
    for info in _discover_catalog_infos():
        if _catalog_id_key(info.model_id) == _catalog_id_key(ast_model_id):
            entry = _build_entry(info)
            if entry.model_id != requested:
                entry = replace(
                    entry,
                    model_id=requested,
                    aliases=resolve._coerce_aliases(*entry.aliases, entry.model_id),
                )
            return entry
    raise KeyError(f"Unknown Studio model id: {model_id}")


def find_entry(model_id: str) -> CatalogEntry:
    model_id = (model_id or "").strip().lower()
    exact_entry = _find_exact_entry_without_full_catalog(model_id)
    if exact_entry is not None:
        return exact_entry
    for entry in discover_catalog():
        if entry.model_id.lower() == model_id:
            return entry
        if model_id in {alias.lower() for alias in entry.aliases}:
            return entry
    raise KeyError(f"Unknown Studio model id: {model_id}")


LIVE_CONTROL_TAGS = frozenset(
    {
        "navigation",
        "camera-control",
        "interaction-heavy",
        "egocentric",
        "scene-context",
        "render-view",
        "trajectory",
    }
)


def template_id_hint(entry: CatalogEntry) -> str:
    """Classify the Studio template from catalog fields."""
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    if entry.category == "Remote API":
        return "hosted-api"
    if entry.category == "Video-to-Video":
        return "video-to-video"
    if entry.category == "Depth / Geometry":
        return "depth-geometry"
    if entry.category == "3D Scene":
        return "scene-3d"
    if entry.category == "Visual Action":
        return "visual-action"
    if entry.category == "Embodied Action":
        return "embodied-policy"
    if task_type in {"t2v", "text-video", "text-to-video", "video-generation"}:
        return "text-video"
    if task_type in {"t2i", "text-to-image", "image-generation", "class-conditional-image-generation"}:
        return "text-video"
    if task_type in {"i2v", "image-video", "image-to-video", "video-to-video", "v2v"}:
        return "conditioned-video"
    if "interactive-world" in entry.tags:
        return "interactive-world"
    if entry.family == "world_model":
        return "interactive-world"
    if "image-to-world" in entry.tags or "video-to-world" in entry.tags:
        return "interactive-world"
    if entry.supports_stream and (entry.default_interactions or set(entry.tags) & LIVE_CONTROL_TAGS):
        return "interactive-world"
    model_id = entry.model_id.lower().replace("_", "-")
    if "text-to-video" in entry.tags or "-t2v" in model_id or model_id.endswith("t2v"):
        return "text-video"
    if entry.category == "Video Generation":
        if "i2v" in entry.tags or "image-to-video" in entry.tags:
            return "conditioned-video"
        if "video-to-video" in entry.tags or "v2v" in entry.tags:
            return "conditioned-video"
        if "images" in entry.call_params:
            return "conditioned-video"
        return "text-video"
    return "text-video"
