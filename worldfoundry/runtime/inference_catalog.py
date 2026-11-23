"""Registry of curated per-model-family inference specs.

The specs themselves live in :mod:`worldfoundry.runtime.inference_specs`, one
module per family, and hold the model identity that
:mod:`worldfoundry.core.execution.inference` must not: checkpoint locations,
demo fixtures, default prompts, call kwargs, and task/variant profiles. This
module registers them and provides the lookups
(:func:`get_model_inference_spec`, :func:`list_model_inference_specs`) and the
generic fallback builder used for uncurated families.

Generation defaults live in ``data/models/runtime/profiles/<model-id>.yaml``.
Task ``call_kwargs`` fields left without a default are filled from that
profile when a spec is looked up, so the same value is not written twice.

The spec dataclasses themselves are core-owned contracts; this module only
populates them.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import cache, lru_cache
from typing import Any

import yaml

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.execution.inference import normalise_infer_id as _normalise_infer_id
from worldfoundry.core.io.paths import package_data_path
from worldfoundry.runtime.inference_specs._common import _TEST_CASES_ROOT, GENERIC_IMAGE_FIXTURE, _artifact, _field
from worldfoundry.runtime.inference_specs.adaworld import ADAWORLD_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.astra import ASTRA_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.bernini import BERNINI_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.cameractrl import CAMERACTRL_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.cosmos3 import COSMOS3_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.cosmos_predict2p5 import COSMOS_PREDICT2P5_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.ctrl_world import CTRL_WORLD_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.cut3r import CUT3R_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.depth_anything3 import DEPTH_ANYTHING3_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.diamond import DIAMOND_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.dino_wm import DINO_WM_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.dreamdojo import DREAMDOJO_STUDENT_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.fantasyworld import (
    FANTASYWORLD_INFERENCE_SPEC,
    FANTASYWORLD_WAN21_INFERENCE_SPEC,
)
from worldfoundry.runtime.inference_specs.fastframes import FASTFRAMES_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.flashworld import FLASHWORLD_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.gen3c import GEN3C_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.genie_envisioner import GENIE_ENVISIONER_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.giga_world_0 import GIGA_WORLD_0_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.h3_tiny import H3_TINY_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.h3_world import H3_WORLD_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.helios import HELIOS_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.hma import HMA_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.hunyuan_gamecraft import HUNYUAN_GAMECRAFT_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.hy_worldplay import HY_WORLDPLAY_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.hyperflow import HYPERFLOW_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.leworld_model import LEWORLD_MODEL_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.lingbot_video import LINGBOT_VIDEO_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.lingbot_world import (
    LINGBOT_WORLD_INFERENCE_SPEC,
    LINGBOT_WORLD_V2_INFERENCE_SPEC,
    LINGBOT_WORLD_V2_SMALL_INFERENCE_SPEC,
)
from worldfoundry.runtime.inference_specs.longcat_video import LONGCAT_VIDEO_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.ltx_v2v import LTX2_V2V_INFERENCE_SPEC, LTX23_V2V_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.lyra1 import LYRA1_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.lyra2 import LYRA2_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.matrix_game4 import MATRIX_GAME4_EGYPT_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.matrix_game_2 import MATRIX_GAME_2_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.matrix_game_3 import MATRIX_GAME_3_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.mira import MIRA_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.neoverse import NEOVERSE_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.pi_h3 import PI_H3_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.pika_h3_flex import PIKA_H3_FLEX_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.rolling_forcing import ROLLING_FORCING_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.rt1 import RT1_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.sana_streaming import (
    SANA_STREAMING_BIDIRECTIONAL_INFERENCE_SPEC,
    SANA_STREAMING_INFERENCE_SPEC,
)
from worldfoundry.runtime.inference_specs.stable_video_infinity import STABLE_VIDEO_INFINITY_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.starwm import STARWM_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.tesseract import TESSERACT_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.uni3c import UNI3C_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.uniworld_view import UNIWORLD_VIEW_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.wan2p2 import WAN2P2_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.warp_as_history import WARP_AS_HISTORY_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.wonderworld import WONDERWORLD_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.worldfm import WORLDFM_INFERENCE_SPEC
from worldfoundry.runtime.inference_specs.wow import WOW_INFERENCE_SPEC
from worldfoundry.runtime.interactive_inference_catalog import INTERACTIVE_INFERENCE_SPECS

GENERIC_3D_FIXTURE = str(_TEST_CASES_ROOT / "vggt" / "examples" / "kitchen" / "images")


GENERIC_GEOMETRY_FIXTURE = str(_TEST_CASES_ROOT / "images" / "000.png")


GENERIC_ACTION_FIXTURE = str(_TEST_CASES_ROOT / "test_vla_case1" / "droid" / "exterior_image_1_left.png")


_STATIC_ASSET_GATED_WORLD_RUNTIME_MODEL_IDS = frozenset(
    {
        "adaworld",
        "causal-rcm",
        "ctrl-world",
        "diamond",
        "dino-wm",
        "genie-envisioner",
        "giga-world-0",
        "happyoyster",
        "hma",
        "hunyuanworld-1",
        "leworldmodel",
        "mineworld",
        "mosaicmem",
        "motionbricks",
        "omniforcing",
        "oasis-500m",
        "pointworld",
        "sana-wm",
        "simworld",
        "starwm",
        "tesseract",
        "uwm",
        "vggt-world",
        "vid2world",
        "wildworld",
        "worldgrow",
        "worldmem",
    }
)


def _asset_gated_world_runtime_model_ids() -> frozenset[str]:
    ids = set(_STATIC_ASSET_GATED_WORLD_RUNTIME_MODEL_IDS)
    if os.getenv("WORLDFOUNDRY_INFERENCE_LOAD_RUNTIME_ASSET_GATES", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return frozenset(ids)
    try:
        from worldfoundry.synthesis.visual_generation.runtime_manifest import WORLD_MODEL_RUNTIME_SPECS
    except Exception:
        return frozenset(ids)
    for model_id, spec in WORLD_MODEL_RUNTIME_SPECS.items():
        if getattr(spec, "blocked_reason", None):
            ids.add(str(model_id))
    return frozenset(ids)


ASSET_GATED_WORLD_RUNTIME_MODEL_IDS = _asset_gated_world_runtime_model_ids()


TEXT_ONLY_DEFAULT_VIDEO_MODEL_IDS = frozenset(
    {
        "causal-forcing",
        "fastframes",
        "rolling-forcing",
        "self-forcing",
    }
)


_MISSING_DEFAULT = object()


def _default_call_value(defaults: Mapping[str, Any], *names: str) -> Any:
    lookup = {_normalise_infer_id(key): value for key, value in defaults.items()}
    for name in names:
        key = _normalise_infer_id(name)
        if key in lookup:
            return lookup[key]
    return _MISSING_DEFAULT


def _infer_field_kind(field_id: str, default: Any = _MISSING_DEFAULT) -> str:
    if default is not _MISSING_DEFAULT:
        if isinstance(default, bool):
            return "boolean"
        if isinstance(default, int) and not isinstance(default, bool):
            return "integer"
        if isinstance(default, float):
            return "number"
        if isinstance(default, (list, tuple, dict)):
            return "json"

    normalized = _normalise_infer_id(field_id)
    if normalized.endswith(("-path", "-dir")):
        return "path"
    if normalized == "grayscale" or normalized.startswith(
        ("allow-", "apply-", "create-", "disable-", "enable-", "force-", "is-", "trim-", "use-", "visualize-")
    ):
        return "boolean"
    if (
        normalized.startswith(("num-", "max-", "min-"))
        or normalized.endswith(("-frames", "-steps", "-clips", "-degree", "-height", "-width", "-fps", "-seed"))
        or normalized in {"height", "width", "fps", "seed"}
    ):
        return "integer"
    if any(token in normalized for token in ("scale", "shift", "speed", "distance", "threshold", "rate", "guidance")):
        return "number"
    return "string"


_MODEL_INFERENCE_SPECS: dict[str, ModelInferenceSpec] = {
    "hyperflow": HYPERFLOW_INFERENCE_SPEC,
    "h3-tiny": H3_TINY_INFERENCE_SPEC,
    "h3-world": H3_WORLD_INFERENCE_SPEC,
    "pi-h3": PI_H3_INFERENCE_SPEC,
    "pika-h3-flex": PIKA_H3_FLEX_INFERENCE_SPEC,
    "fastframes": FASTFRAMES_INFERENCE_SPEC,
    RT1_INFERENCE_SPEC.model_family_id: RT1_INFERENCE_SPEC,
    DREAMDOJO_STUDENT_INFERENCE_SPEC.model_family_id: DREAMDOJO_STUDENT_INFERENCE_SPEC,
    MATRIX_GAME4_EGYPT_INFERENCE_SPEC.model_family_id: MATRIX_GAME4_EGYPT_INFERENCE_SPEC,
    WONDERWORLD_INFERENCE_SPEC.model_family_id: WONDERWORLD_INFERENCE_SPEC,
    ADAWORLD_INFERENCE_SPEC.model_family_id: ADAWORLD_INFERENCE_SPEC,
    UNIWORLD_VIEW_INFERENCE_SPEC.model_family_id: UNIWORLD_VIEW_INFERENCE_SPEC,
    **INTERACTIVE_INFERENCE_SPECS,
    LTX2_V2V_INFERENCE_SPEC.model_family_id: LTX2_V2V_INFERENCE_SPEC,
    LTX23_V2V_INFERENCE_SPEC.model_family_id: LTX23_V2V_INFERENCE_SPEC,
    GENIE_ENVISIONER_INFERENCE_SPEC.model_family_id: GENIE_ENVISIONER_INFERENCE_SPEC,
    GIGA_WORLD_0_INFERENCE_SPEC.model_family_id: GIGA_WORLD_0_INFERENCE_SPEC,
    CTRL_WORLD_INFERENCE_SPEC.model_family_id: CTRL_WORLD_INFERENCE_SPEC,
    DIAMOND_INFERENCE_SPEC.model_family_id: DIAMOND_INFERENCE_SPEC,
    DINO_WM_INFERENCE_SPEC.model_family_id: DINO_WM_INFERENCE_SPEC,
    HMA_INFERENCE_SPEC.model_family_id: HMA_INFERENCE_SPEC,
    TESSERACT_INFERENCE_SPEC.model_family_id: TESSERACT_INFERENCE_SPEC,
    LEWORLD_MODEL_INFERENCE_SPEC.model_family_id: LEWORLD_MODEL_INFERENCE_SPEC,
    LINGBOT_WORLD_INFERENCE_SPEC.model_family_id: LINGBOT_WORLD_INFERENCE_SPEC,
    LINGBOT_WORLD_V2_INFERENCE_SPEC.model_family_id: LINGBOT_WORLD_V2_INFERENCE_SPEC,
    LINGBOT_WORLD_V2_SMALL_INFERENCE_SPEC.model_family_id: LINGBOT_WORLD_V2_SMALL_INFERENCE_SPEC,
    ROLLING_FORCING_INFERENCE_SPEC.model_family_id: ROLLING_FORCING_INFERENCE_SPEC,
    STARWM_INFERENCE_SPEC.model_family_id: STARWM_INFERENCE_SPEC,
    WARP_AS_HISTORY_INFERENCE_SPEC.model_family_id: WARP_AS_HISTORY_INFERENCE_SPEC,
    MATRIX_GAME_2_INFERENCE_SPEC.model_family_id: MATRIX_GAME_2_INFERENCE_SPEC,
    MATRIX_GAME_3_INFERENCE_SPEC.model_family_id: MATRIX_GAME_3_INFERENCE_SPEC,
    MIRA_INFERENCE_SPEC.model_family_id: MIRA_INFERENCE_SPEC,
    ASTRA_INFERENCE_SPEC.model_family_id: ASTRA_INFERENCE_SPEC,
    NEOVERSE_INFERENCE_SPEC.model_family_id: NEOVERSE_INFERENCE_SPEC,
    COSMOS3_INFERENCE_SPEC.model_family_id: COSMOS3_INFERENCE_SPEC,
    COSMOS_PREDICT2P5_INFERENCE_SPEC.model_family_id: COSMOS_PREDICT2P5_INFERENCE_SPEC,
    UNI3C_INFERENCE_SPEC.model_family_id: UNI3C_INFERENCE_SPEC,
    GEN3C_INFERENCE_SPEC.model_family_id: GEN3C_INFERENCE_SPEC,
    FANTASYWORLD_INFERENCE_SPEC.model_family_id: FANTASYWORLD_INFERENCE_SPEC,
    FANTASYWORLD_WAN21_INFERENCE_SPEC.model_family_id: FANTASYWORLD_WAN21_INFERENCE_SPEC,
    FLASHWORLD_INFERENCE_SPEC.model_family_id: FLASHWORLD_INFERENCE_SPEC,
    HELIOS_INFERENCE_SPEC.model_family_id: HELIOS_INFERENCE_SPEC,
    SANA_STREAMING_INFERENCE_SPEC.model_family_id: SANA_STREAMING_INFERENCE_SPEC,
    SANA_STREAMING_BIDIRECTIONAL_INFERENCE_SPEC.model_family_id: SANA_STREAMING_BIDIRECTIONAL_INFERENCE_SPEC,
    BERNINI_INFERENCE_SPEC.model_family_id: BERNINI_INFERENCE_SPEC,
    LYRA1_INFERENCE_SPEC.model_family_id: LYRA1_INFERENCE_SPEC,
    LYRA2_INFERENCE_SPEC.model_family_id: LYRA2_INFERENCE_SPEC,
    LINGBOT_VIDEO_INFERENCE_SPEC.model_family_id: LINGBOT_VIDEO_INFERENCE_SPEC,
    LONGCAT_VIDEO_INFERENCE_SPEC.model_family_id: LONGCAT_VIDEO_INFERENCE_SPEC,
    STABLE_VIDEO_INFINITY_INFERENCE_SPEC.model_family_id: STABLE_VIDEO_INFINITY_INFERENCE_SPEC,
    HUNYUAN_GAMECRAFT_INFERENCE_SPEC.model_family_id: HUNYUAN_GAMECRAFT_INFERENCE_SPEC,
    HY_WORLDPLAY_INFERENCE_SPEC.model_family_id: HY_WORLDPLAY_INFERENCE_SPEC,
    CAMERACTRL_INFERENCE_SPEC.model_family_id: CAMERACTRL_INFERENCE_SPEC,
    DEPTH_ANYTHING3_INFERENCE_SPEC.model_family_id: DEPTH_ANYTHING3_INFERENCE_SPEC,
    WAN2P2_INFERENCE_SPEC.model_family_id: WAN2P2_INFERENCE_SPEC,
    CUT3R_INFERENCE_SPEC.model_family_id: CUT3R_INFERENCE_SPEC,
    WORLDFM_INFERENCE_SPEC.model_family_id: WORLDFM_INFERENCE_SPEC,
    WOW_INFERENCE_SPEC.model_family_id: WOW_INFERENCE_SPEC,
}


@cache
def _profile_call_defaults(model_family_id: str) -> Mapping[str, Any]:
    """Generation defaults declared by the model family's runtime profile."""

    path = package_data_path("models", "runtime", "profiles", f"{model_family_id}.yaml")
    if not path.is_file():
        return {}
    execution = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("execution") or {}
    defaults = execution.get("default_parameters") or execution.get("defaults") or {}
    return defaults if isinstance(defaults, Mapping) else {}


def _with_profile_defaults(spec: ModelInferenceSpec) -> ModelInferenceSpec:
    """Fill task ``call_kwargs`` fields the spec leaves unset from the runtime profile."""

    defaults = _profile_call_defaults(spec.model_family_id)
    if not defaults:
        return spec
    tasks = []
    for task in spec.tasks:
        profiled = {field.field_id for field in task.inputs if field.target == "call_kwargs"} & set(defaults)
        inputs = tuple(
            replace(field, default=copy.deepcopy(defaults[field.field_id]))
            if field.field_id in profiled and field.default is None
            else field
            for field in task.inputs
        )
        call_kwargs = {key: copy.deepcopy(defaults[key]) for key in sorted(profiled)}
        call_kwargs.update(task.default_call_kwargs)
        tasks.append(replace(task, inputs=inputs, default_call_kwargs=call_kwargs))
    return replace(spec, tasks=tuple(tasks))


@lru_cache(maxsize=1)
def _resolved_inference_specs() -> tuple[ModelInferenceSpec, ...]:
    return tuple(_with_profile_defaults(spec) for spec in _MODEL_INFERENCE_SPECS.values())


def list_model_inference_specs() -> tuple[ModelInferenceSpec, ...]:
    """Return curated model inference specs with explicit variants and task profiles."""

    return _resolved_inference_specs()


def get_model_inference_spec(model_family_id: str) -> ModelInferenceSpec | None:
    """Return a curated inference spec for a model family when one exists."""

    key = _normalise_infer_id(model_family_id)
    for spec in _resolved_inference_specs():
        keys = (spec.model_family_id, *spec.aliases)
        if key in {_normalise_infer_id(item) for item in keys}:
            return spec
    return None


def generic_model_inference_spec(
    *,
    model_family_id: str,
    display_name: str | None = None,
    default_model_ref: str = "",
    default_load_kwargs: Mapping[str, Any] | None = None,
    default_call_kwargs: Mapping[str, Any] | None = None,
    supports_stream: bool = False,
    workload_type: str = "",
    supported_call_params: Sequence[str] | None = None,
) -> ModelInferenceSpec:
    """Build a conservative fallback spec for models not yet curated."""

    workload = _normalise_infer_id(workload_type)
    supported = {_normalise_infer_id(item) for item in supported_call_params or ()}
    strict = supported_call_params is not None
    call_defaults = dict(default_call_kwargs or {})
    text_only_default = _normalise_infer_id(model_family_id) in TEXT_ONLY_DEFAULT_VIDEO_MODEL_IDS
    is_asset_gated_world_runtime = _normalise_infer_id(model_family_id) in ASSET_GATED_WORLD_RUNTIME_MODEL_IDS
    if is_asset_gated_world_runtime:
        call_defaults.setdefault("plan_only", True)
        call_defaults.setdefault("timeout_seconds", 21600)

    def has_any(*names: str) -> bool:
        return not strict or bool({_normalise_infer_id(name) for name in names} & supported)

    def field_default(*names: str) -> Any:
        value = _default_call_value(call_defaults, *names)
        return None if value is _MISSING_DEFAULT else value

    def generic_input_default() -> str | None:
        configured = field_default("input_path", "data_path", "image", "image_path", "video", "video_path")
        if configured not in (None, ""):
            return configured
        if workload == "3d":
            return GENERIC_3D_FIXTURE
        if workload in {"geometry", "depth", "depth-geometry"}:
            return GENERIC_GEOMETRY_FIXTURE
        if workload == "action":
            return GENERIC_ACTION_FIXTURE
        wants_video = bool({"video", "videos", "video-path"} & supported)
        wants_image = bool({"image", "images", "image-path"} & supported)
        if wants_video and not wants_image and workload not in {"i2v"}:
            return None
        return GENERIC_IMAGE_FIXTURE

    def add_common_fields(fields: list[InferenceFieldSpec]) -> None:
        if has_any("prompt"):
            fields.append(_field("prompt", "Prompt", target="prompt"))
        if has_any("negative_prompt", "negative-prompt"):
            fields.append(
                _field(
                    "negative_prompt",
                    "Negative Prompt",
                    target="call_kwargs",
                    default=field_default("negative_prompt", "negative-prompt"),
                )
            )
        if (
            has_any(
                "input_path",
                "data_path",
                "image",
                "images",
                "image_path",
                "video",
                "videos",
                "video_path",
                "input",
                "input_",
            )
            and not text_only_default
        ):
            fields.append(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=generic_input_default(),
                )
            )
        if has_any("interactions", "interaction_signal", "interaction", "action"):
            fields.append(
                _field(
                    "interactions",
                    "Interactions",
                    kind="interaction_tokens",
                    target="params",
                    default=field_default("interactions", "interaction_signal", "interaction", "action"),
                )
            )
        if has_any("num_frames", "frames", "video_length"):
            fields.append(
                _field(
                    "frames",
                    "Frames",
                    kind="integer",
                    target="params",
                    default=field_default("num_frames", "frames", "video_length"),
                )
            )
        if has_any("fps"):
            fields.append(_field("fps", "FPS", kind="integer", target="params", default=field_default("fps")))
        if has_any("height", "user_height", "output_h", "resize_h", "image_height"):
            fields.append(
                _field(
                    "height",
                    "Height",
                    kind="integer",
                    target="params",
                    default=field_default("height", "user_height", "output_h", "resize_h", "image_height"),
                )
            )
        if has_any("width", "user_width", "output_w", "resize_w", "image_width"):
            fields.append(
                _field(
                    "width",
                    "Width",
                    kind="integer",
                    target="params",
                    default=field_default("width", "user_width", "output_w", "resize_w", "image_width"),
                )
            )
        if has_any("num_inference_steps", "sampling_steps", "infer_steps", "num_steps", "steps"):
            fields.append(
                _field(
                    "steps",
                    "Steps",
                    kind="integer",
                    target="params",
                    default=field_default(
                        "num_inference_steps",
                        "sampling_steps",
                        "infer_steps",
                        "num_steps",
                        "steps",
                    ),
                )
            )
        if has_any("guidance_scale", "cfg_scale", "scale"):
            fields.append(
                _field(
                    "guidance_scale",
                    "Guidance",
                    kind="number",
                    target="params",
                    default=field_default("guidance_scale", "cfg_scale", "scale"),
                )
            )
        if has_any("seed"):
            fields.append(_field("seed", "Seed", kind="integer", target="params", default=field_default("seed")))
        if has_any("output_path", "output_save_path"):
            fields.append(
                _field(
                    "output_path",
                    "Output Path",
                    kind="path",
                    target="call_kwargs",
                    default=field_default("output_path", "output_save_path"),
                )
            )
        if is_asset_gated_world_runtime and has_any("plan_only"):
            fields.append(
                _field(
                    "plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=field_default("plan_only")
                )
            )
        if is_asset_gated_world_runtime and has_any("timeout_seconds", "timeout-seconds"):
            fields.append(
                _field(
                    "timeout_seconds",
                    "Timeout Seconds",
                    kind="integer",
                    target="call_kwargs",
                    default=field_default("timeout_seconds", "timeout-seconds"),
                )
            )

    def streaming_fallback_fields() -> tuple[InferenceFieldSpec, ...]:
        return (
            _field("prompt", "Prompt", target="prompt"),
            _field(
                "input_path",
                "Input Path",
                kind="path",
                target="input_path",
                required=True,
                default=generic_input_default(),
            ),
            _field(
                "interactions",
                "Interactions",
                kind="interaction_tokens",
                target="params",
                default=("forward",),
            ),
            _field("frames", "Frames", kind="integer", target="params", default=16),
            _field("fps", "FPS", kind="integer", target="params", default=16),
            _field("seed", "Seed", kind="integer", target="params", default=42),
        )

    def extra_fields() -> list[InferenceFieldSpec]:
        if not strict:
            return []
        consumed = {
            "prompt",
            "negative-prompt",
            "input-path",
            "data-path",
            "image",
            "images",
            "image-path",
            "video",
            "videos",
            "video-path",
            "input",
            "input-",
            "interactions",
            "interaction-signal",
            "interaction",
            "action",
            "num-frames",
            "frames",
            "video-length",
            "fps",
            "height",
            "user-height",
            "output-h",
            "resize-h",
            "image-height",
            "width",
            "user-width",
            "output-w",
            "resize-w",
            "image-width",
            "num-inference-steps",
            "sampling-steps",
            "infer-steps",
            "num-steps",
            "steps",
            "guidance-scale",
            "cfg-scale",
            "scale",
            "seed",
            "output-path",
            "output-save-path",
            "return-dict",
            "output-dir",
        }
        rows: list[InferenceFieldSpec] = []
        for raw_name in supported_call_params or ():
            field_id = _normalise_infer_id(raw_name)
            if field_id in consumed or field_id in {item.field_id for item in rows}:
                continue
            default = field_default(raw_name, field_id)
            rows.append(
                _field(
                    field_id,
                    raw_name.replace("_", " ").replace("-", " ").title(),
                    kind=_infer_field_kind(field_id, default if default is not None else _MISSING_DEFAULT),
                    default=default,
                )
            )
        return rows

    is_geometry = workload in {"geometry", "depth", "depth-geometry"}
    is_action = workload in {"action", "embodied", "embodied-policy", "robotics", "visual-action"}
    is_video_to_audio = workload in {"v2a", "video-to-audio"}
    is_image = workload in {
        "image",
        "t2i",
        "text-to-image",
        "image-generation",
        "class-conditional-image-generation",
        "class-conditional-generation",
    }
    is_video = not is_action and (
        not is_image
        and (
            supports_stream
            or workload in {"t2v", "i2v", "v2v", "v2a", "video", "world", "video-to-video", "video-to-audio"}
        )
    )
    task_id = (
        "video-to-audio"
        if is_video_to_audio
        else ("image-generation" if is_image else ("interactive-video" if supports_stream else "default"))
    )
    if is_geometry:
        task_label = "Depth / Geometry Inference"
        fields: list[InferenceFieldSpec] = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(
            fields
            or (
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=generic_input_default(),
                ),
            )
        )
        task_outputs = (
            _artifact("geometry", "geometry", required=True, preview=True),
            _artifact("manifest", "manifest", required=True),
        )
    elif workload == "3d":
        task_label = "3D Reconstruction"
        fields = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(
            fields
            or (
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=generic_input_default(),
                ),
            )
        )
        task_outputs = (
            _artifact("scene", "3d", required=True, preview=True),
            _artifact("manifest", "manifest", required=True),
        )
    elif is_action:
        task_label = "Action Inference"
        fields = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(fields) or (streaming_fallback_fields() if supports_stream else ())
        artifact_kind = (
            "action_tokens"
            if _normalise_infer_id(model_family_id) == "lapa" or workload == "visual-action"
            else "action_trace"
        )
        task_outputs = (
            _artifact(artifact_kind, artifact_kind, required=True, preview=False),
            _artifact("manifest", "manifest", required=True),
        )
    elif is_image:
        task_label = "Image Inference"
        fields = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(field for field in fields if field.target != "input_path")
        task_outputs = (
            _artifact("image", "generated_image", required=True, preview=True),
            _artifact("manifest", "manifest", required=True),
        )
    elif is_video:
        task_label = (
            "Runtime Plan"
            if is_asset_gated_world_runtime
            else (
                "Video-to-Audio Inference"
                if is_video_to_audio
                else ("Interactive Video" if supports_stream else "Video Inference")
            )
        )
        fields = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(fields) or (streaming_fallback_fields() if supports_stream else ())
        if is_asset_gated_world_runtime:
            task_outputs = (
                _artifact(
                    "runtime_plan",
                    "manifest",
                    required=True,
                    description="Runtime requirement plan emitted when official checkpoints or assets are not configured.",
                ),
            )
        elif is_video_to_audio:
            task_outputs = (
                _artifact("audio", "audio", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            )
        else:
            task_outputs = (
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            )
    else:
        task_label = "Default Inference"
        fields = []
        add_common_fields(fields)
        fields.extend(extra_fields())
        task_inputs = tuple(fields)
        task_outputs = (
            _artifact("primary", "artifact", required=True, preview=True),
            _artifact("manifest", "manifest", required=True),
        )
    return ModelInferenceSpec(
        model_family_id=model_family_id,
        display_name=display_name or model_family_id,
        default_variant_id="default",
        default_task_id=task_id,
        variants=(
            InferenceVariantSpec(
                variant_id="default",
                label="Default",
                checkpoints=(
                    (InferenceCheckpointRef(role="primary", uri=default_model_ref, status="configured"),)
                    if default_model_ref
                    else ()
                ),
                status="configured",
                load_kwargs=dict(default_load_kwargs or {}),
                call_kwargs=dict(default_call_kwargs or {}),
            ),
        ),
        tasks=(
            InferenceTaskProfile(
                task_id=task_id,
                label=task_label,
                description="Generic WorldFoundry inference task profile.",
                inputs=task_inputs,
                outputs=task_outputs,
                default_call_kwargs=dict(default_call_kwargs or {}),
            ),
        ),
    )


def model_inference_spec(
    *,
    model_family_id: str,
    display_name: str | None = None,
    default_model_ref: str = "",
    default_load_kwargs: Mapping[str, Any] | None = None,
    default_call_kwargs: Mapping[str, Any] | None = None,
    supports_stream: bool = False,
    workload_type: str = "",
    supported_call_params: Sequence[str] | None = None,
) -> ModelInferenceSpec:
    """Return curated spec for a model family, or a generic fallback."""

    return get_model_inference_spec(model_family_id) or generic_model_inference_spec(
        model_family_id=model_family_id,
        display_name=display_name,
        default_model_ref=default_model_ref,
        default_load_kwargs=default_load_kwargs,
        default_call_kwargs=default_call_kwargs,
        supports_stream=supports_stream,
        workload_type=workload_type,
        supported_call_params=supported_call_params,
    )
