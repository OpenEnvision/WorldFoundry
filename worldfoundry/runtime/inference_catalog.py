"""Curated per-model-family inference specs.

Holds the model identity that :mod:`worldfoundry.core.execution.inference` must not:
checkpoint locations, demo fixtures, default prompts, call kwargs, and the
task/variant profiles for every curated model family, plus the registry
lookups (:func:`get_model_inference_spec`, :func:`list_model_inference_specs`)
and the generic fallback builder used for uncurated families.

The spec dataclasses themselves are core-owned contracts; this module only
populates them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.execution.inference import (
    normalise_infer_id as _normalise_infer_id,
)
from worldfoundry.core.io.paths import (
    checkpoint_root_candidates,
    checkpoint_root_path,
    hfd_root_path,
    local_data_root_path,
    official_runtime_repo_path,
    project_root,
)
from worldfoundry.runtime.hyperflow_inference_spec import HYPERFLOW_INFERENCE_SPEC
from worldfoundry.runtime.interactive_inference_catalog import INTERACTIVE_INFERENCE_SPECS
from worldfoundry.runtime.rt1_inference_spec import RT1_INFERENCE_SPEC
from worldfoundry.runtime.uniworld_view_inference_spec import UNIWORLD_VIEW_INFERENCE_SPEC

LINGBOT_WORLD_MODEL_ID = "lingbot-world"
LINGBOT_VARIANT_FAST = "fast"
LINGBOT_VARIANT_BASE_CAM = "base-cam"
LINGBOT_VARIANT_BASE_ACT_PREVIEW = "base-act-preview"
LINGBOT_OFFICIAL_PROMPT = (
    "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's "
    "blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle "
    "approaches steadily, its stone details becoming clearer against the backdrop of floating islands "
    "and distant waterfalls."
)


def _official_repo_path(repo_name: str, *parts: str) -> str:
    return str(official_runtime_repo_path(repo_name).joinpath(*parts))


def _first_existing_path(*candidates: str | Path) -> str:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return str(Path(candidates[0]).expanduser()) if candidates else ""


def _local_hfd_checkpoint_path(repo_dir_name: str, *parts: str) -> str:
    """Resolve a flat-HFD export while respecting an explicit HFD root.

    Shared Workspace installations commonly keep flat ``owner--repo`` exports
    directly below the adjacent plural ``ckpts`` directory.  The generic HFD
    default, however, is ``ckpt/hfd`` when a singular ``ckpt`` directory also
    exists.  Discover the conventional checkpoint-root sibling only when the
    caller did not explicitly select an HFD root.
    """

    configured = hfd_root_path(repo_dir_name, *parts)
    if os.environ.get("WORLDFOUNDRY_HFD_ROOT"):
        return str(configured)
    return _first_existing_path(
        configured,
        *checkpoint_root_candidates(repo_dir_name, *parts),
    )


def _read_text_file_or_default(path: str | Path, default: str) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return default
    return value or default


_TEST_CASES_ROOT = Path(__file__).resolve().parents[1] / "data" / "test_cases"
_RUNTIME_CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "data" / "models" / "runtime" / "configs"
_PROJECT_ROOT = project_root(__file__)
_WORKSPACE_ROOT = _PROJECT_ROOT.parent
DINO_WM_CHECKPOINT_BASE = str(
    Path(
        os.environ.get("WORLDFOUNDRY_DINO_WM_CKPT_BASE")
        or _first_existing_path(*checkpoint_root_candidates("dino_wm"))
    ).expanduser().resolve()
)
GENERIC_IMAGE_FIXTURE = str(_TEST_CASES_ROOT / "studio_demo" / "00" / "image.jpg")
GENERIC_3D_FIXTURE = str(_TEST_CASES_ROOT / "vggt" / "examples" / "kitchen" / "images")
GENERIC_GEOMETRY_FIXTURE = str(_TEST_CASES_ROOT / "images" / "000.png")
GENERIC_ACTION_FIXTURE = str(_TEST_CASES_ROOT / "test_vla_case1" / "droid" / "exterior_image_1_left.png")
UNI3C_DEMO_ROOT = _TEST_CASES_ROOT / "uni3c" / "synthetic-forward-9f-768x480"
UNI3C_DEMO_REFERENCE_IMAGE = str(UNI3C_DEMO_ROOT / "reference.jpg")
UNI3C_DEMO_RENDER_BUNDLE = str(UNI3C_DEMO_ROOT)
MIRA_DEFAULT_CHECKPOINT = str(checkpoint_root_path("mira"))
MIRA_DEFAULT_DATASET = str(local_data_root_path() / "rocket-science" / "test")
_WOW_OFFICIAL_LOCAL_CHECKPOINT = _WORKSPACE_ROOT / "ckpt" / "WoW-1-Wan-14B-600k"
WOW_LOCAL_CHECKPOINT = (
    str(_WOW_OFFICIAL_LOCAL_CHECKPOINT)
    if _WOW_OFFICIAL_LOCAL_CHECKPOINT.exists()
    else "X-Humanoid/WoW-1-Wan-14B-600k"
)
HELIOS_CHECKPOINT_ROOT = _WORKSPACE_ROOT / "ckpt"
HELIOS_BASE_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Base")
HELIOS_MID_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Mid")
HELIOS_DISTILLED_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Distilled")
SANA_STREAMING_CHECKPOINT = _local_hfd_checkpoint_path(
    "Efficient-Large-Model--SANA-Streaming", "dit", "sana_streaming_ar.pth"
)
SANA_STREAMING_BIDIRECTIONAL_CHECKPOINT = _local_hfd_checkpoint_path(
    "Efficient-Large-Model--SANA-Streaming_bidirectional",
    "dit",
    "sana_bidirectional_short.pth",
)
SANA_STREAMING_VAE = _local_hfd_checkpoint_path("Lightricks--LTX-2")
SANA_STREAMING_TEXT_ENCODER = _local_hfd_checkpoint_path("Efficient-Large-Model--gemma-2-2b-it")
SANA_STREAMING_DEMO_PROMPT = (
    "A man in a dark suit sits at a table beside a bright window in a dim room, lowers his gaze, "
    "touches his forehead, and gestures with his hands; preserve his identity, the fixed camera, "
    "stable geometry, and consistent lighting."
)
ROLLING_FORCING_CHECKPOINT = _local_hfd_checkpoint_path(
    "TencentARC--RollingForcing",
    "checkpoints",
    "rolling_forcing_dmd.pt",
)
ROLLING_FORCING_WAN_CHECKPOINT = _local_hfd_checkpoint_path("Wan-AI--Wan2.1-T2V-1.3B")
ROLLING_FORCING_DEMO_PROMPT = (
    "A golden retriever runs along a sunlit beach beside rolling ocean waves while the camera tracks "
    "smoothly, with coherent motion, stable anatomy, detailed wet sand, and realistic warm lighting."
)
BERNINI_CHECKPOINT = _local_hfd_checkpoint_path("ByteDance--Bernini-Diffusers")
BERNINI_R_CHECKPOINT = _local_hfd_checkpoint_path("ByteDance--Bernini-R-Diffusers")
BERNINI_R_1P3B_CHECKPOINT = _local_hfd_checkpoint_path(
    "ByteDance--Bernini-R-1.3B-Diffusers"
)
BERNINI_WORKSPACE_NPROC = 1
# Product inference must not depend on private or repository test prompts. The
# required field remains empty until the user supplies their own description.
HELIOS_DEMO_PROMPT = (
    "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
    "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
)
WOW_OFFICIAL_IMAGE_FIXTURE = str(_TEST_CASES_ROOT / "test_vla_case1" / "droid" / "exterior_image_1_left.png")
WOW_OFFICIAL_PROMPT = "The Franka robot grasps the red bottle on the table."
LINGBOT_WORLD_DEMO_ROOT = _TEST_CASES_ROOT / "lingbot_world" / "00"
STABLE_VIDEO_INFINITY_DEMO_ROOT = _TEST_CASES_ROOT / "stable-video-infinity" / "svi-2.0"
STABLE_VIDEO_INFINITY_OFFICIAL_PROMPT = (
    "The water shimmers with dancing caustics as gentle bubbles rise; "
    "the camera dolly-ins through a blue haze."
)
LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"
LINGBOT_WORLD_V2_CHECKPOINT = _first_existing_path(
    _WORKSPACE_ROOT / "ckpt" / "lingbot-world-v2-14b-causal-fast",
    _WORKSPACE_ROOT / "ckpt" / "hfd" / "robbyant--lingbot-world-v2-14b-causal-fast",
    "robbyant/lingbot-world-v2-14b-causal-fast",
)
_STATIC_ASSET_GATED_WORLD_RUNTIME_MODEL_IDS = frozenset(
    {
        "adaworld",
        "causal-rcm",
        "ctrl-world",
        "diamond",
        "dino-wm",
        "egowm",
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
        "open-dreamer",
        "pointworld",
        "sana-wm",
        "shotstream",
        "simworld",
        "starwm",
        "tesseract",
        "uwm",
        "vggt-world",
        "vid2world",
        "wilddet3d",
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
        "rolling-forcing",
        "self-forcing",
    }
)

def _field(
    field_id: str,
    label: str,
    *,
    kind: str = "string",
    target: str = "call_kwargs",
    required: bool = False,
    default: Any = None,
    choices: Sequence[str] = (),
    description: str = "",
) -> InferenceFieldSpec:
    return InferenceFieldSpec(
        field_id=field_id,
        label=label,
        kind=kind,
        target=target,
        required=required,
        default=default,
        choices=tuple(choices),
        description=description,
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
    if normalized.endswith("-path") or normalized.endswith("-dir"):
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


def _artifact(
    artifact_id: str,
    kind: str,
    *,
    required: bool = False,
    preview: bool = False,
    description: str = "",
) -> InferenceArtifactSpec:
    return InferenceArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        required=required,
        preview=preview,
        description=description,
    )


# ---------------------------------------------------------------------------
# Curated model-family specs
# ---------------------------------------------------------------------------


LINGBOT_WORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id=LINGBOT_WORLD_MODEL_ID,
    display_name="LingBot-World",
    default_variant_id=LINGBOT_VARIANT_BASE_CAM,
    default_task_id="image-navigation-video",
    aliases=("lingbot", "lingbot-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id=LINGBOT_VARIANT_BASE_CAM,
            label="Base Cam",
            status="confirmed",
            checkpoints=(
                InferenceCheckpointRef(
                    role="base",
                    uri="robbyant/lingbot-world-base-cam",
                    status="confirmed",
                ),
            ),
            aliases=("base-camera", "camera", "cam", "lingbot-world-base-cam", "lingbot_base_camera"),
            notes=("Base camera-control checkpoint; no fast runtime overlay.",),
        ),
        InferenceVariantSpec(
            variant_id=LINGBOT_VARIANT_BASE_ACT_PREVIEW,
            label="Base Act Preview",
            status="confirmed",
            checkpoints=(
                InferenceCheckpointRef(
                    role="base",
                    uri="robbyant/lingbot-world-base-cam",
                    status="confirmed",
                ),
            ),
            call_kwargs={"allow_act2cam": True, "sampling_steps": 20},
            aliases=(
                "base-act",
                "base-action",
                "act",
                "action",
                "act2cam",
                "lingbot-world-base-act-preview",
                "lingbot_base_action",
            ),
            notes=("Official act2cam preview uses the base camera checkpoint with allow_act2cam.",),
        ),
        InferenceVariantSpec(
            variant_id=LINGBOT_VARIANT_FAST,
            label="Fast",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="base",
                    uri="robbyant/lingbot-world-base-cam",
                    status="confirmed",
                ),
                InferenceCheckpointRef(
                    role="fast",
                    uri="robbyant/lingbot-world-fast",
                    status="local_required",
                ),
            ),
            load_kwargs={"runtime_variant": "fast", "fast_model_path": "robbyant/lingbot-world-fast"},
            call_kwargs={"num_frames": 9, "seed": 42, "max_area": 480 * 832, "offload_model": False},
            aliases=("fast-realtime", "realtime", "lingbot-world-fast", "lingbot_fast_realtime"),
            notes=("Fast runtime uses the base checkpoint plus a fast checkpoint overlay.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="image-navigation-video",
            label="Image Navigation Video",
            description="Generate a camera/navigation-controlled world video from a seed image.",
            aliases=("navigation", "camera-control", "i2v-navigation"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=str(LINGBOT_WORLD_DEMO_ROOT / "image.jpg"),
                ),
                _field("prompt", "Prompt", target="prompt", default=LINGBOT_OFFICIAL_PROMPT),
                _field("interactions", "Interactions", kind="interaction_tokens", target="params"),
                _field(
                    "action_path",
                    "Action Path",
                    kind="path",
                    target="call_kwargs",
                    default=str(LINGBOT_WORLD_DEMO_ROOT),
                ),
                _field("frames", "Frames", kind="integer", target="params", default=161),
                _field("steps", "Steps", kind="integer", target="params", default=70),
                _field("seed", "Seed", kind="integer", target="params", default=42),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
                _artifact("camera_controls", "camera_controls"),
            ),
            default_call_kwargs={
                "action_path": str(LINGBOT_WORLD_DEMO_ROOT),
                "max_area": 480 * 832,
                "num_frames": 161,
                "sampling_steps": 70,
                "seed": 42,
                "offload_model": False,
            },
        ),
        InferenceTaskProfile(
            task_id="image-action-video",
            label="Image Action Video",
            description="Generate an action-conditioned world video from a seed image.",
            aliases=("action", "act2cam"),
            inputs=(
                _field("image", "Image", kind="path", target="input_path", required=True),
                _field("prompt", "Prompt", target="prompt"),
                _field("interactions", "Interactions", kind="interaction_tokens", target="params"),
                _field("action_path", "Action Path", kind="path", target="call_kwargs"),
                _field("action_string", "Action String", target="call_kwargs"),
                _field("frames", "Frames", kind="integer", target="params", default=21),
                _field("steps", "Steps", kind="integer", target="params", default=20),
                _field("seed", "Seed", kind="integer", target="params", default=42),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={"num_frames": 21, "sampling_steps": 20, "seed": 42, "allow_act2cam": True},
        ),
    ),
    notes=("Variants are explicit checkpoint/runtime bundles, not separate Studio model ids.",),
)

LINGBOT_WORLD_V2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id=LINGBOT_WORLD_V2_MODEL_ID,
    display_name="LingBot-World-V2",
    default_variant_id="causal-fast-14b",
    default_task_id="image-camera-video",
    aliases=("lingbot-world-infinity", "lingbot-v2", "lingbot_world_v2"),
    variants=(
        InferenceVariantSpec(
            variant_id="causal-fast-14b",
            label="Causal Fast 14B",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=LINGBOT_WORLD_V2_CHECKPOINT,
                    status="configured",
                ),
            ),
            aliases=("14b-causal-fast", "causal-fast", "infinity"),
            notes=("Released causal-fast checkpoint using the in-tree WorldFoundry runtime.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="image-camera-video",
            label="Image + Camera Path to Video",
            description="Generate a causal world video from an image and camera trajectory arrays.",
            aliases=("image-to-world-video", "camera-controlled-video", "image-to-video"),
            inputs=(
                _field(
                    "image",
                    "Initial Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=str(LINGBOT_WORLD_DEMO_ROOT / "image.jpg"),
                ),
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=LINGBOT_OFFICIAL_PROMPT,
                ),
                _field(
                    "action_path",
                    "Camera Path Directory",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=str(LINGBOT_WORLD_DEMO_ROOT),
                    description="Directory containing poses.npy and intrinsics.npy.",
                ),
                _field(
                    "size",
                    "Output Size",
                    target="call_kwargs",
                    default="480*832",
                    choices=("480*832", "832*480", "720*1280", "1280*720"),
                ),
                _field("frame_num", "Frames", kind="integer", target="call_kwargs", default=361),
                _field("chunk_size", "Chunk Size", kind="integer", target="call_kwargs", default=4),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("sample_shift", "Sample Shift", kind="number", target="call_kwargs", default=10.0),
                _field("local_attn_size", "Local Attention", kind="integer", target="call_kwargs", default=18),
                _field("sink_size", "Attention Sink", kind="integer", target="call_kwargs", default=6),
                _field("nproc_per_node", "GPU Processes", kind="integer", target="call_kwargs", default=8),
                _field("t5_fsdp", "T5 FSDP", kind="boolean", target="call_kwargs", default=True),
                _field("dit_fsdp", "DiT FSDP", kind="boolean", target="call_kwargs", default=True),
                _field("offload_model", "Offload Model", kind="boolean", target="call_kwargs", default=False),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "action_path": str(LINGBOT_WORLD_DEMO_ROOT),
                "size": "480*832",
                "frame_num": 361,
                "chunk_size": 4,
                "seed": 42,
                "sample_shift": 10.0,
                "local_attn_size": 18,
                "sink_size": 6,
                "nproc_per_node": 8,
                "t5_fsdp": True,
                "dit_fsdp": True,
                "offload_model": False,
                "return_dict": True,
                "timeout_seconds": 7200,
            },
        ),
    ),
    notes=(
        "The required action directory contains poses.npy [F,4,4] and intrinsics.npy [F,4].",
        "The official profile uses 8 GPUs and 361 output frames.",
    ),
)

WARP_AS_HISTORY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="warp-as-history",
    display_name="Warp-as-History",
    default_variant_id="default",
    default_task_id="official-demo-video",
    aliases=("wah", "warp_as_history", "yyfz233/warp-as-history"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs={"num_frames": 33, "fps": 16, "height": 384, "width": 640},
            aliases=("official", "demo"),
            notes=("Uses the vendored official demo CSV when no custom first frame is provided.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-demo-video",
            label="Official Demo Video",
            description="Run the vendored official Warp-as-History demo CSV through the in-tree runtime.",
            aliases=("default", "interactive-video", "demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "frames",
                    "Frames",
                    kind="integer",
                    target="params",
                    default=33,
                    choices=("33",),
                ),
                _field("fps", "FPS", kind="integer", target="params", default=16, choices=("16",)),
                _field("height", "Height", kind="integer", target="params", default=384),
                _field("width", "Width", kind="integer", target="params", default=640),
                _field("seed", "Seed", kind="integer", target="params", default=42),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={"num_frames": 33, "fps": 16, "height": 384, "width": 640, "seed": 42},
        ),
    ),
    notes=("Custom Warp-as-History runs can still pass images plus runtime kwargs through advanced CLI/API paths.",),
)

GEN3C_OFFICIAL_FIXTURE = str(Path(__file__).resolve().parents[1] / "data" / "test_cases" / "gen3c" / "image.png")
# Catalog copies; canonical strings live in pipelines.gen3c.constants.
GEN3C_OFFICIAL_PROMPT = (
    "A cinematic view of a futuristic science-fiction city with coherent structures, realistic lighting, "
    "and smooth natural camera motion."
)
GEN3C_OFFICIAL_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)
FANTASYWORLD_CAMERA_FIXTURE = str(_TEST_CASES_ROOT / "fantasyworld" / "camera_forward.json")
FANTASYWORLD_PROMPT = "A coherent fantasy harbor world with stable geometry during a forward camera move."

GEN3C_OFFICIAL_CALL_KWARGS = {
    "trajectory": "left",
    "camera_rotation": "center_facing",
    "movement_distance": 0.3,
    "negative_prompt": GEN3C_OFFICIAL_NEGATIVE_PROMPT,
    "guidance": 1.0,
    "num_steps": 35,
    "num_video_frames": 121,
    "fps": 24,
    "height": 704,
    "width": 1280,
    "seed": 1,
}

LYRA1_STATIC_OFFICIAL_FIXTURE = (
    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/lyra/Lyra-1/00172.png"
)
LYRA2_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "lyra" / "Lyra-2" / "00.png")
LYRA2_OFFICIAL_PROMPT = _read_text_file_or_default(
    _TEST_CASES_ROOT / "lyra" / "Lyra-2" / "00.txt",
    "A slow, steady camera push forward through a coherent static 3D world with stable geometry.",
)

LYRA1_STATIC_OFFICIAL_CALL_KWARGS = {
    "mode": "static",
    "trajectory": "zoom_in",
    "num_video_frames": 121,
    "fps": 24,
    "height": 704,
    "width": 1280,
    "seed": 1,
    "num_steps": 35,
    "guidance": 1.0,
    "num_gpus": 1,
    "movement_distance": 0.3,
    "camera_rotation": "center_facing",
    "multi_trajectory": True,
    "total_movement_distance_factor": 1.0,
    "foreground_masking": True,
    "filter_points_threshold": 0.05,
    "disable_prompt_encoder": True,
    "offload_diffusion_transformer": True,
    "offload_tokenizer": True,
    "offload_text_encoder_model": True,
    "offload_prompt_upsampler": True,
    "offload_guardrail_models": True,
    "disable_guardrail": True,
    "execute": True,
    "show_progress": True,
}

LYRA2_OFFICIAL_CALL_KWARGS = {
    "fps": 16,
    "resolution": (480, 832),
    "seed": 1,
    "reconstruct_3d": False,
    "return_dict": True,
    "show_progress": True,
    "execute": True,
}

FANTASYWORLD_WAN22_CALL_KWARGS = {
    "camera_json_path": FANTASYWORLD_CAMERA_FIXTURE,
    "fps": 16,
    "using_scale": True,
    "conf_threshold": 1.5,
    "stride": 4,
    "return_dict": True,
}

FANTASYWORLD_WAN21_CALL_KWARGS = {
    "camera_json_path": FANTASYWORLD_CAMERA_FIXTURE,
    "fps": 16,
    "seed": 1024,
    "using_scale": True,
    "conf_threshold": 1.0,
    "stride": 4,
    "return_dict": True,
}

FLASHWORLD_CALL_KWARGS = {
    "num_frames": 16,
    "fps": 15,
    "image_height": 480,
    "image_width": 704,
    "return_video": True,
}

LONGCAT_VIDEO_OFFICIAL_PROMPT = (
    "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, "
    "wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla "
    "and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice "
    "cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background "
    "features a green lawn and several tall trees, creating a warm and loving scene."
)
LONGCAT_VIDEO_OFFICIAL_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
    "picture, messy background, three legs, many people in the background, walking backwards"
)
LONGCAT_VIDEO_LOCAL_CHECKPOINT = _first_existing_path(
    hfd_root_path("meituan-longcat--LongCat-Video"),
    _WORKSPACE_ROOT / "ckpt" / "LongCat-Video",
    "meituan-longcat/LongCat-Video",
)
LONGCAT_VIDEO_OFFICIAL_LOAD_KWARGS = {
    "python_executable": sys.executable,
}
LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS = {
    "task_type": "t2v",
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 50,
    "distill_num_inference_steps": 16,
    "guidance_scale": 2.0,
    "seed": 0,
    "fps": 15,
    "context_parallel_size": 1,
    "execute": True,
    "timeout_seconds": 7200,
    "return_dict": True,
}

LINGBOT_VIDEO_STRUCTURED_PROMPT = (
    '{"comprehensive_description":"A humanoid robot carefully places a red block into a matching tray on a '
    'clean workbench while the camera remains stable.","camera_info":{"frame_size":"Medium Shot",'
    '"shot_type_angle":"Eye Level","lighting_type":"Daylight"},"world_knowledge":[]}'
)
_LINGBOT_VIDEO_HFD_ROOT = Path(os.getenv("WORLDFOUNDRY_HFD_ROOT", str(_WORKSPACE_ROOT / "ckpt" / "hfd"))).expanduser()
LINGBOT_VIDEO_DENSE_CHECKPOINT = _first_existing_path(
    _LINGBOT_VIDEO_HFD_ROOT / "robbyant--lingbot-video-dense-1.3b",
    _WORKSPACE_ROOT / "ckpt" / "lingbot-video-dense-1.3b",
    "robbyant/lingbot-video-dense-1.3b",
)
LINGBOT_VIDEO_MOE_CHECKPOINT = _first_existing_path(
    _LINGBOT_VIDEO_HFD_ROOT / "robbyant--lingbot-video-moe-30b-a3b",
    _WORKSPACE_ROOT / "ckpt" / "lingbot-video-moe-30b-a3b",
    "robbyant/lingbot-video-moe-30b-a3b",
)
LINGBOT_VIDEO_DEFAULT_CALL_KWARGS = {
    "mode": "t2v",
    "backend": "diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 121,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "shift": 3.0,
    "seed": 42,
    "fps": 24,
    "transformer_dtype": "bf16",
    "text_encoder_dtype": "bf16",
    "vae_dtype": "fp32",
    "execute": True,
    "timeout_seconds": 7200,
    "return_dict": True,
}

HY_WORLDPLAY_OFFICIAL_PROMPT = (
    "A paved pathway leads towards a stone arch bridge spanning a calm body of water.  "
    "Lush green trees and foliage line the path and the far bank of the water. "
    "A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. "
    "The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, "
    "natural light, creating a tranquil and serene atmosphere. The pathway is composed of "
    "large, rectangular stones, and the bridge is constructed of light gray stone.  The overall "
    "composition emphasizes the peaceful and harmonious nature of the landscape."
)
HY_WORLDPLAY_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_worldplay" / "test.png")
# Workspace schedules catalog jobs on an explicitly selected ``cuda:N`` by
# default.  Keep the full upstream generation settings, but use one process so
# the static task default cannot silently turn a single-GPU request into an
# eight-rank torchrun job. Users can still select any supported 2/3/4/6/8-rank
# topology explicitly.
HY_WORLDPLAY_WORKSPACE_NPROC = 1
HY_WORLDPLAY_OFFICIAL_CALL_KWARGS = {
    "num_frames": 125,
    "aspect_ratio": "16:9",
    "num_inference_steps": 4,
    "seed": 1,
    "fps": 24,
    "output_type": "pt",
    "prompt_rewrite": False,
    "enable_sr": False,
    "return_pre_sr_video": False,
    "few_step": True,
    "chunk_latent_frames": 4,
    "model_type": "ar",
    "transformer_resident_ar_rollout": True,
    "user_width": 832,
    "user_height": 480,
}

HUNYUAN_GAMECRAFT_OFFICIAL_PROMPT = (
    "A charming medieval village with cobblestone streets, thatched-roof houses, "
    "and vibrant flower gardens under a bright blue sky."
)
HUNYUAN_GAMECRAFT_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_game_craft" / "village.png")
HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS = {
    "interactions": ("forward", "backward", "right", "left"),
    "interaction_speed": (0.2, 0.2, 0.2, 0.2),
    "size": (704, 1216),
    "num_frames": 129,
    "infer_steps": 50,
    "cfg_scale": 2.0,
}
HUNYUAN_GAMECRAFT_OFFICIAL_LOAD_KWARGS = {
    "seed": 250160,
}

HUNYUAN_WORLD_VOYAGER_OFFICIAL_PROMPT = "An old-fashioned European village with thatched roofs on the houses."
HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR = str(_TEST_CASES_ROOT / "hunyuan_world_voyager" / "case1")
HUNYUAN_WORLD_VOYAGER_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_world_voyager" / "case1" / "ref_image.png")
HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS = {
    "num_frames": 49,
    "condition_dir": HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR,
    "interactions": "forward",
    "seed": 0,
    "infer_steps": 50,
    "flow_shift": 7.0,
    "embedded_cfg_scale": 6.0,
    "i2v_stability": True,
    "ulysses_degree": 1,
    "ring_degree": 1,
}

MATRIX_GAME_2_IN_TREE_ROOT = _TEST_CASES_ROOT / "matrix-game-2"
MATRIX_GAME_2_OFFICIAL_FIXTURE = _first_existing_path(
    MATRIX_GAME_2_IN_TREE_ROOT / "universal" / "0000.png",
)
MATRIX_GAME_2_OFFICIAL_CONFIG = _first_existing_path(
    MATRIX_GAME_2_IN_TREE_ROOT / "configs" / "inference_universal.yaml",
)
MATRIX_GAME_2_OFFICIAL_INTERACTIONS = ()
MATRIX_GAME_2_OFFICIAL_CALL_KWARGS = {
    "num_frames": 150,
    "size": (352, 640),
    "fps": 12,
    "seed": 42,
    "official_bench_actions": True,
    "visualize_ops": False,
    "visualize_warning": False,
}
MATRIX_GAME_2_OFFICIAL_LOAD_KWARGS = {"mode": "universal"}

MATRIX_GAME_3_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "matrix-game-3" / "001" / "image.png"
MATRIX_GAME_3_OFFICIAL_FIXTURE = _first_existing_path(
    MATRIX_GAME_3_FALLBACK_FIXTURE,
)
MATRIX_GAME_3_OFFICIAL_PROMPT = "A colorful, animated cityscape with a gas station and various buildings."
MATRIX_GAME_3_DEFAULT_INTERACTIONS = ("forward", "camera_r")
MATRIX_GAME_3_OFFICIAL_CALL_KWARGS = {
    "num_iterations": 12,
    "num_inference_steps": 3,
    "size": "704*1280",
    "fps": 17,
    "seed": 42,
    "save_name": "test",
    "visualize_ops": True,
    "show_progress": True,
    "use_int8": True,
    "compile_vae": True,
    "lightvae_pruning_rate": 0.5,
    "vae_type": "mg_lightvae",
    "fa_version": "3",
    "use_async_vae": False,
    "async_vae_warmup_iters": 0,
}

MATRIX_GAME_2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="matrix-game-2",
    display_name="Matrix-Game-2",
    default_variant_id="universal",
    default_task_id="official-universal-image",
    aliases=("matrixgame", "matrixgame2", "matrix-game2", "Skywork/Matrix-Game-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="universal",
            label="Universal",
            status="configured",
            load_kwargs=MATRIX_GAME_2_OFFICIAL_LOAD_KWARGS,
            call_kwargs=MATRIX_GAME_2_OFFICIAL_CALL_KWARGS,
            aliases=("default", "official", "demo"),
            notes=("Matches the upstream Matrix-Game-2 inference_universal demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-universal-image",
            label="Official Universal Image",
            description="Run the Matrix-Game-2 universal demo using the upstream demo image and config defaults.",
            aliases=("default", "official-demo", "interactive-video", "navigation-video"),
            inputs=(
                _field(
                    "image",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=MATRIX_GAME_2_OFFICIAL_FIXTURE,
                ),
                _field(
                    "mode",
                    "Mode",
                    target="load_kwargs",
                    default="universal",
                    choices=("universal", "gta_drive", "templerun"),
                ),
                _field(
                    "config_path",
                    "Config Path",
                    kind="path",
                    target="call_kwargs",
                    default=MATRIX_GAME_2_OFFICIAL_CONFIG,
                ),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=MATRIX_GAME_2_OFFICIAL_INTERACTIONS,
                ),
                _field("frames", "Frames", kind="integer", target="params", default=150),
                _field("fps", "FPS", kind="integer", target="params", default=12),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("size", "Size", kind="json", target="call_kwargs", default=(352, 640)),
                _field(
                    "official_bench_actions",
                    "Official Bench Actions",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                    description="Use Matrix-Game-2's official Bench_actions_* trajectory generator instead of the manual action-token list.",
                ),
                _field("visualize_ops", "Visualize Actions", kind="boolean", target="call_kwargs", default=False),
                _field("visualize_warning", "Visualize Warning", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=MATRIX_GAME_2_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image/config path uses packaged in-tree Matrix-Game fixtures.",),
)

MATRIX_GAME_3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="matrix-game-3",
    display_name="Matrix-Game-3",
    default_variant_id="official",
    default_task_id="official-cityscape-image",
    aliases=("matrixgame3", "matrix-game3", "Skywork/Matrix-Game-3.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="official",
            label="Official",
            status="configured",
            call_kwargs=MATRIX_GAME_3_OFFICIAL_CALL_KWARGS,
            aliases=("default", "demo"),
            notes=(
                "Uses the Matrix-Game-3 README demo sampling defaults.",
                "Async VAE is disabled by default because it failed device ordinal validation when GPUs are remapped.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-cityscape-image",
            label="Official Cityscape Image",
            description="Run the Matrix-Game-3 README image-conditioned cityscape demo.",
            aliases=("default", "official-demo", "interactive-video", "navigation-video"),
            inputs=(
                _field(
                    "image",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=MATRIX_GAME_3_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=MATRIX_GAME_3_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=MATRIX_GAME_3_DEFAULT_INTERACTIONS,
                ),
                _field("num_iterations", "Iterations", kind="integer", target="call_kwargs", default=12),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=3),
                _field("size", "Size", target="call_kwargs", default="704*1280", choices=("704*1280", "1280*704")),
                _field("fps", "FPS", kind="integer", target="params", default=17),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("save_name", "Save Name", target="call_kwargs", default="test"),
                _field("visualize_ops", "Visualize Actions", kind="boolean", target="call_kwargs", default=True),
                _field("show_progress", "Show Progress", kind="boolean", target="call_kwargs", default=True),
                _field("use_int8", "INT8", kind="boolean", target="call_kwargs", default=True),
                _field("compile_vae", "Compile VAE", kind="boolean", target="call_kwargs", default=True),
                _field("lightvae_pruning_rate", "LightVAE Pruning", kind="number", target="call_kwargs", default=0.5),
                _field(
                    "vae_type", "VAE Type", target="call_kwargs", default="mg_lightvae", choices=("mg_lightvae", "wan")
                ),
                _field("fa_version", "FlashAttention", target="call_kwargs", default="3", choices=("3", "2", "0")),
                _field("use_async_vae", "Async VAE", kind="boolean", target="call_kwargs", default=False),
                _field("async_vae_warmup_iters", "Async VAE Warmup", kind="integer", target="call_kwargs", default=0),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=MATRIX_GAME_3_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default Matrix-Game-3 demo uses the packaged in-tree image fixture.",),
)

ASTRA_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "astra" / "condition_images" / "garden_1.png"
ASTRA_OFFICIAL_FIXTURE = _first_existing_path(
    ASTRA_FALLBACK_FIXTURE,
)
ASTRA_OFFICIAL_PROMPT = (
    "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, "
    "charming, and inviting atmosphere. The scene shows a picturesque open square paved with red "
    "bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and "
    "dark-painted facades. On the right side, a lush arrangement of potted plants and blooming "
    "flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands "
    "prominently near the center-right, contributing to the timeless character of the street. "
    "Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. "
    "Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely "
    "feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from "
    "the left, enhancing the scene's inviting, peaceful mood."
)
ASTRA_OFFICIAL_CALL_KWARGS = {
    "frames_per_generation": 8,
    "total_frames_to_generate": 24,
    "num_inference_steps": 50,
    "start_frame": 0,
    "initial_condition_frames": 1,
    "modality_type": "sekai",
}

ASTRA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="astra",
    display_name="Astra",
    default_variant_id="default",
    default_task_id="official-sekai-image",
    aliases=("kwai-vgi/astra", "KwaiVGI/Astra"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=ASTRA_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "sekai"),
            notes=(
                "Matches the executable arguments from Astra infer_demo.sh except upstream cam_type, which is represented by direction tokens in this wrapper.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-sekai-image",
            label="Official Sekai Image",
            description="Run the Astra README image-conditioned sekai demo with the packaged garden fixture.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "image",
                    "Condition Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=ASTRA_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=ASTRA_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Camera Direction",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward",),
                ),
                _field(
                    "frames_per_generation", "Frames Per Generation", kind="integer", target="call_kwargs", default=8
                ),
                _field(
                    "total_frames_to_generate", "Generated Frames", kind="integer", target="call_kwargs", default=24
                ),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("start_frame", "Start Frame", kind="integer", target="call_kwargs", default=0),
                _field("initial_condition_frames", "Condition Frames", kind="integer", target="call_kwargs", default=1),
                _field(
                    "modality_type",
                    "Modality",
                    target="call_kwargs",
                    default="sekai",
                    choices=("sekai", "nuscenes", "openx"),
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=ASTRA_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default Astra demo uses the packaged in-tree garden image fixture.",),
)

NEOVERSE_DATA_ROOT = _TEST_CASES_ROOT / "neoverse"
NEOVERSE_CHECKPOINT = str(hfd_root_path("Yuppie1204--NeoVerse"))
NEOVERSE_OFFICIAL_FIXTURE = _first_existing_path(
    NEOVERSE_DATA_ROOT / "videos" / "robot.mp4",
)
NEOVERSE_OFFICIAL_PROMPT = "A two-arm robot assembles parts in front of a table."
NEOVERSE_PREDEFINED_CHOICES = (
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "move_left",
    "move_right",
    "push_in",
    "pull_out",
    "boom_up",
    "boom_down",
    "orbit_left",
    "orbit_right",
    "static",
)
NEOVERSE_OFFICIAL_LOAD_KWARGS = {
    "height": 336,
    "width": 560,
    "num_inference_steps": 4,
    "cfg_scale": 1.0,
    "disable_lora": False,
    # The 14B Wan backbone and WorldMirror reconstructor do not fit together
    # with the official 81-frame reconstruction activations on an 80GB GPU.
    # Keep inactive components CPU-resident; this does not change sampling.
    "enable_vram_management": True,
}
NEOVERSE_OFFICIAL_CALL_KWARGS = {
    "predefined_trajectory": "tilt_up",
    "num_frames": 81,
    "trajectory_mode": "relative",
    "angle": 15,
    "distance": 0,
    "orbit_radius": 0,
    "zoom_ratio": 1.0,
    "alpha_threshold": 1.0,
    "seed": 42,
    "use_first_frame": True,
    "static_scene": False,
    "num_inference_steps": 4,
    "cfg_scale": 1.0,
}

NEOVERSE_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="neoverse",
    display_name="NeoVerse",
    default_variant_id="default",
    default_task_id="official-tilt-up-video",
    aliases=("Yuppie1204/NeoVerse", "neoverse-4d"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=NEOVERSE_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_path": NEOVERSE_CHECKPOINT,
                **NEOVERSE_OFFICIAL_LOAD_KWARGS,
            },
            call_kwargs=NEOVERSE_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "tilt-up"),
            notes=("Matches the NeoVerse README tilt_up CLI example with fast LoRA defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-tilt-up-video",
            label="Official Tilt Up Video",
            description="Run the NeoVerse README tilt_up demo using the packaged robot video.",
            aliases=("default", "official-demo", "interactive-video", "video-to-world"),
            inputs=(
                _field(
                    "video",
                    "Input Video",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=NEOVERSE_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=NEOVERSE_OFFICIAL_PROMPT),
                _field(
                    "predefined_trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    default="tilt_up",
                    choices=NEOVERSE_PREDEFINED_CHOICES,
                ),
                _field("interactions", "Actions", kind="interaction_tokens", target="params"),
                _field("trajectory_file", "Trajectory File", kind="path", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=81),
                _field("height", "Height", kind="integer", target="load_kwargs", default=336),
                _field("width", "Width", kind="integer", target="load_kwargs", default=560),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=4),
                _field("cfg_scale", "CFG Scale", kind="number", target="call_kwargs", default=1.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field(
                    "trajectory_mode",
                    "Trajectory Mode",
                    target="call_kwargs",
                    default="relative",
                    choices=("relative", "global"),
                ),
                _field("angle", "Angle", kind="number", target="call_kwargs", default=15),
                _field("distance", "Distance", kind="number", target="call_kwargs", default=0),
                _field("orbit_radius", "Orbit Radius", kind="number", target="call_kwargs", default=0),
                _field("zoom_ratio", "Zoom Ratio", kind="number", target="call_kwargs", default=1.0),
                _field("alpha_threshold", "Alpha Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("use_first_frame", "Use First Frame", kind="boolean", target="call_kwargs", default=True),
                _field("static_scene", "Static Scene", kind="boolean", target="call_kwargs", default=False),
                _field("disable_lora", "Disable LoRA", kind="boolean", target="load_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=NEOVERSE_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=(
        "The upstream checkout in this workspace does not include examples/videos, so the default video falls back to WorldFoundry packaged test_cases/neoverse.",
    ),
)

COSMOS3_NANO_REPO_ID = "nvidia/Cosmos3-Nano"
COSMOS3_SUPER_REPO_ID = "nvidia/Cosmos3-Super"
COSMOS3_NANO_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
COSMOS3_SUPER_REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"
COSMOS3_DEFAULT_PROMPT = "A robot arm is cleaning a plate in the kitchen"
COSMOS3_T2V_CALL_KWARGS = {
    "task_type": "text-to-video",
    "fps": 24,
    "guidance_scale": 6.0,
    "height": 720,
    "num_inference_steps": 35,
    "num_frames": 189,
    "output_type": "video",
    "seed": 0,
    "width": 1280,
    "flow_shift": 10.0,
    "use_karras_sigmas": False,
    "enable_safety_check": True,
}


def _cosmos3_task_profile(
    task_id: str,
    label: str,
    *,
    input_kind: str | None = None,
    image_output: bool = False,
) -> InferenceTaskProfile:
    """Build one Cosmos3 generator-inference task contract."""

    num_frames = 1 if image_output else 189
    shifted_scheduler = task_id in {"t2v", "v2v"}
    task_types = {
        "t2i": "text-to-image",
        "t2v": "text-to-video",
        "i2v": "image-to-video",
        "v2v": "video-to-video",
    }
    output_type = "pil" if image_output else "video"
    output_type_choices = ("pil", "pt", "np") if image_output else ("video",)
    defaults = {
        "task_type": task_types[task_id],
        "fps": 24,
        "guidance_scale": 6.0,
        "height": 720,
        "num_inference_steps": 35,
        "num_frames": num_frames,
        "output_type": output_type,
        "seed": 0,
        "width": 1280,
        "flow_shift": 10.0 if shifted_scheduler else 1.0,
        "use_karras_sigmas": not shifted_scheduler,
        "enable_safety_check": True,
    }
    supports_sound = not image_output
    if supports_sound:
        defaults["enable_sound"] = False
    fields: list[InferenceFieldSpec] = [
        _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS3_DEFAULT_PROMPT),
        _field(
            "load_sound_tokenizer",
            "Load Sound Tokenizer",
            kind="boolean",
            target="load_kwargs",
            default=True,
            description="Disable for visual-only runs to save memory; keep enabled when enable_sound may be used.",
        ),
    ]
    if input_kind is not None:
        fields.append(
            _field(
                "input_path",
                "Input Image" if input_kind == "image" else "Input Video",
                kind="path",
                target="input_path",
                required=True,
                default="",
            )
        )
    fields.extend(
        (
            _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=num_frames),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
            _field("height", "Height", kind="integer", target="call_kwargs", default=720),
            _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=35),
            _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=6.0),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
            _field(
                "flow_shift",
                "Flow Shift",
                kind="number",
                target="call_kwargs",
                default=defaults["flow_shift"],
            ),
            _field(
                "use_karras_sigmas",
                "Use Karras Sigmas",
                kind="boolean",
                target="call_kwargs",
                default=defaults["use_karras_sigmas"],
            ),
            _field(
                "enable_safety_check",
                "Run Safety Check",
                kind="boolean",
                target="call_kwargs",
                default=True,
            ),
            _field(
                "output_type",
                "Output Type",
                target="call_kwargs",
                default=output_type,
                choices=output_type_choices,
                description=(
                    "Workspace persists single-frame PIL/NumPy/PyTorch results as an image artifact."
                    if image_output
                    else "Workspace persists multi-frame Cosmos3 results as a video artifact."
                ),
            ),
            _field("output_path", "Output Path", kind="path", target="call_kwargs"),
        )
    )
    if supports_sound:
        fields.append(
            _field(
                "enable_sound",
                "Generate Synchronized Sound",
                kind="boolean",
                target="call_kwargs",
                default=False,
                description="Decode the checkpoint's synchronized sound stream and preserve its audio artifact.",
            )
        )
    artifact = (
        _artifact("image", "generated_image", required=True, preview=True)
        if image_output
        else _artifact("video", "video", required=True, preview=True)
    )
    outputs = [artifact]
    if supports_sound:
        outputs.append(
            _artifact(
                "audio",
                "audio",
                description="Optional synchronized audio waveform and muxed video track when enable_sound is true.",
            )
        )
    outputs.append(_artifact("manifest", "manifest", required=True))
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Cosmos3 {label.lower()} through the in-tree Diffusers generator runtime.",
        aliases=(task_types[task_id],),
        inputs=tuple(fields),
        outputs=tuple(outputs),
        default_call_kwargs=defaults,
    )


def _cosmos3_action_task_profile(mode: str, label: str) -> InferenceTaskProfile:
    """Build an official-default Cosmos3 action inference task."""

    task_id = f"action-{mode.replace('_', '-')}"
    defaults: dict[str, Any] = {
        "task_type": task_id,
        "action_mode": mode,
        "action_chunk_size": 16,
        "domain_name": "bridge_orig_lerobot",
        "resolution_tier": 480,
        "view_point": "ego_view",
        "num_frames": 17,
        "fps": 5,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "use_karras_sigmas": False,
        "use_system_prompt": False,
        "enable_safety_check": True,
        "enable_sound": False,
        "output_type": "video",
        "seed": 0,
    }
    fields: list[InferenceFieldSpec] = [
        _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS3_DEFAULT_PROMPT),
        _field(
            "load_sound_tokenizer",
            "Load Sound Tokenizer",
            kind="boolean",
            target="load_kwargs",
            default=False,
            description="Action inference is visual/action-only and does not require the AVAE sound tokenizer.",
        ),
        _field(
            "input_path",
            "Input Video",
            kind="path",
            target="input_path",
            required=True,
            default="",
            description=(
                "Policy and forward dynamics use the first frame; inverse dynamics conditions on the clip."
            ),
        ),
        _field(
            "action_mode",
            "Action Mode",
            target="call_kwargs",
            required=True,
            default=mode,
            choices=(mode,),
        ),
        _field(
            "action_chunk_size",
            "Action Chunk Size",
            kind="integer",
            target="call_kwargs",
            required=True,
            default=16,
            description="The generated clip contains chunk_size + 1 frames.",
        ),
        _field(
            "domain_name",
            "Action Domain",
            target="call_kwargs",
            required=True,
            default="bridge_orig_lerobot",
        ),
        _field(
            "resolution_tier",
            "Action Resolution Tier",
            kind="integer",
            target="call_kwargs",
            default=480,
            choices=(256, 480, 704, 720),
        ),
        _field(
            "view_point",
            "Action Viewpoint",
            target="call_kwargs",
            default="ego_view",
            choices=("ego_view", "third_person_view", "wrist_view", "concat_view"),
        ),
    ]
    if mode == "forward_dynamics":
        fields.append(
            _field(
                "raw_actions",
                "Raw Actions",
                kind="json",
                target="call_kwargs",
                required=True,
                description="A [T, D] action array driving the forward-dynamics rollout.",
            )
        )
    fields.extend(
        (
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=17),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=5),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=30),
            _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
            _field("flow_shift", "Flow Shift", kind="number", target="call_kwargs", default=10.0),
            _field(
                "use_karras_sigmas",
                "Use Karras Sigmas",
                kind="boolean",
                target="call_kwargs",
                default=False,
            ),
            _field(
                "use_system_prompt",
                "Use System Prompt",
                kind="boolean",
                target="call_kwargs",
                default=False,
                description="Official action prompts are plain task descriptions and skip LLM system upsampling.",
            ),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
            _field(
                "enable_safety_check",
                "Run Safety Check",
                kind="boolean",
                target="call_kwargs",
                default=True,
            ),
            _field(
                "output_type",
                "Output Type",
                target="call_kwargs",
                default="video",
                choices=("video",),
            ),
            _field("output_path", "Output Path", kind="path", target="call_kwargs"),
        )
    )
    outputs = [_artifact("video", "video", required=True, preview=True)]
    if mode in {"policy", "inverse_dynamics"}:
        outputs.append(
            _artifact(
                "action_trace",
                "action_trace",
                description="Predicted normalized actions emitted by the policy or inverse-dynamics head.",
            )
        )
    outputs.append(_artifact("manifest", "manifest", required=True))
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Cosmos3 {label.lower()} with the official action-inference defaults.",
        aliases=tuple(dict.fromkeys((mode, mode.replace("_", "-")))),
        inputs=tuple(fields),
        outputs=tuple(outputs),
        default_call_kwargs=defaults,
    )


COSMOS3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cosmos3",
    display_name="Cosmos3",
    default_variant_id="cosmos3-nano",
    default_task_id="t2v",
    aliases=("cosmos-3", "cosmos3-nano", "cosmos-3-nano", "cosmos3-super", "cosmos-3-super"),
    variants=(
        InferenceVariantSpec(
            variant_id="cosmos3-nano",
            label="Cosmos3 Nano",
            status="integrated",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=COSMOS3_NANO_REPO_ID, status="official"),
            ),
            load_kwargs={
                "variant_id": "cosmos3-nano",
                "profile_id": "cosmos3-nano",
                "runtime_profile": "cosmos3-nano",
                "revision": COSMOS3_NANO_REVISION,
                "load_sound_tokenizer": True,
            },
            call_kwargs=COSMOS3_T2V_CALL_KWARGS,
            aliases=("default", "nano", "cosmos-3-nano"),
        ),
        InferenceVariantSpec(
            variant_id="cosmos3-super",
            label="Cosmos3 Super",
            status="checkpoint-required",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=COSMOS3_SUPER_REPO_ID, status="official"),
            ),
            load_kwargs={
                "variant_id": "cosmos3-super",
                "profile_id": "cosmos3-super",
                "runtime_profile": "cosmos3-super",
                "revision": COSMOS3_SUPER_REVISION,
                "device_map": "balanced",
                "load_sound_tokenizer": True,
            },
            call_kwargs=COSMOS3_T2V_CALL_KWARGS,
            aliases=("super", "cosmos-3-super"),
            notes=("Requires a high-memory configuration; use device_map when distributing across GPUs.",),
        ),
    ),
    tasks=(
        _cosmos3_task_profile("t2i", "Text to Image", image_output=True),
        _cosmos3_task_profile("t2v", "Text to Video"),
        _cosmos3_task_profile("i2v", "Image to Video", input_kind="image"),
        _cosmos3_task_profile("v2v", "Video to Video", input_kind="video"),
        _cosmos3_action_task_profile("policy", "Action Policy"),
        _cosmos3_action_task_profile("forward_dynamics", "Action Forward Dynamics"),
        _cosmos3_action_task_profile("inverse_dynamics", "Action Inverse Dynamics"),
    ),
    notes=(
        "The Workspace contract exposes generator inference only: T2I/T2V/I2V/V2V, optional synchronized sound, "
        "and the checkpoint's structured policy/forward-dynamics/inverse-dynamics action modes.",
        "Cosmos3 Reasoner, training, and vLLM/vLLM-Omni serving workflows are intentionally not integrated.",
    ),
)


COSMOS_PREDICT2P5_VALIDATION_PROMPT = (
    "A nighttime city bus terminal gradually shifts from stillness to subtle movement, "
    "with realistic lighting, stable geometry, and smooth camera motion."
)
COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS = {
    "num_frames": 93,
    "fps": 16,
    "height": 704,
    "width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 7.0,
    "seed": 42,
    "output_type": "pt",
}
COSMOS_PREDICT2P5_OFFICIAL_BASE_JSONL = str(_TEST_CASES_ROOT / "cosmos-predict2p5" / "base" / "robot_pouring.jsonl")

COSMOS_PREDICT2P5_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cosmos-predict2p5",
    display_name="Cosmos Predict2.5",
    default_variant_id="default",
    default_task_id="text-to-world-validation",
    aliases=("cosmos-predict2.5", "cosmos-predict-2.5", "nvidia/Cosmos-Predict2.5-2B"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="2B Default",
            status="configured",
            call_kwargs=COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS,
            aliases=("2b", "default", "validation"),
            notes=("Uses the catalog-resolved Cosmos Predict2.5 2B checkpoint and required Reason1/Wan components.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-world-validation",
            label="Text-to-World Validation",
            description=(
                "Run a prompt-only Cosmos Predict2.5 validation job. The official robot_pouring "
                "demo is a video2world recipe and the local media assets are Git LFS pointers in "
                "this checkout, so the default Studio task avoids silently pretending to run it."
            ),
            aliases=("default", "validation", "text-to-world", "interactive-video"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS_PREDICT2P5_VALIDATION_PROMPT),
                _field(
                    "input_path",
                    "Optional Image",
                    kind="path",
                    target="input_path",
                    default="",
                    description="Leave empty for text-to-world; provide an image path for image-to-world conditioning.",
                ),
                _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=93),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=35),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=7.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("output_type", "Output Type", target="call_kwargs", default="pt", choices=("pt", "pil")),
                _field("use_kerras_sigma", "Karras Sigma", kind="boolean", target="call_kwargs", default=True),
                _field("pad_mode", "Pad Mode", target="call_kwargs", default="repeat", choices=("repeat", "zero")),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS,
        ),
    ),
    notes=(
        "Official docs run examples/inference.py with assets/base/robot_pouring.jsonl.",
        f"In this checkout that file is not materialized by Git LFS: {COSMOS_PREDICT2P5_OFFICIAL_BASE_JSONL}",
    ),
)

GEN3C_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="gen3c",
    display_name="GEN3C",
    default_variant_id="default",
    default_task_id="official-single-image",
    aliases=("gen-3c", "gen3c-cosmos-7b", "nvidia/GEN3C-Cosmos-7B"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            checkpoints=(InferenceCheckpointRef(role="primary", uri="gen3c", status="configured"),),
            call_kwargs=GEN3C_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "cosmos-7b"),
            notes=("Matches the upstream single-image demo command in GEN3C README.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-single-image",
            label="Official Single Image",
            description="Run the GEN3C README single-image demo with the packaged official fixture.",
            aliases=("default", "interactive-video", "official-demo", "single-image"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GEN3C_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=GEN3C_OFFICIAL_PROMPT),
                _field(
                    "negative_prompt",
                    "Negative Prompt",
                    target="negative_prompt",
                    default=GEN3C_OFFICIAL_NEGATIVE_PROMPT,
                ),
                _field(
                    "trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    default="left",
                    choices=(
                        "left",
                        "right",
                        "up",
                        "down",
                        "zoom_in",
                        "zoom_out",
                        "clockwise",
                        "counterclockwise",
                    ),
                ),
                _field(
                    "camera_rotation",
                    "Camera Rotation",
                    target="call_kwargs",
                    default="center_facing",
                    choices=("center_facing", "no_rotation", "trajectory_aligned"),
                ),
                _field("movement_distance", "Movement Distance", kind="number", target="call_kwargs", default=0.3),
                _field("num_video_frames", "Frames", kind="integer", target="call_kwargs", default=121),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                _field("num_steps", "Steps", kind="integer", target="call_kwargs", default=35),
                _field("guidance", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=GEN3C_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default input is a byte-identical copy of upstream assets/diffusion/000000.png.",),
)

FANTASYWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="fantasyworld",
    display_name="FantasyWorld",
    default_variant_id="wan21",
    default_task_id="official-camera-world",
    aliases=(),
    variants=(
        InferenceVariantSpec(
            variant_id="wan21",
            label="Wan2.1",
            status="configured",
            call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
            aliases=("default", "official", "wan2.1"),
            notes=("Uses the in-tree FantasyWorld Wan2.1 camera-control runtime.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-world",
            label="Camera World",
            description="Run FantasyWorld image-conditioned camera-control world generation.",
            aliases=("default", "official-demo", "image-to-world"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=FANTASYWORLD_PROMPT),
                _field(
                    "camera_json_path",
                    "Camera JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=FANTASYWORLD_CAMERA_FIXTURE,
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1024),
                _field("using_scale", "Use Scale", kind="boolean", target="call_kwargs", default=True),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("stride", "Point Stride", kind="integer", target="call_kwargs", default=4),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("model", "generated_3d_asset", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
        ),
    ),
    notes=("The camera fixture is an in-tree c2w trajectory because the official demo JSON was not present locally.",),
)

FANTASYWORLD_WAN21_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="fantasyworld-wan21",
    display_name="FantasyWorld Wan2.1",
    default_variant_id="wan21",
    default_task_id="official-camera-world",
    aliases=("fantasyworld-wan2.1",),
    variants=(
        InferenceVariantSpec(
            variant_id="wan21",
            label="Wan2.1",
            status="configured",
            call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
            aliases=("default", "official", "wan2.1"),
            notes=("Uses the in-tree FantasyWorld Wan2.1 camera-control runtime.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-world",
            label="Camera World",
            description="Run FantasyWorld Wan2.1 image-conditioned camera-control world generation.",
            aliases=("default", "official-demo", "image-to-world"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=FANTASYWORLD_PROMPT),
                _field(
                    "camera_json_path",
                    "Camera JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=FANTASYWORLD_CAMERA_FIXTURE,
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1024),
                _field("using_scale", "Use Scale", kind="boolean", target="call_kwargs", default=True),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("stride", "Point Stride", kind="integer", target="call_kwargs", default=4),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("model", "generated_3d_asset", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
        ),
    ),
    notes=("The camera fixture is an in-tree c2w trajectory because the official demo JSON was not present locally.",),
)

FLASHWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="flashworld",
    display_name="FlashWorld",
    default_variant_id="default",
    default_task_id="official-camera-scene",
    aliases=("flash-world",),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=FLASHWORLD_CALL_KWARGS,
            aliases=("official", "demo"),
            notes=("Uses the in-tree FlashWorld pipeline with action-derived camera path.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-scene",
            label="Camera Scene",
            description="Run FlashWorld image-conditioned 3D scene generation.",
            aliases=("default", "official-demo", "image-to-3d"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default="A coherent explorable 3D scene."),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward", "camera_l", "right", "camera_zoom_in"),
                ),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=16),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=15),
                _field("image_height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("image_width", "Width", kind="integer", target="call_kwargs", default=704),
                _field("return_video", "Return Video", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("model", "generated_3d_asset", required=True, preview=True),
                _artifact("video", "video", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FLASHWORLD_CALL_KWARGS,
        ),
    ),
    notes=("Default image is the WorldFoundry studio fixture; official FlashWorld examples are JSON prompts.",),
)

LYRA1_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="lyra-1",
    display_name="Lyra-1",
    default_variant_id="default",
    default_task_id="official-static-sdg",
    aliases=("lyra1", "nvidia/lyra-1"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="checkpoint_bundle_root",
                    uri="${WORLDFOUNDRY_CKPT_DIR}",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="gen3c_transformer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/nvidia--GEN3C-Cosmos-7B",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="cosmos_tokenizer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="text_encoder",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b",
                    status="required",
                ),
            ),
            load_kwargs={"model_id": "lyra-1"},
            call_kwargs=LYRA1_STATIC_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "static-sdg"),
            notes=("Matches the upstream Lyra-1 scripts/bash/static_sdg.sh demo input and SDG defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-static-sdg",
            label="Official Static SDG",
            description="Run the Lyra-1 static single-image SDG demo from the official repository.",
            aliases=("default", "interactive-video", "official-demo", "static"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=LYRA1_STATIC_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=""),
                _field("mode", "Mode", target="call_kwargs", default="static", choices=("static",)),
                _field(
                    "trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    default="zoom_in",
                    choices=("left", "right", "up", "down", "zoom_in", "zoom_out"),
                ),
                _field(
                    "camera_rotation",
                    "Camera Rotation",
                    target="call_kwargs",
                    default="center_facing",
                    choices=("center_facing", "no_rotation", "trajectory_aligned"),
                ),
                _field("movement_distance", "Movement Distance", kind="number", target="call_kwargs", default=0.3),
                _field("multi_trajectory", "Multi Trajectory", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "total_movement_distance_factor",
                    "Movement Factor",
                    kind="number",
                    target="call_kwargs",
                    default=1.0,
                ),
                _field("num_video_frames", "Frames", kind="integer", target="call_kwargs", default=121),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                _field("num_steps", "Steps", kind="integer", target="call_kwargs", default=35),
                _field("guidance", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1),
                _field("num_gpus", "GPUs", kind="integer", target="call_kwargs", default=1),
                _field("foreground_masking", "Foreground Masking", kind="boolean", target="call_kwargs", default=True),
                _field("filter_points_threshold", "Point Filter", kind="number", target="call_kwargs", default=0.05),
                _field(
                    "disable_prompt_encoder",
                    "Disable Prompt Encoder",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field("disable_guardrail", "Disable Guardrail", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "offload_diffusion_transformer",
                    "Offload Transformer",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field("offload_tokenizer", "Offload Tokenizer", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "offload_text_encoder_model",
                    "Offload Text Encoder",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field(
                    "offload_prompt_upsampler",
                    "Offload Prompt Upsampler",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field(
                    "offload_guardrail_models", "Offload Guardrail", kind="boolean", target="call_kwargs", default=True
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LYRA1_STATIC_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image path is the upstream assets/demo/static/diffusion_input/images/00172.png fixture.",),
)

LYRA2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="lyra",
    display_name="Lyra",
    default_variant_id="lyra-2",
    default_task_id="official-zoomgs",
    aliases=("lyra-2", "lyra2", "nvidia/lyra", "nvidia/lyra-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="lyra-2",
            label="Lyra-2",
            status="configured",
            call_kwargs=LYRA2_OFFICIAL_CALL_KWARGS,
            aliases=("default", "official", "zoomgs"),
            notes=("Matches the upstream Lyra-2 assets/samples single-image navigation demo shape.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-zoomgs",
            label="Official ZoomGS",
            description="Run the Lyra-2 single-image action-conditioned world generation demo.",
            aliases=("default", "interactive-video", "official-demo", "zoomgs"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=LYRA2_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=LYRA2_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward",),
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("resolution", "Resolution", kind="json", target="call_kwargs", default=(480, 832)),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1),
                _field("reconstruct_3d", "Reconstruct 3D", kind="boolean", target="call_kwargs", default=False),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
                _field("show_progress", "Show Progress", kind="boolean", target="call_kwargs", default=True),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LYRA2_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image and caption come from upstream Lyra-2 assets/samples/00.*.",),
)

HELIOS_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="helios",
    display_name="Helios",
    default_variant_id="helios-distilled",
    default_task_id="text-to-video",
    aliases=("bestwishysh/helios",),
    variants=(
        InferenceVariantSpec(
            variant_id="helios-distilled",
            label="Distilled",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_DISTILLED_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "distilled", "pyramid_num_inference_steps_list": [2, 2, 2]},
            aliases=("distilled", "helios_distilled", "BestWishYsh/Helios-Distilled"),
            notes=("Official distilled checkpoint and two-step pyramid recipe.",),
        ),
        InferenceVariantSpec(
            variant_id="helios-base",
            label="Base",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_BASE_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "base", "pyramid_num_inference_steps_list": [20, 20, 20]},
            aliases=("base", "helios_base", "BestWishYsh/Helios-Base"),
            notes=("Official base checkpoint.",),
        ),
        InferenceVariantSpec(
            variant_id="helios-mid",
            label="Mid",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_MID_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "mid", "pyramid_num_inference_steps_list": [20, 20, 20]},
            aliases=("mid", "helios_mid", "BestWishYsh/Helios-Mid"),
            notes=("Official mid checkpoint.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-video",
            label="Text to Video",
            description="Generate a video with an official Helios Base, Mid, or Distilled checkpoint.",
            aliases=("t2v", "video", "default"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=HELIOS_DEMO_PROMPT),
                _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=240),
                _field("height", "Height", kind="integer", target="call_kwargs", default=384),
                _field("width", "Width", kind="integer", target="call_kwargs", default=640),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("guidance_scale", "Guidance", target="call_kwargs", default="auto"),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("enable_low_vram_mode", "Low VRAM Mode", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "sample_type": "t2v",
                "num_frames": 240,
                "height": 384,
                "width": 640,
                "fps": 24,
                "num_inference_steps": 50,
                "guidance_scale": "auto",
                "seed": 42,
            },
        ),
    ),
    notes=(
        "Each Workspace variant maps to exactly one official local checkpoint; no synthetic default variant is exposed.",
    ),
)


def _sana_streaming_inference_spec(*, bidirectional: bool) -> ModelInferenceSpec:
    """Build the explicit Workspace contract for one SANA-Streaming route."""

    model_id = (
        "sana-streaming-bidirectional-2b-720p" if bidirectional else "sana-streaming-2b-720p"
    )
    checkpoint = SANA_STREAMING_BIDIRECTIONAL_CHECKPOINT if bidirectional else SANA_STREAMING_CHECKPOINT
    mode = "bidirectional_short" if bidirectional else "long_streaming"
    steps = 50 if bidirectional else 4
    guidance = 6.0 if bidirectional else 1.0
    return ModelInferenceSpec(
        model_family_id=model_id,
        display_name="SANA-Streaming Bidirectional 2B 720p" if bidirectional else "SANA-Streaming 2B 720p",
        default_variant_id=mode,
        default_task_id="video-to-video",
        aliases=(
            ("sana-streaming-bidirectional", "sana-streaming-short")
            if bidirectional
            else ("sana-streaming", "sana-streaming-long", "sana-streaming-720p")
        ),
        variants=(
            InferenceVariantSpec(
                variant_id=mode,
                label="Bidirectional Short" if bidirectional else "Long Streaming",
                status="local_required",
                checkpoints=(
                    InferenceCheckpointRef(role="primary", uri=checkpoint, status="local_required"),
                    InferenceCheckpointRef(role="vae", uri=SANA_STREAMING_VAE, status="local_required"),
                    InferenceCheckpointRef(
                        role="text_encoder",
                        uri=SANA_STREAMING_TEXT_ENCODER,
                        status="local_required",
                    ),
                ),
                load_kwargs={"model_id": model_id},
                notes=(
                    "Uses the official short bidirectional denoising route."
                    if bidirectional
                    else "Uses state-cached autoregressive attention with sink-token caching."
                ,),
            ),
        ),
        tasks=(
            InferenceTaskProfile(
                task_id="video-to-video",
                label="Video to Video",
                description="Edit a source video with the official in-tree SANA-Streaming runtime.",
                aliases=("v2v", "video-editing", "default"),
                inputs=(
                    _field(
                        "prompt",
                        "Prompt",
                        target="prompt",
                        required=True,
                        default=SANA_STREAMING_DEMO_PROMPT,
                    ),
                    _field(
                        "input_path",
                        "Source Video",
                        kind="path",
                        target="input_path",
                        required=True,
                        description="Provide at least num_frames decoded frames; shorter videos are rejected.",
                    ),
                    _field("negative_prompt", "Negative Prompt", target="call_kwargs", default=""),
                    _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=81),
                    _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                    _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                    _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                    _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=steps),
                    _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=guidance),
                    _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                    _field("flow_shift", "Flow Shift", kind="number", target="call_kwargs", default=8.0),
                    _field(
                        "motion_score",
                        "Motion Score",
                        kind="integer",
                        target="call_kwargs",
                        default=10 if bidirectional else 0,
                    ),
                    _field("num_cached_blocks", "Cached Blocks", kind="integer", target="call_kwargs", default=2),
                    _field("sink_token", "Sink Token", kind="boolean", target="call_kwargs", default=True),
                ),
                outputs=(
                    _artifact("video", "video", required=True, preview=True),
                    _artifact("manifest", "manifest", required=True),
                ),
                default_call_kwargs={
                    "num_frames": 81,
                    "height": 704,
                    "width": 1280,
                    "fps": 16,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "seed": 42,
                    "flow_shift": 8.0,
                    "motion_score": 10 if bidirectional else 0,
                    "num_cached_blocks": 2,
                    "sink_token": True,
                },
            ),
        ),
        notes=(
            "The Workspace default is an 81-frame bounded validation; the long-streaming runtime itself retains "
            "the official 969-frame default when called without an override."
        ,),
    )


SANA_STREAMING_INFERENCE_SPEC = _sana_streaming_inference_spec(bidirectional=False)
SANA_STREAMING_BIDIRECTIONAL_INFERENCE_SPEC = _sana_streaming_inference_spec(bidirectional=True)


def _bernini_task_profile(
    task_id: str,
    label: str,
    *,
    input_kind: str | None = None,
    image_output: bool = False,
    reference_image: bool = False,
) -> InferenceTaskProfile:
    """Build one of Bernini's six official generation/editing contracts."""

    fields = [_field("prompt", "Prompt", target="prompt", required=True)]
    if input_kind is not None:
        fields.append(
            _field(
                "input_path",
                "Source Image" if input_kind == "image" else "Source Video",
                kind="path",
                target="input_path",
                required=True,
                default=GENERIC_IMAGE_FIXTURE if input_kind == "image" else None,
            )
        )
    if reference_image:
        fields.append(
            _field(
                "images",
                "Reference Image",
                kind="path",
                target="call_kwargs",
                required=True,
                default=GENERIC_IMAGE_FIXTURE,
            )
        )
    frames = 1 if image_output else 81
    height, width = ((512, 512) if image_output else (480, 848))
    steps = 50 if task_id in {"t2i", "t2v"} else 40
    fields.extend(
        (
            _field("task_type", "Task Type", target="call_kwargs", default=task_id, choices=(task_id,)),
            _field("neg_prompt", "Negative Prompt", target="call_kwargs"),
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=frames),
            _field("height", "Height", kind="integer", target="call_kwargs", default=height),
            _field("width", "Width", kind="integer", target="call_kwargs", default=width),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=steps),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
            _field(
                "nproc_per_node",
                "GPU Processes",
                kind="integer",
                target="call_kwargs",
                default=BERNINI_WORKSPACE_NPROC,
            ),
            _field(
                "ulysses_size",
                "Ulysses Size",
                kind="integer",
                target="call_kwargs",
                default=BERNINI_WORKSPACE_NPROC,
            ),
            _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
        )
    )
    artifact = _artifact("image", "generated_image", required=True, preview=True) if image_output else _artifact(
        "video", "video", required=True, preview=True
    )
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Bernini {label.lower()} through the official in-tree planner/renderer runtime.",
        aliases=(task_id, label.lower().replace(" ", "-")),
        inputs=tuple(fields),
        outputs=(artifact, _artifact("manifest", "manifest", required=True)),
        default_call_kwargs={
            "task_type": task_id,
            "num_frames": frames,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "seed": 42,
            "fps": 16,
            "nproc_per_node": BERNINI_WORKSPACE_NPROC,
            "ulysses_size": BERNINI_WORKSPACE_NPROC,
        },
    )


BERNINI_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="bernini",
    display_name="Bernini",
    default_variant_id="planner-renderer",
    default_task_id="t2v",
    aliases=("bernini-diffusers", "bernini-7b-14b"),
    variants=(
        InferenceVariantSpec(
            variant_id="planner-renderer",
            label="7B Planner + 14B Renderer",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=BERNINI_CHECKPOINT, status="local_required"),
            ),
            load_kwargs={"model_id": "bernini"},
            notes=(
                "Uses the full 81-frame/50-step recipe on one H100 because each additional Ulysses rank "
                "replicates about 179 GiB of host weights; the current cgroup does not leave a safe transient "
                "margin even for two ranks. "
                "The upstream eight-rank topology remains available as an explicit override on larger hosts.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="renderer-14b",
            label="14B Renderer",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=BERNINI_R_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={"model_id": "bernini-r-14b"},
            call_kwargs={"nproc_per_node": 1, "ulysses_size": 1},
            aliases=("bernini-r-14b", "14b-renderer"),
            notes=("Streams the high- and low-noise experts sequentially on one GPU.",),
        ),
        InferenceVariantSpec(
            variant_id="renderer-1.3b",
            label="1.3B Renderer",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=BERNINI_R_1P3B_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={"model_id": "bernini-r-1.3b"},
            call_kwargs={"nproc_per_node": 1, "ulysses_size": 1},
            aliases=("bernini-r-1.3b", "1.3b-renderer"),
        ),
    ),
    tasks=(
        _bernini_task_profile("t2i", "Text to Image", image_output=True),
        _bernini_task_profile("i2i", "Image to Image", input_kind="image", image_output=True),
        _bernini_task_profile("t2v", "Text to Video"),
        _bernini_task_profile("v2v", "Video to Video", input_kind="video"),
        _bernini_task_profile("r2v", "Reference to Video", input_kind="image"),
        _bernini_task_profile(
            "rv2v",
            "Reference Video to Video",
            input_kind="video",
            reference_image=True,
        ),
    ),
    notes=(
        "Checkpoint integrity and host/GPU memory are preflighted before any heavyweight model load.",
    ),
)

LINGBOT_VIDEO_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="lingbot-video",
    display_name="LingBot-Video",
    default_variant_id="dense",
    default_task_id="t2v",
    aliases=("lingbot-video-dense", "lingbot-video-moe", "robbyant/lingbot-video"),
    variants=(
        InferenceVariantSpec(
            variant_id="dense",
            label="Dense 1.3B",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=LINGBOT_VIDEO_DENSE_CHECKPOINT, status="configured"),
            ),
            load_kwargs={"variant": "dense"},
            aliases=("1.3b", "dense-1.3b", "lingbot-video-dense-1.3b"),
        ),
        InferenceVariantSpec(
            variant_id="moe",
            label="MoE 30B-A3B",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=LINGBOT_VIDEO_MOE_CHECKPOINT, status="configured"),
            ),
            load_kwargs={"variant": "moe"},
            aliases=("30b-a3b", "moe-30b-a3b", "lingbot-video-moe-30b-a3b"),
            notes=("The released MoE package also contains the optional refiner component.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="t2v",
            label="Text to Video",
            description="Generate video from a LingBot structured caption.",
            aliases=("text-to-video", "video"),
            inputs=(
                _field(
                    "prompt",
                    "Structured Prompt",
                    target="prompt",
                    required=True,
                    default=LINGBOT_VIDEO_STRUCTURED_PROMPT,
                    description="Structured JSON caption text; use prompt_json for a prepared caption file.",
                ),
                _field("prompt_json", "Prompt JSON", kind="path", target="call_kwargs"),
                _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
                _field(
                    "backend", "Backend", target="call_kwargs", default="diffusers", choices=("diffusers", "sglang")
                ),
                _field("mode", "Mode", target="call_kwargs", default="t2v", choices=("t2v",)),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=121),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=40),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=3.0),
                _field("shift", "Flow Shift", kind="number", target="call_kwargs", default=3.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("batch_cfg", "Batch CFG", kind="boolean", target="call_kwargs", default=False),
                _field("run_refiner", "Run Refiner", kind="boolean", target="call_kwargs", default=False),
                _field(
                    "reuse_condition_features", "Reuse Conditions", kind="boolean", target="call_kwargs", default=False
                ),
                _field("cfg_parallel_degree", "CFG Parallel", kind="integer", target="call_kwargs", default=1),
                _field("context_parallel_degree", "Context Parallel", kind="integer", target="call_kwargs", default=1),
                _field("nproc_per_node", "Processes", kind="integer", target="call_kwargs", default=1),
                _field("enable_fsdp_inference", "FSDP", kind="boolean", target="call_kwargs", default=False),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LINGBOT_VIDEO_DEFAULT_CALL_KWARGS,
        ),
        InferenceTaskProfile(
            task_id="ti2v",
            label="Text Image to Video",
            description="Generate video from a structured caption and first frame.",
            aliases=("image-to-video", "text-image-to-video"),
            inputs=(
                _field("image", "Image", kind="path", target="input_path", required=True),
                _field(
                    "prompt",
                    "Structured Prompt",
                    target="prompt",
                    required=True,
                    default=LINGBOT_VIDEO_STRUCTURED_PROMPT,
                ),
                _field("prompt_json", "Prompt JSON", kind="path", target="call_kwargs"),
                _field(
                    "backend", "Backend", target="call_kwargs", default="diffusers", choices=("diffusers", "sglang")
                ),
                _field("mode", "Mode", target="call_kwargs", default="ti2v", choices=("ti2v",)),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=121),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=40),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=3.0),
                _field("shift", "Flow Shift", kind="number", target="call_kwargs", default=3.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("run_refiner", "Run Refiner", kind="boolean", target="call_kwargs", default=False),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={**LINGBOT_VIDEO_DEFAULT_CALL_KWARGS, "mode": "ti2v"},
        ),
        InferenceTaskProfile(
            task_id="t2i",
            label="Text to Image",
            description="Generate an image from a LingBot structured caption.",
            aliases=("text-to-image", "image"),
            inputs=(
                _field(
                    "prompt",
                    "Structured Prompt",
                    target="prompt",
                    required=True,
                    default=LINGBOT_VIDEO_STRUCTURED_PROMPT,
                ),
                _field("prompt_json", "Prompt JSON", kind="path", target="call_kwargs"),
                _field(
                    "backend", "Backend", target="call_kwargs", default="diffusers", choices=("diffusers", "sglang")
                ),
                _field("mode", "Mode", target="call_kwargs", default="t2i", choices=("t2i",)),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=40),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=3.0),
                _field("shift", "Flow Shift", kind="number", target="call_kwargs", default=3.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("image", "image", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={**LINGBOT_VIDEO_DEFAULT_CALL_KWARGS, "mode": "t2i"},
        ),
    ),
    notes=(
        "The DiT consumes a structured JSON caption. The prompt rewriter is intentionally not part of this inference-only port.",
    ),
)

LONGCAT_VIDEO_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="longcat-video",
    display_name="LongCat-Video",
    default_variant_id="default",
    default_task_id="official-t2v",
    aliases=("longcat", "meituan-longcat/LongCat-Video"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=LONGCAT_VIDEO_LOCAL_CHECKPOINT, status="configured"),
            ),
            load_kwargs=LONGCAT_VIDEO_OFFICIAL_LOAD_KWARGS,
            call_kwargs=LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "t2v"),
            notes=("Matches the DiffSynth LongCat-Video text-to-video official example defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-t2v",
            label="Official T2V",
            description="Run the LongCat-Video official text-to-video demo prompt.",
            aliases=("default", "interactive-video", "official-demo", "text-to-video"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=LONGCAT_VIDEO_OFFICIAL_PROMPT),
                _field(
                    "negative_prompt",
                    "Negative Prompt",
                    target="call_kwargs",
                    default=LONGCAT_VIDEO_OFFICIAL_NEGATIVE_PROMPT,
                ),
                _field("task_type", "Task Type", target="call_kwargs", default="t2v", choices=("t2v",)),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=93),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=15),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field(
                    "distill_num_inference_steps", "Distill Steps", kind="integer", target="call_kwargs", default=16
                ),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=2.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
                _field("context_parallel_size", "Context Parallel", kind="integer", target="call_kwargs", default=1),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The T2V official demo does not require an input image path.",),
)


STABLE_VIDEO_INFINITY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="stable-video-infinity",
    display_name="Stable Video Infinity 2.0",
    default_variant_id="svi-2.0-wan2.1-i2v-14b",
    default_task_id="prompt-stream-i2v",
    aliases=("svi", "svi-2.0", "stable_video_infinity", "stable-video-infinity-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="svi-2.0-wan2.1-i2v-14b",
            label="SVI 2.0 · Wan2.1 I2V 14B",
            status="configured",
            load_kwargs={"lazy": True},
            aliases=("default", "svi-2.0", "wan2.1-i2v-14b"),
            notes=(
                "Loads the locally staged Wan2.1-I2V-14B-480P base and official SVI 2.0 LoRA.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="prompt-stream-i2v",
            label="Prompt Stream Image to Long Video",
            description=(
                "Generate linked SVI 2.0 clips from one reference image and an optional JSON prompt stream."
            ),
            aliases=("default", "i2v", "image-to-video", "long-video", "prompt-stream"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=STABLE_VIDEO_INFINITY_OFFICIAL_PROMPT,
                ),
                _field(
                    "prompt_stream",
                    "Prompt Stream (JSON)",
                    kind="json",
                    target="call_kwargs",
                    description=(
                        'Optional JSON array such as ["enter the forest", "approach the castle"]; '
                        "when provided, it overrides Prompt."
                    ),
                ),
                _field(
                    "image",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=str(STABLE_VIDEO_INFINITY_DEMO_ROOT / "frame.jpg"),
                ),
                _field(
                    "num_clips",
                    "Maximum Clips",
                    kind="integer",
                    target="call_kwargs",
                    default=999,
                    description=(
                        "Official upper bound; normal prompt streams stop after each prompt has been repeated twice."
                    ),
                ),
                _field("num_frames", "Frames per Clip", kind="integer", target="call_kwargs", default=81),
                _field(
                    "num_motion_frames",
                    "Continuation Frames",
                    kind="integer",
                    target="call_kwargs",
                    default=5,
                    description="Tail frames reused to condition the next autoregressive clip.",
                ),
                _field(
                    "num_inference_steps",
                    "Steps per Clip",
                    kind="integer",
                    target="call_kwargs",
                    default=50,
                ),
                _field(
                    "cfg_scale_text",
                    "Text CFG",
                    kind="number",
                    target="call_kwargs",
                    default=5.0,
                ),
                _field("seed", "Base Seed", kind="integer", target="call_kwargs", default=0),
                _field("seed_stride", "Seed Stride", kind="integer", target="call_kwargs", default=42),
                _field(
                    "prompt_repeat_times",
                    "Prompt Repeats",
                    kind="integer",
                    target="call_kwargs",
                    default=2,
                ),
                _field(
                    "ref_pad_num",
                    "Reference Padding",
                    kind="integer",
                    target="call_kwargs",
                    default=-1,
                    description="-1 preserves the official full-reference-padding behavior.",
                ),
                _field("height", "Height", kind="integer", target="call_kwargs"),
                _field("width", "Width", kind="integer", target="call_kwargs"),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("output_path", "Output Path", kind="path", target="call_kwargs"),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "num_clips": 999,
                "num_frames": 81,
                "num_motion_frames": 5,
                "num_inference_steps": 50,
                "cfg_scale_text": 5.0,
                "seed": 0,
                "seed_stride": 42,
                "prompt_repeat_times": 2,
                "ref_pad_num": -1,
                "fps": 24,
            },
        ),
    ),
    notes=(
        "Workspace dispatches this task to the shared cu128 environment and resolves both weight roots from runtime defaults.",
        "Checkpoint-backed parity still requires the Wan2.1 base and SVI LoRA to be staged locally.",
    ),
)

HUNYUAN_GAMECRAFT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-game-craft",
    display_name="Hunyuan Game Craft",
    default_variant_id="default",
    default_task_id="official-interactive-video",
    aliases=("hunyuan-gamecraft", "gamecraft", "tencent/Hunyuan-GameCraft-1.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            load_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_LOAD_KWARGS,
            call_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "gamecraft"),
            notes=("Matches the upstream Hunyuan-GameCraft official village demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-interactive-video",
            label="Official Interactive Video",
            description="Run the Hunyuan-GameCraft official village demo with the default action sequence.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HUNYUAN_GAMECRAFT_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HUNYUAN_GAMECRAFT_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward", "backward", "right", "left"),
                ),
                _field("frames", "Frames", kind="integer", target="params", default=129),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1216),
                _field("steps", "Steps", kind="integer", target="params", default=50),
                _field("guidance_scale", "Guidance", kind="number", target="params", default=2.0),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=250160),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The checkpoint is resolved from the Studio catalog so local /ckpt installs are preferred.",),
)

HUNYUAN_WORLD_VOYAGER_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-world-voyager",
    display_name="HunyuanWorld-Voyager",
    default_variant_id="default",
    default_task_id="official-case1-video",
    aliases=("hunyuan-voyager", "world-voyager", "tencent/HunyuanWorld-Voyager"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "case1"),
            notes=("Matches the upstream HunyuanWorld-Voyager examples/case1 demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-case1-video",
            label="Official Case1 Video",
            description="Run the HunyuanWorld-Voyager official case1 conditioned world-video demo.",
            aliases=("default", "official-demo", "conditioned-video"),
            inputs=(
                _field(
                    "image",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_PROMPT),
                _field(
                    "condition_dir",
                    "Condition Dir",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR,
                ),
                _field(
                    "interactions", "Interactions", kind="interaction_tokens", target="params", default=("forward",)
                ),
                _field("frames", "Requested Frames", kind="integer", target="params", default=49),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("steps", "Steps", kind="integer", target="params", default=50),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
                _field("flow_shift", "Flow Shift", kind="number", target="call_kwargs", default=7.0),
                _field("embedded_cfg_scale", "Embedded CFG", kind="number", target="call_kwargs", default=6.0),
                _field("i2v_stability", "I2V Stability", kind="boolean", target="call_kwargs", default=True),
                _field("ulysses_degree", "Ulysses Degree", kind="integer", target="call_kwargs", default=1),
                _field("ring_degree", "Ring Degree", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The official case1 requests 49 frames and emits 45 after dropping four I2V conditioning frames.",),
)

HY_WORLDPLAY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-worldplay",
    display_name="HY-WorldPlay",
    default_variant_id="default",
    default_task_id="official-image-pose-video",
    aliases=("hy-worldplay", "hyworldplay", "tencent/HY-WorldPlay"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=HY_WORLDPLAY_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "480p-i2v"),
            notes=("Matches the upstream HY-WorldPlay run.sh defaults for the 480p image-to-video demo.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-image-pose-video",
            label="Official Image Pose Video",
            description="Run the HY-WorldPlay official image-to-video demo with the default pose string.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HY_WORLDPLAY_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HY_WORLDPLAY_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Pose",
                    kind="interaction_tokens",
                    target="params",
                    required=True,
                    default=("w-31",),
                ),
                _field("num_frames", "Frames", kind="integer", target="params", default=125),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("height", "Height", kind="integer", target="params", default=480),
                _field("width", "Width", kind="integer", target="params", default=832),
                _field("num_inference_steps", "Steps", kind="integer", target="params", default=4),
                _field("seed", "Seed", kind="integer", target="params", default=1),
                _field("aspect_ratio", "Aspect Ratio", target="call_kwargs", default="16:9", choices=("16:9", "9:16")),
                _field("few_step", "Few Step", kind="boolean", target="call_kwargs", default=True),
                _field("enable_sr", "Super Resolution", kind="boolean", target="call_kwargs", default=False),
                _field("prompt_rewrite", "Prompt Rewrite", kind="boolean", target="call_kwargs", default=False),
                _field(
                    "torchrun_nproc_per_node",
                    "GPUs",
                    kind="integer",
                    target="load_kwargs",
                    default=HY_WORLDPLAY_WORKSPACE_NPROC,
                    description=(
                        "Torchrun workers for sequence-parallel inference. Workspace defaults to one worker "
                        "for cuda:N jobs; select 2, 3, 4, 6, or 8 explicitly for a coordinated multi-GPU run."
                    ),
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HY_WORLDPLAY_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=(
        "The action checkpoint is resolved from the Studio catalog model_ref; the base video model uses the runtime config.",
    ),
)

CAMERACTRL_OFFICIAL_PROMPT = "A serene mountain lake at sunrise, with mist hovering over the water."
CAMERACTRL_OFFICIAL_TRAJECTORY = _first_existing_path(
    Path(__file__).resolve().parents[1] / "data" / "test_cases" / "cameractrl" / "pose_files" / "0f47577ab3441480.txt",
    _official_repo_path("CameraCtrl", "assets", "pose_files", "0f47577ab3441480.txt"),
)

CAMERACTRL_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cameractrl",
    display_name="CameraCtrl",
    default_variant_id="default",
    default_task_id="trajectory-video",
    aliases=("camera-ctrl", "hehao13/CameraCtrl"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            aliases=("official", "animatediff"),
            notes=("Uses the staged CameraCtrl, RealEstate10K LoRA, and SD1.5 paths from the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="trajectory-video",
            label="Trajectory Video",
            description="Run the official CameraCtrl text-to-video trajectory demo.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=CAMERACTRL_OFFICIAL_PROMPT),
                _field(
                    "trajectory_file",
                    "Trajectory File",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=CAMERACTRL_OFFICIAL_TRAJECTORY,
                ),
                _field("frames", "Frames", kind="integer", target="params", default=16),
                _field("fps", "FPS", kind="integer", target="params", default=8),
                _field("height", "Height", kind="integer", target="params", default=256),
                _field("width", "Width", kind="integer", target="params", default=384),
                _field("steps", "Steps", kind="integer", target="params", default=25),
                _field("guidance_scale", "Guidance", kind="number", target="params", default=14.0),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field(
                    "original_pose_width", "Original Pose Width", kind="integer", target="call_kwargs", default=1280
                ),
                _field(
                    "original_pose_height", "Original Pose Height", kind="integer", target="call_kwargs", default=720
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "num_frames": 16,
                "fps": 8,
                "height": 256,
                "width": 384,
                "num_inference_steps": 25,
                "guidance_scale": 14.0,
                "seed": 42,
                "trajectory_file": CAMERACTRL_OFFICIAL_TRAJECTORY,
                "original_pose_width": 1280,
                "original_pose_height": 720,
            },
        ),
    ),
    notes=("The upstream reference videos are not runtime inputs for this wrapper; use the paired pose file.",),
)

DEPTH_ANYTHING3_OFFICIAL_FIXTURE = str(
    Path(__file__).resolve().parents[1] / "data" / "test_cases" / "depth_anything_v3" / "examples" / "SOH"
)

DEPTH_ANYTHING3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="depth-anything-v3",
    display_name="Depth Anything 3",
    default_variant_id="large",
    default_task_id="official-soh-export",
    aliases=("da3", "depth-anything-3", "depthanything3"),
    variants=(
        InferenceVariantSpec(
            variant_id="large",
            label="DA3 Large",
            status="configured",
            call_kwargs={"export_format": "mini_npz", "process_res": 504},
            aliases=("default", "official", "large-1.1"),
            notes=("Uses the staged DA3-LARGE checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-soh-export",
            label="Official SOH Export",
            description="Run the official Depth Anything 3 SOH-style image sequence export.",
            aliases=("default", "official", "official-demo", "interactive-video", "depth"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=DEPTH_ANYTHING3_OFFICIAL_FIXTURE,
                ),
                _field("export_format", "Export Format", target="call_kwargs", default="mini_npz"),
                _field("process_res", "Process Resolution", kind="integer", target="call_kwargs", default=504),
                _field("process_res_method", "Resize Method", target="call_kwargs", default="upper_bound_resize"),
                _field("max_frames", "Max Frames", kind="integer", target="call_kwargs", default=2),
                _field("frame_stride", "Frame Stride", kind="integer", target="call_kwargs", default=1),
                _field("infer_gs", "Infer Gaussians", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("depth", "depth", required=True, preview=True),
                _artifact("point_cloud", "3d"),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "export_format": "mini_npz",
                "process_res": 504,
                "process_res_method": "upper_bound_resize",
                "max_frames": 2,
                "frame_stride": 1,
                "infer_gs": False,
            },
        ),
    ),
    notes=(
        "The default task mirrors the lightweight official SOH demo path; GLB/GS export can be exposed as a separate task.",
    ),
)

WAN2P2_OFFICIAL_PROMPT = _read_text_file_or_default(
    _TEST_CASES_ROOT / "wan2.2" / "ti2v_5b" / "prompt.txt",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
)

WAN2P2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="wan-2p2",
    display_name="Wan 2.2",
    default_variant_id="ti2v-5b",
    default_task_id="ti2v-5b",
    aliases=("wan2.2", "wan-2.2", "wan2p2"),
    variants=(
        InferenceVariantSpec(
            variant_id="ti2v-5b",
            label="TI2V 5B",
            status="configured",
            load_kwargs={"mode": "ti2v-5B"},
            call_kwargs={"size": "1280*704"},
            aliases=("default", "official", "ti2v"),
            notes=("Uses the staged local Wan2.2-TI2V-5B checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="ti2v-5b",
            label="TI2V 5B",
            description="Run Wan 2.2 TI2V-5B with official default size and sampling config.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=WAN2P2_OFFICIAL_PROMPT),
                _field("input_path", "Input Image", kind="path", target="input_path"),
                _field("size", "Size", target="call_kwargs", default="1280*704", choices=("1280*704", "704*1280")),
                _field("frame_num", "Frames", kind="integer", target="call_kwargs"),
                _field("sample_steps", "Sample Steps", kind="integer", target="call_kwargs"),
                _field("sample_guide_scale", "Guidance", kind="number", target="call_kwargs"),
                _field("base_seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("offload_model", "Offload Model", kind="boolean", target="call_kwargs", default=True),
                _field("use_prompt_extend", "Prompt Extend", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "size": "1280*704",
                "base_seed": 42,
                "offload_model": True,
                "use_prompt_extend": False,
            },
        ),
    ),
    notes=("Keep validation jobs tiny by overriding frame_num/sample_steps from the frontend when needed.",),
)

CUT3R_OFFICIAL_FIXTURE = str(Path(__file__).resolve().parents[1] / "data" / "test_cases" / "cut3r" / "examples" / "001")

CUT3R_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cut3r",
    display_name="CUT3R",
    default_variant_id="default",
    default_task_id="official-export",
    aliases=("cut3r-512", "cut3r_512_dpt_4_64"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs={"size": 512, "vis_threshold": 1.5},
            aliases=("official", "demo"),
            notes=("Uses the staged local CUT3R checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-export",
            label="Official Export",
            description="Run the CUT3R official demo-style export on a sequence directory.",
            aliases=("official", "official-demo", "interactive-video"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=CUT3R_OFFICIAL_FIXTURE,
                ),
                _field("size", "Size", kind="integer", target="call_kwargs", default=512),
                _field("vis_threshold", "Visibility Threshold", kind="number", target="call_kwargs", default=1.5),
                _field("revisit", "Revisit", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("official_export", "metadata", required=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={"task_type": "cut3r_official_export", "size": 512, "vis_threshold": 1.5},
        ),
        InferenceTaskProfile(
            task_id="two-stage-3dgs",
            label="Two-stage 3DGS",
            description="Reconstruct a point cloud and render a short 3DGS preview video.",
            aliases=("cut3r_two_stage_3dgs", "two-stage", "3dgs"),
            inputs=(
                _field("input_path", "Input Path", kind="path", target="input_path", required=True),
                _field("interactions", "Interactions", kind="interaction_tokens", target="params"),
                _field("size", "Size", kind="integer", target="call_kwargs", default=512),
                _field("vis_threshold", "Visibility Threshold", kind="number", target="call_kwargs", default=1.5),
                _field("width", "Width", kind="integer", target="params", default=704),
                _field("height", "Height", kind="integer", target="params", default=480),
                _field(
                    "frames_per_interaction", "Frames Per Interaction", kind="integer", target="call_kwargs", default=10
                ),
                _field("num_orbit_frames", "Orbit Frames", kind="integer", target="call_kwargs", default=24),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
                _artifact("scene", "3d"),
            ),
            default_call_kwargs={
                "task_type": "cut3r_two_stage_3dgs",
                "size": 512,
                "vis_threshold": 1.5,
                "image_width": 704,
                "image_height": 480,
            },
        ),
    ),
    notes=("Official export mirrors the upstream demo inputs; 3DGS preview remains available as a separate task.",),
)

WORLDFM_OFFICIAL_META = str(_TEST_CASES_ROOT / "worldfm" / "meta.json")
WORLDFM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="worldfm",
    display_name="WorldFM",
    default_variant_id="default",
    default_task_id="official-meta-demo",
    aliases=("inspatio/worldfm", "world-fm"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            aliases=("official", "demo"),
            notes=("Uses the official WorldFM demo/meta.json input contract by default.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-meta-demo",
            label="Official Meta Demo",
            description="Run the official WorldFM meta.json demo with the bundled reference image, intrinsics, and target poses.",
            aliases=("default", "official-demo", "novel-view"),
            inputs=(
                _field(
                    "input_path",
                    "Optional Input Image",
                    kind="path",
                    target="input_path",
                    default="",
                    description="Leave empty to use the image referenced by meta-path.",
                ),
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "interactions",
                    "Target Camera Poses",
                    kind="interaction_tokens",
                    target="params",
                    description="Leave empty to use c2w poses from meta-path.",
                ),
                _field("fps", "FPS", kind="integer", target="params", default=30),
                _field(
                    "meta-path",
                    "Meta JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=WORLDFM_OFFICIAL_META,
                ),
                _field("panorama-path", "Panorama Path", kind="path", target="call_kwargs"),
                _field("scene-name", "Scene Name", target="call_kwargs"),
                _field("save-mode", "Save Mode", target="call_kwargs", default="video", choices=("video", "image")),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
                _artifact("scene_context", "manifest"),
            ),
            default_call_kwargs={"meta_path": WORLDFM_OFFICIAL_META, "fps": 30, "save_mode": "video"},
        ),
    ),
    notes=("Official usage: python run_pipeline.py --meta demo/meta.json --output_dir outputs.",),
)

WOW_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="wow",
    display_name="WoW",
    default_variant_id="wan-14b-600k",
    default_task_id="official-image-to-world",
    aliases=("wow-world-model", "wow-world-model/wow-1-wan-14b-600k"),
    variants=(
        InferenceVariantSpec(
            variant_id="wan-14b-600k",
            label="Wan 14B 600k",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=WOW_LOCAL_CHECKPOINT,
                    status="configured",
                ),
            ),
            call_kwargs={
                "steps": 50,
                "seed": 42,
                "num_frames": 41,
                "no_tiled": False,
                "enable_vram_management": True,
                "persistent_param_gb": 70,
            },
            aliases=("default", "official", "14b", "wan-14b", "600k"),
            notes=("Matches the official WoW Wan demo checkpoint X-Humanoid/WoW-1-Wan-14B-600k.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-image-to-world",
            label="Image To World",
            description="Run the WoW image-conditioned world-video generation path with explicit prompt and sampling controls.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=WOW_OFFICIAL_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", required=True, default=WOW_OFFICIAL_PROMPT),
                _field("steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=41),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("no_tiled", "Disable Tiling", kind="boolean", target="call_kwargs", default=False),
                _field("enable_vram_management", "VRAM Management", kind="boolean", target="call_kwargs", default=True),
                _field("persistent_param_gb", "Persistent Param GB", kind="integer", target="call_kwargs", default=70),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "steps": 50,
                "seed": 42,
                "num_frames": 41,
                "no_tiled": False,
                "enable_vram_management": True,
                "persistent_param_gb": 70,
            },
        ),
    ),
    notes=("The upstream WoW script uses image conditioning, prompt text, seed 42, and generated video output.",),
)

GENIE_ENVISIONER_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="genie-envisioner",
    display_name="Genie Envisioner",
    default_variant_id="ge-base-fast-v0.1",
    default_task_id="robot-multiview-video",
    aliases=("genie_envisioner", "genieenvisioner"),
    variants=(
        InferenceVariantSpec(
            variant_id="ge-base-fast-v0.1",
            label="GE Base Fast v0.1",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="genie_envisioner_transformer",
                    uri=(
                        "${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/"
                        "GE_base_fast_v0.1.safetensors"
                    ),
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="ltx_video_base_components",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "genie-envisioner",
                "checkpoint_path": (
                    "${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/"
                    "GE_base_fast_v0.1.safetensors"
                ),
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
            },
            aliases=("default", "official", "fast", "ge-base-fast"),
            notes=(
                "Uses the released GE-base-fast multiview transformer with local LTX 0.9 tokenizer, T5, and VAE components.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-multiview-video",
            label="Robot Multiview Video",
            description="Generate a three-view robot video from four-frame camera histories; bundled still images are smoke inputs.",
            aliases=("default", "robot-action-video", "synthetic-three-view-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Smoke Preview Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/aloha/observation_images_cam_high.png"
                    ),
                    description="Bundled ALOHA still image used by the default smoke configuration.",
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="Move the robot gripper toward the objects on the workbench.",
                ),
                _field(
                    "checkpoint_path",
                    "Genie Envisioner Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/"
                        "GE_base_fast_v0.1.safetensors"
                    ),
                ),
                _field(
                    "base_model_dir",
                    "LTX-Video Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
                ),
                _field(
                    "input_views",
                    "Camera History Directories or Smoke Images",
                    kind="json",
                    target="call_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                    ),
                    description=(
                        "For official GE-base history, pass three directories in head, left-wrist, "
                        "right-wrist order; each needs 0.png through 3.png. The default three "
                        "ALOHA still images are repeated across four history slots for smoke use."
                    ),
                ),
                _field(
                    "input_mode",
                    "Input Mode",
                    target="call_kwargs",
                    default="explicit-three-view",
                    choices=("explicit-three-view", "synthetic-three-view"),
                    description=(
                        "Explicit mode accepts three four-frame history directories or three still images; "
                        "the bundled still images and synthetic mode are smoke configurations."
                    ),
                ),
                _field("height", "Height", kind="integer", target="call_kwargs", default=192),
                _field("width", "Per-View Width", kind="integer", target="call_kwargs", default=256),
                _field("n_previous", "History Frames", kind="integer", target="call_kwargs", default=4),
                _field("latent_chunk", "Generated Latent Frames", kind="integer", target="call_kwargs", default=2),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=5),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("decode_timestep", "Decode Timestep", kind="number", target="call_kwargs", default=0.03),
                _field("decode_noise_scale", "Decode Noise", kind="number", target="call_kwargs", default=0.025),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=8),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "input_mode": "explicit-three-view",
                "input_views": [
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                ],
                "height": 192,
                "width": 256,
                "n_previous": 4,
                "latent_chunk": 2,
                "num_inference_steps": 5,
                "guidance_scale": 1.0,
                "decode_timestep": 0.03,
                "decode_noise_scale": 0.025,
                "fps": 8,
                "seed": 42,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The default consumes synchronized ALOHA high/left-wrist/right-wrist observations and repeats four history frames.",
        "Use a checkpoint-matched AgiBot or LeRobot trajectory for action-quality parity.",
    ),
)


GIGA_WORLD_0_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="giga-world-0",
    display_name="GigaWorld-0",
    default_variant_id="gr1-2b",
    default_task_id="robot-image-to-video",
    aliases=("giga_world_0", "gigaworld-0", "gigaworld0"),
    variants=(
        InferenceVariantSpec(
            variant_id="gr1-2b",
            label="GR1 Video 2B",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="giga_world_0_gr1_transformer",
                    uri=(
                        "${WORLDFOUNDRY_CKPT_DIR}/"
                        "open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"
                    ),
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="t5_11b_encoder_safetensors",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="wan_diffusers_vae",
                    uri=(
                        "${WORLDFOUNDRY_CKPT_DIR}/"
                        "Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"
                    ),
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "giga-world-0",
                "transformer_model_dir": (
                    "${WORLDFOUNDRY_CKPT_DIR}/"
                    "open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"
                ),
                "text_encoder_model_dir": (
                    "${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder"
                ),
                "vae_model_dir": (
                    "${WORLDFOUNDRY_CKPT_DIR}/"
                    "Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"
                ),
            },
            aliases=("default", "official", "gr1"),
            notes=(
                "Uses the released GR1 image-conditioned transformer with a safe local T5 encoder and Wan Diffusers VAE.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-image-to-video",
            label="Robot Image To Video",
            description="Generate a short robot-world video from one initial observation and task prompt.",
            aliases=("default", "gr1", "single-image-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/droid/exterior_image_1_left.png"
                    ),
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="A robot arm moves toward the objects on the workbench.",
                ),
                _field(
                    "transformer_model_dir",
                    "GR1 Transformer",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_CKPT_DIR}/"
                        "open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"
                    ),
                ),
                _field(
                    "text_encoder_model_dir",
                    "T5-11B Encoder",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder",
                ),
                _field(
                    "vae_model_dir",
                    "Wan Diffusers VAE",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_CKPT_DIR}/"
                        "Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"
                    ),
                ),
                _field(
                    "attention_backend",
                    "Attention Backend",
                    target="call_kwargs",
                    default="natten",
                    choices=("natten", "torch"),
                    description=(
                        "NATTEN is the official Hopper path. Torch SDPA is a low-resolution "
                        "checkpoint-smoke compatibility backend, not quality parity."
                    ),
                ),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=640),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=61),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=30),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=6666),
                _field("max_text_length", "Text Tokens", kind="integer", target="call_kwargs", default=128),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "attention_backend": "natten",
                "height": 480,
                "width": 640,
                "num_frames": 61,
                "num_inference_steps": 30,
                "guidance_scale": 1.0,
                "fps": 16,
                "seed": 6666,
                "max_text_length": 128,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The default uses released 480x640, 61-frame, 30-step NATTEN sampling settings.",
        "Use a checkpoint-matched GR1 observation for dataset parity.",
    ),
)


ADAWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="adaworld",
    display_name="AdaWorld",
    default_variant_id="official",
    default_task_id="latent-action-world-model",
    variants=(
        InferenceVariantSpec(
            variant_id="official",
            label="Official AdaWorld FDM",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="adaworld_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Little-Podi--AdaWorld/adaworld.safetensors",
                    status="required",
                ),
            ),
            aliases=("default",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="latent-action-world-model",
            label="Latent Action World Model",
            description="Generate an AdaWorld video from a readable source video using the official FDM checkpoint.",
            aliases=("default",),
            inputs=(
                _field(
                    "video",
                    "Source Video",
                    kind="path",
                    target="input_path",
                    required=True,
                    description="Readable RGB video used to extract AdaWorld's latent action conditioning.",
                ),
                _field(
                    "checkpoint_path",
                    "AdaWorld Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    description="Optional explicit adaworld.safetensors path; the default checkpoint root is discovered automatically.",
                ),
                _field("start_index", "Start Frame", kind="integer", default=0),
                _field("video_len", "Video Frames", kind="integer", default=20),
                _field("resolution", "Resolution", kind="integer", default=256),
                _field("context_frame", "Context Frames", kind="integer", default=6),
                _field("num_steps", "Sampling Steps", kind="integer", default=5),
                _field("cfg_scale", "Guidance Scale", kind="number", default=1.1),
                _field("aug_level", "Augmentation Level", kind="number", default=0.1),
                _field("fps", "FPS", kind="integer", default=5),
                _field("seed", "Seed", kind="integer", default=32),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
        ),
    ),
    notes=(
        "The public path uses the released AdaWorld checkpoint and requires a readable source video.",
        "Only short synthetic-video inference has been verified; task quality and long-horizon behavior are unmeasured.",
    ),
)


CTRL_WORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="ctrl-world",
    display_name="Ctrl-World",
    default_variant_id="droid-checkpoint-10000",
    default_task_id="robot-action-video",
    aliases=("ctrl_world", "ctrlworld"),
    variants=(
        InferenceVariantSpec(
            variant_id="droid-checkpoint-10000",
            label="DROID Checkpoint 10000",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="ctrl_world_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="clip_text_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "ctrl-world",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                "clip_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
            },
            aliases=("default", "official", "droid"),
            notes=("Uses the released DROID action-conditioned checkpoint with local SVD and CLIP components.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a three-camera action-conditioned robot rollout.",
            aliases=("default", "droid", "synthetic-three-view-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/aloha/observation_images_cam_high.png"
                    ),
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="Move the robot gripper toward the objects on the workbench.",
                ),
                _field(
                    "checkpoint_path",
                    "Ctrl-World Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                ),
                _field(
                    "base_model_dir",
                    "Stable Video Diffusion Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field(
                    "clip_model_dir",
                    "CLIP Text Model",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
                ),
                _field(
                    "input_views",
                    "Synchronized Camera Views",
                    kind="json",
                    target="call_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                    ),
                    description="Three synchronized high, left-wrist, and right-wrist camera images.",
                ),
                _field(
                    "input_mode",
                    "Three-View Input",
                    target="call_kwargs",
                    default="explicit-three-view",
                    choices=("explicit-three-view", "synthetic-three-view"),
                    description="Consumes three synchronized camera images by default; synthetic mode is retained for smoke tests.",
                ),
                _field(
                    "action_mode",
                    "Action Conditioning",
                    target="call_kwargs",
                    default="absolute-pose",
                    choices=("absolute-pose", "synthetic-zero-smoke"),
                    description="Physical DROID Cartesian pose conditioning is required for action-quality evaluation; zero-pose conditioning is smoke only.",
                ),
                _field(
                    "initial_pose",
                    "Initial Cartesian Pose",
                    kind="json",
                    target="call_kwargs",
                    description="Required in absolute-pose mode: seven physical DROID values [x, y, z, rx, ry, rz, gripper] corresponding to the three camera images.",
                ),
                _field(
                    "action_direction",
                    "Action Direction",
                    target="call_kwargs",
                    default="right",
                    choices=("stationary", "forward", "backward", "left", "right", "up", "down", "open", "close"),
                ),
                _field("action_distance", "XYZ Distance (metres)", kind="number", target="call_kwargs", default=0.08),
                _field("action_scale", "Smoke Normalized Delta", kind="number", target="call_kwargs", description="Only used with synthetic-zero-smoke conditioning."),
                _field("height", "Per-View Height", kind="integer", target="call_kwargs", default=192),
                _field("width", "Per-View Width", kind="integer", target="call_kwargs", default=320),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=5),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=4),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("motion_bucket_id", "Motion Bucket", kind="integer", target="call_kwargs", default=127),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=4),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "input_mode": "explicit-three-view",
                "input_views": [
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                ],
                "action_direction": "right",
                "action_scale": 0.2,
                "height": 192,
                "width": 320,
                "num_frames": 5,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "motion_bucket_id": 127,
                "fps": 4,
                "seed": 42,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The output concatenates three predicted camera views horizontally.",
        "The default uses synchronized ALOHA views; checkpoint-matched DROID actions and views remain necessary for action-quality parity.",
    ),
)

DIAMOND_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="diamond",
    display_name="DIAMOND",
    default_variant_id="atari-pong-pretrained",
    default_task_id="runtime-plan",
    aliases=("diamond-atari", "diamond-wm"),
    variants=(
        InferenceVariantSpec(
            variant_id="atari-pong-pretrained",
            label="Atari Pong Pretrained",
            status="local_checkpoint_confirmed",
            checkpoints=(
                InferenceCheckpointRef(
                    role="pretrained_dir",
                    uri=str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                    status="local_confirmed",
                ),
            ),
            load_kwargs={
                "model_id": "diamond",
                "pretrained": True,
                "pretrained_game": "Pong",
                "pretrained_dir": str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                "fps": 15,
                "size": 640,
                "num_steps_initial_collect": 1000,
            },
            aliases=("default", "pong", "pretrained"),
            notes=(
                "Matches the official DIAMOND Atari pretrained play route; Studio defaults to plan-only execution.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="runtime-plan",
            label="Runtime Plan",
            description="Prepare the official DIAMOND play.py Atari route and report runtime requirements.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=False,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field(
                    "interactions", "Interactions", kind="interaction_tokens", target="params", default=("forward",)
                ),
                _field("frames", "Frames", kind="integer", target="params", default=16),
                _field("fps", "FPS", kind="integer", target="params", default=15),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("pretrained", "Pretrained", kind="boolean", target="load_kwargs", default=True),
                _field(
                    "pretrained_game",
                    "Pretrained Game",
                    target="load_kwargs",
                    default="Pong",
                    choices=("Pong", "Breakout", "Seaquest", "Qbert", "MsPacman", "Freeway"),
                ),
                _field(
                    "pretrained_dir",
                    "Pretrained Dir",
                    kind="path",
                    target="load_kwargs",
                    default=str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                ),
                _field(
                    "num_steps_initial_collect",
                    "Initial Collect Steps",
                    kind="integer",
                    target="load_kwargs",
                    default=1000,
                ),
                _field("size", "Window Size", kind="integer", target="load_kwargs", default=640),
                _field("record", "Record", kind="boolean", target="load_kwargs", default=False),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("generated_world", "generated_world"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Set call_kwargs.plan_only=false for real Studio/API execution after confirming the Atari runtime and GPU budget.",
    ),
)

MIRA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="mira",
    display_name="MIRA",
    default_variant_id="local-checkpoint",
    default_task_id="rocket-science-rollout",
    aliases=("mira-wm", "multiplayer-interactive-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="local-checkpoint",
            label="Local MIRA Checkpoint",
            status="requires_local_checkpoint_and_dataset",
            load_kwargs={"model_id": "mira"},
            aliases=("default", "offline", "rocket-science"),
            notes=("Uses a local MIRA run directory and local Rocket Science test split.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="rocket-science-rollout",
            label="Rocket Science Rollout",
            description="Generate one offline, action-conditioned MIRA rollout from a Rocket Science clip.",
            aliases=("default", "offline-rollout", "game-rollout"),
            inputs=(
                _field(
                    "checkpoint_path",
                    "Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=MIRA_DEFAULT_CHECKPOINT,
                    description="MIRA checkpoint.pth file or run directory.",
                ),
                _field(
                    "dataset_path",
                    "Rocket Science Split",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=MIRA_DEFAULT_DATASET,
                    description="Local split directory containing index.json, or the index.json path.",
                ),
                _field(
                    "actions",
                    "Action Timeline",
                    kind="json",
                    target="load_kwargs",
                    default=(),
                    description=(
                        "JSON segments such as [{\"player\":0,\"keys\":[\"W\",\"D\"],"
                        "\"start\":0,\"frames\":20}]."
                    ),
                ),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=42),
                _field("clip_index", "Dataset Clip", kind="integer", target="load_kwargs", default=0),
                _field(
                    "n_context_frames", "Context Frames", kind="integer", target="load_kwargs", default=38
                ),
                _field(
                    "num_unrolled_frames",
                    "Generated Latent Frames",
                    kind="integer",
                    target="load_kwargs",
                    default=20,
                ),
                _field(
                    "n_diffusion_steps", "Diffusion Steps", kind="integer", target="load_kwargs", default=10
                ),
                _field(
                    "schedule_type",
                    "Schedule",
                    target="load_kwargs",
                    default="linear",
                    choices=("linear", "linear_quadratic"),
                ),
                _field("noise_level", "Cache Noise", kind="number", target="load_kwargs", default=0.0),
                _field("compile", "Compile Model", kind="boolean", target="load_kwargs", default=False),
                _field(
                    "overlay_actions", "Overlay Actions", kind="boolean", target="load_kwargs", default=True
                ),
                _field(
                    "generated_only", "Generated Frames Only", kind="boolean", target="load_kwargs", default=False
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field(
                    "timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("runtime_plan", "manifest", required=True),
                _artifact("result", "manifest"),
            ),
            default_call_kwargs={"plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Inference-only integration; no MIRA training or metric entrypoints are exposed.",
        "The isolated runtime mirrors upstream's torch 2.8, CPU TorchCodec 0.7, and FFmpeg 7 pins.",
    ),
)

DINO_WM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="dino-wm",
    display_name="DINO-WM",
    default_variant_id="wall-official",
    default_task_id="wall-planning",
    aliases=("dino_wm", "dino-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="wall-official",
            label="Wall Planning",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="ckpt_base_path",
                    uri=DINO_WM_CHECKPOINT_BASE,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_id": "dino-wm",
                "config": str(_RUNTIME_CONFIGS_ROOT / "dino_wm" / "conf" / "plan_wall.yaml"),
                "ckpt_base_path": DINO_WM_CHECKPOINT_BASE,
                "model_name": "wall_single",
                "model_epoch": "latest",
            },
            aliases=("default", "wall", "official-demo"),
            notes=(
                "The released wall checkpoint lives at outputs/wall_single/checkpoints/model_latest.pth under ckpt_base_path.",
                "The wall_single integration passed real GPU inference with official trajectories; provide a local checkpoint and DATASET_DIR for execution.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="wall-planning",
            label="Wall Planning",
            description="Run or preflight the official DINO-WM wall planning/evaluation demo.",
            aliases=("default", "official-demo", "plan-wall"),
            inputs=(
                _field(
                    "config",
                    "Config",
                    kind="path",
                    target="load_kwargs",
                    default=str(_RUNTIME_CONFIGS_ROOT / "dino_wm" / "conf" / "plan_wall.yaml"),
                ),
                _field(
                    "ckpt_base_path",
                    "Checkpoint Base Path",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=DINO_WM_CHECKPOINT_BASE,
                ),
                _field(
                    "model_name",
                    "Model Name",
                    target="load_kwargs",
                    required=True,
                    default="wall_single",
                    choices=("wall_single", "pusht", "point_maze"),
                ),
                _field("model_epoch", "Model Epoch", target="load_kwargs", default="latest"),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=99),
                _field("n_evals", "Eval Count", kind="integer", target="load_kwargs", default=1),
                _field("goal_source", "Goal Source", target="load_kwargs", default="dset"),
                _field("goal_H", "Goal Horizon", kind="integer", target="load_kwargs", default=1),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("result", "json"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Stage the official checkpoint under ckpt_base_path/outputs/wall_single and set DATASET_DIR to the extracted official wall trajectories.",
        "Set WORLDFOUNDRY_DINO_WM_CKPT_BASE before importing the catalog to select a nonstandard checkpoint root; a published smoke run used a smaller planner budget than plan_wall.yaml defaults.",
    ),
)

LEWORLD_MODEL_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="leworldmodel",
    display_name="LeWorldModel",
    default_variant_id="pusht-random",
    default_task_id="pusht-eval",
    aliases=("le-wm", "lewm", "leworldmodel"),
    variants=(
        InferenceVariantSpec(
            variant_id="pusht-random",
            label="PushT Random Policy",
            status="local_dataset_smoke_ready",
            checkpoints=(),
            load_kwargs={
                "model_id": "leworldmodel",
                "config_dir": "${WORLDFOUNDRY_DATA_ROOT}/models/runtime/configs/le_wm/config/eval",
                "config_name": "pusht",
                "policy": "random",
                "cache_dir": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
                "dataset_name": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
                "num_eval": 1,
                "eval_budget": 25,
                "goal_offset_steps": 5,
                "seed": 42,
            },
            aliases=("default", "pusht", "random"),
            notes=(
                "Uses the official PushT eval flow with a deterministic locally collected trajectory and random policy; no policy checkpoint is required.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="pusht-eval",
            label="PushT Eval",
            description="Run or preflight the official LeWorldModel PushT evaluation route.",
            aliases=("default", "official-demo", "pusht"),
            inputs=(
                _field(
                    "config_dir",
                    "Config Dir",
                    kind="path",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_DATA_ROOT}/models/runtime/configs/le_wm/config/eval",
                ),
                _field(
                    "config_name",
                    "Config Name",
                    target="load_kwargs",
                    default="pusht",
                    choices=("pusht", "cube", "tworoom", "reacher"),
                ),
                _field("policy", "Policy", target="load_kwargs", default="random"),
                _field(
                    "cache_dir",
                    "Cache Dir",
                    kind="path",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
                ),
                _field(
                    "dataset_name",
                    "Dataset Name",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
                ),
                _field("num_eval", "Eval Count", kind="integer", target="load_kwargs", default=1),
                _field("eval_budget", "Eval Budget", kind="integer", target="load_kwargs", default=25),
                _field("goal_offset_steps", "Goal Offset Steps", kind="integer", target="load_kwargs", default=5),
                _field("img_size", "Image Size", kind="integer", target="load_kwargs"),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("result", "json"),
                _artifact("video", "video"),
            ),
            default_call_kwargs={"plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "The PushT random-policy demo uses a deterministic trajectory prepared under the checkpoint root. Non-random policies still require compatible policy checkpoints.",
    ),
)

EGOWM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="egowm",
    display_name="EgoWM",
    default_variant_id="svd-25dof-nav",
    default_task_id="navigation-video",
    aliases=("ego-wm", "egowm-svd-25dof"),
    variants=(
        InferenceVariantSpec(
            variant_id="svd-25dof-nav",
            label="SVD 25-DoF Navigation",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="egowm_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_25dof_nav.pth",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "egowm",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_25dof_nav.pth",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            aliases=("default", "official", "25dof"),
            notes=(
                "Runs the official action/state-conditioned SVD model from a checkout staged under "
                "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/egowm.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="svd-3dof-nav",
            label="SVD 3-DoF Navigation",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="egowm_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "egowm",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                "model_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            call_kwargs={"variant": "3dof"},
            aliases=("3dof",),
            notes=(
                "Pair an initial image with actions_path containing one normalized [x, y, yaw] action per generated frame.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="navigation-video",
            label="Navigation Video",
            description="Generate an egocentric navigation rollout from a paired initial image, 25-DoF state, and per-frame actions.",
            aliases=("default", "official-demo", "25dof-nav"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Scene Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    description="Supply the first frame paired with the 25-DoF conditions JSON.",
                ),
                _field(
                    "conditions_path",
                    "Paired 25-DoF Conditions JSON",
                    kind="path",
                    target="call_kwargs",
                    description="Required for 25dof; omit for 3dof. Contains physical_initial_state and physical_actions.",
                ),
                _field(
                    "actions_path",
                    "Paired 3-DoF Actions JSON",
                    kind="path",
                    target="call_kwargs",
                    description="For 3dof, supply normalized_actions with one [x, y, yaw] row per generated frame, paired with the initial image.",
                ),
                _field("variant", "Checkpoint Variant", kind="string", target="call_kwargs",
                       choices=("25dof", "3dof")),
                _field(
                    "checkpoint_path",
                    "EgoWM Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                ),
                _field(
                    "base_model_dir",
                    "SVD Base Model",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field("num_frames", "Generated Frames", kind="integer", target="call_kwargs", default=8),
                _field("num_inference_steps", "Inference Steps", kind="integer", target="call_kwargs", default=25),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=7),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("height", "Height", kind="integer", target="call_kwargs", default=512),
                _field("width", "Width", kind="integer", target="call_kwargs", default=512),
                _field("motion_bucket_id", "Motion Bucket", kind="integer", target="call_kwargs", default=180),
                _field("smoke_synthetic", "Synthetic Smoke Test Only", kind="boolean", target="call_kwargs", default=False),
                _field("action_scale", "3-DoF Action Offset", kind="number", target="call_kwargs",
                       description="Constant normalized x offset for a 3dof smoke test; paired inference uses actions_path."),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "variant": "25dof",
                "num_frames": 8,
                "num_inference_steps": 25,
                "fps": 7,
                "seed": 42,
                "height": 512,
                "width": 512,
                "motion_bucket_id": 180,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The compatibility wrapper passes the local SVD base model explicitly and leaves the official checkout unchanged. "
        "Paired 3-DoF actions_path and paired 25-DoF conditions_path use distinct input contracts.",
    ),
)

HMA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hma",
    display_name="HMA",
    default_variant_id="base-cont-langtable",
    default_task_id="robot-action-video",
    aliases=("hma-magvit-362m", "hma-continuous"),
    variants=(
        InferenceVariantSpec(
            variant_id="base-cont-langtable",
            label="Base Continuous LangTable",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="hma_continuous_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_temporal_vae",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "hma",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            aliases=("default", "official", "base-cont", "hma-mar-1b"),
            notes=(
                "Runs the official STMAR continuous sampler from a checkout staged under "
                "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/HMA.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="hma-magvit-362m",
            label="Base Discrete LangTable",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="hma_discrete_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "hma",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
                "model_path": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
            },
            aliases=("base-disc",),
            notes=(
                "Runs the official STMaskGIT discrete sampler and requires the official MAGVIT2 tokenizer "
                "under the staged HMA checkout or hma_magvit_checkpoint override.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a LangTable rollout; the default repeats one image and action for smoke testing.",
            aliases=("default", "official-demo", "langtable"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default="${WORLDFOUNDRY_MODEL_SOURCE_DIR}/HMA/assets/langtable_prompt/frame_00.png",
                ),
                _field(
                    "checkpoint_dir",
                    "HMA Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                ),
                _field(
                    "base_model_dir",
                    "SVD Temporal VAE",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field("generated_frames", "Generated Frames", kind="integer", target="call_kwargs", default=6),
                _field("prompt_horizon", "Prompt Horizon", kind="integer", target="call_kwargs", default=3),
                _field("maskgit_steps", "MaskGIT Steps", kind="integer", target="call_kwargs", default=2),
                _field(
                    "trajectory_path",
                    "Paired LangTable Trajectory JSON",
                    kind="path",
                    target="call_kwargs",
                    description=(
                        "Optional real-history input: action_format=raw_language_table_delta_yx, "
                        "prompt_frames has prompt_horizon image paths, and actions has "
                        "prompt_horizon-1+generated_frames raw 2D increments. "
                        "The checkpoint normalizes these actions internally."
                    ),
                ),
                _field(
                    "direction",
                    "Action Direction",
                    target="call_kwargs",
                    default="right",
                    choices=("right", "left", "down", "up"),
                ),
                _field("action_scale", "Smoke Action Scale", kind="number", target="call_kwargs", default=0.05,
                       description="Repeated raw LangTable 2D delta in the selected direction; ignored for trajectory_path."),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=2),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "generated_frames": 6,
                "prompt_horizon": 3,
                "maskgit_steps": 2,
                "direction": "right",
                "action_scale": 0.05,
                "fps": 2,
                "seed": 42,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The compatibility wrapper uses the official continuous checkpoint, two-dimensional actions, and the "
        "SVD temporal VAE without importing TensorFlow/OpenX dataset tooling.",
    ),
)

TESSERACT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="tesseract",
    display_name="TesserAct",
    default_variant_id="rgbdn-sft",
    default_task_id="robot-action-video",
    aliases=("tesser-act", "tesseract-rgbdn"),
    variants=(
        InferenceVariantSpec(
            variant_id="rgbdn-sft",
            label="RGB Depth Normal SFT",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="rgbdn_sft_transformer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="cogvideox_i2v_base",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "tesseract",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
            },
            aliases=("default", "official", "rgbdn"),
            notes=(
                "Uses the official TesserAct RGB-depth-normal SFT transformer with the local CogVideoX-5b-I2V components.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a robot rollout from RGB, depth, normal, and a task prompt.",
            aliases=("default", "rgbdn", "synthetic-geometry-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/droid/exterior_image_1_left.png"
                    ),
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="The robot moves its gripper toward the object on the table.",
                ),
                _field(
                    "checkpoint_dir",
                    "TesserAct RGBDN Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                ),
                _field(
                    "base_model_dir",
                    "CogVideoX I2V Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
                ),
                _field(
                    "geometry_mode",
                    "Geometry Conditioning",
                    target="call_kwargs",
                    default="synthetic-gradient",
                    choices=("synthetic-gradient", "files"),
                    description=(
                        "Synthetic-gradient is a deterministic smoke aid, not estimated geometry; use files for official inputs."
                    ),
                ),
                _field("depth_path", "Official Depth NPY", kind="path", target="call_kwargs"),
                _field("normal_path", "Official Normal Image", kind="path", target="call_kwargs"),
                _field("height", "Height", kind="integer", target="call_kwargs", default=256),
                _field("width", "Per-Modality Width", kind="integer", target="call_kwargs", default=320),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=49),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=4),
                _field("guidance_scale", "Text Guidance", kind="number", target="call_kwargs", default=7.5),
                _field("image_guidance_scale", "Image Guidance", kind="number", target="call_kwargs", default=1.5),
                _field("use_dynamic_cfg", "Dynamic CFG", kind="boolean", target="call_kwargs", default=False),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=8),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=23),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("memory_efficient", "CPU Offload", kind="boolean", target="call_kwargs", default=True),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "geometry_mode": "synthetic-gradient",
                "height": 256,
                "width": 320,
                "num_frames": 49,
                "num_inference_steps": 4,
                "guidance_scale": 7.5,
                "image_guidance_scale": 1.5,
                "use_dynamic_cfg": False,
                "fps": 8,
                "seed": 23,
                "mixed_precision": "bf16",
                "memory_efficient": True,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The generated video concatenates RGB, depth, and normal predictions horizontally, matching the official pipeline.",
        "Synthetic-gradient conditioning validates checkpoint execution only and must not be interpreted as official input quality.",
    ),
)

SHOTSTREAM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="shotstream",
    display_name="ShotStream",
    default_variant_id="wan2.1-1.3b-causal",
    default_task_id="causal-video",
    aliases=("shot-stream", "interactive-video-generation"),
    variants=(
        InferenceVariantSpec(
            variant_id="wan2.1-1.3b-causal",
            label="Wan2.1 1.3B Causal",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="shotstream_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="wan_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "shotstream",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                "wan_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
            },
            aliases=("default", "official", "causal"),
            notes=(
                "Runs the official four-step causal generator from a checkout staged under ${WORLDFOUNDRY_MODEL_SOURCE_DIR}/ShotStream.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="causal-video",
            label="Causal Video",
            description="Generate one 81-frame shot from a prompt, or run an official multi-shot CSV fixture.",
            aliases=("default", "official-demo", "multi-shot"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=(
                        "A cinematic science-fiction scene in a bioluminescent forest at night. An astronaut in "
                        "a white exploration suit walks between glowing blue plants."
                    ),
                ),
                _field(
                    "checkpoint_dir",
                    "ShotStream Checkpoint Dir",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                ),
                _field(
                    "wan_model_dir",
                    "Wan Base Model Dir",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
                ),
                _field(
                    "data_path",
                    "Official Input CSV",
                    kind="path",
                    target="call_kwargs",
                    description="Optional official ShotStream CSV. Omit it to synthesize a one-shot fixture from the prompt.",
                ),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={"seed": 42, "fps": 16, "plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "The compatibility launcher uses per-run checkpoint symlinks and handles FlashAttention/PyAV API drift without editing upstream source.",
    ),
)

STARWM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="starwm",
    display_name="StarWM",
    default_variant_id="offline-client",
    default_task_id="starcraft-offline-client",
    aliases=("star-wm", "starcraft-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="offline-client",
            label="Offline Client",
            status="requires_endpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="model_dir",
                    uri=str(_WORKSPACE_ROOT / "ckpt" / "StarWM"),
                    status="local_confirmed",
                ),
            ),
            load_kwargs={
                "model_id": "starwm",
                "input_file": str(_RUNTIME_CONFIGS_ROOT / "starwm" / "data" / "wm_test_horizon5_1traj.json"),
                "mode": "nothink",
                "api_base": "http://localhost:12000",
                "api_key": "sk-11223344",
                "served_model_id": "StarWM-demo",
                "max_workers": 8,
            },
            aliases=("default", "official-demo", "nothink"),
            notes=("Uses the official offline inference client against an OpenAI-compatible StarWM endpoint.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="starcraft-offline-client",
            label="StarCraft Offline Client",
            description="Run or preflight the official StarWM offline client prompt fixture.",
            aliases=("default", "official-demo", "starcraft"),
            inputs=(
                _field(
                    "input_file",
                    "Input File",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=str(_RUNTIME_CONFIGS_ROOT / "starwm" / "data" / "wm_test_horizon5_1traj.json"),
                ),
                _field("mode", "Mode", target="load_kwargs", default="nothink", choices=("nothink", "think")),
                _field("api_base", "API Base", target="load_kwargs", default="http://localhost:12000"),
                _field("api_key", "API Key", target="load_kwargs", default="sk-11223344"),
                _field(
                    "served_model_id", "Served Model ID", target="load_kwargs", required=True, default="StarWM-demo"
                ),
                _field("max_workers", "Max Workers", kind="integer", target="load_kwargs", default=8),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("predictions", "jsonl"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=("Set call_kwargs.plan_only=false only after a StarWM/vLLM OpenAI-compatible endpoint is running.",),
)

ROLLING_FORCING_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="rolling-forcing",
    display_name="RollingForcing",
    default_variant_id="dmd-ema",
    default_task_id="text-to-long-video",
    aliases=("rolling_forcing", "rollingforcing", "rolling"),
    variants=(
        InferenceVariantSpec(
            variant_id="dmd-ema",
            label="Official DMD EMA",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=ROLLING_FORCING_CHECKPOINT,
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="base",
                    uri=ROLLING_FORCING_WAN_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_id": "rolling-forcing",
                "checkpoint_path": ROLLING_FORCING_CHECKPOINT,
                "wan_models_root": str(Path(ROLLING_FORCING_WAN_CHECKPOINT).parent),
            },
            call_kwargs={
                "num_frames": 126,
                "seed": 0,
                "num_samples": 1,
                "use_ema": True,
                "save_with_index": True,
            },
            aliases=("default", "official", "ema"),
            notes=(
                "Uses the official five-step DMD checkpoint and three-latent-frame rolling block.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-long-video",
            label="Text to Long Video",
            description="Generate a long text-conditioned video with the official rolling-window schedule.",
            aliases=("default", "text-to-video", "t2v", "long-video"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=ROLLING_FORCING_DEMO_PROMPT,
                ),
                _field(
                    "frames",
                    "Latent Frames",
                    kind="integer",
                    target="params",
                    default=126,
                    description="Must be a positive multiple of the official three-frame block size.",
                ),
                _field("seed", "Seed", kind="integer", target="params", default=0),
                _field("num_samples", "Samples", kind="integer", target="call_kwargs", default=1),
                _field("use_ema", "Use EMA", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "save_with_index",
                    "Indexed Filenames",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field(
                    "report_timing",
                    "Report CUDA Timing",
                    kind="boolean",
                    target="call_kwargs",
                    default=False,
                ),
                _field("extended_prompt", "Extended Prompt", target="call_kwargs"),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "num_frames": 126,
                "seed": 0,
                "num_samples": 1,
                "use_ema": True,
                "save_with_index": True,
            },
        ),
    ),
    notes=(
        "Academic use only; the upstream license prohibits commercial and production use.",
        "The released checkpoint supports text-to-video only.",
    ),
)


UNI3C_DEMO_PROMPT = (
    "A close-up of a wet white cat wearing oversized amber sunglasses in a sunlit swimming pool, "
    "while the camera smoothly pushes forward toward the cat."
)
UNI3C_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="uni3c",
    display_name="Uni3C",
    default_variant_id="camera",
    default_task_id="image-camera-control-video",
    aliases=("pcdcontroller", "alibaba-damo-academy/Uni3C"),
    variants=(
        InferenceVariantSpec(
            variant_id="camera",
            label="PCDController Camera",
            status="input-assets-required",
            checkpoints=(
                InferenceCheckpointRef(role="controller", uri="ewrfcas/Uni3C", status="official"),
                InferenceCheckpointRef(
                    role="base",
                    uri="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                    status="official",
                ),
            ),
            call_kwargs={
                "mode": "camera",
                "render_path": UNI3C_DEMO_RENDER_BUNDLE,
                "num_frames": 9,
                "max_area": 114688,
                "fps": 16,
                "seed": 0,
                "num_gpus": 1,
            },
            aliases=("default", "pcd", "pcd-controller"),
            notes=("Requires the official stage-one pre-rendered camera-control bundle.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="image-camera-control-video",
            label="Image + Camera Control Video",
            description=(
                "Run the official Uni3C stage-two PCDController path from a reference image and "
                "a pre-rendered camera-control bundle."
            ),
            aliases=("default", "camera", "image-to-video"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=UNI3C_DEMO_PROMPT,
                ),
                _field(
                    "input_path",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=UNI3C_DEMO_REFERENCE_IMAGE,
                    description="Reference image used to produce the stage-one render bundle.",
                ),
                _field(
                    "render_path",
                    "Stage-One Render Bundle",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=UNI3C_DEMO_RENDER_BUNDLE,
                    description="Directory containing render.mp4, render_mask.mp4, and cam_info.json.",
                ),
                _field("mode", "Mode", target="call_kwargs", default="camera", choices=("camera",)),
                _field("num_frames", "Frames", kind="integer", target="params", default=9),
                _field("fps", "FPS", kind="integer", target="params", default=16),
                _field("max_area", "Maximum Area", kind="integer", target="call_kwargs", default=114688),
                _field("seed", "Seed", kind="integer", target="params", default=0),
                _field("num_gpus", "GPUs", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "mode": "camera",
                "render_path": UNI3C_DEMO_RENDER_BUNDLE,
                "num_frames": 9,
                "max_area": 114688,
                "fps": 16,
                "seed": 0,
                "num_gpus": 1,
            },
        ),
    ),
    notes=(
        "Uni3C stage two does not synthesize its own camera controls; a matching stage-one render bundle is required.",
    ),
)

def _ltx_ic_lora_v2v_inference_spec(
    model_id: str,
    base_repo: str,
    adapter_repo: str,
    control_types: tuple[str, ...],
) -> ModelInferenceSpec:
    """Describe the typed control video required by the official IC-LoRA route."""

    control_label = "Pose" if control_types == ("pose",) else "Canny, Depth, or Pose"
    return ModelInferenceSpec(
        model_family_id=model_id,
        display_name=model_id.replace("ltx-", "LTX-").replace("-v2v", " V2V"),
        default_variant_id=model_id,
        default_task_id="control-video-to-video",
        variants=(
            InferenceVariantSpec(
                variant_id=model_id,
                label="Official IC-LoRA control",
                status="configured",
                checkpoints=(
                    InferenceCheckpointRef(role="distilled_base", uri=base_repo, status="official"),
                    InferenceCheckpointRef(role="control_ic_lora", uri=adapter_repo, status="official"),
                ),
            ),
        ),
        tasks=(
            InferenceTaskProfile(
                task_id="control-video-to-video",
                label="Control Video to Video",
                description=f"Generate video from a {control_label.lower()} control video using official IC-LoRA.",
                inputs=(
                    _field("prompt", "Prompt", target="prompt", required=True),
                    _field(
                        "input_path", "Control Video", kind="path", target="input_path", required=True,
                        description=f"A {control_label.lower()} control video, not ordinary RGB footage.",
                    ),
                    _field(
                        "control_type", "Control Type", required=True, choices=control_types,
                        description="Type of control encoded in the supplied video.",
                    ),
                    _field("num_frames", "Frames", kind="integer", default=121),
                    _field("fps", "FPS", kind="integer", default=24),
                    _field("height", "Height", kind="integer", default=512),
                    _field("width", "Width", kind="integer", default=768),
                    _field("seed", "Seed", kind="integer", default=171198),
                    _field("control_strength", "Control Strength", kind="number", default=1.0),
                    _field("lora_strength", "LoRA Strength", kind="number", default=1.0),
                    _field("offload", "Offload", choices=("none", "cpu", "disk"), default="cpu"),
                    _field("skip_stage_2", "Skip Stage 2", kind="boolean", default=False),
                    _field("checkpoint_path", "Distilled Checkpoint", kind="path", target="load_kwargs"),
                    _field("spatial_upsampler_path", "Spatial Upsampler", kind="path", target="load_kwargs"),
                    _field("gemma_root", "Gemma Root", kind="path", target="load_kwargs"),
                    _field("ic_lora_path", "Control IC-LoRA", kind="path", target="load_kwargs"),
                    _field("python_executable", "Official Runtime Python", kind="path", target="load_kwargs"),
                ),
                outputs=(
                    _artifact("video", "video", required=True, preview=True),
                    _artifact("manifest", "manifest", required=True),
                ),
            ),
        ),
        notes=("An explicit control type and matching versioned IC-LoRA are required; RGB source video is not a control video.",),
    )


LTX2_V2V_INFERENCE_SPEC = _ltx_ic_lora_v2v_inference_spec(
    "ltx-2-v2v", "Lightricks/LTX-2", "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", ("pose",),
)
LTX23_V2V_INFERENCE_SPEC = _ltx_ic_lora_v2v_inference_spec(
    "ltx-2.3-v2v", "Lightricks/LTX-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    ("canny", "depth", "pose"),
)


_MODEL_INFERENCE_SPECS: dict[str, ModelInferenceSpec] = {
    "hyperflow": HYPERFLOW_INFERENCE_SPEC,
    RT1_INFERENCE_SPEC.model_family_id: RT1_INFERENCE_SPEC,
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
    EGOWM_INFERENCE_SPEC.model_family_id: EGOWM_INFERENCE_SPEC,
    HMA_INFERENCE_SPEC.model_family_id: HMA_INFERENCE_SPEC,
    TESSERACT_INFERENCE_SPEC.model_family_id: TESSERACT_INFERENCE_SPEC,
    LEWORLD_MODEL_INFERENCE_SPEC.model_family_id: LEWORLD_MODEL_INFERENCE_SPEC,
    SHOTSTREAM_INFERENCE_SPEC.model_family_id: SHOTSTREAM_INFERENCE_SPEC,
    LINGBOT_WORLD_INFERENCE_SPEC.model_family_id: LINGBOT_WORLD_INFERENCE_SPEC,
    LINGBOT_WORLD_V2_INFERENCE_SPEC.model_family_id: LINGBOT_WORLD_V2_INFERENCE_SPEC,
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
    HUNYUAN_WORLD_VOYAGER_INFERENCE_SPEC.model_family_id: HUNYUAN_WORLD_VOYAGER_INFERENCE_SPEC,
    HY_WORLDPLAY_INFERENCE_SPEC.model_family_id: HY_WORLDPLAY_INFERENCE_SPEC,
    CAMERACTRL_INFERENCE_SPEC.model_family_id: CAMERACTRL_INFERENCE_SPEC,
    DEPTH_ANYTHING3_INFERENCE_SPEC.model_family_id: DEPTH_ANYTHING3_INFERENCE_SPEC,
    WAN2P2_INFERENCE_SPEC.model_family_id: WAN2P2_INFERENCE_SPEC,
    CUT3R_INFERENCE_SPEC.model_family_id: CUT3R_INFERENCE_SPEC,
    WORLDFM_INFERENCE_SPEC.model_family_id: WORLDFM_INFERENCE_SPEC,
    WOW_INFERENCE_SPEC.model_family_id: WOW_INFERENCE_SPEC,
}


# ---------------------------------------------------------------------------
# Spec registry queries
# ---------------------------------------------------------------------------


def list_model_inference_specs() -> tuple[ModelInferenceSpec, ...]:
    """Return curated model inference specs with explicit variants and task profiles."""

    return tuple(_MODEL_INFERENCE_SPECS.values())


def get_model_inference_spec(model_family_id: str) -> ModelInferenceSpec | None:
    """Return a curated inference spec for a model family when one exists."""

    key = _normalise_infer_id(model_family_id)
    for spec in _MODEL_INFERENCE_SPECS.values():
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
        task_inputs = tuple(fields) or (streaming_fallback_fields() if supports_stream else tuple())
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
        task_inputs = tuple(fields) or (streaming_fallback_fields() if supports_stream else tuple())
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
