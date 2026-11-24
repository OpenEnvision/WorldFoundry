# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``lingbot_world`` model family."""

from __future__ import annotations

from dataclasses import replace

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _TEST_CASES_ROOT, _WORKSPACE_ROOT, _artifact, _field, _first_existing_path

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


LINGBOT_WORLD_DEMO_ROOT = _TEST_CASES_ROOT / "lingbot_world" / "00"


LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"


LINGBOT_WORLD_V2_CHECKPOINT = _first_existing_path(
    _WORKSPACE_ROOT / "ckpt" / "lingbot-world-v2-14b-causal-fast",
    _WORKSPACE_ROOT / "ckpt" / "hfd" / "robbyant--lingbot-world-v2-14b-causal-fast",
    "robbyant/lingbot-world-v2-14b-causal-fast",
)


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
                    choices=("480*832", "832*480", "720*1280", "1280*720"),
                ),
                _field("frame_num", "Frames", kind="integer", target="call_kwargs"),
                _field("chunk_size", "Chunk Size", kind="integer", target="call_kwargs"),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("sample_shift", "Sample Shift", kind="number", target="call_kwargs", default=10.0),
                _field("local_attn_size", "Local Attention", kind="integer", target="call_kwargs"),
                _field("sink_size", "Attention Sink", kind="integer", target="call_kwargs"),
                _field("nproc_per_node", "GPU Processes", kind="integer", target="call_kwargs"),
                _field("t5_fsdp", "T5 FSDP", kind="boolean", target="call_kwargs"),
                _field("dit_fsdp", "DiT FSDP", kind="boolean", target="call_kwargs"),
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
                "seed": 42,
                "sample_shift": 10.0,
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


_LINGBOT_V2_SMALL_DEFAULTS = {
    **LINGBOT_WORLD_V2_INFERENCE_SPEC.tasks[0].default_call_kwargs,
    "frame_num": 317,
    "sample_shift": 5.0,
    "nproc_per_node": 1,
    "t5_fsdp": False,
    "dit_fsdp": False,
}


LINGBOT_WORLD_V2_SMALL_INFERENCE_SPEC = replace(
    LINGBOT_WORLD_V2_INFERENCE_SPEC,
    model_family_id="lingbot-world-v2-1.3b",
    display_name="LingBot-World-V2 1.3B",
    default_variant_id="causal-fast-1.3b",
    aliases=("lingbot-v2-1.3b",),
    variants=(
        InferenceVariantSpec(
            variant_id="causal-fast-1.3b",
            label="Causal Fast 1.3B",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary", uri="robbyant/lingbot-world-v2-1.3b-causal-fast", status="configured"
                ),
            ),
        ),
    ),
    tasks=(
        replace(
            LINGBOT_WORLD_V2_INFERENCE_SPEC.tasks[0],
            inputs=tuple(
                replace(field, default=_LINGBOT_V2_SMALL_DEFAULTS.get(field.field_id, field.default))
                for field in LINGBOT_WORLD_V2_INFERENCE_SPEC.tasks[0].inputs
            ),
            default_call_kwargs=_LINGBOT_V2_SMALL_DEFAULTS,
        ),
    ),
    notes=("1.3B, four-latent causal-fast runtime with shared Wan T5/tokenizer/VAE assets.",),
)
