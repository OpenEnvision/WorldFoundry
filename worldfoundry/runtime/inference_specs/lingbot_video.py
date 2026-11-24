# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``lingbot_video`` model family."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _WORKSPACE_ROOT, _artifact, _field, _first_existing_path

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
