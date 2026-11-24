# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for FastFrames-S (1.3B) and FastFrames-L (14B)."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field, _local_hfd_checkpoint_path

FASTFRAMES_S_REPO = _local_hfd_checkpoint_path("apple--FastFrames-S")
FASTFRAMES_L_REPO = _local_hfd_checkpoint_path("apple--FastFrames-L")
FASTFRAMES_S_DIT = _local_hfd_checkpoint_path(
    "zhuhz22--Causal-Forcing",
    "causal-forcing++",
    "framewise-2step.pt",
)
FASTFRAMES_WAN_1_3B = _local_hfd_checkpoint_path("Wan-AI--Wan2.1-T2V-1.3B")
FASTFRAMES_WAN_14B = _local_hfd_checkpoint_path("Wan-AI--Wan2.1-T2V-14B")

FASTFRAMES_DEMO_PROMPT = "A corgi surfing a wave at sunset"

FASTFRAMES_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="fastframes",
    display_name="FastFrames",
    default_variant_id="fastframes-s",
    default_task_id="text-to-video",
    aliases=("FastFrames", "fastframes-s", "fastframes-l", "ff-s", "ff-l", "fastframes_s", "fastframes_l"),
    variants=(
        InferenceVariantSpec(
            variant_id="fastframes-s",
            label="FastFrames-S (1.3B)",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="refiner",
                    uri=f"{FASTFRAMES_S_REPO}/fastframes-s/kv_refiner.safetensors",
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="decoder",
                    uri=f"{FASTFRAMES_S_REPO}/fastframes-s/pixel_decoder.safetensors",
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="dit",
                    uri=FASTFRAMES_S_DIT,
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="wan_base",
                    uri=FASTFRAMES_WAN_1_3B,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_id": "fastframes",
                "variant": "fastframes-s",
                "dit_path": FASTFRAMES_S_DIT,
                "refiner_path": f"{FASTFRAMES_S_REPO}/fastframes-s/kv_refiner.safetensors",
                "decoder_path": f"{FASTFRAMES_S_REPO}/fastframes-s/pixel_decoder.safetensors",
                "wan_dir": FASTFRAMES_WAN_1_3B,
            },
            call_kwargs={
                "context": "refiner",
                "seed": 0,
                "num_frames": 81,
                "height": 480,
                "width": 832,
                "fps": 16,
                "return_dict": True,
            },
            aliases=("default", "ff-s", "fastframes_s", "1.3b"),
            notes=(
                "Uses the frozen Causal Forcing++ 2-step 1.3B generator with FastFrames KV refiner and pixel decoder.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="fastframes-l",
            label="FastFrames-L (14B)",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="dit",
                    uri=f"{FASTFRAMES_L_REPO}/fastframes-l/dit.safetensors",
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="refiner",
                    uri=f"{FASTFRAMES_L_REPO}/fastframes-l/kv_refiner.safetensors",
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="decoder",
                    uri=f"{FASTFRAMES_L_REPO}/fastframes-l/pixel_decoder.safetensors",
                    status="local_required",
                ),
                InferenceCheckpointRef(
                    role="wan_base",
                    uri=FASTFRAMES_WAN_14B,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_id": "fastframes",
                "variant": "fastframes-l",
                "dit_path": f"{FASTFRAMES_L_REPO}/fastframes-l/dit.safetensors",
                "refiner_path": f"{FASTFRAMES_L_REPO}/fastframes-l/kv_refiner.safetensors",
                "decoder_path": f"{FASTFRAMES_L_REPO}/fastframes-l/pixel_decoder.safetensors",
                "wan_dir": FASTFRAMES_WAN_14B,
            },
            call_kwargs={
                "context": "refiner",
                "seed": 0,
                "num_frames": 81,
                "height": 480,
                "width": 832,
                "fps": 16,
                "return_dict": True,
            },
            aliases=("ff-l", "fastframes_l", "14b"),
            notes=(
                "Distilled 14B two-step causal Wan2.1 generator with fused FastOps, KV refiner, and bridged pixel decoder.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-video",
            label="Text to Video",
            description="Generate 832x480 interactive-latency video with two-step causal DiT, KV refinement, and direct pixel decoding.",
            aliases=("default", "t2v"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=FASTFRAMES_DEMO_PROMPT,
                ),
                _field(
                    "context",
                    "KV Context Mode",
                    target="call_kwargs",
                    default="refiner",
                    choices=("refiner", "dit"),
                    description="'refiner' uses the KV refinement network; 'dit' runs a clean t=0 DiT pass per frame.",
                ),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
                _field(
                    "num_frames",
                    "Pixel Frames",
                    kind="integer",
                    target="call_kwargs",
                    default=81,
                    description="Output pixel frames (4k+1; default 81 from 21 latent frames).",
                ),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field(
                    "noise",
                    "Transition Noise Mode",
                    target="call_kwargs",
                    choices=("sequential", "pregenerated", "reseeded"),
                    description="Optional override for transition noise sampling order.",
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "context": "refiner",
                "seed": 0,
                "num_frames": 81,
                "height": 480,
                "width": 832,
                "fps": 16,
                "return_dict": True,
            },
        ),
    ),
    notes=(
        "FastFrames code and weights are distributed under the Apple Sample Code License / Apple AMLR license; Wan2.1 derivatives remain Apache-2.0.",
        "Text-to-video only in the released FastFrames-S and FastFrames-L routes.",
    ),
)
