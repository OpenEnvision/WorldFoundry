# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``sana_streaming`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field, _local_hfd_checkpoint_path

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


def _sana_streaming_inference_spec(*, bidirectional: bool) -> ModelInferenceSpec:
    """Build the explicit Workspace contract for one SANA-Streaming route."""

    model_id = "sana-streaming-bidirectional-2b-720p" if bidirectional else "sana-streaming-2b-720p"
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
                    else "Uses state-cached autoregressive attention with sink-token caching.",
                ),
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
            "the official 969-frame default when called without an override.",
        ),
    )


SANA_STREAMING_INFERENCE_SPEC = _sana_streaming_inference_spec(bidirectional=False)


SANA_STREAMING_BIDIRECTIONAL_INFERENCE_SPEC = _sana_streaming_inference_spec(bidirectional=True)
