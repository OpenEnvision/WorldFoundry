# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``rolling_forcing`` model family."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field, _local_hfd_checkpoint_path

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
            notes=("Uses the official five-step DMD checkpoint and three-latent-frame rolling block.",),
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
                _field("num_samples", "Samples", kind="integer", target="call_kwargs"),
                _field("use_ema", "Use EMA", kind="boolean", target="call_kwargs"),
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
                "save_with_index": True,
            },
        ),
    ),
    notes=(
        "Academic use only; the upstream license prohibits commercial and production use.",
        "The released checkpoint supports text-to-video only.",
    ),
)
