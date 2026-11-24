# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared pi-H3 input and variant contracts for CLI, TUI, and Studio."""

from __future__ import annotations

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.runtime.interactive_inference_catalog import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

_BASE = str(hfd_root_path("multimodalart--MiniMax-H3-Pruned"))
_ADAPTERS = str(hfd_root_path("Lakonik--pi-H3"))

_COMMON = (
    InferenceFieldSpec("prompt", "Prompt", target="prompt", required=True),
    InferenceFieldSpec("num_frames", "Frames", kind="integer"),
    InferenceFieldSpec("short_edge", "Short edge", kind="integer", default=768),
    InferenceFieldSpec("aspect_ratio", "Aspect ratio", default="16:9"),
    InferenceFieldSpec("seed", "Seed", kind="integer", default=42),
    InferenceFieldSpec("variant", "Variant", target="load_kwargs", default="4step-dwd"),
    InferenceFieldSpec("base_model_path", "MiniMax-H3-Pruned base", kind="path", target="load_kwargs", default=_BASE),
)

_OUTPUTS = (
    InferenceArtifactSpec(
        "video",
        "video",
        required=True,
        preview=True,
        description="24 fps video with generated 32 kHz stereo audio.",
    ),
)


def _task(task_id: str, label: str, extra: tuple[InferenceFieldSpec, ...] = ()) -> InferenceTaskProfile:
    return InferenceTaskProfile(
        task_id,
        label,
        (*_COMMON, *extra),
        _OUTPUTS,
        default_call_kwargs={"task": task_id, "return_dict": True},
    )


def _variant(
    variant_id: str,
    label: str,
    *,
    variant_key: str,
    subfolder: str,
    aliases: tuple[str, ...] = (),
) -> InferenceVariantSpec:
    return InferenceVariantSpec(
        variant_id,
        label,
        checkpoints=(
            InferenceCheckpointRef("base_model", _BASE),
            InferenceCheckpointRef("adapter", f"{_ADAPTERS}/{subfolder}"),
        ),
        load_kwargs={
            "base_model_path": _BASE,
            "adapter_path": f"{_ADAPTERS}/{subfolder}",
            "variant": variant_key,
        },
        aliases=aliases,
    )


PI_H3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="pi-h3",
    display_name="pi-H3",
    variants=(
        _variant(
            "pi-h3-4step-dwd",
            "pi-H3 4-step DWD (no video VAE decoder)",
            variant_key="4step-dwd",
            subfolder="pi_h3_4step_dwd_adapter",
            aliases=("pi-h3", "pi_h3", "pih3", "pi-h3-dwd", "minimax-h3-pi"),
        ),
        _variant(
            "pi-h3-2step-dwd",
            "pi-H3 2-step DWD (no video VAE decoder)",
            variant_key="2step-dwd",
            subfolder="pi_h3_2step_dwd_adapter",
            aliases=("pih3-2step-dwd",),
        ),
        _variant(
            "pi-h3-4step",
            "pi-H3 4-step (video VAE decoder)",
            variant_key="4step",
            subfolder="pi_h3_4step_adapter",
            aliases=("pih3-4step",),
        ),
        _variant(
            "pi-h3-2step",
            "pi-H3 2-step (video VAE decoder)",
            variant_key="2step",
            subfolder="pi_h3_2step_adapter",
            aliases=("pih3-2step",),
        ),
    ),
    tasks=(
        _task("t2va", "Text to video + audio"),
        _task("i2va", "First frame to video + audio", (InferenceFieldSpec("images", "First image", kind="path"),)),
        _task("l2va", "Last frame to video + audio", (InferenceFieldSpec("images", "Last image", kind="path"),)),
        _task(
            "fl2va",
            "First and last frame to video + audio",
            (InferenceFieldSpec("images", "Endpoint images", kind="json"),),
        ),
    ),
    default_variant_id="pi-h3-4step-dwd",
    default_task_id="t2va",
    aliases=("pi_h3", "pih3", "pi-h3-dwd", "minimax-h3-pi"),
)

__all__ = ["PI_H3_INFERENCE_SPEC"]
