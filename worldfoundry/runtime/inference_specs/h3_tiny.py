# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared H3-Tiny input contracts for CLI, TUI and Studio."""

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.runtime.interactive_inference_catalog import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

_BASE = str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
_FEWSTEP_17B = str(hfd_root_path("H3-Tiny--MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1"))
_PRUNED_17B_REF = str(hfd_root_path("H3-Tiny--MiniMax-H3-Tiny-17B-Pruned-Ref2VA"))
_PRUNED_17B_FL = str(hfd_root_path("H3-Tiny--MiniMax-H3-Tiny-17B-Pruned-FL2VA"))
_OKD_14B_REF = str(hfd_root_path("H3-Tiny--MiniMax-H3-Tiny-14B-OKD-Ref2VA-v0.1"))

_COMMON = (
    InferenceFieldSpec("prompt", "Prompt", target="prompt", required=True),
    InferenceFieldSpec(
        "num_frames",
        "Frames",
        kind="integer",
        default=124,
        description="Aligned to 17n+5 frames at 24 fps (124 ≈ 5.2 s, 243 ≈ 10.1 s).",
    ),
    InferenceFieldSpec(
        "num_inference_steps",
        "Steps",
        kind="integer",
        description="Defaults to 6 for MeanFlow few-step checkpoints (>=4) and 50 for pruned/OKD checkpoints.",
    ),
    InferenceFieldSpec("height", "Height", kind="integer", description="Optional positive multiple of 32."),
    InferenceFieldSpec("width", "Width", kind="integer", description="Optional positive multiple of 32."),
    InferenceFieldSpec("short_edge", "Short edge", kind="integer", default=768),
    InferenceFieldSpec("aspect_ratio", "Aspect ratio", default="16:9"),
    InferenceFieldSpec("seed", "Seed", kind="integer", default=1234),
    InferenceFieldSpec("weights_path", "H3-Tiny weights", kind="path", target="load_kwargs", default=_FEWSTEP_17B),
    InferenceFieldSpec("base_model_path", "MiniMax-H3 base", kind="path", target="load_kwargs", default=_BASE),
)

_OUTPUTS = (
    InferenceArtifactSpec(
        "video",
        "video",
        required=True,
        preview=True,
        description="24 fps video with synchronized 32 kHz stereo audio.",
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


H3_TINY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="h3-tiny",
    display_name="H3-Tiny",
    variants=(
        InferenceVariantSpec(
            "17b-fewstep-ref2va",
            "H3-Tiny 17B FewStep Ref2VA v0.1 (MeanFlow 6-step)",
            checkpoints=(
                InferenceCheckpointRef("weights", _FEWSTEP_17B),
                InferenceCheckpointRef("base_model", _BASE),
            ),
            load_kwargs={
                "variant": "17b-fewstep-ref2va",
                "workflow": "ref2va",
                "weights_path": _FEWSTEP_17B,
                "base_model_path": _BASE,
            },
            aliases=("h3-tiny", "minimax-h3-tiny", "h3-tiny-17b-fewstep"),
        ),
        InferenceVariantSpec(
            "17b-pruned-ref2va",
            "H3-Tiny 17B Pruned Ref2VA (50-step)",
            checkpoints=(
                InferenceCheckpointRef("weights", _PRUNED_17B_REF),
                InferenceCheckpointRef("base_model", _BASE),
            ),
            load_kwargs={
                "variant": "17b-pruned-ref2va",
                "workflow": "ref2va",
                "weights_path": _PRUNED_17B_REF,
                "base_model_path": _BASE,
            },
            aliases=("h3-tiny-17b-pruned",),
        ),
        InferenceVariantSpec(
            "17b-pruned-fl2va",
            "H3-Tiny 17B Pruned FL2VA / T2VA (50-step)",
            checkpoints=(
                InferenceCheckpointRef("weights", _PRUNED_17B_FL),
                InferenceCheckpointRef("base_model", _BASE),
            ),
            load_kwargs={
                "variant": "17b-pruned-fl2va",
                "workflow": "fl2va",
                "weights_path": _PRUNED_17B_FL,
                "base_model_path": _BASE,
            },
            aliases=("h3-tiny-17b-fl2va",),
        ),
        InferenceVariantSpec(
            "14b-okd-ref2va",
            "H3-Tiny 14B OKD Ref2VA v0.1 (50-step)",
            checkpoints=(
                InferenceCheckpointRef("weights", _OKD_14B_REF),
                InferenceCheckpointRef("base_model", _BASE),
            ),
            load_kwargs={
                "variant": "14b-okd-ref2va",
                "workflow": "ref2va",
                "weights_path": _OKD_14B_REF,
                "base_model_path": _BASE,
            },
            aliases=("h3-tiny-14b-okd",),
        ),
    ),
    tasks=(
        _task(
            "ref2va",
            "Image references to video + audio",
            (InferenceFieldSpec("references", "Ordered image references", kind="json", required=True),),
        ),
        _task("t2va", "Text to video + audio"),
        _task("i2va", "First frame to video + audio", (InferenceFieldSpec("images", "First image", kind="path"),)),
        _task("l2va", "Last frame to video + audio", (InferenceFieldSpec("images", "Last image", kind="path"),)),
        _task(
            "fl2va",
            "First and last frame to video + audio",
            (InferenceFieldSpec("images", "Endpoint images", kind="json"),),
        ),
    ),
    default_variant_id="17b-fewstep-ref2va",
    default_task_id="ref2va",
    aliases=("minimax-h3-tiny", "h3-tiny-17b-fewstep", "h3-tiny-17b-pruned", "h3-tiny-17b-fl2va", "h3-tiny-14b-okd"),
)

__all__ = ["H3_TINY_INFERENCE_SPEC"]
