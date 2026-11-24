# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``ltx_v2v`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field


def _ltx_ic_lora_v2v_inference_spec(
    model_id: str,
    base_repo: str,
    adapter_repo: str,
    control_types: tuple[str, ...],
) -> ModelInferenceSpec:
    """Describe the typed control video required by the native IC-LoRA route."""

    control_label = "Pose" if control_types == ("pose",) else "Canny, Depth, or Pose"
    return ModelInferenceSpec(
        model_family_id=model_id,
        display_name=model_id.replace("ltx-", "LTX-").replace("-v2v", " V2V"),
        default_variant_id=model_id,
        default_task_id="control-video-to-video",
        variants=(
            InferenceVariantSpec(
                variant_id=model_id,
                label="Native IC-LoRA control",
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
                description=f"Generate video from a {control_label.lower()} control video using native IC-LoRA.",
                inputs=(
                    _field("prompt", "Prompt", target="prompt", required=True),
                    _field(
                        "input_path",
                        "Control Video",
                        kind="path",
                        target="input_path",
                        required=True,
                        description=f"A {control_label.lower()} control video, not ordinary RGB footage.",
                    ),
                    _field(
                        "control_type",
                        "Control Type",
                        required=True,
                        choices=control_types,
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
                ),
                outputs=(
                    _artifact("video", "video", required=True, preview=True),
                    _artifact("manifest", "manifest", required=True),
                ),
            ),
        ),
        notes=(
            "An explicit control type and matching versioned IC-LoRA are required; RGB source video is not a control video.",
        ),
    )


LTX2_V2V_INFERENCE_SPEC = _ltx_ic_lora_v2v_inference_spec(
    "ltx-2-v2v",
    "Lightricks/LTX-2",
    "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
    ("pose",),
)


LTX23_V2V_INFERENCE_SPEC = _ltx_ic_lora_v2v_inference_spec(
    "ltx-2.3-v2v",
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    ("canny", "depth", "pose"),
)
