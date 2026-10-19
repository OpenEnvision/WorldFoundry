# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free CLI and Studio contract for explicit DreamDojo student assets."""

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path

DREAMDOJO_STUDENT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="dreamdojo-student",
    display_name="DreamDojo Student",
    default_variant_id="2b-gr1-distilled",
    default_task_id="robot-world-generation",
    variants=(
        InferenceVariantSpec(
            variant_id="2b-gr1-distilled",
            label="2B GR-1 self-forcing student",
            aliases=("dreamdojo-2b-gr1-distilled",),
            status="native_student_checkpoint_required",
            checkpoints=(
                InferenceCheckpointRef(
                    "student",
                    str(hfd_root_path("nvidia--DreamDojo", "student", "model_ema_bf16.pt")),
                    status="operator_supplied_required",
                ),
            ),
            notes=("The pinned DreamDojo Hub snapshot contains no distilled student weights.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-world-generation",
            label="Robot world prediction",
            inputs=(
                InferenceFieldSpec("input_path", "Initial RGB image", kind="path", target="input_path", required=True),
                InferenceFieldSpec("prompt", "Text conditioning", target="prompt", default=""),
                InferenceFieldSpec("action_file", "Normalized 384-D actions", kind="path", required=True),
                InferenceFieldSpec(
                    "student_checkpoint_path",
                    "Distilled student checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                ),
                InferenceFieldSpec(
                    "vae_path",
                    "Cosmos video tokenizer",
                    kind="path",
                    target="load_kwargs",
                    default=str(checkpoint_root_path("nvidia--Cosmos-Predict2.5-2B", "tokenizer.pth")),
                ),
                InferenceFieldSpec(
                    "text_encoder_path",
                    "Reason1 text encoder",
                    kind="path",
                    target="load_kwargs",
                    description="Use this or precomputed text_embeddings_path.",
                ),
                InferenceFieldSpec(
                    "text_embeddings_path",
                    "Precomputed Reason1 features",
                    kind="path",
                    target="load_kwargs",
                    description="Actual [1,S,100352] features; replaces online text loading.",
                ),
                InferenceFieldSpec("height", "Height", kind="integer", default=480),
                InferenceFieldSpec("width", "Width", kind="integer", default=640),
                InferenceFieldSpec(
                    "num_inference_steps",
                    "Student sampling steps",
                    kind="integer",
                    default=4,
                    description="Integer 1–4; fixed released TrigFlow schedule.",
                ),
                InferenceFieldSpec(
                    "guidance_scale", "Distilled guidance", kind="number", default=1.0, description="Must remain 1."
                ),
                InferenceFieldSpec(
                    "fps",
                    "Playback FPS",
                    kind="integer",
                    default=10,
                    description="Playback clock, not measured generation throughput.",
                ),
                InferenceFieldSpec("seed", "Seed", kind="integer", default=42),
            ),
            outputs=(InferenceArtifactSpec("dreamdojo_world.mp4", "generated_world", required=True, preview=True),),
            default_call_kwargs={
                "height": 480,
                "width": 640,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "fps": 10,
                "seed": 42,
                "return_dict": True,
            },
            description="Offline normalized-action rollout: 12 initial actions, then complete four-action chunks. Resident configure/STEP/reset is available through the pipeline API.",
        ),
    ),
    notes=("Student-weight GPU replay, official parity, and realtime inference throughput remain pending.",),
)
