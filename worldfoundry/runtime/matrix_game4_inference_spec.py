# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free public task contract for Matrix-Game 4 Ancient Egypt."""

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.matrix_game4 import MATRIX_GAME4_EGYPT_PROMPT
from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

MATRIX_GAME4_EGYPT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="matrix-game-4-egypt",
    display_name="Matrix-Game 4 Ancient Egypt",
    default_variant_id="matrix-game-4-egypt",
    default_task_id="low-poly-rendering",
    variants=(
        InferenceVariantSpec(
            variant_id="matrix-game-4-egypt",
            label="Ancient Egypt · Wan VAE · two steps",
            aliases=("matrix-game-4-ancient-egypt",),
            status="implemented",
            checkpoints=(InferenceCheckpointRef("dit", "Skywork/Matrix-Game-4.0:fps-egypt/model.safetensors"),),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="low-poly-rendering",
            label="Offline low-poly video rendering",
            inputs=(
                InferenceFieldSpec("input_path", "Low-poly RGB video", kind="path", target="input_path", required=True),
                InferenceFieldSpec(
                    "prompt",
                    "Fixed scene prompt",
                    target="prompt",
                    default=MATRIX_GAME4_EGYPT_PROMPT,
                    description="Empty uses the released Egypt prompt; one prompt per clip.",
                ),
                InferenceFieldSpec(
                    "fps",
                    "Playback FPS override",
                    kind="number",
                    description="Omit to preserve source FPS; tensors default to 20 FPS.",
                ),
                InferenceFieldSpec("seed", "Seed", kind="integer", default=42),
            ),
            outputs=(InferenceArtifactSpec("matrix-game-4-egypt.mp4", "generated_video", required=True, preview=True),),
            default_call_kwargs={
                "height": 480,
                "width": 864,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "seed": 42,
                "return_dict": True,
            },
            description="One RGB frame then four-frame causal chunks; tail padding is removed from output.",
        ),
    ),
    notes=(
        "The game engine is optional. Accepted-reference release replay, gameplay quality and throughput remain pending.",
    ),
)
