# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``matrix_game_3`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field, _first_existing_path

MATRIX_GAME_3_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "matrix-game-3" / "001" / "image.png"


MATRIX_GAME_3_OFFICIAL_FIXTURE = _first_existing_path(
    MATRIX_GAME_3_FALLBACK_FIXTURE,
)


MATRIX_GAME_3_OFFICIAL_PROMPT = "A colorful, animated cityscape with a gas station and various buildings."


MATRIX_GAME_3_DEFAULT_INTERACTIONS = ("forward", "camera_r")


MATRIX_GAME_3_OFFICIAL_CALL_KWARGS = {
    "num_iterations": 12,
    "num_inference_steps": 3,
    "size": "704*1280",
    "fps": 17,
    "seed": 42,
    "save_name": "test",
    "visualize_ops": True,
    "show_progress": True,
    "use_int8": True,
    "compile_vae": True,
    "lightvae_pruning_rate": 0.5,
    "vae_type": "mg_lightvae",
    "fa_version": "3",
    "use_async_vae": False,
    "async_vae_warmup_iters": 0,
}


MATRIX_GAME_3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="matrix-game-3",
    display_name="Matrix-Game-3",
    default_variant_id="official",
    default_task_id="official-cityscape-image",
    aliases=("matrixgame3", "matrix-game3", "Skywork/Matrix-Game-3.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="official",
            label="Official",
            status="configured",
            call_kwargs=MATRIX_GAME_3_OFFICIAL_CALL_KWARGS,
            aliases=("default", "demo"),
            notes=(
                "Uses the Matrix-Game-3 README demo sampling defaults.",
                "Async VAE is disabled by default because it failed device ordinal validation when GPUs are remapped.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-cityscape-image",
            label="Official Cityscape Image",
            description="Run the Matrix-Game-3 README image-conditioned cityscape demo.",
            aliases=("default", "official-demo", "interactive-video", "navigation-video"),
            inputs=(
                _field(
                    "image",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=MATRIX_GAME_3_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=MATRIX_GAME_3_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=MATRIX_GAME_3_DEFAULT_INTERACTIONS,
                ),
                _field("num_iterations", "Iterations", kind="integer", target="call_kwargs", default=12),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=3),
                _field("size", "Size", target="call_kwargs", default="704*1280", choices=("704*1280", "1280*704")),
                _field("fps", "FPS", kind="integer", target="params", default=17),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("save_name", "Save Name", target="call_kwargs", default="test"),
                _field("visualize_ops", "Visualize Actions", kind="boolean", target="call_kwargs", default=True),
                _field("show_progress", "Show Progress", kind="boolean", target="call_kwargs", default=True),
                _field("use_int8", "INT8", kind="boolean", target="call_kwargs", default=True),
                _field("compile_vae", "Compile VAE", kind="boolean", target="call_kwargs", default=True),
                _field("lightvae_pruning_rate", "LightVAE Pruning", kind="number", target="call_kwargs", default=0.5),
                _field(
                    "vae_type", "VAE Type", target="call_kwargs", default="mg_lightvae", choices=("mg_lightvae", "wan")
                ),
                _field("fa_version", "FlashAttention", target="call_kwargs", default="3", choices=("3", "2", "0")),
                _field("use_async_vae", "Async VAE", kind="boolean", target="call_kwargs", default=False),
                _field("async_vae_warmup_iters", "Async VAE Warmup", kind="integer", target="call_kwargs", default=0),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=MATRIX_GAME_3_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default Matrix-Game-3 demo uses the packaged in-tree image fixture.",),
)
