# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``matrix_game_2`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field, _first_existing_path

MATRIX_GAME_2_IN_TREE_ROOT = _TEST_CASES_ROOT / "matrix-game-2"


MATRIX_GAME_2_OFFICIAL_FIXTURE = _first_existing_path(
    MATRIX_GAME_2_IN_TREE_ROOT / "universal" / "0000.png",
)


MATRIX_GAME_2_OFFICIAL_CONFIG = _first_existing_path(
    MATRIX_GAME_2_IN_TREE_ROOT / "configs" / "inference_universal.yaml",
)


MATRIX_GAME_2_OFFICIAL_INTERACTIONS = ()


MATRIX_GAME_2_OFFICIAL_CALL_KWARGS = {
    "num_frames": 150,
    "size": (352, 640),
    "fps": 12,
    "seed": 42,
    "official_bench_actions": True,
    "visualize_ops": False,
    "visualize_warning": False,
}


MATRIX_GAME_2_OFFICIAL_LOAD_KWARGS = {"mode": "universal"}


MATRIX_GAME_2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="matrix-game-2",
    display_name="Matrix-Game-2",
    default_variant_id="universal",
    default_task_id="official-universal-image",
    aliases=("matrixgame", "matrixgame2", "matrix-game2", "Skywork/Matrix-Game-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="universal",
            label="Universal",
            status="configured",
            load_kwargs=MATRIX_GAME_2_OFFICIAL_LOAD_KWARGS,
            call_kwargs=MATRIX_GAME_2_OFFICIAL_CALL_KWARGS,
            aliases=("default", "official", "demo"),
            notes=("Matches the upstream Matrix-Game-2 inference_universal demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-universal-image",
            label="Official Universal Image",
            description="Run the Matrix-Game-2 universal demo using the upstream demo image and config defaults.",
            aliases=("default", "official-demo", "interactive-video", "navigation-video"),
            inputs=(
                _field(
                    "image",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=MATRIX_GAME_2_OFFICIAL_FIXTURE,
                ),
                _field(
                    "mode",
                    "Mode",
                    target="load_kwargs",
                    default="universal",
                    choices=("universal", "gta_drive", "templerun"),
                ),
                _field(
                    "config_path",
                    "Config Path",
                    kind="path",
                    target="call_kwargs",
                    default=MATRIX_GAME_2_OFFICIAL_CONFIG,
                ),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=MATRIX_GAME_2_OFFICIAL_INTERACTIONS,
                ),
                _field("frames", "Frames", kind="integer", target="params", default=150),
                _field("fps", "FPS", kind="integer", target="params", default=12),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("size", "Size", kind="json", target="call_kwargs", default=(352, 640)),
                _field(
                    "official_bench_actions",
                    "Official Bench Actions",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                    description="Use Matrix-Game-2's official Bench_actions_* trajectory generator instead of the manual action-token list.",
                ),
                _field("visualize_ops", "Visualize Actions", kind="boolean", target="call_kwargs", default=False),
                _field("visualize_warning", "Visualize Warning", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=MATRIX_GAME_2_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image/config path uses packaged in-tree Matrix-Game fixtures.",),
)
