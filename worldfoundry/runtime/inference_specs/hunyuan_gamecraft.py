# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``hunyuan_gamecraft`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

HUNYUAN_GAMECRAFT_OFFICIAL_PROMPT = (
    "A charming medieval village with cobblestone streets, thatched-roof houses, "
    "and vibrant flower gardens under a bright blue sky."
)


HUNYUAN_GAMECRAFT_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_game_craft" / "village.png")


HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS = {
    "interactions": ("forward", "backward", "right", "left"),
    "interaction_speed": (0.2, 0.2, 0.2, 0.2),
    "size": (704, 1216),
    "num_frames": 129,
    "infer_steps": 50,
    "cfg_scale": 2.0,
}


HUNYUAN_GAMECRAFT_OFFICIAL_LOAD_KWARGS = {
    "seed": 250160,
}


HUNYUAN_GAMECRAFT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-game-craft",
    display_name="Hunyuan Game Craft",
    default_variant_id="default",
    default_task_id="official-interactive-video",
    aliases=("hunyuan-gamecraft", "gamecraft", "tencent/Hunyuan-GameCraft-1.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            load_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_LOAD_KWARGS,
            call_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "gamecraft"),
            notes=("Matches the upstream Hunyuan-GameCraft official village demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-interactive-video",
            label="Official Interactive Video",
            description="Run the Hunyuan-GameCraft official village demo with the default action sequence.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HUNYUAN_GAMECRAFT_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HUNYUAN_GAMECRAFT_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward", "backward", "right", "left"),
                ),
                _field("frames", "Frames", kind="integer", target="params", default=129),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1216),
                _field("steps", "Steps", kind="integer", target="params", default=50),
                _field("guidance_scale", "Guidance", kind="number", target="params", default=2.0),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=250160),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HUNYUAN_GAMECRAFT_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The checkpoint is resolved from the Studio catalog so local /ckpt installs are preferred.",),
)
