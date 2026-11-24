# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``worldfm`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

WORLDFM_OFFICIAL_META = str(_TEST_CASES_ROOT / "worldfm" / "meta.json")


WORLDFM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="worldfm",
    display_name="WorldFM",
    default_variant_id="default",
    default_task_id="official-meta-demo",
    aliases=("inspatio/worldfm", "world-fm"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            aliases=("official", "demo"),
            notes=("Uses the official WorldFM demo/meta.json input contract by default.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-meta-demo",
            label="Official Meta Demo",
            description="Run the official WorldFM meta.json demo with the bundled reference image, intrinsics, and target poses.",
            aliases=("default", "official-demo", "novel-view"),
            inputs=(
                _field(
                    "input_path",
                    "Optional Input Image",
                    kind="path",
                    target="input_path",
                    default="",
                    description="Leave empty to use the image referenced by meta-path.",
                ),
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "interactions",
                    "Target Camera Poses",
                    kind="interaction_tokens",
                    target="params",
                    description="Leave empty to use c2w poses from meta-path.",
                ),
                _field("fps", "FPS", kind="integer", target="params", default=30),
                _field(
                    "meta-path",
                    "Meta JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=WORLDFM_OFFICIAL_META,
                ),
                _field("panorama-path", "Panorama Path", kind="path", target="call_kwargs"),
                _field("scene-name", "Scene Name", target="call_kwargs"),
                _field("save-mode", "Save Mode", target="call_kwargs", default="video", choices=("video", "image")),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
                _artifact("scene_context", "manifest"),
            ),
            default_call_kwargs={"meta_path": WORLDFM_OFFICIAL_META, "fps": 30, "save_mode": "video"},
        ),
    ),
    notes=("Official usage: python run_pipeline.py --meta demo/meta.json --output_dir outputs.",),
)
