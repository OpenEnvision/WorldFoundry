# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``fantasyworld`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, GENERIC_IMAGE_FIXTURE, _artifact, _field

FANTASYWORLD_CAMERA_FIXTURE = str(_TEST_CASES_ROOT / "fantasyworld" / "camera_forward.json")


FANTASYWORLD_PROMPT = "A coherent fantasy harbor world with stable geometry during a forward camera move."


FANTASYWORLD_WAN21_CALL_KWARGS = {
    "camera_json_path": FANTASYWORLD_CAMERA_FIXTURE,
    "fps": 16,
    "seed": 1024,
    "using_scale": True,
    "conf_threshold": 1.0,
    "stride": 4,
    "return_dict": True,
}


FANTASYWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="fantasyworld",
    display_name="FantasyWorld",
    default_variant_id="wan21",
    default_task_id="official-camera-world",
    aliases=(),
    variants=(
        InferenceVariantSpec(
            variant_id="wan21",
            label="Wan2.1",
            status="configured",
            call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
            aliases=("default", "official", "wan2.1"),
            notes=("Uses the in-tree FantasyWorld Wan2.1 camera-control runtime.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-world",
            label="Camera World",
            description="Run FantasyWorld image-conditioned camera-control world generation.",
            aliases=("default", "official-demo", "image-to-world"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=FANTASYWORLD_PROMPT),
                _field(
                    "camera_json_path",
                    "Camera JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=FANTASYWORLD_CAMERA_FIXTURE,
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1024),
                _field("using_scale", "Use Scale", kind="boolean", target="call_kwargs", default=True),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("stride", "Point Stride", kind="integer", target="call_kwargs", default=4),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("model", "generated_3d_asset", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
        ),
    ),
    notes=("The camera fixture is an in-tree c2w trajectory because the official demo JSON was not present locally.",),
)


FANTASYWORLD_WAN21_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="fantasyworld-wan21",
    display_name="FantasyWorld Wan2.1",
    default_variant_id="wan21",
    default_task_id="official-camera-world",
    aliases=("fantasyworld-wan2.1",),
    variants=(
        InferenceVariantSpec(
            variant_id="wan21",
            label="Wan2.1",
            status="configured",
            call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
            aliases=("default", "official", "wan2.1"),
            notes=("Uses the in-tree FantasyWorld Wan2.1 camera-control runtime.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-world",
            label="Camera World",
            description="Run FantasyWorld Wan2.1 image-conditioned camera-control world generation.",
            aliases=("default", "official-demo", "image-to-world"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=FANTASYWORLD_PROMPT),
                _field(
                    "camera_json_path",
                    "Camera JSON",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=FANTASYWORLD_CAMERA_FIXTURE,
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1024),
                _field("using_scale", "Use Scale", kind="boolean", target="call_kwargs", default=True),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("stride", "Point Stride", kind="integer", target="call_kwargs", default=4),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("model", "generated_3d_asset", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FANTASYWORLD_WAN21_CALL_KWARGS,
        ),
    ),
    notes=("The camera fixture is an in-tree c2w trajectory because the official demo JSON was not present locally.",),
)
