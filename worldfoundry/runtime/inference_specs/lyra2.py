# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``lyra2`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field, _read_text_file_or_default

LYRA2_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "lyra" / "Lyra-2" / "00.png")


LYRA2_OFFICIAL_PROMPT = _read_text_file_or_default(
    _TEST_CASES_ROOT / "lyra" / "Lyra-2" / "00.txt",
    "A slow, steady camera push forward through a coherent static 3D world with stable geometry.",
)


LYRA2_OFFICIAL_CALL_KWARGS = {
    "fps": 16,
    "resolution": (480, 832),
    "seed": 1,
    "reconstruct_3d": False,
    "return_dict": True,
    "show_progress": True,
    "execute": True,
}


LYRA2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="lyra",
    display_name="Lyra",
    default_variant_id="lyra-2",
    default_task_id="official-zoomgs",
    aliases=("lyra-2", "lyra2", "nvidia/lyra", "nvidia/lyra-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="lyra-2",
            label="Lyra-2",
            status="configured",
            call_kwargs=LYRA2_OFFICIAL_CALL_KWARGS,
            aliases=("default", "official", "zoomgs"),
            notes=("Matches the upstream Lyra-2 assets/samples single-image navigation demo shape.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-zoomgs",
            label="Official ZoomGS",
            description="Run the Lyra-2 single-image action-conditioned world generation demo.",
            aliases=("default", "interactive-video", "official-demo", "zoomgs"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=LYRA2_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=LYRA2_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward",),
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("resolution", "Resolution", kind="json", target="call_kwargs", default=(480, 832)),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1),
                _field("reconstruct_3d", "Reconstruct 3D", kind="boolean", target="call_kwargs", default=False),
                _field("return_dict", "Return Dict", kind="boolean", target="call_kwargs", default=True),
                _field("show_progress", "Show Progress", kind="boolean", target="call_kwargs", default=True),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LYRA2_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image and caption come from upstream Lyra-2 assets/samples/00.*.",),
)
