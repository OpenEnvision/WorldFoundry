# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``wan2p2`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field, _read_text_file_or_default

WAN2P2_OFFICIAL_PROMPT = _read_text_file_or_default(
    _TEST_CASES_ROOT / "wan2.2" / "ti2v_5b" / "prompt.txt",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
)


WAN2P2_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="wan-2p2",
    display_name="Wan 2.2",
    default_variant_id="ti2v-5b",
    default_task_id="ti2v-5b",
    aliases=("wan2.2", "wan-2.2", "wan2p2"),
    variants=(
        InferenceVariantSpec(
            variant_id="ti2v-5b",
            label="TI2V 5B",
            status="configured",
            load_kwargs={"mode": "ti2v-5B"},
            call_kwargs={"size": "1280*704"},
            aliases=("default", "official", "ti2v"),
            notes=("Uses the staged local Wan2.2-TI2V-5B checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="ti2v-5b",
            label="TI2V 5B",
            description="Run Wan 2.2 TI2V-5B with official default size and sampling config.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=WAN2P2_OFFICIAL_PROMPT),
                _field("input_path", "Input Image", kind="path", target="input_path"),
                _field("size", "Size", target="call_kwargs", default="1280*704", choices=("1280*704", "704*1280")),
                _field("frame_num", "Frames", kind="integer", target="call_kwargs"),
                _field("sample_steps", "Sample Steps", kind="integer", target="call_kwargs"),
                _field("sample_guide_scale", "Guidance", kind="number", target="call_kwargs"),
                _field("base_seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("offload_model", "Offload Model", kind="boolean", target="call_kwargs", default=True),
                _field("use_prompt_extend", "Prompt Extend", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "size": "1280*704",
                "base_seed": 42,
                "offload_model": True,
                "use_prompt_extend": False,
            },
        ),
    ),
    notes=("Keep validation jobs tiny by overriding frame_num/sample_steps from the frontend when needed.",),
)
