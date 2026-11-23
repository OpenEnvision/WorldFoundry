"""Shared H3-World input contracts for CLI, TUI, and Studio."""

from __future__ import annotations

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.pipelines.h3_world.action_script import H3_WORLD_ACTION_PRESETS
from worldfoundry.runtime.interactive_inference_catalog import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

_BASE = str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
_ADAPTER = str(hfd_root_path("DANNY621--H3-World") / "step-10000.safetensors")

_FIELDS = (
    InferenceFieldSpec("prompt", "Scene prompt", target="prompt", required=True),
    InferenceFieldSpec("images", "First frame", kind="path", required=True),
    InferenceFieldSpec(
        "action_preset",
        "Action preset",
        choices=tuple(sorted(H3_WORLD_ACTION_PRESETS)),
    ),
    InferenceFieldSpec("interactions", "Action schedule / keys", kind="json"),
    InferenceFieldSpec("action_script", "Per-latent action clauses", kind="json"),
    InferenceFieldSpec("num_frames", "Frames (17k+5)", kind="integer"),
    InferenceFieldSpec("height", "Height", kind="integer"),
    InferenceFieldSpec("width", "Width", kind="integer"),
    InferenceFieldSpec("num_inference_steps", "Steps", kind="integer"),
    InferenceFieldSpec("cfg_scale", "CFG scale", kind="number"),
    InferenceFieldSpec("seed", "Seed", kind="integer"),
    InferenceFieldSpec("base_model_path", "MiniMax-H3 FL2VA base", kind="path", target="load_kwargs", default=_BASE),
    InferenceFieldSpec("adapter_path", "H3-World LoRA checkpoint", kind="path", target="load_kwargs", default=_ADAPTER),
)
_OUTPUTS = (
    InferenceArtifactSpec(
        "video",
        "video",
        required=True,
        preview=True,
        description="Action-conditioned world video with synchronized 32 kHz stereo audio.",
    ),
)

H3_WORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="h3-world",
    display_name="H3-World",
    variants=(
        InferenceVariantSpec(
            "h3-world",
            "H3-World rank-32 LoRA over MiniMax-H3 FL2VA",
            checkpoints=(
                InferenceCheckpointRef("base_model", _BASE),
                InferenceCheckpointRef("h3_world_lora", _ADAPTER),
            ),
            load_kwargs={
                "base_model_path": _BASE,
                "adapter_path": _ADAPTER,
            },
            aliases=("h3_world", "minimax-h3-world"),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            "action-video",
            "First frame + actions to world video",
            _FIELDS,
            _OUTPUTS,
            default_call_kwargs={"return_dict": True},
        ),
    ),
    default_variant_id="h3-world",
    default_task_id="action-video",
    aliases=("h3_world", "minimax-h3-world"),
)
