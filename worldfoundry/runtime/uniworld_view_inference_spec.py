"""Inference-only image and video routes for official UniWorld-View."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import official_runtime_repo_path


def _field(name: str, label: str, kind: str = "string", *, target: str = "call_kwargs", default=None, required=False):
    return InferenceFieldSpec(name, label, kind=kind, target=target, default=default, required=required)


_SOURCE = str(official_runtime_repo_path("UniWorld-View", specific_env="UNIWORLD_VIEW_REPO"))
_COMMON = (
    _field("repo_root", "Official Source", "path", target="load_kwargs", default=_SOURCE),
    _field("python_executable", "Official Python", "path", target="load_kwargs"),
    _field("checkpoint_root", "Checkpoint Root", "path", target="load_kwargs"),
    _field("render_method", "Renderer", default="hybrid"),
    _field("height", "Height", "integer", default=480),
    _field("width", "Width", "integer", default=832),
    _field("keep_aspect_ratio", "Preserve Input Aspect Ratio", "boolean", default=True),
    _field("num_frames", "Frames", "integer", default=81),
    _field("num_inference_steps", "Steps", "integer", default=8),
    _field("guidance_scale", "Guidance", "number", default=4.0),
    _field("fps", "FPS", "integer", default=16),
    _field("seed", "Seed", "integer", default=43),
    _field("traj_type", "Camera Trajectory", default="custom"),
    _field("d_phi", "Horizontal Rotation", "number", default=50.0),
    _field("d_theta", "Vertical Rotation", "number", default=0.0),
    _field("x_offset", "Horizontal Translation", "number", default=0.0),
    _field("y_offset", "Vertical Translation", "number", default=0.0),
    _field("z_offset", "Depth Translation", "number", default=0.0),
    _field("radius_scale", "Radius Scale", "number", default=1.0),
    _field("low_gpu_memory_mode", "Low GPU Memory", "boolean", default=False),
    _field("plan_only", "Plan Only", "boolean", default=False),
    _field("timeout_seconds", "Timeout Seconds", "integer", default=21600),
)
_OUTPUT = (InferenceArtifactSpec("video", "video", required=True, preview=True),)


def _task(task_id: str, label: str, mode: str) -> InferenceTaskProfile:
    fields = (
        _field(
            "input_path", "Source Video" if mode == "dynamic_view" else "Source Image", "path",
            target="input_path", required=True,
            default=f"{_SOURCE}/test/videos/2.mp4" if mode == "dynamic_view" else f"{_SOURCE}/test/images/fruit.jpg",
        ),
        _field("mode", "View Mode", default=mode),
        *_COMMON,
    )
    defaults = {item.field_id: item.default for item in fields if item.target == "call_kwargs" and item.default is not None}
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        inputs=fields,
        outputs=_OUTPUT,
        default_call_kwargs=defaults,
        description="Generate camera-controlled novel views using the official BLIP2-captioned UniWorld-View pipeline.",
    )


UNIWORLD_VIEW_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="uniworld-view",
    display_name="UniWorld-View",
    default_variant_id="official",
    default_task_id="image-to-video",
    aliases=("uniworld_view", "uniview"),
    variants=(
        InferenceVariantSpec(
            variant_id="official",
            label="Official UniView + Wan VACE 14B",
            status="requires_local_source_and_checkpoints",
            checkpoints=(
                InferenceCheckpointRef("primary", "${WORLDFOUNDRY_CKPT_DIR}/Drexubery--UniView"),
                InferenceCheckpointRef("wan_vace_diffusers", "${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-VACE-14B-diffusers"),
                InferenceCheckpointRef("blip2", "${WORLDFOUNDRY_CKPT_DIR}/Salesforce--blip2-opt-2.7b"),
                InferenceCheckpointRef("moge", "${WORLDFOUNDRY_CKPT_DIR}/Ruicheng--moge-2-vitl-normal/model.pt"),
                InferenceCheckpointRef("sam2", "${WORLDFOUNDRY_CKPT_DIR}/facebook--sam2-hiera-large/sam2_hiera_large.pt"),
                InferenceCheckpointRef("tracer", "${WORLDFOUNDRY_CKPT_DIR}/Carve--tracer_b7/tracer_b7.pth"),
                InferenceCheckpointRef("causvid_lora", "${WORLDFOUNDRY_CKPT_DIR}/Kijai--WanVideo_comfy/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"),
                InferenceCheckpointRef("stream3r", "${WORLDFOUNDRY_CKPT_DIR}/yslan--STream3R", required=False),
            ),
            load_kwargs={"repo_root": _SOURCE},
            aliases=("default",),
            notes=("Official source, STream3R, an isolated Python environment, and multiple local weight bundles are required.",),
        ),
    ),
    tasks=(
        _task("image-to-video", "Single Image Novel View", "single_view"),
        _task("video-to-video", "Monocular Video Novel View", "dynamic_view"),
    ),
    notes=(
        "The official CLI ignores a user prompt and generates its own BLIP2 caption.",
        "Dynamic view needs at least 13 frames; default aspect-ratio preservation may change the output dimensions.",
    ),
)


__all__ = ["UNIWORLD_VIEW_INFERENCE_SPEC"]
