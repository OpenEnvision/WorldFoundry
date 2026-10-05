"""Shared UI/CLI contracts for inference-only interactive world pipelines.

Keep lightweight: discovery must not import CUDA models or upstream runtimes.
"""

from __future__ import annotations

from typing import Any

import yaml

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import hfd_root_path, official_runtime_repo_path, package_data_path


def _field(name, kind="string", default=None, *, target="call_kwargs", required=False, description=""):
    return InferenceFieldSpec(
        name,
        name.replace("_", " ").title(),
        kind=kind,
        target=target,
        default=default,
        required=required,
        description=description,
    )


def _asset(name, *, required=False, description=""):
    return _field(name, "path", target="load_kwargs", required=required, description=description)


def _variant(name, repo, *, aliases=(), **load):
    checkpoint = str(load.get("checkpoint_dir") or hfd_root_path(repo.replace("/", "--")))
    return InferenceVariantSpec(
        name,
        name.replace("-", " ").title(),
        checkpoints=(InferenceCheckpointRef("primary", checkpoint),),
        load_kwargs={"checkpoint_dir": checkpoint, **load},
        aliases=aliases,
        notes=("CUDA, checkpoints and the matching inference environment are required.",),
    )


_PROMPT = _field("prompt", target="prompt", required=True)
_IMAGE = _field("input_path", "path", target="input_path", description="Reference image path.")
_REQUIRED_IMAGE = _field("input_path", "path", target="input_path", required=True, description="Reference image path.")
_EXTERNAL = (_asset("source_root"), _asset("python_executable"))
_OUTPUTS = (InferenceArtifactSpec("video", "video", required=True, preview=True),)


def _spec(model, name, variants, fields, *, aliases=(), external=True, task="image-to-video"):
    defaults = {
        field.field_id: field.default for field in fields if field.target == "call_kwargs" and field.default is not None
    }
    if external:
        defaults.update(execute=True, return_dict=True)
        fields = (*fields, *_EXTERNAL, _field("execute", "boolean", True), _field("timeout_seconds", "integer", 7200))
    return ModelInferenceSpec(
        model,
        name,
        tuple(variants),
        (
            InferenceTaskProfile(
                task, task.replace("-", " ").title(), tuple(fields), _OUTPUTS, default_call_kwargs=defaults
            ),
        ),
        default_variant_id=variants[0].variant_id,
        default_task_id=task,
        aliases=aliases,
        notes=("Inference only. Checkpoint-backed GPU generation is required for end-to-end validation.",),
    )


ZING = _spec(
    "zing",
    "Zing-0.5",
    [_variant("zing-0.5", "seedleap/zing-0.5")],
    (
        _PROMPT,
        _IMAGE,
        _asset("checkpoint_dir", required=True),
        _field("num_frames", "integer", 81, description="Total pixel frames, including reference: 1 + 4*N."),
        _field("height", "integer", 480),
        _field("width", "integer", 832),
        _field("fps", "integer", 24),
        _field("seed", "integer", 0),
        _field(
            "controls",
            "json",
            description="List of frame-aligned keyboard_direction_frame_interval or text_prompt_interval controls.",
        ),
        _field("local_attn_size", "integer"),
        _field("sink_size", "integer"),
    ),
    aliases=("zing-0.5", "zing-world-model"),
    external=False,
    task="interactive-world-model",
)
ASTRONEX = _spec(
    "astronex-world",
    "Astronex-World",
    [
        _variant(
            mode,
            "Astronex-Lab/Astronex-World",
            mode=mode,
            source_root=str(official_runtime_repo_path("Astronex-Robotics--Astronex-World")),
        )
        for mode in ("causal", "bidirectional")
    ],
    (
        _PROMPT,
        _IMAGE,
        _asset("checkpoint_dir", required=True),
        _field(
            "latent_frames",
            "integer",
            description="Latent frames, not video frames. Leave unset for mode/image-dependent length.",
        ),
        _field("trajectory", description="Camera DSL, e.g. w*12,h*11; lengths must match latent_frames."),
        _field("steps", "integer"),
        _field("fps", "integer", 24),
        _field("seed", "integer", 0),
        _field("height", "integer"),
        _field("width", "integer"),
        _field("event_prompt"),
        _field("event_start_frame", "integer"),
        _field("action_tokens", "path"),
        _field("action_keys", "boolean"),
        _field("action_scale", "number"),
        _field("embodiment_id", "integer"),
        _field("vram_limit_gb", "number"),
        _field("consumer", "boolean", target="load_kwargs", description="Causal mode only."),
    ),
    aliases=("astronex",),
    task="interactive-world-model",
)

_SOLAR_ROUTES = yaml.safe_load(package_data_path("models/runtime/configs/solarwm/routes.yaml").read_text())["variants"]
SOLAR = _spec(
    "solarwm",
    "SolarWM",
    [
        _variant(
            name,
            route["repo_id"],
            variant=name,
            checkpoint_dir=str(hfd_root_path(route["repo_id"].replace("/", "--"), route["checkpoint_subdir"])),
            source_root=str(official_runtime_repo_path("Junchao-cs--SolarWM")),
        )
        for name, route in _SOLAR_ROUTES.items()
    ],
    (
        _asset("checkpoint_dir", required=True),
        _asset("base_path", required=True),
        _asset("data_root", required=True),
        _field(
            "assets",
            "json",
            target="load_kwargs",
            description="LTX requires model.codec.video_vae_path and inference.negative_caption_cache; H3 requires data.silence_latents_path and data.encoder_contract_path. H3 long also needs inference.plan and inference.dataset_root.",
        ),
        _field(
            "overrides",
            "json",
            description='Indexed inference overrides, e.g. {"inference.sample_count": 1}. Images, prompts and cameras come from the index.',
        ),
    ),
    task="indexed-inference",
)
ALAYA_V11 = _spec(
    "alayaworld-v1.1",
    "AlayaWorld v1.1",
    [
        _variant(
            name,
            "AlayaLab/AlayaWorld-v1.1-stage2b",
            variant=name,
            source_root=str(official_runtime_repo_path("AlayaLab--AlayaWorld")),
            student_dir=str(hfd_root_path("AlayaLab--AlayaWorld-v1.1-stage3")),
        )
        for name in ("dmd", "ar")
    ],
    (
        _PROMPT,
        _REQUIRED_IMAGE,
        _asset("checkpoint_dir", required=True),
        _asset("base_checkpoint", required=True),
        _asset("gemma_path", required=True),
        _asset("vigeo_checkpoint", required=True),
        _asset("vigeo_source_root"),
        _asset("student_dir", description="Stage 3 LoRA required for DMD; AR uses stage 2b."),
        _field("rounds", "integer", 5),
        _field("seed", "integer", 42),
        _field("camera_path", "path", description="C2W .npy, .npz or .txt trajectory."),
        _field("intrinsic", "json"),
        _field("forward", "number"),
        _field("yaw", "number"),
        _field("pitch", "number"),
    ),
    aliases=("alayaworld-v11", "alaya-world-v1.1"),
)
ECHO = _spec(
    "joyai-echo-wm",
    "JoyAI Echo-WM",
    [
        _variant(
            name,
            "Echo-Team/Echo-WM",
            variant=name,
            source_root=str(official_runtime_repo_path("jd-opensource--JoyAI-Echo")),
            gemma_path=str(hfd_root_path("google--gemma-3-12b-it-qat-q4_0-unquantized")),
        )
        for name in ("base", "flash")
    ],
    (
        _PROMPT,
        _REQUIRED_IMAGE,
        _asset("checkpoint_dir", required=True),
        _asset("gemma_path", required=True),
        _field("action_str", default="w-60,a-60", required=True, description="Camera actions, e.g. w-60,a-60."),
        _field("num_frames", "integer", 241, description="Base: 1+8*N; Flash: 1+8*video_chunk_size*N."),
        _field("width", "integer", 1280),
        _field("height", "integer", 704),
        _field("fps", "integer", 24),
        _field("seed", "integer", 4),
        *(_field(key, "number") for key in ("fov_deg", "translation_speed", "rotation_speed_deg", "pitch_limit_deg")),
        *(
            _field(key, "boolean", True if key == "action_overlay" else False)
            for key in ("auto_fov", "no_audio", "action_overlay")
        ),
        *(
            _field(key, kind, description="Base variant only; leave unset for Flash.")
            for key, kind in (
                ("steps", "integer"),
                ("guidance_scale", "number"),
                ("video_cfg", "number"),
                ("audio_cfg", "number"),
                ("negative_prompt", "string"),
                ("stg_scale", "number"),
                ("stg_blocks", "json"),
            )
        ),
        *(
            _field(key, kind, description="Flash variant only; leave unset for Base.")
            for key, kind in (
                ("video_local_attn_size", "integer"),
                ("video_sink_size", "integer"),
                ("video_chunk_size", "integer"),
                ("timesteps", "json"),
            )
        ),
    ),
    aliases=("echo-wm",),
)

INTERACTIVE_INFERENCE_SPECS = {spec.model_family_id: spec for spec in (ZING, ASTRONEX, SOLAR, ALAYA_V11, ECHO)}


def studio_contract_override(model_id: str) -> dict[str, Any]:
    """Derive discovery metadata from the same contract consumed by CLI/Workspace."""
    spec = INTERACTIVE_INFERENCE_SPECS.get(model_id)
    if spec is None:
        return {}
    fields = spec.task().inputs
    call = [field.field_id for field in fields if field.target == "call_kwargs"]
    call += ["prompt"] if any(field.target == "prompt" for field in fields) else []
    call += ["images"] if any(field.target == "input_path" for field in fields) else []
    call += ["output_path", "return_dict"]
    if model_id != "zing":
        call += ["execute", "output_dir"]
    load = [field.field_id for field in fields if field.target == "load_kwargs"]
    return {
        "display_name": spec.display_name,
        "supports_from_pretrained": True,
        "supports_stream": False,
        "stream_params": (),
        "call_params": tuple(dict.fromkeys(call)),
        "load_params": tuple(
            dict.fromkeys(("model_path", "required_components", "device", *load, *spec.variant().load_kwargs))
        ),
        "default_load_kwargs": {},
        "default_call_kwargs": {"return_dict": True},
        "default_model_ref": "",
        "default_input_path": "",
        "default_interactions": (),
        "default_prompt": "" if model_id == "solarwm" else "A camera explores a coherent scene.",
        "default_task_type": "image-to-video" if model_id in {"alayaworld-v1.1", "joyai-echo-wm"} else "",
        "suggested_task_types": (spec.default_task_id,),
        "aliases": spec.aliases,
        "tags": ("world-model", "inference"),
    }


def interactive_task_for_variant(spec, variant, task):
    """Resolve applicable controls and asset defaults before rendering a surface."""
    from dataclasses import replace

    if spec.model_family_id not in INTERACTIVE_INFERENCE_SPECS:
        return task
    excluded = set()
    if spec.model_family_id == "joyai-echo-wm":
        excluded = (
            {"steps", "guidance_scale", "video_cfg", "audio_cfg", "negative_prompt", "stg_scale", "stg_blocks"}
            if variant.variant_id == "flash"
            else {"video_local_attn_size", "video_sink_size", "video_chunk_size", "timesteps"}
        )
    if spec.model_family_id == "astronex-world" and variant.variant_id != "causal":
        excluded.add("consumer")
    inputs = []
    for field in task.inputs:
        if field.field_id in excluded:
            continue
        defaults = variant.load_kwargs if field.target == "load_kwargs" else variant.call_kwargs
        inputs.append(replace(field, default=defaults.get(field.field_id, field.default)))
    return replace(task, inputs=tuple(inputs))
