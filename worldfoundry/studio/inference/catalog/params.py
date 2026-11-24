"""Parameter templates and variant folding shared by several model families."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import resolve

DISPATCH_LOAD_PARAMS: tuple[str, ...] = (
    "cuda_visible_devices",
    "visible_devices",
    "cuda_devices",
    "gpu_ids",
)
_SANA_IMAGE_RUNNERS = frozenset({"image", "sprint", "controlnet"})
COGVIDEOX_STUDIO_PARENT_ID = "cogvideox_5b_t2v"
COGVIDEOX_DEFAULT_VARIANT_ID = "cogvideox_5b_t2v"
_COGVIDEOX_VARIANT_IDS = frozenset(
    {
        "cogvideox_2b_t2v",
        "cogvideox_5b_t2v",
        "cogvideox_5b_i2v",
    }
)
_COGVIDEOX_ALIAS_TO_VARIANT = {
    "cogvideox": "cogvideox_5b_t2v",
    "cogvideox_2b": "cogvideox_2b_t2v",
    "cogvideox_5b": "cogvideox_5b_t2v",
    "cogvideox_2b_t2v": "cogvideox_2b_t2v",
    "cogvideox_5b_t2v": "cogvideox_5b_t2v",
    "cogvideox_5b_i2v": "cogvideox_5b_i2v",
}


@lru_cache(maxsize=1)
def _sana_folded_studio_model_ids() -> frozenset[str]:
    """Return image-family Sana checkpoint ids that belong on one Studio card."""

    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    return frozenset(variant.model_id for variant in SANA_VARIANTS.values() if variant.runner in _SANA_IMAGE_RUNNERS)


def _cogvideox_canonical_id(model_id: str) -> str:
    return _normalize_model_id(model_id).replace("-", "_")


def _cogvideox_variant_id(model_id: str) -> str | None:
    key = _cogvideox_canonical_id(model_id)
    if key in _COGVIDEOX_VARIANT_IDS:
        return key
    return _COGVIDEOX_ALIAS_TO_VARIANT.get(key)


def _cogvideox_folded_studio_model_ids() -> frozenset[str]:
    """Return CogVideoX pipeline ids that are variants on the shared Studio card."""

    folded = _COGVIDEOX_VARIANT_IDS - {COGVIDEOX_STUDIO_PARENT_ID}
    return frozenset(item for model_id in folded for item in (model_id, model_id.replace("_", "-")))


def _cogvideox_studio_parent_id(model_id: str) -> str | None:
    """Return the CogVideoX card id when ``model_id`` is a folded T2V/I2V pipeline."""

    if _cogvideox_variant_id(model_id):
        return COGVIDEOX_STUDIO_PARENT_ID
    return None


def cogvideox_runtime_model_id(model_id: str, variant_id: str = "") -> str | None:
    """Return the concrete CogVideoX pipeline id for a Studio card + variant."""

    if _cogvideox_studio_parent_id(model_id) is None:
        return None
    return _cogvideox_variant_id(variant_id or model_id) or COGVIDEOX_DEFAULT_VARIANT_ID


def _wan_2p1_t2v_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan-AI--Wan2.1-T2V-1.3B",
        "Wan2.1-T2V-1.3B",
        fallback="Wan-AI/Wan2.1-T2V-1.3B",
    )


def _wan_2p1_i2v_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan2.1-I2V-14B-480P",
        "Wan-AI--Wan2.1-I2V-14B-480P",
        fallback="Wan-AI/Wan2.1-I2V-14B-480P",
    )


def _official_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "checkpoint_path",
        "checkpoint_candidates",
        "kind",
        "pipeline_target",
        "torch_dtype",
        "fps",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        *DISPATCH_LOAD_PARAMS,
    )


def _official_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "video",
        "image_path",
        "video_path",
        "output_path",
        "return_dict",
        "plan_only",
        "num_frames",
        "max_frames",
        "num_blocks",
        "task",
        "num_inference_steps",
        "num_sampling_steps",
        "sampling_steps",
        "infer_steps",
        "base_num_frames",
        "ar_step",
        "causal_block_size",
        "addnoise_condition",
        "fps",
        "target_fps",
        "height",
        "width",
        "resolution",
        "size",
        "aspect_ratio",
        "i2v_mode",
        "i2v_image_path",
        "i2v_resolution",
        "i2v_stability",
        "i2v_condition_type",
        "seed",
        "guidance_scale",
        "cfg_scale",
        "shift",
        "negative_prompt",
        "quality",
        "tiled",
        "prompt_prefix",
        "lora_path",
        "state_dict",
        "time_shift",
        "tensor_parallel_degree",
        "ulysses_degree",
        "ulysses_size",
        "ring_degree",
        "ring_size",
        "parallel",
        "seconds",
        "duration",
        "latent_window_size",
        "num_steps",
        "steps",
        "aes",
        "flow",
        "sample_method",
        "max_sequence_length",
        "config",
        "version",
        "use_usp",
        "nproc_per_node",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "model_subdir",
        "vae_subdir",
        "cache_dir",
        "text_encoder_name_1",
        "ae",
        "rewrite",
        "cfg_distilled",
        "enable_step_distill",
        "sparse_attn",
        "use_sageattn",
        "enable_cache",
        "sr",
        "save_pre_sr_video",
        "overlap_group_offloading",
        "model_resolution",
        "flow_shift",
        "embedded_cfg_scale",
        "mode",
        "variant",
        "gpu_memory_preservation",
        "cuda_visible_devices",
        "visible_devices",
        "cuda_devices",
        "gpu_ids",
    )


def _native_diffusion_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "checkpoint_path",
        "checkpoint_overrides",
        "release_text_encoders_after_encode",
        "torch_dtype",
        "weight_dtype",
        "dtype",
        "offload_mode",
        "attention_backend",
        "quantization",
        "quantization_compute_dtype",
        "quantization_exclude",
        "fp8_min_features",
        "fp8_scaling",
        "scaling",
        "keep_dense_fallback",
        "use_fast_accum",
        "fuse_qkv",
        "qkv_strategy",
        "qkv_split_threshold",
        "inplace_residual",
        "static_cross_kv",
        "fused_rope",
        "rope_precision",
        "rms_norm_precision",
        "compile",
        "compile_scope",
        "compile_backend",
        "compile_mode",
        "compile_fullgraph",
        "compile_dynamic",
        "compile_options",
        "cuda_graph",
        "warmup_requests",
        "approximate_attention",
        "vae_tiling",
        *DISPATCH_LOAD_PARAMS,
    )


def _native_diffusion_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "image",
        "image_path",
        "video",
        "interactions",
        "negative_prompt",
        "output_path",
        "output_dir",
        "num_frames",
        "frames",
        "frame_num",
        "height",
        "width",
        "num_inference_steps",
        "infer_steps",
        "guidance_scale",
        "cfg_scale",
        "embedded_guidance_scale",
        "shift",
        "seed",
        "fps",
        "output_type",
        "return_dict",
    )


def _t2v_turbo_base_ref() -> str:
    return resolve._checkpoint_model_ref(
        "VideoCrafter--VideoCrafter2/model.ckpt",
        "VideoCrafter2/model.ckpt",
        fallback="VideoCrafter/VideoCrafter2",
    )


def _t2v_turbo_lora_ref() -> str:
    return resolve._checkpoint_model_ref(
        "jiachenli-ucsb--T2V-Turbo-VC2/unet_lora.pt",
        "T2V-Turbo-VC2/unet_lora.pt",
        fallback="jiachenli-ucsb/T2V-Turbo-VC2",
    )


def _t2v_turbo_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_checkpoint": _t2v_turbo_base_ref(),
        "lora_checkpoint": _t2v_turbo_lora_ref(),
    }


def _cogvideox_variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "cogvideox_2b_t2v": {
            "generation_type": "t2v",
            "aliases": ("cogvideox-2b-t2v", "cogvideox-2b"),
            "label": "2B T2V",
            "default_prompt": "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting.",
            "default_call_kwargs": {
                "fps": 8,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "seed": 42,
                "width": 720,
            },
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "CogVideoX-2b",
                "THUDM--CogVideoX-2b",
                fallback="THUDM/CogVideoX-2b",
            ),
        },
        "cogvideox_5b_i2v": {
            "generation_type": "i2v",
            "aliases": ("cogvideox-5b-i2v",),
            "label": "5B I2V",
            "default_prompt": (
                "A young man holding a bright sparkler gently turns his head and smiles as "
                "sparks scatter through the night air, with a subtle cinematic camera dolly, "
                "stable facial details, and natural motion."
            ),
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "fps": 8,
                "guidance_scale": 6.0,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "seed": 45,
                "width": 720,
            },
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "CogVideoX-5b-I2V",
                "THUDM--CogVideoX-5b-I2V",
                fallback="THUDM/CogVideoX-5b-I2V",
            ),
        },
        "cogvideox_5b_t2v": {
            "generation_type": "t2v",
            "aliases": ("cogvideox", "cogvideox-5b-t2v", "cogvideox-5b"),
            "label": "5B T2V",
            "default_prompt": "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting.",
            "default_call_kwargs": {
                "fps": 8,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "seed": 43,
                "width": 720,
            },
            "default_model_ref": lambda: resolve._checkpoint_model_ref(
                "CogVideoX-5b",
                "THUDM--CogVideoX-5b",
                fallback="THUDM/CogVideoX-5b",
            ),
        },
    }


def _cogvideox_aliases() -> tuple[str, ...]:
    aliases = [COGVIDEOX_STUDIO_PARENT_ID, *sorted(_cogvideox_folded_studio_model_ids())]
    for spec in _cogvideox_variant_specs().values():
        aliases.extend(spec.get("aliases") or ())
    return resolve._coerce_aliases(*aliases)


def _cogvideox_workspace_variants() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for variant_id, spec in _cogvideox_variant_specs().items():
        model_ref = str(resolve._resolve_override_value(spec.get("default_model_ref", "")) or "")
        rows.append(
            {
                "variant_id": variant_id,
                "label": spec.get("label") or variant_id,
                "status": "checkpoint-configured",
                "workload_type": spec.get("generation_type") or "t2v",
                "default_prompt": spec.get("default_prompt") or "",
                "default_input_path": spec.get("default_input_path") or "",
                "aliases": tuple(spec.get("aliases") or ()),
                "checkpoints": (
                    (
                        {
                            "role": "primary",
                            "uri": model_ref,
                            "required": True,
                            "status": "checkpoint-configured",
                        },
                    )
                    if model_ref
                    else ()
                ),
                "load_kwargs": {},
                "call_kwargs": dict(resolve._resolve_override_value(spec.get("default_call_kwargs", {})) or {}),
            }
        )
    return tuple(rows)


OFFICIAL_ACTION_CALL_PARAMS: tuple[str, ...] = (
    "prompt",
    "images",
    "video",
    "interactions",
    "output_path",
    "fps",
    "timeout_seconds",
    "checkpoint_path",
    "checkpoint_dir",
    "ckpt_path",
    "checkpoint_ref",
    "repo_id",
    "runtime_config_path",
    "plan_only",
    "official_policy_observation",
    "observation",
    "action_space",
    "policy_controls",
)
OFFICIAL_ACTION_LOAD_PARAMS: tuple[str, ...] = (
    "pretrained_model_path",
    "args",
    "device",
    "model_id",
    "profile_path",
    "manifest_path",
    "acquisition_root",
    "hf_models_root",
    "command_template",
    "checkpoint_path",
    "checkpoint_dir",
    "ckpt_path",
    "checkpoint_ref",
    "repo_id",
    "runtime_config_path",
)


def _normalize_model_id(model_id: str) -> str:
    if model_id == "matrix-game-1":
        return model_id
    if model_id == "matrix-game-2":
        return model_id
    if model_id == "matrix-game-3":
        return model_id
    if model_id == "depth-anything-v3":
        return model_id
    if model_id == "vggt-omega":
        return model_id
    return model_id
