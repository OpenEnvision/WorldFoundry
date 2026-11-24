"""Curated overrides and defaults for 3D scene, novel-view, depth, and reconstruction models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    resolve_local_hf_model_path,
)

from .. import resolve
from ..params import DISPATCH_LOAD_PARAMS, _wan_2p1_t2v_default_ref


def _pi3_default_ref() -> str:
    try:
        return str(
            resolve_local_hf_model_path(
                "yyfz233/Pi3X",
                required_files=("config.json", "model.safetensors"),
            )
        )
    except FileNotFoundError:
        return "yyfz233/Pi3X"


def _depth_anything_v2_default_ref() -> str:
    repo_id = "depth-anything/Depth-Anything-V2-Large"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("depth_anything_v2_vitl.pth",),
            )
        )
    except FileNotFoundError:
        return repo_id


def _cosmos_predict2p5_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Cosmos-Predict2.5-2B",
        "nvidia--Cosmos-Predict2.5-2B",
        "huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B",
        fallback="nvidia/Cosmos-Predict2.5-2B",
    )


def _cosmos_predict2_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "nvidia--Cosmos-Predict2-2B-Video2World",
        "Cosmos-Predict2-2B-Video2World",
        fallback="nvidia/Cosmos-Predict2-2B-Video2World",
    )


def _cosmos_predict2_default_load_kwargs() -> dict[str, Any]:
    model_root = Path(_cosmos_predict2_default_ref())
    return {
        "checkpoint_path": str(model_root),
        "text_encoder_model_path": str(model_root / "text_encoder"),
        "text_tokenizer_path": str(model_root / "tokenizer"),
        "offload_mode": "block",
    }


def _cosmos_predict2p5_default_load_kwargs() -> dict[str, Any]:
    model_root = _cosmos_predict2p5_default_ref()
    return {
        "required_components": {
            "text_encoder_model_path": resolve._checkpoint_model_ref(
                "Cosmos-Reason1-7B",
                fallback="nvidia/Cosmos-Reason1-7B",
            ),
            # The Cosmos checkpoint ships its own tokenizer.pth VAE. A Wan
            # checkpoint directory does not satisfy the Cosmos codec contract.
            "vae_model_path": model_root,
        }
    }


def _cosmos_predict2p5_default_call_kwargs() -> dict[str, Any]:
    return {
        "fps": 16,
        "guidance_scale": 7.0,
        "height": 704,
        "num_frames": 93,
        "num_inference_steps": 35,
        "output_type": "pt",
        "seed": 42,
        "width": 1280,
    }


def _cosmos_transfer2p5_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "nvidia--Cosmos-Transfer2.5-2B",
        "Cosmos-Transfer2.5-2B",
        fallback="nvidia/Cosmos-Transfer2.5-2B",
    )


def _cosmos_transfer2p5_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "text_encoder_model_path": resolve._checkpoint_model_ref(
                "Cosmos-Reason1-7B",
                fallback="nvidia/Cosmos-Reason1-7B",
            ),
            "vae_model_path": _cosmos_predict2p5_default_ref(),
        }
    }


def _cosmos_transfer2p5_default_call_kwargs() -> dict[str, Any]:
    return {
        **_cosmos_predict2p5_default_call_kwargs(),
        "controlnet_variant": "edge",
        "control_context_scale": 1.0,
        "output_type": "video",
    }


def _moverse_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Orange-3DV-Team--MoVerse",
        fallback="Orange-3DV-Team/MoVerse",
    )


def _moverse_default_load_kwargs() -> dict[str, Any]:
    return {
        "flux_path": resolve._checkpoint_model_ref(
            "black-forest-labs--FLUX.1-Fill-dev",
            "FLUX.1-Fill-dev",
            fallback="black-forest-labs/FLUX.1-Fill-dev",
        ),
        "da3_path": resolve._checkpoint_model_ref(
            "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
            "DA3NESTED-GIANT-LARGE-1.1",
            fallback="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ),
        "wan_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
            fallback="Wan-AI/Wan2.1-T2V-1.3B",
        ),
    }


def _depth_anything_v3_default_ref() -> str:
    repo_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("config.json", "model.safetensors"),
            )
        )
    except FileNotFoundError:
        return repo_id


def _inspatio_world_default_ref() -> str:
    repo_id = "inspatio/world"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("InSpatio-World-1.3B.safetensors",),
            )
        )
    except FileNotFoundError:
        return repo_id


def _inspatio_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_path": _wan_2p1_t2v_default_ref(),
        "da3_model_path": _depth_anything_v3_default_ref(),
        "florence_model_path": resolve._checkpoint_model_ref(
            "Florence-2-large",
            fallback="microsoft/Florence-2-large",
        ),
        "default_traj_txt_path": str(
            resolve._data_path("models", "runtime", "configs", "inspatio_world", "traj", "x_y_circle_cycle.txt")
        ),
    }


def _worldcam_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("worldcam--worldcam", "worldcam"),
        "worldcam/worldcam",
    )


def _worldcam_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _wan_2p1_t2v_default_ref(),
        },
    }


def _worldcam_default_call_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "conditioning_frames": 65,
        "num_ar_steps": 50,
        "cfg_scale": 4,
        "seed": 0,
        "num_inference_steps": 50,
        "long_term_memory_start_step": 30,
        "long_term_memory_num_clips": 4,
        "long_term_memory_ref_indices": [48, 52, 56, 60],
        "attention_sink_inference": False,
        "trim_conditioning": False,
        "height": 480,
        "width": 832,
        "fps": 30,
    }
    demo_dir = resolve._data_path("test_cases", "worldcam")
    video_path = demo_dir / "0.mp4"
    intrinsics_path = demo_dir / "0_intrinsics_palindrome.npy"
    extrinsics_path = demo_dir / "0_poses_palindrome.npy"
    if video_path.is_file() and intrinsics_path.is_file() and extrinsics_path.is_file():
        kwargs.update(
            {
                "video_path": str(video_path),
                "intrinsics_path": str(intrinsics_path),
                "extrinsics_path": str(extrinsics_path),
            }
        )
    return kwargs


def _stream_vggt_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("lch01--StreamVGGT", "StreamVGGT"),
        str(checkpoint_root_path("StreamVGGT")),
        "lch01/StreamVGGT",
    )


def _loger_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("Junyi42--LoGeR", "LoGeR"),
        str(checkpoint_root_path("LoGeR")),
        "Junyi42/LoGeR",
    )


def _flash_world_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("imlixinyang--FlashWorld", "FlashWorld"),
        "imlixinyang/FlashWorld",
    )


def _flash_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": resolve._prefer_existing_model_ref(
                *resolve._cache_candidates("Wan-AI--Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers"),
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            )
        }
    }


def _fantasy_world_camera_fixture() -> str:
    return str(resolve._data_path("test_cases", "fantasyworld", "camera_forward.json"))


def _fantasy_world_wan21_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "FantasyWorld-Wan2.1-I2V-14B-480P",
        "acvlab--FantasyWorld-Wan2.1-I2V-14B-480P",
        fallback="acvlab/FantasyWorld-Wan2.1-I2V-14B-480P",
    )


def _fantasy_world_wan22_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
        "acvlab--FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
        fallback="acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
    )


def _fantasy_world_wan21_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": resolve._checkpoint_model_ref(
                "Wan2.1-I2V-14B-480P",
                "Wan-AI--Wan2.1-I2V-14B-480P",
                fallback="Wan-AI/Wan2.1-I2V-14B-480P",
            ),
            "moge_pretrained": resolve._checkpoint_model_ref(
                "moge-2-vitl-normal",
                "Ruicheng--moge-2-vitl-normal",
                fallback="Ruicheng/moge-2-vitl-normal",
            ),
        },
        "sample_steps": 50,
        "frames": 81,
        "fps": 16,
        "height": 336,
        "width": 592,
    }


def _fantasy_world_wan22_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": resolve._checkpoint_model_ref(
                "Wan2.2-Fun-A14B-Control-Camera",
                "alibaba-pai--Wan2.2-Fun-A14B-Control-Camera",
                fallback="alibaba-pai/Wan2.2-Fun-A14B-Control-Camera",
            ),
            "lora_path": resolve._checkpoint_model_ref(
                "Wan2.2-Fun-Reward-LoRAs",
                "alibaba-pai--Wan2.2-Fun-Reward-LoRAs",
                "Wan-AI--Wan2.2-Fun-Reward-LoRAs",
                fallback="alibaba-pai/Wan2.2-Fun-Reward-LoRAs",
            ),
            "moge_pretrained": resolve._checkpoint_model_ref(
                "moge-2-vitl-normal",
                "Ruicheng--moge-2-vitl-normal",
                fallback="Ruicheng/moge-2-vitl-normal",
            ),
        },
        "sample_steps": 50,
        "frames": 81,
        "fps": 16,
        "height": 480,
        "width": 832,
    }


def _fantasy_world_default_load_kwargs() -> dict[str, Any]:
    return _fantasy_world_wan21_load_kwargs()


def _fantasy_world_default_call_kwargs(conf_threshold: float = 1.5) -> dict[str, Any]:
    return {
        "camera_json_path": _fantasy_world_camera_fixture(),
        "fps": 16,
        "using_scale": True,
        "conf_threshold": conf_threshold,
        "stride": 4,
        "return_dict": True,
    }


def _vggt_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("facebook--VGGT-1B", "VGGT-1B"),
        "facebook/VGGT-1B",
    )


def _lagernvs_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        str(checkpoint_root_path("lagernvs_general_512")),
        *resolve._cache_candidates("facebook--lagernvs_general_512", "facebook--lagernvs-general-512"),
        "facebook/lagernvs_general_512",
    )


def _pixelsplat_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        str(checkpoint_root_path("pixelSplat", "re10k.ckpt")),
        str(checkpoint_root_path("hfd", "3DGeneration--pixelSplat", "re10k.ckpt")),
        *resolve._cache_candidates("dylanebert--pixelSplat", "3DGeneration--pixelSplat", "pixelSplat"),
        "dylanebert/pixelSplat",
    )


def _lingbot_map_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        str(checkpoint_root_path("lingbot-map", "lingbot-map-long.pt")),
        str(checkpoint_root_path("lingbot-map", "lingbot-map.pt")),
        str(checkpoint_root_path("hfd", "custom--lingbot-map", "lingbot-map-long.pt")),
        str(checkpoint_root_path("hfd", "custom--lingbot-map", "lingbot-map.pt")),
        *resolve._cache_candidates("robbyant--lingbot-map", "lingbot-map"),
        "robbyant/lingbot-map",
    )


_IN_TREE_REPO_PARTS: dict[str, tuple[str, ...]] = {
    "lagernvs": ("base_models", "three_dimensions", "general_3d", "lagernvs", "lagernvs_runtime"),
    "MVDiffusion": ("base_models", "three_dimensions", "general_3d", "mvdiffusion", "mvdiffusion_runtime"),
    "WonderJourney": ("synthesis", "visual_generation", "wonderjourney", "wonderjourney_runtime"),
    "WorldGen": ("synthesis", "visual_generation", "worldgen", "worldgen_runtime"),
}


def _in_tree_repo_ref(repo_name: str) -> str:
    parts = _IN_TREE_REPO_PARTS.get(repo_name)
    if parts is None:
        known = ", ".join(sorted(_IN_TREE_REPO_PARTS))
        raise ValueError(f"Unknown in-tree repo {repo_name!r}; known repos: {known}")
    return str(Path(__file__).resolve().parents[4] / Path(*parts))


def _three_d_four_d_runtime_call_params(*model_params: str) -> tuple[str, ...]:
    base_params = (
        "output_path",
        "python_executable",
        "source_root",
        "runtime_root",
        "repo_root",
        "return_dict",
        "gpu",
        "timeout_seconds",
    )
    deduped: list[str] = []
    for name in (*model_params, *base_params):
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def _three_d_four_d_runtime_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "repo_root",
        "source_root",
        "runtime_root",
    )


def _three_d_four_d_runtime_call_kwargs(**kwargs: Any) -> dict[str, Any]:
    call_kwargs = {
        "python_executable": resolve._worldfoundry_unified_python_executable(),
        "return_dict": True,
    }
    call_kwargs.update(kwargs)
    return call_kwargs


def _geometry_prior_call_params(*model_params: str) -> tuple[str, ...]:
    base_params = (
        "images",
        "image_path",
        "input_path",
        "video",
        "video_path",
        "output_path",
        "output_dir",
        "execute",
        "return_dict",
        "max_points",
        "focal_length",
    )
    deduped: list[str] = []
    for name in (*model_params, *base_params):
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def _geometry_prior_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "profile_id",
        "runtime_profile",
        "base_model_target",
    )


def _geometry_prior_default_call_kwargs(**kwargs: Any) -> dict[str, Any]:
    call_kwargs = {"execute": True, "return_dict": True, "max_points": 5000}
    call_kwargs.update(kwargs)
    return call_kwargs


def _geometry_prior_image_fixture() -> str:
    return str(resolve._data_path("test_cases", "images", "000.png"))


def _geometry_prior_video_fixture() -> str:
    return str(resolve._data_path("test_cases", "neoverse", "videos", "robot.mp4"))


def _cut3r_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "cut3r",
        "CUT3R",
        "liguang0115--cut3r",
        "hfd/liguang0115--cut3r",
        fallback="liguang0115/cut3r",
    )


def _worldfm_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("worldfm", "inspatio--worldfm"),
        "inspatio/worldfm",
    )


def _worldfm_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "moge_pretrained": resolve._prefer_existing_model_ref(
                *resolve._cache_candidates(
                    "moge-2-vitl-normal", "Ruicheng--moge-2-vitl-normal", "Ruicheng--moge-2-vitl"
                ),
            ),
        }
    }


def _worldfm_demo_meta_path() -> str:
    return str(resolve._data_path("test_cases", "worldfm", "meta.json"))


def _worldfm_demo_panorama_path() -> str:
    return str(resolve._data_path("test_cases", "worldfm", "mario.png"))


def _lyra1_default_call_kwargs() -> dict[str, Any]:
    return {
        "mode": "static",
        "trajectory": "zoom_in",
        "num_video_frames": 121,
        "fps": 24,
        "height": 704,
        "width": 1280,
        "seed": 1,
        "num_steps": 35,
        "guidance": 1.0,
        "num_gpus": 1,
        "movement_distance": 0.3,
        "camera_rotation": "center_facing",
        "multi_trajectory": True,
        "total_movement_distance_factor": 1.0,
        "foreground_masking": True,
        "filter_points_threshold": 0.05,
        "disable_prompt_encoder": True,
        "offload_diffusion_transformer": True,
        "offload_tokenizer": True,
        "offload_text_encoder_model": True,
        "offload_prompt_upsampler": True,
        "offload_guardrail_models": True,
        "disable_guardrail": True,
        "execute": True,
        "show_progress": True,
    }


def _lyra2_default_call_kwargs() -> dict[str, Any]:
    return {
        "fps": 16,
        "resolution": (480, 832),
        "seed": 1,
        "reconstruct_3d": False,
        "return_dict": True,
        "show_progress": True,
        "execute": True,
    }


def _lyra2_default_load_kwargs() -> dict[str, Any]:
    required_components: dict[str, Any] = {"load_runtime": True}
    checkpoint_root = None
    for candidate in (
        hfd_root_path("Lyra-2.0"),
        hfd_root_path("nvidia--Lyra-2.0"),
        checkpoint_root_path("Lyra-2.0"),
        checkpoint_root_path("Lyra-2"),
    ):
        for root in (candidate, candidate / "Lyra-2.0", candidate / "Lyra-2"):
            if (
                (root / "checkpoints" / "model").is_dir()
                and (root / "checkpoints" / "text_encoder" / "negative_prompt.pt").is_file()
                and (root / "checkpoints" / "recon" / "model.pt").exists()
            ):
                checkpoint_root = root
                break
        if checkpoint_root is not None:
            break
    if checkpoint_root is not None:
        required_components.update(
            {
                "checkpoint_dir": str(checkpoint_root / "checkpoints" / "model"),
                "negative_prompt_path": str(checkpoint_root / "checkpoints" / "text_encoder" / "negative_prompt.pt"),
                "da3_model_path_custom": str(checkpoint_root / "checkpoints" / "recon" / "model.pt"),
            }
        )
    return {"required_components": required_components}


OVERRIDES: dict[str, dict[str, Any]] = {
    "uniworld-view": {
        "display_name": "UniWorld-View",
        "category": "Video Generation",
        "summary": "Official UniView novel-view synthesis from one image or a monocular video.",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "UniWorld-View", fallback=str(checkpoint_root_path("UniWorld-View"))
        ),
        "default_input_path": str(resolve._data_path("test_cases", "uniworld_view", "images", "fruit.jpg")),
        "default_prompt": "",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_call_kwargs": {
            "mode": "single_view",
            "geometry_backend": "stream3r",
            "render_method": "hybrid",
            "height": 480,
            "width": 832,
            "keep_aspect_ratio": True,
            "num_frames": 81,
            "num_inference_steps": 8,
            "guidance_scale": 4.0,
            "fps": 16,
            "seed": 43,
            "d_phi": 50.0,
            "timeout_seconds": 21600,
        },
        "input_params": ("image_path", "video_path"),
        "call_params": (
            "prompt",
            "images",
            "video",
            "image_path",
            "video_path",
            "mode",
            "geometry_backend",
            "render_method",
            "mosca_ws",
            "height",
            "width",
            "keep_aspect_ratio",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "fps",
            "seed",
            "stride",
            "traj_type",
            "d_phi",
            "d_theta",
            "x_offset",
            "y_offset",
            "z_offset",
            "radius_scale",
            "low_gpu_memory_mode",
            "output_path",
            "timeout_seconds",
            "plan_only",
            "return_dict",
        ),
        "load_params": (
            "python_executable",
            "checkpoint_root",
            "transformer_path",
            "model_name",
            "blip_path",
            "moge_path",
            "sam2_checkpoint",
            "segnet_path",
            "stream3r_path",
            "lora_path",
            "device",
        ),
        "tags": ("novel-view", "camera-control", "image-to-video", "video-to-video", "official-runtime"),
    },
    "worldcam": {
        "display_name": "WorldCam",
        "default_model_ref": _worldcam_default_ref,
        "default_load_kwargs": _worldcam_default_load_kwargs,
        "default_call_kwargs": _worldcam_default_call_kwargs,
        "default_prompt": "a first-person view within an industrial setting, likely a power plant, with concrete buildings, chain-link fences, and parked vehicles.",
        "call_params": (
            "prompt",
            "video",
            "input_path",
            "video_path",
            "output_path",
            "fps",
            "return_dict",
            "height",
            "width",
            "intrinsics",
            "extrinsics",
            "intrinsics_path",
            "extrinsics_path",
            "conditioning_frames",
            "num_ar_steps",
            "num_inference_steps",
            "negative_prompt",
            "cfg_scale",
            "seed",
            "long_term_memory_start_step",
            "long_term_memory_num_clips",
            "long_term_memory_ref_indices",
            "attention_sink_inference",
            "trim_conditioning",
            "tiled",
        ),
        "stream_params": (),
        "tags": ("interactive-world", "official-runtime", "video-to-world"),
    },
    "lyra-1": {
        "display_name": "Lyra-1",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_model_ref": lambda: str(checkpoint_root_path()),
        "default_input_path": str(resolve._data_path("test_cases", "lyra", "Lyra-1", "00172.png")),
        "default_prompt": "",
        "default_interactions": ("zoom_in", "left", "right"),
        "default_call_kwargs": _lyra1_default_call_kwargs,
        "call_params": (
            "images",
            "videos",
            "interactions",
            "prompt",
            "mode",
            "trajectory",
            "reconstruct_3d",
            "output_dir",
            "return_dict",
            "num_video_frames",
            "fps",
            "height",
            "width",
            "seed",
            "num_steps",
            "guidance",
            "num_gpus",
            "movement_distance",
            "camera_rotation",
            "multi_trajectory",
            "total_movement_distance_factor",
            "vipe_path",
            "vipe_starting_frame_idx",
            "filter_points_threshold",
            "foreground_masking",
            "center_depth_quantile",
            "flip_supervision",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
            "disable_prompt_encoder",
            "disable_guardrail",
            "show_progress",
            "execute",
            "plan_path",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra1",),
        "tags": ("novel-view", "camera-control", "official-runtime"),
    },
    "lyra": {
        "display_name": "Lyra",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_prompt": (
            "A slow, steady camera push forward through a coherent static 3D world with stable geometry."
        ),
        "default_interactions": ("forward",),
        "default_load_kwargs": _lyra2_default_load_kwargs,
        "default_call_kwargs": _lyra2_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "camera_path",
            "prompt",
            "region_hint",
            "fps",
            "resolution",
            "reconstruct_3d",
            "output_dir",
            "seed",
            "return_dict",
            "show_progress",
            "execute",
            "guidance",
            "shift",
            "num_sampling_step",
            "offload",
            "offload_when_prompt",
        ),
        "supports_stream": True,
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "checkpoint_dir",
            "negative_prompt_path",
            "da3_model_path_custom",
            "load_runtime",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra2", "nvidia/lyra"),
        "tags": (
            "novel-view",
            "camera-control",
            "official-runtime",
        ),
    },
    "lyra-2": {
        "display_name": "Lyra-2",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_prompt": (
            "A slow, steady camera push forward through a coherent static 3D world with stable geometry."
        ),
        "default_interactions": ("forward",),
        "default_load_kwargs": _lyra2_default_load_kwargs,
        "default_call_kwargs": _lyra2_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "camera_path",
            "prompt",
            "region_hint",
            "fps",
            "resolution",
            "reconstruct_3d",
            "output_dir",
            "seed",
            "return_dict",
            "show_progress",
            "execute",
            "guidance",
            "shift",
            "num_sampling_step",
            "offload",
            "offload_when_prompt",
        ),
        "supports_stream": True,
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "checkpoint_dir",
            "negative_prompt_path",
            "da3_model_path_custom",
            "load_runtime",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra2",),
        "tags": (
            "novel-view",
            "camera-control",
            "official-runtime",
        ),
    },
    "cosmos-predict2": {
        "display_name": "Cosmos Predict2 2B Video2World",
        "default_model_ref": _cosmos_predict2_default_ref,
        "default_load_kwargs": _cosmos_predict2_default_load_kwargs,
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_prompt": (
            "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, "
            "with coherent motion, stable facial details, and consistent nighttime architecture."
        ),
        "default_task_type": "image-to-world",
        "default_call_kwargs": {
            "guidance_scale": 7.0,
            "num_inference_steps": 35,
            "fps": 16,
            "num_frames": 93,
            "height": 704,
            "width": 1280,
            "seed": 0,
            "num_latent_conditional_frames": 1,
            "output_type": "video",
            "return_dict": True,
        },
        "tags": ("image-to-world", "video-to-world", "cosmos", "native-diffusion"),
    },
    "cosmos-predict2.5": {
        "display_name": "Cosmos Predict2.5",
        "default_model_ref": _cosmos_predict2p5_default_ref,
        "default_load_kwargs": _cosmos_predict2p5_default_load_kwargs,
        "default_prompt": (
            "A nighttime city bus terminal gradually shifts from stillness to subtle movement, "
            "with realistic lighting, stable geometry, and smooth camera motion."
        ),
        "default_task_type": "text-to-world",
        "default_call_kwargs": _cosmos_predict2p5_default_call_kwargs,
        "aliases": ("cosmos-predict2.5", "cosmos-predict-2.5"),
        "tags": ("text-to-world", "image-to-world", "video", "cosmos"),
    },
    "cosmos-transfer-2.5": {
        "default_model_ref": _cosmos_transfer2p5_default_ref,
        "default_load_kwargs": _cosmos_transfer2p5_default_load_kwargs,
        "default_input_path": str(resolve._data_path("test_cases", "longcat_video", "motorcycle.mp4")),
        "default_prompt": (
            "A cinematic night scene follows the control video edges with stable geometry, "
            "realistic lighting, and smooth coherent motion."
        ),
        "default_task_type": "video-to-world",
        "default_call_kwargs": _cosmos_transfer2p5_default_call_kwargs,
        "aliases": ("cosmos-transfer2.5", "cosmos-transfer2p5"),
        "tags": ("video-to-world", "controlled-video", "edge-control", "cosmos"),
    },
    "cosmos3": {
        "display_name": "Cosmos3",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref_at_revision(
            "Cosmos3-Nano",
            "nvidia/Cosmos3-Nano",
            "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
        ),
        "default_load_kwargs": {
            "revision": "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
            "load_sound_tokenizer": True,
        },
        "default_prompt": "A robot arm is cleaning a plate in the kitchen",
        "default_task_type": "text-to-video",
        "default_call_kwargs": {
            "fps": 24,
            "guidance_scale": 6.0,
            "height": 720,
            "num_inference_steps": 35,
            "num_frames": 189,
            "output_type": "video",
            "seed": 0,
            "width": 1280,
        },
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "torch_dtype",
            "device_map",
            "enable_safety_checker",
            "load_sound_tokenizer",
            "revision",
        ),
        "extra_variants": (
            {
                "variant_id": "cosmos3-super",
                "label": "Cosmos 3 Super",
                "status": "checkpoint-required",
                "aliases": ("cosmos-3-super",),
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref_at_revision(
                            "Cosmos3-Super",
                            "nvidia/Cosmos3-Super",
                            "e0262be9d8f7586bc24c069a2aed2b665bdff266",
                        ),
                        "required": True,
                        "status": "official",
                    },
                ),
                "load_kwargs": {
                    "variant_id": "cosmos3-super",
                    "profile_id": "cosmos3-super",
                    "revision": "e0262be9d8f7586bc24c069a2aed2b665bdff266",
                    "runtime_profile": "cosmos3-super",
                    "device_map": "balanced",
                    "load_sound_tokenizer": True,
                },
                "notes": (
                    "Requires a high-memory multi-GPU configuration; the default safe map shards decoder layers only.",
                ),
            },
        ),
        "aliases": ("cosmos3-nano", "cosmos3-super", "cosmos-3", "cosmos-3-nano", "cosmos-3-super"),
        "tags": (
            "text-to-image",
            "text-to-video",
            "image-to-video",
            "video-to-video",
            "action-policy",
            "forward-dynamics",
            "inverse-dynamics",
            "world-generation",
            "video",
            "cosmos",
        ),
        "notes": "Uses the in-tree official Diffusers generator. Nano is the default; Super is selectable as a separate variant. Exact-revision Nano task-matrix and safety-enabled smokes plus a four-A100 Super T2I smoke passed; official-quality/full-resolution parity remains pending.",
    },
    "inspatio-world-1p5": {
        "display_name": "InSpatio World 1.5",
        "aliases": ("inspatio-world-v1.5", "inspatio-world-1.5"),
        "category": "Image-to-Video",
        "supports_stream": False,
        "default_task_type": "image-to-video",
        "suggested_task_types": ("image-to-video", "video-to-video"),
        "default_model_ref": "inspatio/world-1.5",
        "default_call_kwargs": {"seed": 0, "return_dict": True},
        "call_params": (
            "images",
            "videos",
            "scene_dir",
            "prompt",
            "traj_txt_path",
            "output_dir",
            "output_path",
            "seed",
            "return_dict",
            "return_frames",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "wan_model_path",
            "da3_model_path",
            "device",
            "weight_dtype",
            "compile_dit",
        ),
        "tags": ("world-model", "novel-view", "camera-control", "multi-image", "native-runtime"),
        "notes": "One image, four images or video with OpenCV target Tcw; released resolution is 832x480.",
    },
    "inspatio-world": {
        "display_name": "InSpatio-World",
        "category": "Video-to-Video",
        "supports_stream": False,
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": _inspatio_world_default_ref,
        "default_load_kwargs": _inspatio_world_default_load_kwargs,
        "default_call_kwargs": {"skip_step1": True, "skip_step2": False, "skip_step3": False},
        "call_params": (
            "videos",
            "prompt",
            "traj_txt_path",
            "output_dir",
            "return_dict",
            "skip_step1",
            "skip_step2",
            "skip_step3",
            "relative_to_source",
            "rotation_only",
            "adaptive_frame",
            "freeze_repeat",
            "freeze_frame",
            "compile_dit",
        ),
        "tags": ("novel-view", "video-to-video", "camera-control", "trajectory"),
        "notes": "Video-to-video model that re-renders an input clip along a new camera trajectory.",
    },
    "lingbot-map": {
        "category": "3D Scene",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "streaming-3d-reconstruction",
        "default_model_ref": _lingbot_map_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("point_cloud_generation",),
        "call_params": (
            "images",
            "interactions",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("3d", "point-reconstruction", "lingbot"),
        "notes": "Streaming 3D map reconstruction from images; not video generation.",
    },
    "worldfm": {
        "runtime_kind": "worldfm",
        "category": "3D Scene",
        "default_model_ref": _worldfm_default_ref,
        "default_load_kwargs": _worldfm_default_load_kwargs,
        "default_interactions": ("forward", "left", "camera_r"),
        "default_call_kwargs": {
            "fps": 30,
            "save_mode": "video",
            "meta_path": _worldfm_demo_meta_path(),
            "panorama_path": _worldfm_demo_panorama_path(),
            "return_dict": True,
        },
        "tags": ("novel-view", "poses", "scene-context"),
        "notes": "Supports pose-matrix JSON directly. Navigation tokens can also be converted into pose trajectories.",
    },
    "vggt": {
        "display_name": "VGGT",
        "runtime_kind": "two_stage_3dgs",
        "category": "3D Scene",
        "default_model_ref": _vggt_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_task_type": "vggt_two_stage_3dgs",
        "suggested_task_types": ("vggt_two_stage_3dgs", "vggt_base", "official"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {
            "point_conf_threshold": 0.2,
            "resolution": 518,
            "preprocess_mode": "crop",
            "image_width": 704,
            "image_height": 480,
            "fps": 12,
        },
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "camera_view",
            "task_type",
            "output_dir",
            "output_path",
            "return_dict",
            "point_conf_threshold",
            "resolution",
            "preprocess_mode",
            "image_width",
            "image_height",
            "fps",
            "conf_thres",
            "frame_filter",
            "mask_black_bg",
            "mask_white_bg",
            "show_cam",
            "prediction_mode",
            "output_name",
        ),
        "tags": ("3dgs", "reconstruction", "camera-control"),
    },
    "fantasyworld": {
        "display_name": "FantasyWorld",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan21_default_ref,
        "default_load_kwargs": _fantasy_world_default_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": lambda: _fantasy_world_default_call_kwargs(conf_threshold=1.0),
        "call_params": (
            "images",
            "end_image",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "using_scale",
            "conf_threshold",
            "stride",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "cfg_scale",
            "timestep_boundary",
            "frames",
            "fps",
            "height",
            "width",
            "base_seed",
            "high_model_device",
            "low_model_device",
            "moge_device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("fantasyworld-wan21-default",),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "fantasyworld-wan22": {
        "display_name": "FantasyWorld Wan2.2",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan22_default_ref,
        "default_load_kwargs": _fantasy_world_wan22_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": _fantasy_world_default_call_kwargs,
        "call_params": (
            "images",
            "end_image",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "using_scale",
            "conf_threshold",
            "stride",
            "neg_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "cfg_scale",
            "timestep_boundary",
            "frames",
            "fps",
            "height",
            "width",
            "base_seed",
            "high_model_device",
            "low_model_device",
            "moge_device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "fantasyworld-wan21": {
        "display_name": "FantasyWorld Wan2.1",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan21_default_ref,
        "default_load_kwargs": _fantasy_world_wan21_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": lambda: _fantasy_world_default_call_kwargs(conf_threshold=1.0),
        "call_params": (
            "images",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "seed",
            "using_scale",
            "conf_threshold",
            "stride",
            "neg_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "sample_guide_scale",
            "frames",
            "fps",
            "height",
            "width",
            "start_index",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "flashworld": {
        "display_name": "FlashWorld",
        "category": "3D Scene",
        "default_model_ref": _flash_world_default_ref,
        "default_load_kwargs": _flash_world_default_load_kwargs,
        "default_interactions": ("forward", "camera_l", "right", "camera_zoom_in"),
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 15,
            "image_height": 480,
            "image_width": 704,
            "return_video": True,
        },
        "tags": ("3dgs", "spz", "world-model", "camera-control"),
        "notes": "Set WORLDFOUNDRY_DISABLE_SAGEATTENTION=1 on driver/Triton stacks where SageAttention kernels fail.",
    },
    "infinite-vggt": {
        "display_name": "Infinite VGGT",
        "runtime_kind": "default",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _stream_vggt_default_ref,
        "default_task_type": "3d-reconstruction",
        "default_input_path": str(resolve._data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_call_kwargs": {"output_format": "ply"},
        "call_params": (
            "data_path",
            "input_path",
            "images",
            "image_path",
            "interaction",
            "output_format",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "representation_path",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("stream-vggt", "streamvggt", "lch01/streamvggt"),
        "tags": ("3d", "reconstruction", "streaming-3d"),
        "notes": "Runs the in-tree StreamVGGT/Infinite-VGGT point-cloud reconstruction path.",
    },
    "pixelsplat": {
        "display_name": "pixelSplat",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _pixelsplat_default_ref,
        "default_task_type": "3d-reconstruction",
        "call_params": (
            "images",
            "input_path",
            "interactions",
            "output_path",
            "output_dir",
            "return_dict",
            "checkpoint_path",
            "index_path",
            "dataset_roots",
            "experiment",
            "overrides",
            "python_executable",
        ),
        "load_params": (
            "checkpoint_path",
            "ckpt_path",
            "model_path",
            "pretrained_model_path",
            "dataset_roots",
            "index_path",
            "experiment",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("pixel-splat", "dylanebert/pixelsplat"),
        "tags": ("3dgs", "reconstruction"),
        "notes": "Runs the in-tree pixelSplat runtime against the packaged demo index and staged re10k checkpoint.",
    },
    "vggt-omega": {
        "display_name": "VGGT-Omega",
        "runtime_kind": "default",
        "category": "3D Scene",
        "default_model_ref": resolve._prefer_existing_model_ref(
            *resolve._cache_candidates("facebook--VGGT-Omega", "VGGT-Omega"),
            str(resolve._project_root() / "assets" / "checkpoints" / "facebook__VGGT-Omega"),
            "facebook/VGGT-Omega",
        ),
        "default_task_type": "vggt_omega_official_scene_export",
        "suggested_task_types": ("vggt_omega_official_scene_export", "vggt_base"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {
            "task_type": "official",
            "image_resolution": 512,
            "preprocess_mode": "crop",
            "video_sample_fps": 1.0,
            "max_points_k": 200,
        },
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "camera_view",
            "task_type",
            "output_dir",
            "return_dict",
            "image_resolution",
            "preprocess_mode",
            "patch_size",
            "video_sample_fps",
            "conf_thres",
            "mask_black_bg",
            "mask_white_bg",
            "show_cam",
            "max_points_k",
            "output_name",
        ),
        "aliases": ("vggt_omega", "facebook/vggt-omega"),
        "tags": ("3dgs", "reconstruction", "camera-control", "omega"),
        "notes": "Uses the official VGGT-Omega scene-export path and emits GLB artifacts from local checkpoint trees.",
    },
    "lagernvs": {
        "display_name": "LagrNVS",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("lagernvs"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "novel_view_synthesis",
        "suggested_task_types": ("novel_view_synthesis",),
        "default_interactions": ("camera_path",),
        "default_call_kwargs": lambda: _three_d_four_d_runtime_call_kwargs(
            video_length=16,
            fps=25,
            model_repo=_lagernvs_default_ref(),
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "images",
            "video_length",
            "fps",
            "model_repo",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "aliases": ("lager-nvs", "facebook/lagernvs_general_512", "facebookresearch/lagernvs"),
        "tags": ("novel-view", "nvs", "vggt", "gated-checkpoint"),
        "env_hints": ("HF_TOKEN",),
        "notes": "Uses the local official LagrNVS minimal_inference.py entrypoint when weights and dependencies are staged.",
    },
    "mvdiffusion": {
        "display_name": "MVDiffusion",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("MVDiffusion"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted.",
        "default_input_path": str(resolve._data_path("test_cases", "mvdiffusion", "outpaint_example.png")),
        "default_task_type": "mvdiffusion_demo",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(
            gen_video=True,
            steps=20,
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "images",
            "text_path",
            "gen_video",
            "steps",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("multi-view", "diffusion", "3d"),
        "notes": "Runs the official MVDiffusion demo.py entrypoint from the local official repo. The 20-step default produces a visually structured panorama; one step is suitable only for runtime smoke testing.",
    },
    "wonderjourney": {
        "display_name": "WonderJourney",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("WonderJourney"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a village landscape",
        "default_task_type": "wonderjourney_run",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(
            config=str(resolve._data_path("models", "runtime", "configs", "wonderjourney", "village.yaml"))
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "config",
            "frames",
            "num_scenes",
            "num_keyframes",
            "save_fps",
            "skip_interp",
            "rotation_path",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("world-generation", "3d", "camera-navigation"),
        "notes": "Runs the official WonderJourney run.py entrypoint from the local official repo.",
    },
    "wonderworld": {
        "display_name": "WonderWorld",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: str(checkpoint_root_path()),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a village landscape",
        "default_task_type": "image-to-3d-world",
        "default_call_kwargs": {},
        "call_params": (
            "prompt",
            "images",
            "input_path",
            "camera_path",
            "interactions",
            "seed",
            "depth_steps",
            "inpaint_steps",
            "normal_steps",
            "refinement_steps",
            "focal_length",
            "gen_layer",
            "negative_prompt",
        ),
        "load_params": (
            "checkpoint_root",
            "inpainting_path",
            "depth_path",
            "normals_path",
            "oneformer_path",
            "repvit_checkpoint",
            "dtype",
        ),
        "tags": ("world-generation", "3dgs", "camera-navigation"),
        "notes": "Native offline layered Gaussian generation; explicit image and local shared component weights are required.",
    },
    "worldgen": {
        "display_name": "WorldGen",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("WorldGen"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a small furnished room",
        "default_task_type": "worldgen_inference",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(save_scene=True, low_vram=False),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "images",
            "pano_image",
            "viewer",
            "return_mesh",
            "save_scene",
            "low_vram",
            "resolution",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("world-generation", "3d", "mesh"),
        "notes": "Runs the official WorldGen demo.py entrypoint from the local official repo.",
    },
    "cut3r": {
        "display_name": "CUT3R",
        "runtime_kind": "two_stage_3dgs",
        "category": "3D Scene",
        "default_model_ref": _cut3r_default_ref,
        "aliases": ("cut3r-512", "cut3r_512_dpt_4_64"),
        "default_task_type": "cut3r_two_stage_3dgs",
        "suggested_task_types": ("cut3r_two_stage_3dgs", "cut3r_base", "cut3r_official_export"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {"image_width": 704, "image_height": 480, "size": 512, "vis_threshold": 1.5},
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "task_type",
            "size",
            "vis_threshold",
            "revisit",
            "update",
            "use_pose",
            "frames_per_interaction",
            "camera_radius",
            "camera_yaw",
            "camera_pitch",
            "image_width",
            "image_height",
            "num_orbit_frames",
            "yaw_step",
        ),
        "tags": ("3dgs", "reconstruction"),
    },
    "pi3": {
        "display_name": "Pi3",
        "runtime_kind": "pointcloud_nav",
        "category": "3D Scene",
        "default_model_ref": _pi3_default_ref,
        "default_task_type": "reconstruction",
        "suggested_task_types": ("reconstruction", "render_view", "render_trajectory"),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"mode": "pi3x"},
        "default_call_kwargs": {"interval": 10},
        "call_params": (
            "images",
            "videos",
            "image_path",
            "video_path",
            "task_type",
            "interactions",
            "camera_view",
            "visualize_ops",
            "interval",
            "frames_per_interaction",
            "hold_frames",
            "output_path",
            "return_dict",
        ),
        "tags": ("point-cloud", "render-view", "trajectory"),
    },
    "loger": {
        "display_name": "LoGeR",
        "runtime_kind": "pointcloud_nav",
        "category": "3D Scene",
        "default_model_ref": _loger_default_ref,
        "default_task_type": "reconstruction",
        "suggested_task_types": ("reconstruction", "render_view", "render_trajectory"),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"mode": "loger"},
        "default_call_kwargs": {"interval": 10},
        "call_params": (
            "images",
            "videos",
            "image_path",
            "video_path",
            "task_type",
            "interactions",
            "camera_view",
            "visualize_ops",
            "interval",
            "frames_per_interaction",
            "hold_frames",
            "output_path",
            "return_dict",
        ),
        "tags": ("point-cloud", "render-view", "trajectory"),
    },
    "depth-anything-v2": {
        "display_name": "Depth Anything 2",
        "category": "Depth / Geometry",
        "default_model_ref": _depth_anything_v2_default_ref,
        "aliases": ("da2", "depth-anything-2", "depth-anything2", "depthanything2"),
        "tags": ("depth", "geometry"),
        "notes": "Relative depth wrapper around the vendored official Depth-Anything-V2 runtime.",
    },
    "depth-anything-v3": {
        "display_name": "Depth Anything 3",
        "category": "Depth / Geometry",
        "default_model_ref": _depth_anything_v3_default_ref,
        "suggested_task_types": ("official_soh_export", "single_view_depth", "multi_view_depth", "video_depth"),
        "default_call_kwargs": {"export_format": "mini_npz", "process_res": 504, "max_frames": 2},
        "call_params": (
            "input_data",
            "images",
            "videos",
            "video",
            "interactions",
            "output_dir",
            "return_dict",
            "export_format",
            "export_feat_layers",
            "process_res",
            "process_res_method",
            "use_ray_pose",
            "infer_gs",
            "ref_view_strategy",
            "align_to_input_ext_scale",
            "render_exts",
            "render_ixts",
            "render_hw",
            "data_type",
            "max_frames",
            "frame_stride",
            "conf_thresh_percentile",
            "num_max_points",
            "show_cameras",
            "feat_vis_fps",
            "export_kwargs",
        ),
        "aliases": ("da3", "depth-anything-3", "depth-anything3", "depthanything3"),
        "tags": ("depth", "geometry"),
    },
    "hy-world-2p0": {
        "display_name": "HY-World-2.0",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": resolve._prefer_existing_model_ref(
            *resolve._cache_candidates("tencent--HY-World-2.0", "HY-World-2.0"),
            "tencent/HY-World-2.0",
        ),
        "default_input_path": str(resolve._data_path("test_cases", "images")),
        "default_prompt": "Reconstruct the scene and surface geometry from the image sequence.",
        "default_call_kwargs": {
            "task": "worldrecon",
            "confidence_percentile": 10.0,
            "edge_normal_threshold": 5.0,
            "edge_depth_threshold": 0.03,
            "apply_confidence_mask": True,
            "apply_edge_mask": True,
            "apply_sky_mask": False,
            "save_points": True,
            "save_depth": True,
            "save_normal": True,
            "save_gs": True,
            "save_camera": True,
            "save_rendered": False,
            "save_colmap": True,
        },
        "input_params": ("image_path",),
        "call_params": (
            "task",
            "image_path",
            "images",
            "input",
            "output_path",
            "return_dict",
            "confidence_percentile",
            "edge_normal_threshold",
            "edge_depth_threshold",
            "apply_confidence_mask",
            "apply_edge_mask",
            "apply_sky_mask",
            "save_points",
            "save_depth",
            "save_normal",
            "save_gs",
            "save_camera",
            "save_rendered",
            "save_colmap",
            "prior_cam_path",
            "prior_depth_path",
        ),
        "load_params": (
            "task",
            "backend",
            "subfolder",
            "torch_dtype",
            "device_map",
        ),
        "aliases": ("hy-world-2.0", "hyworld2.0", "hyworld-2.0", "hy-world2.0"),
        "tags": ("reconstruction", "world-mirror"),
    },
    "hunyuanworld-mirror": {
        "display_name": "HunyuanWorld-Mirror",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": resolve._prefer_existing_model_ref(
            *resolve._cache_candidates("tencent--HunyuanWorld-Mirror", "HunyuanWorld-Mirror"),
            "tencent/HunyuanWorld-Mirror",
        ),
        "default_input_path": str(resolve._data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_prompt": "Reconstruct the scene and surface geometry from the image sequence.",
        "default_call_kwargs": {
            "confidence_percentile": 10.0,
            "edge_normal_threshold": 5.0,
            "edge_depth_threshold": 0.03,
            "apply_confidence_mask": True,
            "apply_edge_mask": True,
            "apply_sky_mask": False,
            "save_pointmap": True,
            "save_depth": True,
            "save_normal": True,
            "save_gs": True,
            "save_rendered": False,
            "save_colmap": True,
        },
        "input_params": ("image_path",),
        "call_params": (
            "image_path",
            "images",
            "input",
            "output_path",
            "return_dict",
            "confidence_percentile",
            "edge_normal_threshold",
            "edge_depth_threshold",
            "apply_confidence_mask",
            "apply_edge_mask",
            "apply_sky_mask",
            "cond_flags",
            "save_pointmap",
            "save_depth",
            "save_normal",
            "save_gs",
            "save_rendered",
            "save_colmap",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "local_model_path",
            "output_path",
            "device",
            "local_files_only",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("hunyuan-mirror", "hy-world-mirror", "worldmirror", "world-mirror"),
        "tags": ("reconstruction", "world-mirror", "point-cloud", "gaussian-splat"),
    },
}

OVERRIDES.update(
    {
        "moverse": {
            "display_name": "MoVerse",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "default_model_ref": _moverse_default_ref,
            "default_load_kwargs": _moverse_default_load_kwargs,
            "default_task_type": "image-to-navigable-world",
            "default_input_path": str(resolve._data_path("test_cases", "images", "000.png")),
            "default_prompt": (
                "Move forward through the scene while preserving geometry, appearance, and coherent camera motion."
            ),
            "default_call_kwargs": {"num_frames": 161, "fps": 16, "return_dict": True},
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "moverse_path",
                "flux_path",
                "da3_path",
                "wan_path",
                "python_executable",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "prompt",
                "images",
                "interactions",
                "output_path",
                "num_frames",
                "fps",
                "trajectory_path",
                "return_dict",
            ),
            "tags": ("image-to-world", "camera-control", "gaussian-splat", "in-tree-runtime"),
        },
        "dap": {
            "display_name": "Depth Any Panoramas",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "panoramic-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                input_size=518,
            ),
            "call_params": _geometry_prior_call_params("input_size"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("depth-any-panoramas", "insta360-dap"),
            "tags": ("depth", "geometry", "panorama", "prior"),
        },
        "depth-anything-v2-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "monocular-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                model="vitl",
            ),
            "call_params": _geometry_prior_call_params("model"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("dav2-prior", "depth-anything-v2"),
            "tags": ("depth", "geometry", "prior"),
        },
        "depth-anything-v3-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                focal_length=720.0,
            ),
            "call_params": _geometry_prior_call_params("focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("dav3-prior", "da3metric-large-prior"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "geocalib-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "camera-calibration",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                camera_model="pinhole",
            ),
            "call_params": _geometry_prior_call_params("camera_model"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("geocalib", "cvg-geocalib"),
            "tags": ("camera-calibration", "geometry", "prior"),
        },
        "metric3d-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                model="giant2",
                version=2,
                focal_length=720.0,
            ),
            "call_params": _geometry_prior_call_params("model", "version", "focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("metric3d", "yvanyin-metric3d"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "prior-depth-anything": {
            "display_name": "Prior Depth Anything",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "depth-completion",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
            ),
            "call_params": _geometry_prior_call_params("prompt_depth_path", "prior_depth_path"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("priorda", "prior-depth-anything-prior"),
            "tags": ("depth", "geometry", "depth-completion", "prior"),
        },
        "track-anything-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_prompt": "robot.",
            "default_task_type": "video-object-segmentation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_video_fixture(),
                max_frames=4,
            ),
            "call_params": _geometry_prior_call_params("max_frames"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("segment-and-track-anything", "track-anything", "aot-deaot-l"),
            "tags": ("tracking", "segmentation", "geometry", "prior"),
        },
        "unidepth-v2-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                type="l",
                focal_length=720.0,
                max_points=50_000,
            ),
            "call_params": _geometry_prior_call_params("type", "focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("unidepth-v2", "unidepth-v2-vitl14", "unidepth_v2_vitl14"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "video-depth-anything-prior": {
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "video-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_video_fixture(),
                model="vitl",
                input_size=518,
                max_frames=4,
            ),
            "call_params": _geometry_prior_call_params("model", "input_size", "max_frames"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("video-depth-anything", "vda-prior"),
            "tags": ("depth", "geometry", "video-depth", "prior"),
        },
    }
)
