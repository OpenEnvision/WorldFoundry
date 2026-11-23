"""Curated overrides and defaults for text/image/video-to-video generation models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    checkpoint_root_candidates,
    checkpoint_root_path,
    hfd_root_path,
)

from .. import resolve
from ..params import (
    _SANA_IMAGE_RUNNERS,
    DISPATCH_LOAD_PARAMS,
    _cogvideox_aliases,
    _cogvideox_workspace_variants,
    _native_diffusion_video_call_params,
    _native_diffusion_video_load_params,
    _official_video_call_params,
    _official_video_load_params,
    _sana_folded_studio_model_ids,
    _t2v_turbo_base_ref,
    _t2v_turbo_default_load_kwargs,
    _wan_2p1_i2v_default_ref,
    _wan_2p1_t2v_default_ref,
)

SANA_DEFAULT_IMAGE_VARIANT_ID = "sana-1600m-1024px-bf16"


def _hunyuanvideo15_default_load_kwargs(*, image_to_video: bool) -> dict[str, Any]:
    """Compose the complete HunyuanVideo 1.5 demo from staged local roles.

    The release checkpoint and the HY-WorldPlay checkout are complementary in
    common offline installations: the former contains the full non-distilled
    720p DiTs and materializer-facing text resources, while the latter contains
    the SigLIP assets. Keep those roles explicit instead of copying tens of
    gigabytes or silently falling back to the available 480p model.
    """

    checkpoint_ref = resolve._checkpoint_model_ref(
        "tencent--HunyuanVideo-1.5",
        "HunyuanVideo-1.5",
        fallback="tencent/HunyuanVideo-1.5",
    )
    transformer_dir = "720p_i2v" if image_to_video else "720p_t2v"
    weights_root = resolve._local_root_with_files(
        (
            checkpoint_ref,
            *checkpoint_root_candidates("tencent--HunyuanVideo-1.5"),
            *checkpoint_root_candidates("HunyuanVideo-1.5"),
        ),
        (
            f"transformer/{transformer_dir}/config.json",
            f"transformer/{transformer_dir}/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )

    resource_ref = resolve._checkpoint_model_ref(
        "HunyuanVideo-1.5",
        "tencent--HunyuanVideo-1.5",
        fallback=checkpoint_ref,
    )
    worldplay_root = resolve._project_root().parent / "world_models" / "hy-worldplay" / "HunyuanVideo-1.5"
    resources_root = resolve._local_root_with_files(
        (
            resource_ref,
            worldplay_root,
            weights_root,
        ),
        (
            "config.json",
            "text_encoder/llm/config.json",
            "text_encoder/byt5-small/config.json",
            "text_encoder/byt5-small/model.safetensors",
            "text_encoder/Glyph-SDXL-v2/checkpoints/model.safetensors",
            "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
            "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
        ),
    )

    checkpoint_overrides: dict[str, str] = {}
    if weights_root:
        checkpoint_overrides.update({"transformer": weights_root, "vae": weights_root})
    if resources_root:
        checkpoint_overrides["resources"] = resources_root
        if image_to_video:
            vision_root = resolve._local_root_with_files(
                (
                    Path(resource_ref) / "vision_encoder" / "siglip",
                    worldplay_root / "vision_encoder" / "siglip",
                    Path(resources_root) / "vision_encoder" / "siglip",
                ),
                (
                    "image_encoder/config.json",
                    "image_encoder/model.safetensors",
                    "feature_extractor/preprocessor_config.json",
                ),
            )
            if vision_root:
                checkpoint_overrides["vision"] = vision_root

    load_kwargs: dict[str, Any] = {"attention_backend": "flash"}
    if checkpoint_overrides:
        load_kwargs["checkpoint_overrides"] = checkpoint_overrides
    return load_kwargs


def _stable_video_infinity_default_ref() -> str:
    filename = "version-2.0/SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors"
    return resolve._checkpoint_model_ref(
        f"vita-video-gen--svi-model/{filename}",
        f"epfl-vita--svi-model/{filename}",
        fallback="epfl-vita/svi-model",
    )


def _stable_video_infinity_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_dir": resolve._checkpoint_model_ref(
            "Wan2.1-I2V-14B-480P",
            "Wan-AI--Wan2.1-I2V-14B-480P",
            fallback="Wan-AI/Wan2.1-I2V-14B-480P",
        ),
        "svi_lora_path": _stable_video_infinity_default_ref(),
        "dtype": "bfloat16",
        "enable_vram_management": True,
    }


def _hydra_default_ref() -> str:
    checkpoint = resolve._checkpoint_model_ref("H-EmbodVis--HyDRA/hydra.ckpt")
    if checkpoint and Path(checkpoint).is_file():
        return checkpoint
    from worldfoundry.synthesis.visual_generation.hydra.worldfoundry_runtime import default_hydra_checkpoint

    return str(default_hydra_checkpoint())


def _hydra_default_load_kwargs() -> dict[str, str]:
    base = resolve._checkpoint_model_ref("Wan-AI--Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B")
    if base and Path(base).is_dir():
        return {"base_model_path": base}
    from worldfoundry.synthesis.visual_generation.hydra.worldfoundry_runtime import default_hydra_base_model

    return {"base_model_path": str(default_hydra_base_model(checkpoint_root_path() / "Wan2.1-T2V-1.3B"))}


def _vmem_default_load_kwargs() -> dict[str, Any]:
    runtime_root = str(resolve._package_root() / "synthesis" / "visual_generation" / "vmem" / "vmem_runtime")
    surfel_model_path = resolve._checkpoint_model_ref(
        "cut3r",
        "CUT3R",
        "liguang0115--cut3r",
        "hfd/liguang0115--cut3r",
        fallback="liguang0115/cut3r",
    )
    return {
        "required_components": {
            "runtime_root": runtime_root,
            "surfel_model_path": surfel_model_path,
        }
    }


def _vmem_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "vmem",
        "VMem",
        "liguang0115--vmem",
        "hfd/liguang0115--vmem",
        fallback="liguang0115/vmem",
    )


def _wan_2p1_i2v_720p_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan-AI--Wan2.1-I2V-14B-720P",
        "Wan2.1-I2V-14B-720P",
        fallback="Wan-AI/Wan2.1-I2V-14B-720P",
    )


def _dualcamctrl_base_default_ref() -> str:
    repo_id = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    model_root = os.getenv("WORLDFOUNDRY_MODEL_DIR", "").strip()
    candidates: list[str] = []
    if model_root:
        candidates.append(str(Path(model_root) / "alibaba-pai" / "Wan2.1-Fun-V1.1-1.3B-Control-Camera"))
    candidates.extend(
        resolve._cache_candidates(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            "Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        )
    )
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_dir() and any(path.glob("diffusion_pytorch_model*.safetensors")):
            return str(path)
    return repo_id


def _dualcamctrl_default_load_kwargs() -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {
        "base_model_repo": "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        "tokenizer_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "dualcamctrl_repo": "FayeHongfeiZhang/DualCamCtrl",
        "checkpoint_name": "checkpoints/dualcamctrl_diffusion_transformer.pt",
        "config_path": str(
            resolve._data_path("models", "runtime", "configs", "dualcamctrl", "controlnet_gate_asym_5_10.yaml")
        ),
        "torch_dtype": "bfloat16",
        "download_resource": "HuggingFace",
        "allow_download": True,
        "copy_control_weights": True,
        "redirect_common_files": True,
    }
    base_model_path = Path(_dualcamctrl_base_default_ref()).expanduser()
    checkpoint_root = checkpoint_root_path()
    tokenizer_root = Path(
        resolve._prefer_existing_model_ref(
            *resolve._cache_candidates("Wan-AI--Wan2.1-T2V-1.3B", "Wan-AI/Wan2.1-T2V-1.3B"),
            str(checkpoint_root / "Wan-AI" / "Wan2.1-T2V-1.3B"),
            str(checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B"),
        )
    )
    clip_root = Path(
        resolve._prefer_existing_model_ref(
            *resolve._cache_candidates(
                "Wan-AI--Wan2.1-I2V-14B-720P",
                "Wan-AI--Wan2.1-I2V-14B-480P",
                "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            ),
            str(checkpoint_root / "Wan-AI--Wan2.1-I2V-14B-720P"),
            str(checkpoint_root / "Wan-AI--Wan2.1-I2V-14B-480P"),
            str(checkpoint_root / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera"),
            str(checkpoint_root / "Wan-AI" / "Wan2.1-I2V-14B-480P"),
        )
    )
    dualcam_root = Path(
        resolve._prefer_existing_model_ref(
            *resolve._cache_candidates("FayeHongfeiZhang--DualCamCtrl", "FayeHongfeiZhang/DualCamCtrl"),
            str(checkpoint_root / "FayeHongfeiZhang" / "DualCamCtrl"),
            str(checkpoint_root / "FayeHongfeiZhang--DualCamCtrl"),
        )
    )
    local_files = (
        next(iter(base_model_path.glob("diffusion_pytorch_model*.safetensors")), None)
        if base_model_path.is_dir()
        else None,
        tokenizer_root / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_root / "Wan2.1_VAE.pth",
        tokenizer_root / "google" / "umt5-xxl",
        clip_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        dualcam_root / "checkpoints" / "dualcamctrl_diffusion_transformer.pt",
    )
    if all(path is not None and Path(path).exists() for path in local_files):
        load_kwargs.update(
            base_model_path=str(base_model_path),
            local_model_path=str(checkpoint_root),
            allow_download=False,
        )
    return load_kwargs


def _dualcamctrl_default_call_kwargs() -> dict[str, Any]:
    demo_root = resolve._data_path("test_cases", "dualcamctrl", "demo_pic")
    return {
        "demo_name": "seaside",
        "image_path": str(demo_root / "seaside.png"),
        "depth_image": str(demo_root / "seaside_depth.png"),
        "camera_path": str(demo_root / "seaside.torch"),
        "num_frames": 61,
        "height": 320,
        "width": 480,
        "num_inference_steps": 50,
        "fps": 10,
        "seed": 42,
        "cfg_scale": 5.0,
        "return_control_latents": True,
        "original_height": 360,
        "original_width": 640,
        "tiled": True,
    }


def _wan_2p2_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan-AI--Wan2.2-TI2V-5B",
        "Wan2.2-TI2V-5B",
        fallback="Wan-AI/Wan2.2-TI2V-5B",
    )


def _wan_2p2_t2v_a14b_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan2.2-T2V-A14B",
        "Wan-AI--Wan2.2-T2V-A14B",
        fallback="Wan-AI/Wan2.2-T2V-A14B",
    )


def _wan_2p2_i2v_a14b_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Wan2.2-I2V-A14B",
        "Wan-AI--Wan2.2-I2V-A14B",
        fallback="Wan-AI/Wan2.2-I2V-A14B",
    )


def _ltx23_default_load_kwargs() -> dict[str, Any]:
    """Expose every runtime-required LTX-2.3 checkpoint component to Studio."""

    return {
        "required_components": {
            "spatial_upsampler_path": resolve._checkpoint_model_ref(
                "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "Lightricks--LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "Lightricks--LTX-2.3",
                        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                    )
                ),
            ),
            "gemma_root": resolve._checkpoint_model_ref(
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "google--gemma-3-12b-it-qat-q4_0-unquantized",
                fallback=str(checkpoint_root_path("gemma-3-12b-it-qat-q4_0-unquantized")),
            ),
        }
    }


def _ltx25_checkpoint_dir() -> str:
    return resolve._checkpoint_model_ref(
        "Lightricks--LTX-2.5",
        fallback=str(checkpoint_root_path("Lightricks--LTX-2.5")),
    )


def _ltx25_default_load_kwargs() -> dict[str, Any]:
    """Map Studio to the native LTX-2.5 recipe's local split checkpoints."""

    checkpoint_dir = Path(_ltx25_checkpoint_dir())
    return {
        "required_components": {
            "checkpoint_dir": str(checkpoint_dir),
            "transformer_path": str(
                checkpoint_dir / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
            ),
            "text_encoder_path": str(
                checkpoint_dir / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
            ),
            "video_vae_path": str(checkpoint_dir / "vae" / "ltx-2.5-video-vae-bf16.safetensors"),
            "audio_vae_path": str(checkpoint_dir / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"),
            "spatial_upsampler_path": str(
                checkpoint_dir / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            ),
            "duration_head_path": str(checkpoint_dir / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors"),
            "offload_mode": "cpu",
            "diffvae_optimization": "chunked_eager",
        }
    }


def _echo_infinity_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Echo-Infinity/echo_infinity.pt",
        "Echo-Infinity/checkpoints/echo_infinity.pt",
        fallback="Echo-AI/Echo-Infinity",
    )


def _echo_infinity_default_load_kwargs() -> dict[str, Any]:
    wan_root = checkpoint_root_path()
    wan_model_name = "Wan2.1-T2V-1.3B"
    found = False
    for candidate_root in checkpoint_root_candidates():
        for candidate_name in ("Wan-AI--Wan2.1-T2V-1.3B", wan_model_name):
            if (candidate_root / candidate_name).is_dir():
                wan_root = candidate_root
                wan_model_name = candidate_name
                found = True
                break
        if found:
            break
    return {
        "wan_root": str(wan_root),
        "wan_model_name": wan_model_name,
        "generator_ckpt": _echo_infinity_default_ref(),
        "model_kwargs": {
            "model_name": wan_model_name,
            "wan_root": str(wan_root),
            "local_attn_size": 12,
            "timestep_shift": 5.0,
            "sink_size": 3,
        },
    }


def _echo_memory_default_ref(checkpoint_file: str) -> str:
    relative = Path(checkpoint_file)
    return resolve._checkpoint_model_ref(
        str(Path("Echo-Team--Echo-Memory") / relative),
        str(Path("Echo-Memory") / relative),
        fallback=str(hfd_root_path("Echo-Team--Echo-Memory", *relative.parts)),
    )


def _echo_memory_context_k1_default_ref() -> str:
    return _echo_memory_default_ref("context_k1/epoch-0.safetensors")


def _echo_memory_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_base_dir": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
            fallback=str(hfd_root_path("Wan-AI--Wan2.1-T2V-1.3B")),
        )
    }


def _echo_memory_context_k1_load_kwargs() -> dict[str, Any]:
    return _echo_memory_default_load_kwargs()


def _recammaster_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "KlingTeam--ReCamMaster-Wan2.1",
        "ReCamMaster-Wan2.1",
        fallback="KlingTeam/ReCamMaster-Wan2.1",
    )


def _recammaster_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_path": _wan_2p1_t2v_default_ref(),
        "recammaster_ckpt_path": _recammaster_default_ref(),
    }


def _versecrafter_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "TencentARC--VerseCrafter",
        "VerseCrafter",
        fallback="TencentARC/VerseCrafter",
    )


def _versecrafter_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_model_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-14B",
            "Wan2.1-T2V-14B",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
        "moge_model_path": resolve._checkpoint_model_ref(
            "moge-2-vitl-normal",
            "Ruicheng--moge-2-vitl-normal",
            fallback="Ruicheng/moge-2-vitl-normal",
        ),
    }


def _cameractrl_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "hehao13--CameraCtrl/CameraCtrl.ckpt",
        "CameraCtrl/CameraCtrl.ckpt",
        fallback="hehao13/CameraCtrl",
    )


def _cameractrl_default_load_kwargs() -> dict[str, Any]:
    sd15_path = resolve._checkpoint_model_ref(
        "stable-diffusion-v1-5--stable-diffusion-v1-5",
        "stable-diffusion-v1-5",
        fallback="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    return {
        "sd15_path": sd15_path,
        "pose_adaptor_ckpt": _cameractrl_default_ref(),
        "image_lora_ckpt": resolve._checkpoint_model_ref(
            "hehao13--CameraCtrl/RealEstate10K_LoRA.ckpt",
            "CameraCtrl/RealEstate10K_LoRA.ckpt",
            fallback="hehao13/CameraCtrl",
        ),
        "motion_module_ckpt": resolve._checkpoint_model_ref(
            "guoyww--animatediff/v3_sd15_mm.ckpt",
            "animatediff/v3_sd15_mm.ckpt",
            fallback="guoyww/animatediff",
        ),
        "motion_adapter_ckpt": resolve._checkpoint_model_ref(
            "guoyww--animatediff/v3_sd15_adapter.ckpt",
            "animatediff/v3_sd15_adapter.ckpt",
            fallback="guoyww/animatediff",
        ),
        "unet_subfolder": "unet",
    }


def _longcat_video_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "LongCat-Video",
        "meituan-longcat--LongCat-Video",
        fallback="meituan-longcat/LongCat-Video",
    )


def _lingbot_video_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "lingbot-video-dense-1.3b",
        "robbyant--lingbot-video-dense-1.3b",
    )


def _dreamdojo_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "nvidia--DreamDojo/2B_GR1_post-train",
        fallback="nvidia/DreamDojo",
    )


def _dreamdojo_default_load_kwargs() -> dict[str, Any]:
    root = Path(_dreamdojo_default_ref())
    checkpoint_dir = root / "iter_000050000"
    return {
        "vae_path": str(checkpoint_root_path("nvidia--Cosmos-Predict2.5-2B", "tokenizer.pth")),
        "text_encoder_path": str(checkpoint_root_path("nvidia--Cosmos-Reason1-7B")),
        # Matrix readiness audits the distributed EMA inputs consumed by conversion.
        "checkpoint_shards": [str(checkpoint_dir / "model" / f"__{index}_0.distcp") for index in range(8)],
    }


def _longcat_video_default_load_kwargs() -> dict[str, Any]:
    return {
        "python_executable": resolve._current_python_executable(),
    }


def _motionctrl_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "MotionCtrl/motionctrl.pth",
        "custom--MotionCtrl/motionctrl.pth",
        "TencentARC--MotionCtrl/motionctrl.pth",
        fallback="TencentARC/MotionCtrl",
    )


def _astra_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Astra",
        "custom--Astra",
        "EvanEternal--Astra",
        fallback="EvanEternal/Astra",
    )


def _astra_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": resolve._checkpoint_model_ref(
                "Wan2.1-T2V-1.3B",
                "Wan-AI--Wan2.1-T2V-1.3B",
                fallback="Wan-AI/Wan2.1-T2V-1.3B",
            )
        }
    }


ASTRA_OFFICIAL_PROMPT = (
    "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, "
    "charming, and inviting atmosphere. The scene shows a picturesque open square paved with red "
    "bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and "
    "dark-painted facades. On the right side, a lush arrangement of potted plants and blooming "
    "flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands "
    "prominently near the center-right, contributing to the timeless character of the street. "
    "Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. "
    "Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely "
    "feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from "
    "the left, enhancing the scene's inviting, peaceful mood."
)


def _astra_default_call_kwargs() -> dict[str, Any]:
    return {
        "frames_per_generation": 8,
        "total_frames_to_generate": 24,
        "num_inference_steps": 50,
        "start_frame": 0,
        "initial_condition_frames": 1,
        "modality_type": "sekai",
        "fps": 20,
    }


def _pusa_vidgen_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Pusa-Wan2.2-V1",
        "RaphaelLiu--Pusa-Wan2.2-V1",
        fallback="RaphaelLiu/Pusa-Wan2.2-V1",
    )


def _pusa_vidgen_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_model_root": resolve._checkpoint_model_ref(
            "Wan2.2-T2V-A14B",
            "Wan-AI--Wan2.2-T2V-A14B",
            fallback="Wan-AI/Wan2.2-T2V-A14B",
        ),
        "lightx2v_root": resolve._checkpoint_model_ref(
            "lightx2v--Wan2.2-Lightning",
            "Wan2.2-Lightning",
            fallback="lightx2v/Wan2.2-Lightning",
        ),
        "python_executable": resolve._current_python_executable(),
        "runner_version": "v1",
    }


def _sana_checkpoint_ref(repo_dir: str, checkpoint_name: str, repo_id: str) -> str:
    hfd_repo_dir = repo_id.replace("/", "--")
    return resolve._checkpoint_model_ref(
        f"{hfd_repo_dir}/checkpoints/{checkpoint_name}",
        f"{repo_dir}/checkpoints/{checkpoint_name}",
        f"{repo_dir}/{checkpoint_name}",
        fallback=f"hf://{repo_id}/checkpoints/{checkpoint_name}",
    )


def _sana_variant_default_size(variant: Any) -> tuple[int, int]:
    if variant.default_height is not None and variant.default_width is not None:
        return int(variant.default_height), int(variant.default_width)
    if variant.resolution == "720p":
        return 720, 1280
    if variant.resolution == "480p":
        return 480, 832
    if str(variant.resolution).endswith("px"):
        side = int(str(variant.resolution).removesuffix("px"))
        return side, side
    return 1024, 1024


def _sana_variant_checkpoint_ref(variant: Any) -> str:
    filename = Path(str(variant.model_path).split("?", 1)[0]).name
    repo_dir = str(variant.repo_id).rsplit("/", 1)[-1]
    return _sana_checkpoint_ref(repo_dir, filename, variant.repo_id)


@lru_cache(maxsize=1)
def _sana_image_aliases() -> tuple[str, ...]:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_ALIASES

    image_ids = _sana_folded_studio_model_ids()
    aliases = sorted(image_ids)
    for alias, target in SANA_ALIASES.items():
        if alias != "sana" and target in image_ids:
            aliases.append(alias)
    return resolve._coerce_aliases(*aliases)


def _sana_default_image_variant() -> Any:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    return SANA_VARIANTS[SANA_DEFAULT_IMAGE_VARIANT_ID]


def _sana_image_default_ref() -> str:
    return _sana_variant_checkpoint_ref(_sana_default_image_variant())


def _sana_image_workspace_variants() -> tuple[dict[str, Any], ...]:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    rows: list[dict[str, Any]] = []
    for variant in SANA_VARIANTS.values():
        if variant.runner not in _SANA_IMAGE_RUNNERS:
            continue
        height, width = _sana_variant_default_size(variant)
        rows.append(
            {
                "variant_id": variant.model_id,
                "label": variant.display_name,
                "status": "checkpoint-configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": _sana_variant_checkpoint_ref(variant),
                        "required": True,
                        "status": "checkpoint-configured",
                    },
                ),
                "load_kwargs": {"model_id": variant.model_id},
                "call_kwargs": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": variant.default_steps or 20,
                    "guidance_scale": variant.default_cfg_scale or 4.5,
                    "seed": 42,
                },
                "notes": (variant.notes,) if variant.notes else (),
            }
        )
    return tuple(rows)


def _sana_video_480p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_480p",
        "SANA_Video_2B_480p.pth",
        "Efficient-Large-Model/SANA-Video_2B_480p",
    )


def _sana_video_720p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_720p",
        "SANA_Video_2B_720p.pth",
        "Efficient-Large-Model/SANA-Video_2B_720p",
    )


def _sana_streaming_default_ref(*, bidirectional: bool) -> str:
    repo_dir = (
        "Efficient-Large-Model--SANA-Streaming_bidirectional"
        if bidirectional
        else "Efficient-Large-Model--SANA-Streaming"
    )
    repo_id = (
        "Efficient-Large-Model/SANA-Streaming_bidirectional"
        if bidirectional
        else "Efficient-Large-Model/SANA-Streaming"
    )
    return resolve._checkpoint_model_ref(repo_dir, fallback=repo_id)


def _sana_streaming_default_load_kwargs(*, bidirectional: bool) -> dict[str, Any]:
    dit_root = _sana_streaming_default_ref(bidirectional=bidirectional)
    gemma_root = resolve._checkpoint_model_ref(
        "Efficient-Large-Model--gemma-2-2b-it",
        fallback="Efficient-Large-Model/gemma-2-2b-it",
    )
    codec_root = resolve._checkpoint_model_ref(
        "Lightricks--LTX-2",
        fallback="Lightricks/LTX-2",
    )
    return {
        "model_id": ("sana-streaming-bidirectional-2b-720p" if bidirectional else "sana-streaming-2b-720p"),
        "checkpoint_overrides": {
            "dit": dit_root,
            "text-encoder": gemma_root,
            "tokenizer": gemma_root,
            "codec": codec_root,
        },
    }


def _longsana_video_480p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_480p_LongLive",
        "SANA_Video_2B_480p_LongLive.pth",
        "Efficient-Large-Model/SANA-Video_2B_480p_LongLive",
    )


def _sana_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "video",
        "output_path",
        "fps",
        "return_dict",
        "seed",
        "cfg_scale",
        "guidance_scale",
        "step",
        "num_inference_steps",
        "num_frames",
        "frames",
        "height",
        "width",
        "max_sequence_length",
        "plan_only",
    )


def _longsana_video_call_params() -> tuple[str, ...]:
    return (
        "negative_prompt",
        *(param for param in _sana_video_call_params() if param != "cfg_scale"),
    )


def _sana_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "checkpoint_path",
        "config_path",
        "config",
        "python_executable",
        "python",
        "work_dir",
        "default_work_dir",
        "required_components",
        "device",
        "model_id",
        "variant",
        "profile_id",
        *DISPATCH_LOAD_PARAMS,
    )


def _zeroscope_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "zeroscope_v2_576w",
        "cerspense--zeroscope_v2_576w",
        fallback="cerspense/zeroscope_v2_576w",
    )


def _animatediff_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "animatediff/mm_sd_v15_v2.ckpt",
        "guoyww--animatediff/mm_sd_v15_v2.ckpt",
        fallback="guoyww/animatediff",
    )


def _animatediff_default_load_kwargs() -> dict[str, Any]:
    return {
        "motion_module_path": _animatediff_default_ref(),
        "base_model_path": resolve._checkpoint_model_ref(
            "stable-diffusion-v1-5",
            "stable-diffusion-v1-5--stable-diffusion-v1-5",
            "runwayml--stable-diffusion-v1-5",
            fallback="runwayml/stable-diffusion-v1-5",
        ),
        "dreambooth_model_path": resolve._checkpoint_model_ref(
            "animatediff_t2i_backups/realisticVisionV60B1_v51VAE.safetensors",
            "guoyww--animatediff_t2i_backups/realisticVisionV60B1_v51VAE.safetensors",
        ),
    }


def _open_magvit2_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Open-MAGVIT2/AR_256_L.ckpt",
        "TencentARC--Open-MAGVIT2/AR_256_L.ckpt",
        fallback="TencentARC/Open-MAGVIT2",
    )


def _skyreels_v3_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "SkyReels-V3-R2V-14B",
        "Skywork--SkyReels-V3-R2V-14B",
        fallback="Skywork/SkyReels-V3-R2V-14B",
    )


def _kairos_sensenova_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "kairos-agi--kairos-sensenova-4B-480P-pretrained",
        "kairos-sensenova-4B-480P-pretrained",
        fallback="kairos-agi/kairos-sensenova-4B-480P-pretrained",
    )


def _kairos_sensenova_default_load_kwargs() -> dict[str, Any]:
    model_root = Path(_kairos_sensenova_default_ref())
    return {
        "models_root": str(model_root),
        "variant": "pretrained",
        "pretrained_dit": str(model_root / "kairos-common-4B-480P.safetensors"),
        "text_encoder_path": resolve._checkpoint_model_ref(
            "Qwen--Qwen2.5-VL-7B-Instruct",
            fallback="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        "vae_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
    }


def _neoverse_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Yuppie1204--NeoVerse",
        "NeoVerse",
        fallback="Yuppie1204/NeoVerse",
    )


NEOVERSE_OFFICIAL_PROMPT = "A two-arm robot assembles parts in front of a table."


def _neoverse_default_call_kwargs() -> dict[str, Any]:
    return {
        "predefined_trajectory": "tilt_up",
        "num_frames": 81,
        "trajectory_mode": "relative",
        "angle": 15,
        "distance": 0,
        "orbit_radius": 0,
        "zoom_ratio": 1.0,
        "alpha_threshold": 1.0,
        "seed": 42,
        "use_first_frame": True,
        "static_scene": False,
        "num_inference_steps": 4,
        "cfg_scale": 1.0,
    }


def _neoverse_default_load_kwargs() -> dict[str, Any]:
    return {
        "height": 336,
        "width": 560,
        "num_inference_steps": 4,
        "cfg_scale": 1.0,
        "disable_lora": False,
        "enable_vram_management": True,
    }


def _show_o_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "showlab--show-o",
        "show-o",
        fallback="showlab/show-o",
    )


def _show_o_default_load_kwargs() -> dict[str, Any]:
    return {
        "vq_model_path": resolve._checkpoint_model_ref(
            "showlab--magvitv2",
            "magvitv2",
            fallback="showlab/magvitv2",
        ),
        "llm_model_path": resolve._checkpoint_model_ref(
            "microsoft--phi-1_5",
            "phi-1_5",
            fallback="microsoft/phi-1_5",
        ),
        "resolution": 256,
    }


def _ac3d_dataset_ready(root: Path) -> bool:
    return (
        (root / "annotations" / "test.json").is_file()
        and (root / "pose_files").is_dir()
        and (root / "video_clips").is_dir()
    )


def _ac3d_default_video_root_dir() -> str:
    candidates = [
        os.environ.get("WORLDFOUNDRY_AC3D_VIDEO_ROOT_DIR", ""),
        str(resolve._data_path("test_cases", "ac3d", "RealEstate10K")),
        str(resolve._project_root().parent / "datasets" / "RealEstate10K"),
        str(resolve._project_root().parent / "data" / "RealEstate10K"),
        str(resolve._project_root().parent / "ckpt" / "ac3d" / "RealEstate10K"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if _ac3d_dataset_ready(path):
            return str(path)
    return ""


def _ac3d_default_call_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "annotation_json": "annotations/test.json",
        "num_frames": 49,
        "stride_min": 2,
        "stride_max": 2,
        "height": 480,
        "width": 720,
        "start_camera_idx": 0,
        "end_camera_idx": 1,
    }
    video_root_dir = _ac3d_default_video_root_dir()
    if video_root_dir:
        kwargs["video_root_dir"] = video_root_dir
    return kwargs


def _magi2_python_executable() -> str:
    workspace_root = resolve._project_root().parent
    return resolve._first_existing_path(
        workspace_root / "conda" / "envs" / "magi2-preview" / "bin" / "python",
        workspace_root / "envs" / "magi2-preview" / "bin" / "python",
    )


def _magi2_default_load_kwargs() -> dict[str, Any]:
    return {"required_components": {"python_executable": _magi2_python_executable()}}


def _warp_as_history_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "BestWishYsh--Helios-Distilled",
        "Helios-Distilled",
        "helios-distilled",
        fallback=str(checkpoint_root_path("BestWishYsh--Helios-Distilled")),
    )


def _warp_as_history_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "lora_path": resolve._checkpoint_model_ref(
                "yyfz233--warp-as-history/visible_lora_state_step1000.safetensors",
                "warp-as-history/visible_lora_state_step1000.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "yyfz233--warp-as-history",
                        "visible_lora_state_step1000.safetensors",
                    )
                ),
            ),
            "pi3x_ckpt_path": resolve._checkpoint_model_ref(
                "yyfz233--Pi3X/model.safetensors",
                "Pi3X/model.safetensors",
                fallback=str(checkpoint_root_path("yyfz233--Pi3X", "model.safetensors")),
            ),
            "python_executable": resolve._worldfoundry_unified_python_executable(),
        }
    }


def _joyai_echo_longvideo_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "jdopensource--JoyAI-Echo",
        fallback="jdopensource/JoyAI-Echo",
    )


def _joyai_echo_longvideo_default_load_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _joyai_echo_longvideo_default_ref(),
        "gemma_path": resolve._checkpoint_model_ref(
            "google--gemma-3-12b-it",
            fallback="google/gemma-3-12b-it",
        ),
        "python_executable": resolve._joyai_echo_python_executable("joyai-echo-longvideo"),
        "precision": "bf16",
    }


OVERRIDES: dict[str, dict[str, Any]] = {
    "wan21-fun-1p3b-cam": {
        "display_name": "Wan2.1-Fun V1.1 1.3B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(resolve._data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(resolve._data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.1-fun-1.3b-camera",),
        "tags": ("camera-control", "i2v", "wan2.1", "in-tree-runtime"),
        "notes": "Split official transformer/base assets are composed with symlinks inside each run directory; the full upstream camera demo uses 49 frames at 832x480 with 50 denoising steps.",
    },
    "wan21-fun-14b-cam": {
        "display_name": "Wan2.1-Fun V1.1 14B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun 14B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-14B-Control-Camera",
            "alibaba-pai/Wan2.1-Fun-V1.1-14B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(resolve._data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(resolve._data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.1-fun-14b-camera",),
        "tags": ("camera-control", "i2v", "wan2.1", "in-tree-runtime"),
        "notes": "The full upstream Wan2.1 camera demo uses 49 frames at 832x480 with 50 denoising steps.",
    },
    "wan22-fun-5b-cam": {
        "display_name": "Wan2.2-Fun 5B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun 5B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.2-Fun-5B-Control-Camera",
            "alibaba-pai/Wan2.2-Fun-5B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(resolve._data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(resolve._data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 1280,
            "height": 704,
            "num_frames": 121,
            "fps": 24,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.2-fun-5b-camera",),
        "tags": ("camera-control", "i2v", "wan2.2", "in-tree-runtime"),
        "notes": "The full upstream Wan2.2 5B camera demo uses 121 frames at 1280x704 and 24 FPS with 50 denoising steps.",
    },
    "wan22-fun-a14b-cam": {
        "display_name": "Wan2.2-Fun A14B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun A14B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.2-Fun-A14B-Control-Camera",
            "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(resolve._data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(resolve._data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 81,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.2-fun-a14b-camera",),
        "tags": ("camera-control", "i2v", "wan2.2", "in-tree-runtime"),
        "notes": "The full upstream Wan2.2 A14B camera demo uses 81 frames at 832x480 with 50 denoising steps.",
    },
    "stable-video-infinity": {
        "display_name": "Stable Video Infinity 2.0",
        "category": "Video Generation",
        "summary": "Generate coherent long video from one image and a segment-level prompt stream.",
        "default_model_ref": _stable_video_infinity_default_ref,
        "default_load_kwargs": _stable_video_infinity_default_load_kwargs,
        "default_prompt": (
            "The water shimmers with dancing caustics as gentle bubbles rise; the camera dolly-ins through a blue haze."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "stable-video-infinity", "svi-2.0", "frame.jpg")),
        "default_task_type": "i2v",
        "default_call_kwargs": {
            "sigma_shift": 5.0,
            "prompt_repeat_times": 2,
            "base_seed": 0,
        },
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "wan_model_dir",
            "svi_lora_path",
            "dtype",
            "enable_vram_management",
            "num_persistent_param_in_dit",
            "use_usp",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "prompt",
            "images",
            "prompt_stream",
            "output_path",
            "fps",
            "return_dict",
            "num_clips",
            "num_frames",
            "num_motion_frames",
            "num_inference_steps",
            "cfg_scale_text",
            "guidance_scale",
            "sigma_shift",
            "shift",
            "ref_pad_cfg",
            "ref_pad_num",
            "prompt_repeat_times",
            "use_first_prompt_only",
            "repeat_first_clip",
            "prompt_prefix",
            "base_seed",
            "seed",
            "seed_stride",
            "max_width",
            "height",
            "width",
            "tiled",
            "tile_size",
            "tile_stride",
            "negative_prompt",
        ),
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("svi", "svi-2.0", "stable_video_infinity", "stable-video-infinity-2.0"),
        "tags": ("video", "i2v", "long-video", "prompt-stream", "in-tree-runtime"),
        "notes": ("Uses the in-tree SVI 2.0 continuation runtime; Wan2.1 and SVI weights remain in local HFD staging."),
    },
    "longvie-1": {
        "display_name": "LongVie",
        "category": "Video Generation",
        # Upstream has not published a first-generation checkpoint.  In
        # particular, Vchitect/LongVie2 must not be used here: its control and
        # DiT weights describe a different model variant.  Advanced users may
        # still pass an explicitly staged LongVie-1 control checkpoint.
        "default_model_ref": "",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "execute": True,
            "dense_video": str(resolve._data_path("test_cases", "longvie", "dense_control.mp4")),
            "sparse_video": str(resolve._data_path("test_cases", "longvie", "sparse_control.mp4")),
            "num_frames": 81,
            "height": 352,
            "width": 640,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "ref_image_path",
            "operator_kwargs",
            "execute",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "seed",
            "dense_video",
            "sparse_video",
            "tiled",
            "negative_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "longvie_weight_dir",
            "weight_dir",
            "wan_base_dir",
            "tokenizer_dir",
            "control_weight_path",
            "dit_weight_path",
            "torch_dtype",
            "use_usp",
            "ring_degree",
            "ulysses_degree",
            "enable_vram_management",
            "control_layers",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("longvie", "longvie1", "longvie-v1"),
        "tags": ("image-to-video", "long-video", "depth-controlled-video", "official-runtime", "in-tree-runtime"),
        "notes": (
            "The first-generation runtime adapter is available, but upstream has not published a LongVie-1 "
            "checkpoint. Pass an explicitly staged first-generation control weight; LongVie2 weights are "
            "rejected to prevent cross-variant demo reuse. Use longvie-2 for the public checkpoint-backed route."
        ),
    },
    "longvie-2": {
        "display_name": "LongVie 2",
        "category": "Video Generation",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "LongVie2",
            "Vchitect--LongVie2",
            "custom--LongVie2",
            fallback="Vchitect/LongVie2",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "execute": True,
            "dense_video": str(resolve._data_path("test_cases", "longvie", "dense_control.mp4")),
            "sparse_video": str(resolve._data_path("test_cases", "longvie", "sparse_control.mp4")),
            "num_frames": 81,
            "height": 352,
            "width": 640,
            "fps": 16,
            "seed": 0,
            # Native 352x640 segments fit comfortably on the supported A100
            # profile. Full-frame VAE encode/decode is both faster and avoids
            # tile-boundary artifacts; callers on smaller GPUs can opt in.
            "tiled": False,
            "num_inference_steps": 50,
        },
        "default_load_kwargs": {
            "model_id": "longvie-2",
            "use_usp": True,
            "ring_degree": 1,
            "ulysses_degree": 4,
            "torchrun_nproc_per_node": 4,
            # The supported A100/H100 80GB quality profile fits the complete
            # 14B stack per rank. CPU layer swapping otherwise dominates the
            # four-rank USP path and can leave the first denoise step running
            # for minutes while every GPU holds less than 8GB.
            "enable_vram_management": False,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "ref_image_path",
            "operator_kwargs",
            "execute",
            "num_frames",
            "height",
            "width",
            "seed",
            "dense_video",
            "sparse_video",
            "tiled",
            "negative_prompt",
            "num_inference_steps",
            "allow_control_padding",
        ),
        "stream_params": (
            "prompt",
            "dense_video",
            "sparse_video",
            "execute",
            "num_frames",
            "fps",
            "seed",
            "tiled",
            "negative_prompt",
            "num_inference_steps",
            "allow_control_padding",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "longvie_weight_dir",
            "weight_dir",
            "wan_base_dir",
            "tokenizer_dir",
            "control_weight_path",
            "dit_weight_path",
            "torch_dtype",
            "use_usp",
            "ring_degree",
            "ulysses_degree",
            "enable_vram_management",
            "control_layers",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "nproc_per_node",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("longvie2", "longvie-v2"),
        "tags": (
            "image-to-video",
            "long-video",
            "depth-controlled-video",
            "queued-segment-generation",
            "not-realtime",
            "official-runtime",
            "in-tree-runtime",
        ),
        "notes": (
            "Queued full-quality 81-frame segments at 16 FPS. This is not WASD realtime: "
            "the user supplies a prompt, initial image, dense depth control video, and sparse "
            "pointmap/track control video. Continue reuses the previous final frame, eight-frame "
            "history, noise, and resident weights."
        ),
    },
    "show-o": {
        "category": "Video Generation",
        "default_model_ref": _show_o_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "text-to-image",
        "default_load_kwargs": _show_o_default_load_kwargs,
        "default_call_kwargs": {
            "mode": "t2i",
            "generation_timesteps": 4,
            "guidance_scale": 1.0,
            "resolution": 256,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "mode",
            "generation_timesteps",
            "guidance_scale",
            "batch_size",
            "resolution",
            "generation_temperature",
            "mask_schedule",
            "noise_type",
        ),
        "load_params": (
            "pretrained_model_path",
            "model_path",
            "checkpoint_path",
            "device",
            "vq_model_path",
            "vq_model_name",
            "llm_model_path",
            "resolution",
            "batch_size",
            "guidance_scale",
            "generation_timesteps",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("show_o", "showo"),
        "tags": ("text-to-image", "official-runtime", "in-tree-runtime"),
        "notes": "Current in-tree Show-O wrapper supports t2i image generation; video/MMU modes are not marked as verified.",
    },
    "dualcamctrl": {
        "display_name": "DualCamCtrl",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _dualcamctrl_base_default_ref,
        "default_task_type": "image-depth-camera-to-video",
        "default_prompt": (
            "High aerial view over a British seaside town on a sunny afternoon. Turquoise sea with gentle waves, "
            "a long wooden pier stretching into the water, sandy beach and a bustling promenade with parked cars. "
            "Foreground: curved coastal road and pastel, terraced houses with orange roofs. Midground: beach, pools "
            "and small buildings. Background: white cliffs and rolling hills fading into haze. Few fluffy clouds, "
            "bright natural light, crisp visibility. Slow tilt-down and rightward pan, subtle zoom for parallax. "
            "Natural color grade, 4K, 24fps, steady gimbal/drone feel, light wind ambience and distant seagulls."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "dualcamctrl", "demo_pic", "seaside.png")),
        "default_interactions": (str(resolve._data_path("test_cases", "dualcamctrl", "demo_pic", "seaside.torch")),),
        "default_load_kwargs": _dualcamctrl_default_load_kwargs,
        "default_call_kwargs": _dualcamctrl_default_call_kwargs,
        "input_params": ("image_path", "depth_image", "camera_path"),
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "demo_name",
            "image_path",
            "depth_image",
            "depth_path",
            "camera_path",
            "trajectory_file",
            "num_frames",
            "frame_len",
            "height",
            "width",
            "num_inference_steps",
            "infer_steps",
            "seed",
            "cfg_scale",
            "negative_prompt",
            "return_control_latents",
            "original_height",
            "original_width",
            "tiled",
            "quality",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "base_model_path",
            "base_model_repo",
            "tokenizer_repo",
            "dualcamctrl_repo",
            "checkpoint_path",
            "ckpt_path",
            "checkpoint_name",
            "config_path",
            "model_config",
            "local_model_path",
            "download_resource",
            "allow_download",
            "torch_dtype",
            "weight_dtype",
            "copy_control_weights",
            "redirect_common_files",
            "use_usp",
            "load_model",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("dual-cam-ctrl", "dualcam", "dual-camera-control"),
        "tags": ("image-to-video", "depth-conditioned-video", "camera-control", "official-demo", "in-tree-runtime"),
        "notes": "Defaults mirror the official DualCamCtrl seaside demo: RGB image, depth image, camera .torch path, 61 frames, 320x480, 50 denoising steps, fps 10, seed 42.",
    },
    "warp-as-history": {
        "display_name": "Warp-as-History",
        "default_model_ref": _warp_as_history_default_ref,
        "default_load_kwargs": _warp_as_history_default_load_kwargs,
        "default_prompt": "A cyclist riding past a colorful graffiti wall beside trees and tall grass, urban BMX scene, with camera movement.",
        "default_call_kwargs": {"num_frames": 33, "fps": 16, "height": 384, "width": 640},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "output_dir",
            "fps",
            "num_frames",
            "height",
            "width",
            "seed",
            "return_dict",
            "camera_poses",
            "camera_poses_path",
            "warp_video",
            "warp_video_path",
            "warp_visibility_mask",
            "warp_visibility_mask_path",
            "demo_csv_path",
            "plan_only",
            "dtype",
            "lora_path",
            "no_lora",
            "enable_optional_attention",
            "warp_debug_dir",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "lazy",
            "lora_path",
            "pi3x_ckpt_path",
            "python_executable",
        ),
        "aliases": ("wah", "warp_as_history", "yyfz233/warp-as-history"),
        "tags": ("image-to-video", "camera-control", "in-tree-runtime"),
        "notes": "In-tree official Warp-as-History runtime. Uses vendored demo conditioning by default; real custom runs should pass a first frame plus camera_poses_path or warp_video_path.",
    },
    "ac3d": {
        "display_name": "AC3D",
        "category": "Video Generation",
        "default_prompt": (
            "Three fluffy sheep sit side by side at a rustic wooden table, each eagerly digging into their bowls of "
            "spaghetti. The pasta is tangled playfully around their woolly faces, and the bright red sauce splatters "
            "across their fur. The scene takes place in a lush, green meadow surrounded by rolling hills, with a few "
            "grazing cows in the background."
        ),
        "default_call_kwargs": _ac3d_default_call_kwargs,
        "call_params": (
            "prompt",
            "output_path",
            "return_dict",
            "video_root_dir",
            "annotation_json",
            "num_frames",
            "height",
            "width",
            "stride_min",
            "stride_max",
            "start_camera_idx",
            "end_camera_idx",
            "controlnet_weights",
            "controlnet_guidance_start",
            "controlnet_guidance_end",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
        ),
        "tags": ("camera-control", "text-to-video", "official-runtime"),
        "notes": "Official AC3D inference requires a RealEstate10K-style fixture with annotations/test.json, pose_files, and video_clips.",
    },
    "uni3c": {
        "display_name": "Uni3C",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A close-up of a wet white cat wearing oversized amber sunglasses in a sunlit swimming pool, "
            "while the camera smoothly pushes forward toward the cat."
        ),
        "default_input_path": str(
            resolve._data_path("test_cases", "uni3c", "synthetic-forward-9f-768x480", "reference.jpg")
        ),
        "default_call_kwargs": {
            "render_path": str(resolve._data_path("test_cases", "uni3c", "synthetic-forward-9f-768x480")),
            "mode": "camera",
            "num_frames": 9,
            "max_area": 114688,
            "seed": 0,
            "fps": 16,
            "num_gpus": 1,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "render_path",
            "mode",
            "num_frames",
            "max_area",
            "seed",
            "fps",
            "num_gpus",
            "output_path",
            "plan_only",
            "return_dict",
        ),
        "load_params": (
            "controller_path",
            "base_model_path",
            "model_path",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("image-to-video", "camera-control", "official-stage-two-runtime"),
        "notes": (
            "Requires a reference image and the matching official stage-one render bundle containing "
            "render.mp4, render_mask.mp4, and cam_info.json."
        ),
    },
    "cameractrl": {
        "display_name": "CameraCtrl",
        "default_model_ref": _cameractrl_default_ref,
        "default_load_kwargs": _cameractrl_default_load_kwargs,
        "default_interactions": ("camera_path",),
        "default_call_kwargs": {
            "trajectory_file": "worldfoundry/data/test_cases/cameractrl/pose_files/0f47577ab3441480.txt"
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "trajectory_file",
            "height",
            "width",
            "image_height",
            "image_width",
            "original_pose_width",
            "original_pose_height",
            "num_frames",
            "video_length",
            "num_inference_steps",
            "infer_steps",
            "guidance_scale",
            "cfg_scale",
            "negative_prompt",
            "seed",
        ),
        "tags": ("camera-control", "text-to-video", "official-runtime"),
    },
    "motionctrl": {
        "display_name": "MotionCtrl",
        "default_model_ref": _motionctrl_default_ref,
        "default_interactions": ("camera_motion",),
        "default_call_kwargs": {
            "height": 256,
            "width": 256,
            "unconditional_guidance_scale": 7.5,
            "ddim_steps": 50,
            "condtype": "camera_motion",
            "camera_pose_path": str(
                resolve._data_path(
                    "test_cases", "motionctrl_conditions", "camera_poses", "test_camera_Round-ZoomIn.json"
                )
            ),
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "camera_pose_file",
            "camera_pose_path",
            "trajectory_file",
            "object_trajectory_file",
            "cond_dir",
            "condtype",
            "height",
            "width",
            "num_frames",
            "seed",
            "n_samples",
            "batch_size",
            "bs",
            "unconditional_guidance_scale",
            "unconditional_guidance_scale_temporal",
            "guidance_scale",
            "cfg_scale",
            "ddim_steps",
            "infer_steps",
            "ddim_eta",
            "cond_T",
        ),
        "tags": ("camera-control", "object-motion", "official-runtime"),
    },
    "astra": {
        "display_name": "Astra",
        "default_model_ref": _astra_default_ref,
        "default_prompt": ASTRA_OFFICIAL_PROMPT,
        "default_input_path": str(resolve._data_path("test_cases", "astra", "condition_images", "garden_1.png")),
        "default_load_kwargs": _astra_default_load_kwargs,
        "default_interactions": ("forward_left",),
        "default_call_kwargs": _astra_default_call_kwargs,
        "call_params": (
            "prompt",
            "image_path",
            "images",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "frames_per_generation",
            "total_frames_to_generate",
            "num_inference_steps",
            "start_frame",
            "initial_condition_frames",
            "modality_type",
            "height",
            "width",
        ),
        "tags": ("camera-control", "image-to-video", "official-runtime"),
    },
    "hunyuanvideo-t2v": {
        "display_name": "HunyuanVideo T2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "HunyuanVideo",
            "tencent--HunyuanVideo",
            fallback="tencent/HunyuanVideo",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cat walks on the grass, realistic style.",
        "call_params": _native_diffusion_video_call_params(),
        "input_params": (
            "prompt",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "fps",
            "guidance_scale",
            "embedded_guidance_scale",
            "seed",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "default_call_kwargs": {
            "num_frames": 129,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "fps": 24,
            "guidance_scale": 6.0,
            "embedded_guidance_scale": 6.0,
            "seed": 42,
        },
        "aliases": ("hunyuanvideo", "hunyuan-video", "hunyuanvideo-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "hunyuanvideo-i2v": {
        "display_name": "HunyuanVideo I2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "HunyuanVideo-I2V",
            "tencent--HunyuanVideo-I2V",
            fallback="tencent/HunyuanVideo-I2V",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick.",
        "call_params": _native_diffusion_video_call_params(),
        "input_params": (
            "prompt",
            "image_path",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "fps",
            "guidance_scale",
            "embedded_guidance_scale",
            "seed",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "default_call_kwargs": {
            "image_path": str(resolve._data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
            "num_frames": 129,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "fps": 24,
            "guidance_scale": 6.0,
            "embedded_guidance_scale": 6.0,
            "seed": 0,
        },
        "aliases": ("hunyuanvideo-i2v", "hunyuan-video-i2v"),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint"),
    },
    "framepack": {
        "display_name": "FramePack",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "FramePackI2V_HY",
            "lllyasviel--FramePackI2V_HY",
            "lllyasviel--FramePack_F1_I2V_HY_20250503",
            fallback="lllyasviel/FramePackI2V_HY",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"torch_dtype": "bfloat16"},
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("camera_path",),
        "call_params": (
            *_official_video_call_params(),
            "cfg",
            "gs",
            "rs",
            "use_teacache",
            "mp4_crf",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("frame-pack", "framepack-i2v"),
        "tags": ("image-to-video", "official-runtime"),
    },
    "hunyuanvideo-1.5-t2v": {
        "display_name": "HunyuanVideo 1.5 T2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "HunyuanVideo-1.5",
            fallback="tencent/HunyuanVideo-1.5",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cat walks on a snowy street, cinematic, high quality.",
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "default_load_kwargs": lambda: _hunyuanvideo15_default_load_kwargs(image_to_video=False),
        "default_call_kwargs": {
            "height": 720,
            "width": 1280,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 24,
            "seed": 42,
        },
        "aliases": ("hunyuanvideo-1.5", "hunyuanvideo15", "hunyuanvideo15-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "hunyuanvideo-1.5-i2v": {
        "display_name": "HunyuanVideo 1.5 I2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "HunyuanVideo-1.5",
            fallback="tencent/HunyuanVideo-1.5",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "A young man holds a sparkling firework at night, cinematic lighting, realistic motion.",
        "default_input_path": str(resolve._data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "default_load_kwargs": lambda: _hunyuanvideo15_default_load_kwargs(image_to_video=True),
        "default_call_kwargs": {
            "image_path": str(resolve._data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
            "height": 720,
            "width": 544,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 24,
            "seed": 0,
        },
        "aliases": ("hunyuanvideo-1.5-i2v", "hunyuanvideo15-i2v"),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint"),
    },
    "i2vgen-xl": {
        "display_name": "I2VGen-XL",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "i2vgen-xl",
            "ali-vilab--i2vgen-xl",
            fallback="ali-vilab/i2vgen-xl",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A slow forward camera glide through a lush jungle clearing, with leaves moving gently in the wind "
            "and stable trees, rocks, and background geometry."
        ),
        "default_call_kwargs": {
            "image_path": str(resolve._data_path("test_cases", "neoverse", "videos", "jungle.png")),
            "height": 704,
            "width": 1280,
            "target_fps": 16,
            "num_frames": 16,
            "num_inference_steps": 50,
            "guidance_scale": 9.0,
            "negative_prompt": (
                "distorted, discontinuous, blurry, low resolution, static, disfigured, "
                "disconnected objects, inconsistent geometry"
            ),
            "seed": 8888,
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "tags": ("image-to-video", "diffusers", "official-runtime"),
    },
    "magi-1": {
        "display_name": "MAGI-1",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "MAGI-1",
            "sand-ai--MAGI-1",
            fallback="sand-ai/MAGI-1",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("magi", "magi1"),
        "tags": ("image-to-video", "official-runtime"),
    },
    "magi2-preview": {
        "display_name": "MAGI-2 Preview",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "magi2-preview",
            "sand-ai--MAGI-2-preview",
            fallback="sand-ai/MAGI-2-preview",
        ),
        "default_load_kwargs": _magi2_default_load_kwargs,
        "default_prompt": (
            "A cinematic tracking shot follows a red fox running through a snowy forest, "
            "with coherent motion, stable anatomy, and natural ambient sound."
        ),
        "default_call_kwargs": {
            "short_edge": 512,
            "aspect_ratio": "16:9",
            "duration_seconds": 10.0,
            "num_inference_steps": 100,
            "seed": 0,
            "use_refiner": False,
        },
        "call_params": (
            "prompt",
            "negative_prompt",
            "short_edge",
            "aspect_ratio",
            "duration_seconds",
            "num_inference_steps",
            "seed",
            "use_refiner",
            "refiner_height",
            "refiner_width",
            "refiner_num_inference_steps",
            "output_path",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "checkpoint_path",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("magi2", "magi2_preview", "magi-2-preview", "sand-ai/MAGI-2-preview"),
        "tags": ("text-to-video", "audio-video", "moe", "in-tree-runtime", "local-checkpoint"),
        "notes": "114B MoE text-to-video with audio, per-layer GPU offload, and optional video refinement; image conditioning is unsupported.",
    },
    "mochi-1-preview-t2v": {
        "display_name": "Mochi-1 Preview T2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "mochi-1-preview",
            "genmo--mochi-1-preview",
            fallback="genmo/mochi-1-preview",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "Close-up of a chameleon's eye as its detailed scaly skin slowly changes color, "
            "photorealistic natural motion, shallow depth of field, ultra high resolution 4K."
        ),
        "default_call_kwargs": {
            "height": 480,
            "width": 848,
            "num_frames": 84,
            "num_inference_steps": 64,
            "guidance_scale": 4.5,
            "fps": 30,
            "seed": 12345,
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("mochi-1", "mochi", "mochi-1-preview"),
        "tags": ("text-to-video", "diffusers", "official-runtime"),
    },
    "modelscope-t2v": {
        "display_name": "ModelScopeT2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "modelscope-damo-text-to-video-synthesis",
            "ali-vilab--modelscope-damo-text-to-video-synthesis",
            fallback="ali-vilab/modelscope-damo-text-to-video-synthesis",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A greenhouse full of tropical plants during a gentle rainstorm, cinematic lighting, stable camera movement.",
        "default_call_kwargs": {"fps": 8, "seed": 302},
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("modelscope", "modelscope-text-to-video"),
        "tags": ("text-to-video", "diffusers", "official-runtime"),
    },
    "open-sora-plan": {
        "display_name": "Open-Sora-Plan",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "LanguageBind--Open-Sora-Plan-v1.3.0",
            "Open-Sora-Plan-v1.5.0",
            "LanguageBind--Open-Sora-Plan-v1.5.0",
            fallback="LanguageBind/Open-Sora-Plan-v1.3.0",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and "
            "reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        ),
        "default_call_kwargs": lambda: {
            "version": "v1_3",
            "model_subdir": "any93x640x640",
            "vae_subdir": "vae",
            "num_frames": 93,
            "height": 352,
            "width": 640,
            "fps": 18,
            "guidance_scale": 7.5,
            "num_sampling_steps": 100,
            "max_sequence_length": 512,
            "sample_method": "EulerAncestralDiscrete",
            "seed": 1234,
            "nproc_per_node": 1,
            "ae": "WFVAEModel_D8_4x8x8",
            "text_encoder_name_1": resolve._checkpoint_model_ref(
                "mt5-xxl", fallback=str(checkpoint_root_path("mt5-xxl"))
            ),
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("opensora-plan", "open-sora-plan-v1.3", "open-sora-plan-v1.5"),
        "tags": ("text-to-video", "official-runtime"),
    },
    "open-sora": {
        "display_name": "Open-Sora",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "OpenSora-STDiT-v3",
            "hpcai-tech--OpenSora-STDiT-v3",
            fallback="hpcai-tech/OpenSora-STDiT-v3",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "The street is damp and reflective, creating a mirror effect of the colorful lights. "
            "Many pedestrians walk about as the camera follows her with stable cinematic motion."
        ),
        "input_params": (
            "prompt",
            "num_frames",
            "resolution",
            "aspect_ratio",
            "num_sampling_steps",
            "guidance_scale",
            "aes",
            "seed",
        ),
        "default_call_kwargs": {
            "num_frames": 51,
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "num_sampling_steps": 30,
            "guidance_scale": 7.0,
            "aes": 6.5,
            "seed": 1024,
        },
        "default_load_kwargs": lambda: {
            "required_components": {
                "vae_path": resolve._checkpoint_model_ref(
                    "hpcai-tech--OpenSora-VAE-v1.2",
                    "OpenSora-VAE-v1.2",
                    fallback="hpcai-tech/OpenSora-VAE-v1.2",
                ),
                "text_encoder_path": resolve._checkpoint_model_ref(
                    "DeepFloyd--t5-v1_1-xxl",
                    fallback="DeepFloyd/t5-v1_1-xxl",
                ),
            }
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("opensora", "open-sora-v3"),
        "tags": ("text-to-video", "official-runtime"),
    },
    "wan2.1-vace": {
        "display_name": "Wan2.1 VACE",
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "Wan2.1-VACE-14B",
            "Wan2.1-VACE-1.3B-diffusers",
            "Wan-AI--Wan2.1-VACE-14B",
            fallback="Wan-AI/Wan2.1-VACE-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。"
            "她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。"
            "蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。"
            "小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。"
            "周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
        ),
        "call_params": (
            *_official_video_call_params(),
            "src_ref_images",
            "src_video",
            "src_mask",
            "vace_context_scale",
        ),
        "input_params": (
            "prompt",
            "src_ref_images",
            "src_video",
            "src_mask",
            "num_frames",
            "size",
            "num_inference_steps",
            "guidance_scale",
            "vace_context_scale",
            "seed",
            "nproc_per_node",
            "ulysses_size",
            "ring_size",
        ),
        "load_params": _official_video_load_params(),
        "default_load_kwargs": {"torch_dtype": "bfloat16"},
        "default_call_kwargs": {
            "num_frames": 81,
            "size": "1280*720",
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "seed": -1,
            "nproc_per_node": 1,
            "ulysses_size": 1,
            "ring_size": 1,
        },
        "aliases": ("wan-vace", "wan2.1-vace-14b", "wan-vace-14b"),
        "tags": ("video-to-video", "wan2.1", "official-runtime", "local-checkpoint"),
    },
    "unianimate-dit": {
        "display_name": "UniAnimate-DiT",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "UniAnimate-DiT",
            "ZheWang123--UniAnimate-DiT",
            fallback="ZheWang123/UniAnimate-DiT",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {"max_frames": 81, "num_inference_steps": 50, "seed": 0, "height": 832, "width": 480},
        "aliases": ("unianimate", "unianimate-dit-wan"),
        "tags": ("image-to-video", "human-animation", "official-runtime", "local-checkpoint"),
    },
    "sama-14b": {
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "SAMA-14B",
            "syxbb--SAMA-14B",
            fallback="syxbb/SAMA-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": str(resolve._data_path("test_cases", "sama", "1526909-hd_1920_1080_24fps.mp4")),
        "default_prompt": "Replace the spotted baby seal on the sand with a red crab.",
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {
            "max_frames": 49,
            "height": 480,
            "width": 832,
            "fps": 16,
            "seed": 1,
            "tiled": True,
            "prompt_prefix": True,
        },
        "aliases": ("sama", "sama-video"),
        "tags": ("video-to-video", "official-runtime", "local-checkpoint"),
    },
    "krea-realtime-video": {
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "krea-realtime-video",
            "krea--krea-realtime-video",
            fallback="krea/krea-realtime-video",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {"num_blocks": 2, "height": 480, "width": 832, "fps": 16, "seed": 42},
        "aliases": ("krea-video", "realtime-video"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "mmaudio": {
        "display_name": "MMAudio",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "MMAudio",
            "hkchengrex--MMAudio",
            fallback="hkchengrex/MMAudio",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "video-to-audio",
        "default_input_path": str(resolve._data_path("test_cases", "longcat_video", "motorcycle.mp4")),
        "default_prompt": "A motorcycle engine revs and passes along a road with realistic outdoor ambience.",
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {
            "video_path": str(resolve._data_path("test_cases", "longcat_video", "motorcycle.mp4")),
            "duration": 8,
            "num_steps": 25,
            "variant": "small_44k",
        },
        "aliases": ("mm-audio", "hkchengrex/MMAudio"),
        "tags": ("video-to-audio", "audio-generation", "official-runtime", "local-checkpoint"),
    },
    "step-video-t2v": {
        "display_name": "Step-Video-T2V",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "stepvideo-t2v",
            "stepfun-ai--stepvideo-t2v",
            fallback="stepfun-ai/stepvideo-t2v",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": (
            "prompt",
            "images",
            "video",
            "negative_prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "time_shift",
            "positive_magic",
            "negative_magic",
            "seed",
            "output_type",
        ),
        "load_params": (
            "model_path",
            "checkpoint_dir",
            "device",
            "torch_dtype",
            "offload_mode",
            *DISPATCH_LOAD_PARAMS,
        ),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "fps": 25,
            "num_frames": 204,
            "height": 544,
            "width": 992,
            "num_inference_steps": 50,
            "guidance_scale": 9.0,
            "time_shift": 13.0,
        },
        "aliases": ("stepvideo", "step-video"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "skyreels-v2": {
        "display_name": "SkyReels-V2",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "SkyReels-V2-DF-1.3B-540P",
            "Skywork--SkyReels-V2-DF-1.3B-540P",
            "Skywork--SkyReels-V2-T2V-14B-720P",
            fallback="Skywork/SkyReels-V2-DF-1.3B-540P",
        ),
        "default_prompt": (
            "A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway "
            "at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun "
            "casting long shadows across the barren landscape."
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_call_kwargs": {
            "num_frames": 97,
            "num_inference_steps": 30,
            "guidance_scale": 6.0,
            "shift": 8.0,
            "fps": 24,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "negative_prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "shift",
            "seed",
            "output_type",
        ),
        "load_params": (
            "model_path",
            "checkpoint_dir",
            "device",
            "torch_dtype",
            "offload_mode",
            *DISPATCH_LOAD_PARAMS,
        ),
        "default_load_kwargs": {"offload_mode": "block"},
        "aliases": ("skyreels2", "skyreels-v2-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "helios": {
        "display_name": "Helios",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "BestWishYsh--Helios-Distilled",
            "Helios-Distilled",
            fallback="BestWishYsh/Helios-Distilled",
        ),
        "default_prompt": (
            "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
            "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
        ),
        "default_task_type": "t2v",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_call_kwargs": {
            "num_frames": 240,
            "height": 384,
            "width": 640,
            "fps": 24,
            "num_inference_steps": 50,
            "guidance_scale": "auto",
            "seed": 42,
        },
        "call_params": (
            *_official_video_call_params(),
            "interactions",
            "realtime_segments",
            "pyramid_num_inference_steps_list",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments"),
        "load_params": _official_video_load_params(),
        "aliases": ("bestwishysh/helios",),
        "tags": (
            "text-to-video",
            "prompt-scheduled",
            "stream",
            "in-tree-runtime",
            "local-checkpoint",
        ),
        "notes": (
            "The resident Distilled runtime advances in native 33-frame segments and accepts prompt updates only "
            "at segment boundaries. Helios does not provide keyboard or camera controls."
        ),
    },
    "sana-streaming-2b-720p": {
        "display_name": "SANA-Streaming 2B 720p",
        "category": "Video Generation",
        "default_model_ref": lambda: _sana_streaming_default_ref(bidirectional=False),
        "default_prompt": (
            "A man in a dark suit sits at a table beside a bright window in a dim room, lowers his gaze, "
            "touches his forehead, and gestures with his hands; preserve his identity, the fixed camera, "
            "stable geometry, and consistent lighting."
        ),
        "default_task_type": "video-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": None,
        "default_load_kwargs": lambda: _sana_streaming_default_load_kwargs(bidirectional=False),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 704,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "seed": 42,
            "num_cached_blocks": 2,
            "sink_token": True,
        },
        "call_params": (
            *_official_video_call_params(),
            "negative_prompt",
            "flow_shift",
            "motion_score",
            "num_cached_blocks",
            "sink_token",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("sana-streaming", "sana-streaming-long", "sana-streaming-720p"),
        "tags": ("video-to-video", "streaming", "state-cache", "official-runtime", "in-tree-runtime"),
    },
    "sana-streaming-bidirectional-2b-720p": {
        "display_name": "SANA-Streaming Bidirectional 2B 720p",
        "category": "Video Generation",
        "default_model_ref": lambda: _sana_streaming_default_ref(bidirectional=True),
        "default_prompt": (
            "A man in a dark suit sits at a table beside a bright window in a dim room, lowers his gaze, "
            "touches his forehead, and gestures with his hands; preserve his identity, the fixed camera, "
            "stable geometry, and consistent lighting."
        ),
        "default_task_type": "video-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": None,
        "default_load_kwargs": lambda: _sana_streaming_default_load_kwargs(bidirectional=True),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 704,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": (
            *_official_video_call_params(),
            "negative_prompt",
            "flow_shift",
            "motion_score",
            "num_cached_blocks",
            "sink_token",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("sana-streaming-bidirectional", "sana-streaming-short"),
        "tags": ("video-to-video", "bidirectional", "official-runtime", "in-tree-runtime"),
    },
    "bernini": {
        "display_name": "Bernini",
        "category": "Video Generation",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "ByteDance--Bernini-Diffusers",
            fallback="ByteDance/Bernini-Diffusers",
        ),
        "default_task_type": "text-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"model_id": "bernini"},
        "default_call_kwargs": {
            "task_type": "t2v",
            "num_frames": 81,
            "height": 480,
            "width": 848,
            "num_inference_steps": 50,
            "fps": 16,
            "seed": 42,
            # The complete 81-frame/50-step profile fits one H100, while two
            # ranks duplicate roughly 179 GiB of host weights and leave too
            # little cgroup headroom for transient planner/renderer buffers.
            "nproc_per_node": 1,
            "ulysses_size": 1,
        },
        "call_params": (
            *_official_video_call_params(),
            "task_type",
            "neg_prompt",
            "max_image_size",
            "guidance_mode",
            "omega_vid",
            "omega_img",
            "omega_txt",
            "omega_tgt",
            "omega_scale",
            "planning_step",
            "vit_txt_cfg",
            "vit_img_cfg",
            "vit_denoising_step",
            "eta",
            "norm_threshold",
            "momentum",
            "system_prompt",
            "use_truncate",
            "max_sequence_length",
            "ulysses_size",
        ),
        "load_params": _official_video_load_params(),
        "aliases": (
            "bernini-diffusers",
            "bernini-7b-14b",
            "bernini-r-14b",
            "bernini-r-1.3b",
        ),
        "tags": ("text-to-video", "multi-task", "ulysses", "official-runtime", "in-tree-runtime"),
    },
    "wan-2p1-t2v": {
        "display_name": "Wan 2.1 T2V",
        "default_model_ref": _wan_2p1_t2v_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.1", "wan-2.1", "wan2p1", "wan2.1-t2v", "wan2p1-t2v"),
        "tags": ("text-to-video", "wan2.1", "native-diffusion", "local-checkpoint"),
    },
    "wan-2p1-i2v": {
        "display_name": "Wan 2.1 I2V",
        "default_model_ref": _wan_2p1_i2v_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan-2.1-i2v", "wan2p1-i2v", "wan2.1-i2v"),
        "tags": ("image-to-video", "wan2.1", "native-diffusion", "local-checkpoint"),
    },
    "wan2.2-t2v-a14b": {
        "display_name": "Wan2.2 T2V A14B",
        "default_model_ref": _wan_2p2_t2v_a14b_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cinematic white cat runs through a sunlit field, realistic motion.",
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "low_noise_guidance_scale": 3.0,
            "high_noise_guidance_scale": 4.0,
            "shift": 12.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": (
            *_native_diffusion_video_call_params(),
            "low_noise_guidance_scale",
            "high_noise_guidance_scale",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan22-t2v-a14b", "Wan-AI/Wan2.2-T2V-A14B"),
        "tags": ("text-to-video", "wan2.2", "dual-expert", "native-diffusion"),
    },
    "wan2.2-i2v-a14b": {
        "display_name": "Wan2.2 I2V A14B",
        "default_model_ref": _wan_2p2_i2v_a14b_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "The subject turns toward the camera as the scene moves naturally.",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 3.5,
            "low_noise_guidance_scale": 3.5,
            "high_noise_guidance_scale": 3.5,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": (
            *_native_diffusion_video_call_params(),
            "low_noise_guidance_scale",
            "high_noise_guidance_scale",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan22-i2v-a14b", "Wan-AI/Wan2.2-I2V-A14B"),
        "tags": ("image-to-video", "wan2.2", "dual-expert", "native-diffusion"),
    },
    "wan2.2-ti2v-5b": {
        "display_name": "Wan 2.2",
        "default_model_ref": _wan_2p2_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        # The 5B DiT fits comfortably on the Studio target H100.  Keep it
        # resident by default so the browser path does not pay one H2D copy
        # per transformer layer.  Lower-memory deployments can explicitly
        # select ``async-block``/``block`` and receive audited prefetch.
        "default_load_kwargs": {
            "offload_mode": "resident",
            # Exact production profile established by the Wan2.2 TI2V
            # framework A/B harness. Every acceleration remains explicitly
            # overrideable from the Studio runtime controls/load JSON.
            "fuse_qkv": False,
            "qkv_strategy": "auto",
            "qkv_split_threshold": 8192,
            "inplace_residual": False,
            "static_cross_kv": True,
            "fused_rope": False,
            "rope_precision": "fp32",
            "rms_norm_precision": "input",
        },
        "default_call_kwargs": {
            "height": 704,
            "width": 1280,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "fps": 24,
            "seed": 42,
        },
        "suggested_task_types": ("ti2v-5b", "interactive-video"),
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.2", "wan-2.2", "wan2p2"),
        "tags": ("video", "image-to-video", "text-to-video", "native-diffusion"),
    },
    "echo-infinity": {
        "display_name": "Echo-Infinity",
        "default_model_ref": _echo_infinity_default_ref,
        "default_load_kwargs": _echo_infinity_default_load_kwargs,
        "default_call_kwargs": {"fps": 16},
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "model_name",
            "wan_root",
            "generator_ckpt",
            "wan_model_name",
            "lora_ckpt",
            "adapter",
            "use_ema",
            "seed",
            "num_samples",
            "num_output_frames",
            "denoising_step_list",
            "warp_denoising_step",
            "num_frame_per_block",
            "context_noise",
            "global_sink",
            "model_kwargs",
            "memory_kwargs",
            "dtype",
            "low_memory",
        ),
        "tags": ("text-to-video", "in-tree-runtime"),
    },
    "echo-memory-context-k1": {
        "display_name": "Echo-Memory Context K=1",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _echo_memory_context_k1_default_ref,
        "default_load_kwargs": _echo_memory_context_k1_load_kwargs,
        "default_task_type": "i2v",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_prompt": (
            "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air; "
            "preserve his identity, stable architecture, coherent lighting, and smooth forward camera motion."
        ),
        "default_call_kwargs": {
            "num_frames": 81,
            "steps": 50,
            "guidance_scale": 5.0,
            "seed": 42,
            "camera_trajectory": "z*80",
            "return_dict": True,
        },
        "tags": ("image-to-video", "memory", "native-diffusion", "public-checkpoint"),
    },
    "recammaster": {
        "display_name": "ReCamMaster",
        "category": "Video-to-Video",
        "default_prompt": (
            "A first-person motorcycle ride along an elevated highway bridge, preserving "
            "the rider, road, guardrails, and distant landscape with smooth camera motion."
        ),
        "default_model_ref": _recammaster_default_ref,
        "default_load_kwargs": _recammaster_default_load_kwargs,
        "default_task_type": "video-to-video",
        "default_interactions": ("camera_path",),
        "default_call_kwargs": {
            "camera_trajectory": [100, 100, 0, 0, 30],
            "num_frames": 81,
            "num_inference_steps": 50,
            "cfg_scale": 5.0,
            "fps": 8,
            "video_path": "worldfoundry/data/test_cases/longcat_video/motorcycle.mp4",
        },
        "call_params": (
            "prompt",
            "video_path",
            "output_path",
            "camera_trajectory",
            "num_frames",
            "num_inference_steps",
            "cfg_scale",
            "fps",
            "max_num_frames",
            "frame_interval",
            "size",
        ),
        "tags": ("video-to-video", "camera-control", "official-runtime"),
    },
    "lingbot-video": {
        "display_name": "LingBot-Video",
        "default_model_ref": _lingbot_video_default_ref,
        "default_load_kwargs": {"variant": "dense"},
        "default_prompt": (
            '{"comprehensive_description":"A worker carefully places a red block into a matching tray on a '
            'clean workbench while the camera remains stable.","camera_info":{"frame_size":"Medium Shot",'
            '"shot_type_angle":"Eye Level","lighting_type":"Daylight"},"world_knowledge":[]}'
        ),
        "default_task_type": "t2v",
        "default_call_kwargs": {
            "mode": "t2v",
            "backend": "diffusers",
            "height": 480,
            "width": 832,
            "num_frames": 121,
            "num_inference_steps": 40,
            "guidance_scale": 3.0,
            "shift": 3.0,
            "seed": 42,
            "fps": 24,
            "execute": True,
            "timeout_seconds": 7200,
        },
        "load_params": ("variant", "python_executable", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "prompt",
            "prompt_json",
            "images",
            "negative_prompt",
            "negative_prompt_json",
            "mode",
            "backend",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "shift",
            "seed",
            "fps",
            "batch_cfg",
            "run_refiner",
            "reuse_condition_features",
            "refiner_height",
            "refiner_width",
            "refiner_steps",
            "refiner_guidance_scale",
            "refiner_shift",
            "cfg_parallel_degree",
            "context_parallel_degree",
            "nproc_per_node",
            "enable_fsdp_inference",
            "execute",
            "timeout_seconds",
            "return_dict",
            "output_dir",
            "output_path",
        ),
        "aliases": ("lingbot-video-dense", "lingbot-video-moe"),
        "tags": ("text-to-video", "image-to-video", "text-to-image", "official-runtime"),
        "notes": "Inference-only in-tree runtime. Structured JSON captions are required for release-quality generation.",
    },
    "longcat-video": {
        "display_name": "LongCat-Video",
        "default_model_ref": _longcat_video_default_ref,
        "default_load_kwargs": _longcat_video_default_load_kwargs,
        "default_call_kwargs": {
            "task_type": "t2v",
            "cpu_offload": True,
        },
        "load_params": ("python_executable", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "prompt",
            "negative_prompt",
            "task_type",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "distill_num_inference_steps",
            "guidance_scale",
            "seed",
            "fps",
            "context_parallel_size",
            "enable_compile",
            "cpu_offload",
            "run_base",
            "run_distill",
            "run_refiner",
            "max_sequence_length",
            "execute",
            "timeout_seconds",
            "return_dict",
            "output_dir",
            "output_path",
        ),
        "tags": ("text-to-video", "official-runtime"),
    },
    "pusa-vidgen": {
        "display_name": "Pusa VidGen",
        "default_model_ref": _pusa_vidgen_default_ref,
        "default_load_kwargs": _pusa_vidgen_default_load_kwargs,
        "default_call_kwargs": {
            "mode": "t2v",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "lightx2v": True,
            "high_lora_alpha": 1.5,
            "low_lora_alpha": 1.4,
            "seed": 0,
            "execute": True,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "output_dir",
            "return_dict",
            "task_type",
            "mode",
            "height",
            "width",
            "resolution",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "cfg_scale",
            "lightx2v",
            "high_lora_alpha",
            "low_lora_alpha",
            "cond_position",
            "noise_multipliers",
            "switch_dit_boundary",
            "fps",
            "quality",
            "negative_prompt",
            "seed",
            "execute",
            "timeout_seconds",
        ),
        "tags": ("text-to-video", "image-to-video", "official-runtime"),
    },
    "sana": {
        "display_name": "Sana",
        "default_model_ref": _sana_image_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "text-to-image",
        "default_prompt": (
            "A photorealistic still of a sunlit ceramic bowl of citrus on a wooden table, "
            "sharp focus, natural lighting."
        ),
        "default_load_kwargs": {"model_id": SANA_DEFAULT_IMAGE_VARIANT_ID},
        "default_call_kwargs": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
            "seed": 42,
        },
        "load_params": _sana_video_load_params(),
        "extra_variants": _sana_image_workspace_variants,
        "aliases": _sana_image_aliases(),
        "tags": ("text-to-image", "controlnet", "sprint", "official-runtime", "local-checkpoint"),
        "notes": (
            "Image, Sprint, and ControlNet checkpoints share this Studio card. "
            "Pick a variant for size, resolution, few-step Sprint, or HED ControlNet. "
            "Sana Video, LongSANA, and SANA-Streaming stay on their own cards."
        ),
        "summary": (
            "Sana text-to-image family. Select a checkpoint variant for parameter "
            "size, resolution, Sprint, or ControlNet."
        ),
    },
    "cogvideox_5b_t2v": {
        "display_name": "CogVideoX",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting."
        ),
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
        "extra_variants": _cogvideox_workspace_variants,
        "aliases": _cogvideox_aliases(),
        "tags": ("video", "t2v", "i2v", "official-runtime", "local-checkpoint"),
        "notes": (
            "2B T2V, 5B T2V, and 5B I2V share this Studio card. "
            "Pick a variant for size and text-to-video vs image-to-video."
        ),
        "summary": ("CogVideoX text-to-video and image-to-video family. Select 2B T2V, 5B T2V, or 5B I2V."),
    },
    "sana-video-2b-480p": {
        "display_name": "Sana Video 2B 480p",
        "default_model_ref": _sana_video_480p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "A cinematic tiger walks through a misty green forest, photorealistic, detailed lighting.",
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 480,
            "width": 832,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": _sana_video_call_params(),
        "stream_params": _sana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("sana-video", "sana-video-480p"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "sana-video-2b-720p": {
        "display_name": "Sana Video 2B 720p",
        "default_model_ref": _sana_video_720p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "A cinematic tiger walks through a misty green forest, photorealistic, detailed lighting.",
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": _sana_video_call_params(),
        "stream_params": _sana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("sana-video-720p",),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "longsana-video-2b-480p": {
        "display_name": "LongSANA Video 2B 480p",
        "default_model_ref": _longsana_video_480p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "Evening backlight and soft side lighting in a forest. A young man in a light shirt stands among the "
            "trees while golden sunlight filters through the leaves; wind gently moves his hair and collar, the "
            "camera holds a clean emotional mid-shot with a softly blurred background."
        ),
        "default_call_kwargs": {
            "negative_prompt": (
                "Bright tones, overexposed, static, blurred details, subtitles, paintings, overall gray, worst "
                "quality, low quality, JPEG artifacts, deformed limbs, fused fingers, messy background, walking backwards"
            ),
            "num_frames": 161,
            "height": 480,
            "width": 832,
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _longsana_video_call_params(),
        "stream_params": _longsana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("longsana-video", "sana-video-longlive"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "wan2.1-t2v-14b": {
        "display_name": "Wan2.1 T2V 14B",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "Wan2.1-T2V-14B",
            "Wan-AI--Wan2.1-T2V-14B",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "Two anthropomorphic cats in comfortable boxing gear and bright gloves fight intensely on a "
            "spotlighted stage, cinematic motion, detailed fur, stable anatomy."
        ),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.1-t2v-14b-720p",),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint", "720p"),
    },
    "wan2.1-i2v-14b-720p": {
        "display_name": "Wan2.1 I2V 14B 720P",
        "default_model_ref": _wan_2p1_i2v_720p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "The subject gently turns toward the camera while the scene moves naturally.",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 40,
            "guidance_scale": 5.0,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("Wan-AI/Wan2.1-I2V-14B-720P",),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint", "720p"),
    },
    "zeroscope": {
        "display_name": "ZeroScope",
        "default_model_ref": _zeroscope_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "research rover red desert under dusty sunset, slow cinematic pan, stable vehicle geometry",
        "default_call_kwargs": {
            "num_frames": 24,
            "fps": 8,
            "height": 320,
            "width": 576,
            "num_inference_steps": 40,
            "guidance_scale": 9.0,
            "seed": 301,
        },
        "default_load_kwargs": lambda: {"model_path": _zeroscope_default_ref()},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "infer_steps",
            "num_inference_steps",
            "cfg_scale",
            "guidance_scale",
            "height",
            "width",
            "seed",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("text-to-video", "diffusers", "local-checkpoint"),
    },
    "animatediff": {
        "display_name": "AnimateDiff",
        "default_model_ref": _animatediff_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "snowy street market at night with lanterns, cinematic camera movement, detailed people silhouettes",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "height": 256,
            "width": 256,
            "num_inference_steps": 25,
            "guidance_scale": 8.0,
            "seed": 303,
        },
        "default_load_kwargs": _animatediff_default_load_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "official_config_path",
            "config_path",
            "config",
            "timeout_seconds",
            "negative_prompt",
            "num_frames",
            "video_length",
            "infer_steps",
            "num_inference_steps",
            "cfg_scale",
            "guidance_scale",
            "height",
            "width",
            "seed",
            "allow_validation_fallback",
        ),
        "load_params": (
            "model_path",
            "motion_adapter_path",
            "motion_module_path",
            "motion_module",
            "base_model_path",
            "sd15_path",
            "dreambooth_model_path",
            "dreambooth_path",
            "official_python",
            "hf_hub_cache",
            "integrated_runtime_root",
            "inference_config",
            "negative_prompt",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "open-magvit2": {
        "display_name": "Open-MAGVIT2",
        "default_model_ref": _open_magvit2_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "class-conditional-image-generation",
        "default_load_kwargs": lambda: {
            "checkpoint_path": _open_magvit2_default_ref(),
            "config_path": "imagenet_conditional_llama_L.yaml",
        },
        "default_call_kwargs": {"class_id": 207, "batch_size": 1, "steps": 256},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "class_id",
            "batch_size",
            "steps",
            "temperature",
            "top_k",
            "top_p",
            "cfg_scale",
        ),
        "load_params": (
            "model_path",
            "checkpoint_path",
            "ckpt_path",
            "config_path",
            "config",
            "class_id",
            "batch_size",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("image-generation", "in-tree-runtime", "local-checkpoint"),
    },
    "skyreels-v3": {
        "display_name": "SkyReels V3",
        "default_model_ref": _skyreels_v3_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "reference_to_video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "task_type": "reference_to_video",
            "duration": 5,
            "resolution": "720P",
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "task_type",
            "duration",
            "seed",
            "resolution",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "image_guidance_scale",
            "negative_prompt",
            "fps",
            "output_type",
            "shift",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("skyreels-v3-r2v", "skyreels-v3-reference-to-video"),
        "tags": ("reference-to-video", "native-diffusion", "local-checkpoint"),
    },
    "kairos-sensenova": {
        "display_name": "Kairos Sensenova",
        "default_model_ref": _kairos_sensenova_default_ref,
        "default_load_kwargs": _kairos_sensenova_default_load_kwargs,
        "default_prompt": (
            "A grand natural scene unfolds in an open landscape, beginning with a wide, distant view of a massive "
            "waterfall cascading down a towering cliff that dominates the frame, its immense scale emphasized by the "
            "surrounding rock formations and mist-filled air. From the cliff's edge, enormous volumes of water surge "
            "downward in continuous streams, breaking into layered sheets and turbulent ribbons as gravity accelerates "
            "the flow, while countless fine droplets are torn from the main body of water and dispersed into the air, "
            "forming a drifting veil of mist that catches the sunlight and shimmers faintly. The waterfall's surface "
            "exhibits complex motion, with faster central currents plunging straight down and thinner side streams "
            "clinging briefly to the rock face before peeling away, creating visible variations in speed, thickness, "
            "and texture. As the camera slowly advances forward and downward, the perspective tightens, bringing the "
            "lower section of the waterfall into clearer view, where the falling water collides violently with the "
            "surface below, generating bursts of white foam and outward-splashing sprays. The camera continues to move "
            "closer, transitioning into a medium and then near-field view of the plunge pool at the base of the "
            "waterfall, revealing a circular water basin whose surface churns constantly under the impact. Within the "
            "pool, swirling currents rotate in overlapping patterns, with darker, deeper green-blue water visible "
            "beneath the frothy white surface, and concentric ripples propagate outward toward the rocky edges. "
            "Individual droplets and splashes are now discernible, arcing briefly through the air before rejoining the "
            "pool, while fine mist rises continuously and drifts laterally with subtle air movement. Wet rock surfaces "
            "around the pool glisten under natural light, reflecting soft highlights that shift with the camera's "
            "motion. Throughout the shot, the camera movement remains smooth and deliberate, steadily closing the "
            "distance from the monumental wide shot to an intimate, detailed view of the water's surface, emphasizing "
            "the physical continuity of flowing water, gravity-driven motion, and the dynamic interaction between "
            "falling streams, airborne droplets, and the turbulent pool below."
        ),
        "default_call_kwargs": {
            "negative_prompt": (
                "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra "
                "fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, walking backwards, "
                "contorted human joints, objects floating against natural forces, abrupt shot changes"
            ),
            "seed": 0,
            "tiled": True,
            "height": 480,
            "width": 640,
            "num_frames": 81,
            "cfg_scale": 5,
            "use_prompt_rewriter": False,
            "nproc_per_node": 1,
        },
        "call_params": (
            "prompt",
            "output_path",
            "negative_prompt",
            "seed",
            "tiled",
            "height",
            "width",
            "num_frames",
            "cfg_scale",
            "use_prompt_rewriter",
            "save_fps",
            "nproc_per_node",
            "master_port",
        ),
        "tags": ("robot-video", "official-runtime"),
    },
    "hydra": {
        "display_name": "HyDRA",
        "default_model_ref": _hydra_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "hydra", "condition.mp4")),
        "default_prompt": (
            "The video begins with a close-up of an individual clad in black armor with red and yellow lights, "
            "walking toward the camera on a city street."
        ),
        "default_load_kwargs": _hydra_default_load_kwargs,
        "default_call_kwargs": lambda: {
            "video_path": str(resolve._data_path("test_cases", "hydra", "condition.mp4")),
            "camera_json": str(resolve._data_path("test_cases", "hydra", "camera.json")),
            "num_frames": 77,
            "fps": 15,
            "width": 832,
            "height": 480,
            "num_inference_steps": 50,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "video",
            "video_path",
            "camera_json",
            "output_path",
            "num_frames",
            "fps",
            "width",
            "height",
            "num_inference_steps",
            "seed",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "checkpoint_path",
            "base_model_path",
            "python_executable",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("video-to-video", "camera-control", "official-runtime"),
    },
    "videocrafter1-i2v": {
        "display_name": "VideoCrafter1 I2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter1 image-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "VideoCrafter--Image2Video-512/model.ckpt",
            "VideoCrafter/videocrafter_i2v_512_v1.ckpt",
            "video_models/videocrafter_i2v_512_v1.ckpt",
        ),
        "default_prompt": "horses are walking on the grassland",
        "default_input_path": str(resolve._data_path("test_cases", "videocrafter", "i2v_prompts", "horse.png")),
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "num_inference_steps": 50,
            "height": 320,
            "width": 512,
            "seed": 123,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "fps",
            "return_dict",
            "frames",
            "num_frames",
            "frame_num",
            "ddim_steps",
            "num_inference_steps",
            "steps",
            "sample_steps",
            "infer_steps",
            "sampling_steps",
            "ddim_eta",
            "height",
            "width",
            "seed",
            "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter1_i2v",),
        "tags": ("video-generation", "image-to-video", "official-runtime"),
    },
    "videocrafter1-t2v": {
        "display_name": "VideoCrafter1 T2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter1 text-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "VideoCrafter--Text2Video-1024/model.ckpt",
            "VideoCrafter/videocrafter_t2v_1024_v1.ckpt",
            "video_models/videocrafter_t2v_1024_v1.ckpt",
        ),
        "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "num_inference_steps": 50,
            "height": 576,
            "width": 1024,
            "seed": 123,
        },
        "call_params": (
            "prompt",
            "output_path",
            "fps",
            "return_dict",
            "frames",
            "num_frames",
            "frame_num",
            "ddim_steps",
            "num_inference_steps",
            "steps",
            "sample_steps",
            "infer_steps",
            "sampling_steps",
            "ddim_eta",
            "height",
            "width",
            "seed",
            "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter1_t2v",),
        "tags": ("video-generation", "text-to-video", "official-runtime"),
    },
    "videocrafter2-t2v": {
        "display_name": "VideoCrafter2 T2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter2 text-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "VideoCrafter--VideoCrafter2/model.ckpt",
            "VideoCrafter/videocrafter_t2v_512_v2.ckpt",
            "video_models/videocrafter_t2v_512_v2.ckpt",
        ),
        "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 28,
            "num_inference_steps": 50,
            "height": 320,
            "width": 512,
            "seed": 123,
        },
        "call_params": (
            "prompt",
            "output_path",
            "fps",
            "return_dict",
            "frames",
            "num_frames",
            "frame_num",
            "ddim_steps",
            "num_inference_steps",
            "steps",
            "sample_steps",
            "infer_steps",
            "sampling_steps",
            "ddim_eta",
            "height",
            "width",
            "seed",
            "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter", "videocrafter2_t2v"),
        "tags": ("video-generation", "text-to-video", "official-runtime"),
    },
    "joyai-echo-longvideo": {
        "display_name": "JoyAI-Echo 1.5 / Echo-LongVideo",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _joyai_echo_longvideo_default_ref,
        "default_load_kwargs": _joyai_echo_longvideo_default_load_kwargs,
        "default_input_path": lambda: str(Path(_joyai_echo_longvideo_default_ref()) / "assets" / "image.png"),
        "default_prompt": (
            "A cinematic live-action traveler crosses a windswept coastal platform at golden hour while the "
            "camera tracks smoothly; preserve identity, clothing, environmental sound, and temporal continuity."
        ),
        "default_call_kwargs": {
            "execute": True,
            "seed": 42,
            "return_dict": True,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "memory_slots",
            "request_path",
            "num_frames",
            "width",
            "height",
            "seed",
            "fps",
            "output_path",
            "output_dir",
            "execute",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "checkpoint_dir",
            "gemma_path",
            "python_executable",
            "precision",
            "consumer",
            "low_vram",
            "device",
        ),
        "tags": ("audio-video", "long-video", "reference-to-video", "official-runtime"),
    },
    "vmem": {
        "display_name": "VMem",
        "default_model_ref": _vmem_default_ref,
        "default_load_kwargs": _vmem_default_load_kwargs,
        "default_prompt": (
            "Move forward along an empty Route 66 highway at sunset while preserving the road, "
            "horizon, and desert geometry."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "dualcamctrl", "demo_pic", "route66.jpg")),
        "default_interactions": ("forward",),
        "tags": ("stream", "memory"),
    },
    "neoverse": {
        "display_name": "NeoVerse",
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": _neoverse_default_ref,
        "default_load_kwargs": _neoverse_default_load_kwargs,
        "default_prompt": NEOVERSE_OFFICIAL_PROMPT,
        "default_input_path": str(resolve._data_path("test_cases", "neoverse", "videos", "robot.mp4")),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_call_kwargs": _neoverse_default_call_kwargs,
        "call_params": (
            "prompt",
            "videos",
            "images",
            "interactions",
            "predefined_trajectory",
            "trajectory_file",
            "trajectory_data",
            "output_path",
            "fps",
            "return_dict",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "cfg_scale",
            "seed",
            "trajectory_mode",
            "angle",
            "distance",
            "orbit_radius",
            "zoom_ratio",
            "alpha_threshold",
            "use_first_frame",
            "static_scene",
        ),
        "load_params": (
            "height",
            "width",
            "num_inference_steps",
            "cfg_scale",
            "disable_lora",
            "enable_vram_management",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("stream", "camera-control"),
    },
    "ltx-2.3-t2v": {
        "display_name": "LTX-2.3 T2V",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "t2v",
        "default_prompt": "Ocean waves break against black rocks at sunset with coherent water motion and cinematic light.",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            fallback="Lightricks/LTX-2.3",
        ),
        "default_load_kwargs": _ltx23_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 121,
            "fps": 24,
            "height": 512,
            "width": 768,
            "num_inference_steps": 11,
            "guidance_scale": 1.0,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "num_inference_steps",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "offload_mode",
            "quantization",
        ),
        "aliases": ("ltx2.3-t2v", "ltx2_3_t2v"),
        "tags": ("text-to-video", "joint-audio-video", "native-diffusion", "local-checkpoint"),
    },
    "ltx-2.3-i2v": {
        "display_name": "LTX-2.3 I2V",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "i2v",
        "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            fallback="Lightricks/LTX-2.3",
        ),
        "default_load_kwargs": _ltx23_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 121,
            "fps": 24,
            "height": 512,
            "width": 768,
            "num_inference_steps": 11,
            "guidance_scale": 1.0,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "num_inference_steps",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "offload_mode",
            "quantization",
        ),
        "aliases": ("ltx-2.x", "ltx2.3", "ltx-2.3", "ltx2.3-i2v", "ltx2_3_i2v"),
        "tags": ("image-to-video", "joint-audio-video", "native-diffusion", "local-checkpoint"),
    },
    "hyperflow": {
        "display_name": "HyperFlow for MiniMax-H3",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_backend": "from_pretrained",
        "default_task_type": "t2va",
        "suggested_task_types": ("t2va", "fl2va", "ref2va"),
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "minimax-h3", "MiniMaxAI--MiniMax-H3", fallback=str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
        ),
        "default_load_kwargs": {"gpus": 1},
        "default_call_kwargs": {"num_frames": 124, "seed": 42, "return_dict": True},
        "call_params": (
            "prompt",
            "images",
            "image",
            "last_image",
            "references",
            "workflow",
            "output_path",
            "return_dict",
            "num_frames",
            "height",
            "width",
            "seed",
            "fps",
            "num_inference_steps",
            "memory_reserve_margin",
            "offload",
            "attention_backend",
            "sol_attn",
            "timeout_seconds",
        ),
        "load_params": ("model_path", "weights_path", "python_executable", "device", "gpus"),
        "aliases": ("hyperflow-h3", "minimax-h3-hyperflow"),
        "tags": ("text-to-video", "image-to-video", "reference-to-video", "joint-audio-video", "official-runtime"),
        "notes": "Eight-step HyperFlow adapter; local H3 components and the dedicated inference environment are required.",
    },
    "pika-h3-flex": {
        "display_name": "Pika H3 Flex",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_backend": "from_pretrained",
        "default_task_type": "t2va",
        "suggested_task_types": ("t2va", "i2va", "l2va", "fl2va", "ref2va"),
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "pika-h3-flex", "MiniMaxAI--MiniMax-H3", fallback=str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
        ),
        "default_load_kwargs": {"adapter_task": "fl2va"},
        "default_call_kwargs": {"num_frames": 124, "seed": 0, "return_dict": True},
        "call_params": (
            "prompt",
            "images",
            "references",
            "task",
            "output_path",
            "return_dict",
            "num_frames",
            "short_edge",
            "aspect_ratio",
            "seed",
            "fps",
            "num_inference_steps",
        ),
        "load_params": ("base_model_path", "adapter_path", "adapter_task", "device"),
        "aliases": ("pika-h3", "pika-h3-flex-fl2va", "pika-h3-flex-ref2va"),
        "tags": ("text-to-video", "image-to-video", "reference-to-video", "joint-audio-video", "native-adapter"),
        "notes": "Six-step Pika H3 Flex adapter over the shared in-tree MiniMax-H3 runtime; GPU artifact parity is pending.",
    },
    "ltx-2.5": {
        "display_name": "LTX-2.5 Distilled",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "t2v",
        "default_prompt": "Ocean waves break against black rocks at sunset with coherent water motion, synchronized natural audio, and cinematic light.",
        "default_model_ref": _ltx25_checkpoint_dir,
        "default_load_kwargs": _ltx25_default_load_kwargs,
        "default_call_kwargs": {
            "execute": True,
            "timeout_seconds": 7200,
            "return_dict": True,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "execute",
            "timeout_seconds",
            "fps",
            "num_frames",
            "height",
            "width",
            "seed",
            "offload_mode",
            "quantization",
            "diffvae_optimization",
            "auto_duration",
            "num_generated_keyframes",
            "enhance_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "transformer_path",
            "text_encoder_path",
            "video_vae_path",
            "audio_vae_path",
            "spatial_upsampler_path",
            "duration_head_path",
            "prompt_enhancer_gemma_root",
            "offload_mode",
            "quantization",
            "diffvae_optimization",
        ),
        "aliases": ("ltx2.5", "ltx25", "ltx-2.5-distilled"),
        "tags": ("text-to-video", "image-to-video", "joint-audio-video", "native"),
        "notes": "Runs the native LTX-2.5 recipe in-process in the ltx-2.5 environment.",
    },
}

OVERRIDES.update(
    {
        "dreamdojo": {
            "display_name": "DreamDojo",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "default_model_ref": _dreamdojo_default_ref,
            "default_load_kwargs": _dreamdojo_default_load_kwargs,
            "default_task_type": "robot-world-generation",
            "default_prompt": "Predict the next GR-1 robot-view sequence from the recorded action trajectory.",
            "default_call_kwargs": {
                "seed": 42,
                "return_dict": True,
            },
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoints_dir",
                "vae_path",
                "text_encoder_path",
                "text_embeddings_path",
                "offload_mode",
                "checkpoint_shards",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "prompt",
                "images",
                "video",
                "interactions",
                "action_file",
                "output_path",
                "height",
                "width",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "return_dict",
            ),
            "tags": ("robot-world-model", "action-conditioned-video", "in-tree-runtime"),
        },
        "dreamdojo-student": {
            "display_name": "DreamDojo Student",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "supports_stream": True,
            "default_load_kwargs": lambda: {
                "vae_path": str(checkpoint_root_path("nvidia--Cosmos-Predict2.5-2B", "tokenizer.pth")),
                "text_encoder_path": str(checkpoint_root_path("nvidia--Cosmos-Reason1-7B")),
            },
            "default_task_type": "robot-world-generation",
            "default_prompt": "Predict the GR-1 robot-view sequence from normalized robot actions.",
            "default_call_kwargs": {
                "seed": 42,
                "return_dict": True,
            },
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "student_checkpoint_path",
                "vae_path",
                "text_encoder_path",
                "text_embeddings_path",
                "offload_mode",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "prompt",
                "images",
                "interactions",
                "action_file",
                "output_path",
                "height",
                "width",
                "fps",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "return_dict",
            ),
            "stream_params": (
                "prompt",
                "images",
                "interactions",
                "actions",
                "action_file",
                "height",
                "width",
                "fps",
                "num_inference_steps",
                "guidance_scale",
                "seed",
            ),
            "tags": ("robot-world-model", "action-conditioned-video", "streaming", "distilled", "in-tree-runtime"),
            "notes": "Explicit distilled student weights are required. Resident streaming consumes normalized 384-D arrays, not browser WASD controls; 10 fps is the playback clock, not measured generation throughput.",
        },
        "versecrafter": {
            "display_name": "VerseCrafter",
            "category": "Video Generation",
            "default_task_type": "i2v",
            "default_model_ref": _versecrafter_default_ref,
            "default_load_kwargs": _versecrafter_default_load_kwargs,
            "default_prompt": "A cinematic mountain valley with a gently moving camera and stable natural geometry.",
            "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "seed": 2025,
                "return_dict": True,
            },
            "call_params": (
                "prompt",
                "images",
                "image_path",
                "trajectory_npz",
                "output_path",
                "width",
                "height",
                "num_frames",
                "fps",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "return_dict",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoint_path",
                "base_model_path",
                "moge_model_path",
                "python_executable",
                "env",
            ),
            "aliases": ("verse-crafter", "TencentARC/VerseCrafter"),
            "tags": ("image-to-video", "camera-control", "4d-control", "in-tree-runtime"),
        },
        "t2v_turbo_t2v": {
            "generation_type": "t2v",
            "aliases": ("t2v-turbo-t2v", "t2v-turbo"),
            "default_prompt": "An astronaut riding a horse.",
            "default_model_ref": _t2v_turbo_base_ref,
            "default_load_kwargs": _t2v_turbo_default_load_kwargs,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "base_checkpoint",
                "model_ckpt",
                "lora_checkpoint",
                "lora_path",
                "torch_dtype",
                "weight_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 320,
                "width": 512,
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 8,
                "guidance_scale": 7.5,
                "seed": 0,
                "lcm_origin_steps": 200,
            },
        },
    }
)
