"""Curated overrides and defaults for interactive and action-conditioned world models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    checkpoint_root_candidates,
    checkpoint_root_path,
    hfd_root_path,
    local_data_root_path,
)

from .. import resolve
from ..params import DISPATCH_LOAD_PARAMS, _native_diffusion_video_call_params, _native_diffusion_video_load_params


def _gamma_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "mode": "causal_few_step",
        "model_path": resolve._checkpoint_model_ref(
            "Gamma-World",
            "chijw--Gamma-World",
            fallback="chijw/Gamma-World",
        ),
        "text_encoder_path": resolve._checkpoint_model_ref(
            "Cosmos-Reason1-7B",
            "nvidia--Cosmos-Reason1-7B",
            fallback="nvidia/Cosmos-Reason1-7B",
        ),
    }


def _yume_1p5_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Yume-5B-720P",
        "stdstu123--Yume-5B-720P",
        fallback="stdstu123/Yume-5B-720P",
    )


def _yume_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Yume-I2V-540P",
        "stdstu123--Yume-I2V-540P",
        fallback="stdstu123/Yume-I2V-540P",
    )


def _magicworld_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "LuckyLiGY--MagicWorld",
        fallback=str(hfd_root_path() / "LuckyLiGY--MagicWorld"),
    )


def _magicworld_default_load_kwargs() -> dict[str, str]:
    return {
        "base_model_path": resolve._checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP",
            fallback=str(hfd_root_path() / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP"),
        )
    }


def _lingbot_world_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates(
            "robbyant--lingbot-world-base-cam",
            "lingbot-world-base-cam",
        ),
        "robbyant/lingbot-world-base-cam",
    )


def _matrix_game_2_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("Skywork--Matrix-Game-2.0", "Matrix-Game-2.0"),
        "Skywork/Matrix-Game-2.0",
    )


def _matrix_game_2_default_config_path() -> str:
    return resolve._first_existing_path(
        resolve._data_path(
            "models", "runtime", "configs", "matrix_game_2", "inference_yaml", "inference_universal.yaml"
        ),
    )


def _matrix_game_2_default_load_kwargs() -> dict[str, Any]:
    return {"mode": "universal"}


def _matrix_game_2_default_call_kwargs() -> dict[str, Any]:
    return {
        "num_frames": 150,
        "size": [352, 640],
        "fps": 12,
        "seed": 42,
        "official_bench_actions": True,
        "visualize_ops": False,
        "visualize_warning": False,
        "config_path": _matrix_game_2_default_config_path(),
    }


def _matrix_game_3_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("Matrix-Game-3.0", "Skywork--Matrix-Game-3.0"),
        "Skywork/Matrix-Game-3.0",
    )


MATRIX_GAME_3_OFFICIAL_PROMPT = "A colorful, animated cityscape with a gas station and various buildings."


def _matrix_game_3_default_call_kwargs() -> dict[str, Any]:
    return {
        "num_iterations": 12,
        "num_inference_steps": 3,
        "size": "704*1280",
        "fps": 17,
        "seed": 42,
        "save_name": "test",
        "visualize_ops": True,
        "show_progress": True,
        "use_int8": True,
        "compile_vae": True,
        "lightvae_pruning_rate": 0.5,
        "vae_type": "mg_lightvae",
        "fa_version": "3",
        "use_async_vae": False,
        "async_vae_warmup_iters": 0,
    }


def _matrix_game_35_third_person_default_camera() -> dict[str, Any]:
    """Return a deterministic one-block c2w dolly trajectory for the Studio demo.

    Matrix-Game 3.5 consumes 84 generated poses plus its anchor/padding poses.
    Keeping the camera payload as JSON lets Workspace expose a complete default
    without checking a generated binary NPZ file into the source tree; the
    runtime materializes the mapping as ``camera.npz`` in the run directory.
    Intrinsics are expressed in pixels of the 1152x1536 portrait demo image.
    """

    pose_count = 86
    extrinsics_c2w = []
    for index in range(pose_count):
        progress = index / (pose_count - 1)
        # A gentle right-and-forward camera dolly keeps the reference subject
        # in frame while exercising non-static third-person camera control.
        extrinsics_c2w.append(
            [
                [1.0, 0.0, 0.0, 0.08 * progress],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.35 * progress],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return {
        "extrinsics_c2w": extrinsics_c2w,
        "intrinsics": [1100.0, 1100.0, 576.0, 768.0],
    }


def _matrix_game_35_third_person_default_call_kwargs() -> dict[str, Any]:
    return {
        "camera": _matrix_game_35_third_person_default_camera(),
        "fps": 16,
    }


def _matrix_game_35_default_ref(person: str) -> str:
    filename = f"{person}-person.safetensors"
    return resolve._checkpoint_model_ref(
        f"RiemannDynamics--Matrix-Game-3.5-Base/{filename}",
        f"Matrix-Game-3.5/Matrix-Game-3.5-Base/{filename}",
        f"Matrix-Game-3.5/{filename}",
        fallback="RiemannDynamics/Matrix-Game-3.5-Base",
    )


def _matrix_game_35_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "wan_dir": resolve._checkpoint_model_ref(
                "Wan-AI--Wan2.2-TI2V-5B",
                "Wan2.2-TI2V-5B",
                fallback="Wan-AI/Wan2.2-TI2V-5B",
            ),
            "da3_dir": resolve._checkpoint_model_ref(
                "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
                "DA3NESTED-GIANT-LARGE-1.1",
                fallback="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            ),
            "python_executable": resolve._worldfoundry_unified_python_executable(),
        }
    }


def _matrix_game_3_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "checkpoint_dir": resolve._prefer_existing_model_ref(
                *resolve._cache_candidates("Matrix-Game-3.0", "Skywork--Matrix-Game-3.0"),
            ),
            "vae_type": "mg_lightvae",
            "lightvae_pruning_rate": 0.5,
            "use_int8": True,
            "compile_vae": True,
            "use_async_vae": False,
            "async_vae_warmup_iters": 0,
        }
    }


def _matrix_game_1_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        *resolve._cache_candidates("Skywork--Matrix-Game", "Matrix-Game"),
        "Skywork/Matrix-Game",
    )


def _matrix_game_1_default_load_kwargs() -> dict[str, Any]:
    return {
        "conda_dir": resolve._prefer_existing_model_ref(
            os.getenv("WORLDFOUNDRY_MATRIX_GAME_1_CONDA_DIR", ""),
            str(resolve._project_root().parent / "envs" / "matrix-game-1"),
            str(resolve._project_root().parent / "envs" / "matrix-game-1.0"),
            str(resolve._project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128"),
            str(resolve._project_root().parent / "conda" / "envs" / "matrix-game-1.0"),
        ),
    }


def _causal_rcm_default_load_kwargs() -> dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("rcm")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "dit_path": str(checkpoint_dir / "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
        "vae_path": str(checkpoint_dir / "Wan2.1_VAE.pth"),
        "text_encoder_path": str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
    }


def _causal_rcm_default_call_kwargs() -> dict[str, Any]:
    return {
        # Frame-wise c1-1 chunks: the lowest-latency streaming schedule in the
        # Causal-rCM recipe, with the recommended 4-step distilled midpoints.
        "distilled": True,
        "first_chunk_t": 1,
        "chunk_t": 1,
        "num_steps": 4,
        "mid_t": ["15/16", "5/6", "5/8"],
        "num_frames": 81,
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "model_size": "1.3B",
        "seed": 0,
        # The checkpoint is not bundled with WorldFoundry, so Studio plans by
        # default. Set plan_only=False after staging a DiT checkpoint.
        "plan_only": True,
        "timeout_seconds": 21600,
    }


def _lingbot_world_v2_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "lingbot-world-v2-14b-causal-fast",
        "robbyant--lingbot-world-v2-14b-causal-fast",
        fallback="robbyant/lingbot-world-v2-14b-causal-fast",
    )


def _lingbot_world_v2_small_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "lingbot-world-v2-1.3b-causal-fast",
        "robbyant--lingbot-world-v2-1.3b-causal-fast",
        fallback="robbyant/lingbot-world-v2-1.3b-causal-fast",
    )


def _lingbot_world_v2_small_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "lingbot-world-v2-1.3b",
        "nproc_per_node": 1,
        "t5_fsdp": False,
        "dit_fsdp": False,
        "shared_assets_path": str(checkpoint_root_path("Wan-AI--Wan2.1-T2V-1.3B")),
    }


def _sana_wm_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "SANA-WM_bidirectional",
        "Efficient-Large-Model--SANA-WM_bidirectional",
        fallback="Efficient-Large-Model/SANA-WM_bidirectional",
    )


def _sana_wm_streaming_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Efficient-Large-Model--SANA-WM_streaming",
        "SANA-WM_streaming",
        fallback="Efficient-Large-Model/SANA-WM_streaming",
    )


def _sana_wm_streaming_default_load_kwargs() -> dict[str, Any]:
    streaming_root = _sana_wm_streaming_default_ref()
    gemma_root = resolve._checkpoint_model_ref(
        "Efficient-Large-Model--gemma-2-2b-it",
        fallback="Efficient-Large-Model/gemma-2-2b-it",
    )
    return {
        "checkpoint_overrides": {
            "dit": streaming_root,
            "text-encoder": gemma_root,
            "tokenizer": gemma_root,
            "codec": streaming_root,
            "refiner": streaming_root,
            "refiner-connectors": streaming_root,
            "refiner-text-encoder": streaming_root,
        }
    }


def _dreamx_world_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "DreamX-World-5B-Cam",
        "GD-ML--DreamX-World-5B-Cam",
        fallback="GD-ML/DreamX-World-5B-Cam",
    )


def _dreamx_world_ar_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "DreamX-World-5B",
        "GD-ML--DreamX-World-5B",
        fallback="GD-ML/DreamX-World-5B",
    )


def _dreamx_world_ar_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
    }


def _dreamx_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
    }


def _lingbot_world_v2_default_load_kwargs() -> dict[str, Any]:
    # Conda dispatch already selects the dedicated in-tree LingBot-V2
    # environment.  Leaving python_executable unset makes any fallback runner
    # use that child interpreter instead of escaping back to the unified env.
    return {
        "t5_fsdp": True,
        "dit_fsdp": True,
        "t5_cpu": False,
    }


def _solaris_default_ref() -> str:
    # The GitHub URL identifies the vendored runtime source, not a loadable
    # checkpoint.  Keep model_ref local so readiness checks and the runtime
    # both address the official Orbax snapshot.
    return resolve._checkpoint_model_ref(
        "nyu-visionx--solaris",
        "solaris",
        fallback=str(checkpoint_root_path("nyu-visionx--solaris")),
    )


def _solaris_default_load_kwargs() -> dict[str, Any]:
    # Keep the default data/work base; inference code is loaded from the installed package.
    runtime_root = str(
        resolve._project_root() / "worldfoundry" / "synthesis" / "visual_generation" / "solaris" / "solaris_runtime"
    )
    checkpoint_root = _solaris_default_ref()
    eval_data_root = resolve._checkpoint_model_ref(
        "nyu-visionx--solaris-eval-datasets",
        "solaris/datasets",
        fallback=str(checkpoint_root_path("nyu-visionx--solaris-eval-datasets")),
    )
    return {
        "required_components": {
            "runtime_root": runtime_root,
            "pretrained_model_dir": checkpoint_root,
            "eval_data_dir": eval_data_root,
            "output_dir": str(resolve._project_root() / "tmp" / "solaris_output"),
            "checkpoint_dir": checkpoint_root,
            "jax_cache_dir": str(resolve._project_root() / "tmp" / "solaris_jax_cache"),
            "model_weights_path": str(Path(checkpoint_root) / "solaris.pt"),
            "python_executable": resolve._worldfoundry_unified_python_executable(),
        }
    }


def _wow_default_ref() -> str:
    return resolve._hf_checkpoint_model_ref(
        "X-Humanoid--WoW-1-Wan-14B-600k",
        "X-Humanoid/WoW-1-Wan-14B-600k",
    )


def _minwm_default_ref() -> str:
    return resolve._hf_checkpoint_model_ref("MIN-Lab--minWM", "MIN-Lab/minWM")


def _minwm_hy_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_model_path": resolve._hf_checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "tencent/HunyuanVideo-1.5",
        )
    }


def _minwm_wan_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_model_path": resolve._hf_checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-1.3B",
        )
    }


def _lingbot_world_fast_load_kwargs() -> dict[str, Any]:
    fast_ref = resolve._prefer_existing_model_ref(
        *resolve._cache_candidates(
            "robbyant--lingbot-world-fast",
            "lingbot-world-fast",
        ),
        "robbyant/lingbot-world-fast",
    )
    return {
        "runtime_variant": "fast",
        "fast_model_path": fast_ref,
    }


def _lingbot_world_default_load_kwargs() -> dict[str, Any]:
    # FSDP is the trigger for adaptive 8/4-rank Workspace launch.  The
    # dispatcher injects ulysses_size=WORLD_SIZE after selecting visible GPUs.
    return {
        "t5_fsdp": True,
        "dit_fsdp": True,
        "t5_cpu": False,
    }


def lingbot_world_fast_load_kwargs() -> dict[str, Any]:
    return _lingbot_world_fast_load_kwargs()


def _infinite_world_default_ref() -> str:
    candidates = resolve._cache_candidates("MeiGen-AI--Infinite-World", "Infinite-World")
    for value in candidates:
        candidate = Path(value).expanduser()
        if (candidate / "checkpoints" / "infinite_world_model.ckpt").is_file():
            return str(candidate.resolve())
        if candidate.name == "checkpoints" and (candidate / "infinite_world_model.ckpt").is_file():
            return str(candidate.parent.resolve())
    return "MeiGen-AI/Infinite-World"


def _hunyuan_gamecraft_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Hunyuan-GameCraft-1.0",
        "tencent--Hunyuan-GameCraft-1.0",
        "hfd/tencent--Hunyuan-GameCraft-1.0",
        fallback="tencent/Hunyuan-GameCraft-1.0",
    )


def _gen3c_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "nvidia--GEN3C-Cosmos-7B",
        "GEN3C-Cosmos-7B",
        fallback="nvidia/GEN3C-Cosmos-7B",
    )


def _gen3c_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "transformer_model_path": _gen3c_default_ref(),
            "tokenizer_model_path": resolve._checkpoint_model_ref(
                "nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
                "Cosmos-Tokenize1-CV8x8x8-720p",
                fallback="nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
            ),
            "text_encoder_model_path": resolve._checkpoint_model_ref(
                "google-t5--t5-11b",
                "google-t5/t5-11b",
                "t5-11b",
                fallback="google-t5/t5-11b",
            ),
            "moge_pretrained": resolve._checkpoint_model_ref(
                "Ruicheng--moge-vitl",
                "Ruicheng/moge-vitl",
                "moge-vitl",
                fallback="Ruicheng/moge-vitl",
            ),
        }
    }


def _gen3c_default_call_kwargs() -> dict[str, Any]:
    """Native GEN3C quality defaults; legacy official-subprocess flags are omitted."""

    from worldfoundry.pipelines.gen3c.constants import (
        DEFAULT_GEN3C_NEGATIVE_PROMPT,
    )

    return {
        "negative_prompt": DEFAULT_GEN3C_NEGATIVE_PROMPT,
    }


def _alayaworld_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "checkpoint_path": resolve._checkpoint_model_ref(
                "AlayaWorld/merged_infer.safetensors",
                "AlayaLab--AlayaWorld/merged_infer.safetensors",
                fallback=str(checkpoint_root_path("AlayaWorld", "merged_infer.safetensors")),
            ),
            "gemma_root": resolve._checkpoint_model_ref(
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "google--gemma-3-12b-it-qat-q4_0-unquantized",
                fallback=str(checkpoint_root_path("gemma-3-12b-it-qat-q4_0-unquantized")),
            ),
            "da3_path": resolve._checkpoint_model_ref(
                "DA3NESTED-GIANT-LARGE-1.1",
                "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
                "Lyra-2.0/checkpoints/recon/model.pt",
                fallback=str(checkpoint_root_path("DA3NESTED-GIANT-LARGE-1.1")),
            ),
        }
    }


def _hunyuan_worldplay_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "HY-WorldPlay",
        "tencent--HY-WorldPlay",
        fallback="tencent/HY-WorldPlay",
    )


def _hunyuan_worldplay_default_call_kwargs() -> dict[str, Any]:
    return {
        "num_frames": 125,
        "aspect_ratio": "16:9",
        "num_inference_steps": 4,
        "seed": 1,
        "fps": 24,
        "output_type": "pt",
        "prompt_rewrite": False,
        "enable_sr": False,
        "return_pre_sr_video": False,
        "few_step": True,
        "chunk_latent_frames": 4,
        "model_type": "ar",
        "transformer_resident_ar_rollout": True,
        "user_width": 832,
        "user_height": 480,
    }


def _forcing_release_ref(*repo_dir_names: str) -> str:
    """Return the staged forcing release directory, or ``""`` so the native recipe resolves its checkpoint.

    The pipeline maps this directory to the generator role and reads the selected
    variant's file from it, so every Causal-Forcing variant shares one reference.
    """

    staged = resolve._checkpoint_model_ref(*repo_dir_names)
    return staged if staged and Path(staged).is_dir() else ""


def _self_forcing_default_ref() -> str:
    return _forcing_release_ref("gdhe17--Self-Forcing", "Self-Forcing")


def _causal_forcing_default_ref() -> str:
    return _forcing_release_ref("zhuhz22--Causal-Forcing", "Causal-Forcing")


_FORCING_DEFAULT_CALL_KWARGS: dict[str, Any] = {
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "fps": 16,
    "seed": 0,
    "return_dict": True,
}
_FORCING_T2V_CALL_PARAMS: tuple[str, ...] = (
    "prompt",
    "output_path",
    "output_dir",
    "num_frames",
    "height",
    "width",
    "num_inference_steps",
    "seed",
    "fps",
    "output_type",
    "return_dict",
)


def _oasis_500m_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Etched--oasis-500m/oasis500m.safetensors",
        fallback="Etched/oasis-500m",
    )


def _oasis_500m_default_load_kwargs() -> dict[str, str]:
    checkpoint_root = Path(
        resolve._checkpoint_model_ref(
            "Etched--oasis-500m",
            fallback="Etched/oasis-500m",
        )
    )
    return {
        "checkpoint_path": str(checkpoint_root / "oasis500m.safetensors"),
        "vae_ckpt": str(checkpoint_root / "vit-l-20.safetensors"),
        "prompt_path": str(resolve._data_path("test_cases", "oasis", "sample_image_0.png")),
        "actions_path": str(resolve._data_path("test_cases", "oasis", "forward_4f.one_hot_actions.json")),
    }


def _multiworld_ittakestwo_default_load_kwargs() -> dict[str, Any]:
    runtime_root = resolve._project_root() / "worldfoundry" / "synthesis" / "visual_generation" / "multiworld"
    config_path = resolve._data_path(
        "models",
        "runtime",
        "configs",
        "multiworld",
        "ittakestwo",
        "configs",
        "inference_480P_toy.yaml",
    )
    unified_python = resolve._project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128" / "bin" / "python"
    return {
        "required_components": {
            "runtime_root": str(runtime_root),
            "config_path": str(config_path),
            "checkpoint_path": resolve._checkpoint_model_ref(
                "MultiWorld/multiworld_480p_toydata.safetensors",
                "multiworld/multiworld_480p_toydata.safetensors",
                "Haoyuwu--MultiWorldCheckpoint/multiworld_480p_toydata.safetensors",
                "hfd/Haoyuwu--MultiWorldCheckpoint/multiworld_480p_toydata.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "MultiWorld",
                        "multiworld_480p_toydata.safetensors",
                    )
                ),
            ),
            "base_model_root": resolve._checkpoint_model_ref(
                "Wan-AI--Wan2.2-TI2V-5B",
                "Wan2.2-TI2V-5B",
                "hfd/Wan-AI--Wan2.2-TI2V-5B",
                fallback=str(checkpoint_root_path("Wan-AI--Wan2.2-TI2V-5B")),
            ),
            "vggt_root": resolve._checkpoint_model_ref(
                "VGGT-1B",
                "facebook--VGGT-1B",
                "hfd/facebook--VGGT-1B",
                fallback=str(checkpoint_root_path("facebook--VGGT-1B")),
            ),
            "python_executable": str(unified_python)
            if unified_python.is_file()
            else resolve._worldfoundry_unified_python_executable(),
            "derive_env_obv_from_image": True,
            "num_inference_steps": 35,
            "inference_seed": 0,
            "fps": 60,
        }
    }


def _multiworld_ittakestwo_default_call_kwargs() -> dict[str, Any]:
    return {
        "action_path": str(resolve._data_path("test_cases", "multiworld_ittakestwo", "action.csv")),
        "num_frames": 81,
        "num_inference_steps": 35,
        "inference_seed": 0,
        "derive_env_obv_from_image": True,
        "return_dict": True,
        "save_name": "multiworld_ittakestwo",
        "fps": 60,
    }


def _evoke_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Evoke",
        "AlayaLab--Evoke",
        fallback="AlayaLab/Evoke",
    )


def _evoke_default_load_kwargs() -> dict[str, Any]:
    return {
        "required_components": {
            "vigeo_path": resolve._checkpoint_model_ref(
                "Evoke/ViGeo1.1",
                "pkqbajng--ViGeo1.1",
                fallback=str(checkpoint_root_path("Evoke", "ViGeo1.1")),
            ),
            "python_executable": resolve._worldfoundry_unified_python_executable(),
        }
    }


def _leworldmodel_default_load_kwargs() -> dict[str, Any]:
    candidates = checkpoint_root_candidates("leworldmodel", "pusht_worldfoundry_smoke.h5")
    dataset_path = next((path for path in candidates if path.is_file()), candidates[0])
    return {
        "model_id": "leworldmodel",
        "config_name": "pusht",
        "policy": "random",
        "cache_dir": str(dataset_path.parent),
        "dataset_name": str(dataset_path),
        "num_eval": 1,
        "eval_budget": 25,
        "goal_offset_steps": 5,
        "seed": 42,
    }


def _hma_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "hma",
        "checkpoint_dir": str(checkpoint_root_path("liruiw--hma-base-cont")),
        "base_model_dir": str(checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")),
    }


def _tesseract_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "tesseract",
        "checkpoint_dir": str(checkpoint_root_path("anyeZHY--tesseract", "tesseract_v01e_rgbdn_sft")),
        "base_model_dir": str(checkpoint_root_path("THUDM--CogVideoX-5b-I2V")),
    }


def _ctrl_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "ctrl-world",
        "checkpoint_path": str(checkpoint_root_path("yjguo--Ctrl-World", "checkpoint-10000.pt")),
        "base_model_dir": str(checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")),
        "clip_model_dir": str(checkpoint_root_path("openai--clip-vit-base-patch32")),
    }


def _genie_envisioner_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "genie-envisioner",
        "checkpoint_path": str(checkpoint_root_path("agibot-world--Genie-Envisioner", "GE_base_fast_v0.1.safetensors")),
        "base_model_dir": str(checkpoint_root_path("Lightricks--LTX-Video")),
    }


def _giga_world_0_default_load_kwargs() -> dict[str, Any]:
    return {
        "model_id": "giga-world-0",
        "transformer_model_dir": str(checkpoint_root_path("open-gigaai--GigaWorld-0-Video-GR1-2b", "transformer")),
        "text_encoder_model_dir": str(checkpoint_root_path("google-t5--t5-11b-encoder")),
        "vae_model_dir": str(checkpoint_root_path("Wan-AI--Wan2.1-T2V-1.3B-Diffusers", "vae")),
    }


def _spatia_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Jinjing713--Spatia",
        fallback="Jinjing713/Spatia",
    )


def _spatia_default_load_kwargs() -> dict[str, Any]:
    return {
        "base_model_path": resolve._checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
        "map_model_path": resolve._checkpoint_model_ref(
            "facebook--map-anything",
            fallback="facebook/map-anything",
        ),
    }


def _spatia_default_call_kwargs() -> dict[str, Any]:
    return {
        "w2c_trajectory_file": str(resolve._data_path("test_cases", "spatia", "w2c_pan_right_121.jsonl")),
        "intrinsics": [[[768.0, 0.0, 624.0], [0.0, 768.0, 352.0], [0.0, 0.0, 1.0]]],
        "num_frames": 121,
        "fps": 24,
        "width": 1248,
        "height": 704,
        "num_inference_steps": 40,
        "seed": 20917,
        "return_dict": True,
    }


def _joyai_echo_wm_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Echo-Team--Echo-WM",
        fallback="Echo-Team/Echo-WM",
    )


def _joyai_echo_wm_default_load_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _joyai_echo_wm_default_ref(),
        "gemma_path": resolve._checkpoint_model_ref(
            "google--gemma-3-12b-it-qat-q4_0-unquantized",
            fallback="google/gemma-3-12b-it-qat-q4_0-unquantized",
        ),
        "python_executable": resolve._joyai_echo_python_executable("joyai-echo-wm"),
        "variant": "base",
    }


OVERRIDES: dict[str, dict[str, Any]] = {
    "hy-nomas": {
        "display_name": "Hy-Nomas",
        "default_backend": "from_pretrained",
        "default_model_ref": str(checkpoint_root_path("Hy-Nomas")),
        "default_prompt": "A camera explores a coherent landscape.",
        "default_task_type": "text-to-video",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "call_params": (
            *_native_diffusion_video_call_params(),
            "segments",
            "segment_latent_lengths",
            "camera_extrinsics",
            "kv_cache_quantization",
            "max_cache_segments",
            "cache_noise_sigma",
        ),
        "load_params": (*_native_diffusion_video_load_params(), "text_encoder_path", "low_memory"),
        "init_params": ("prompt", "seed", "height", "width"),
        "stream_params": ("prompt", "interactions", "realtime_segments"),
        "default_call_kwargs": {
            "height": 720,
            "width": 1280,
            "num_frames": 120,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "fps": 24,
        },
        "aliases": ("hynomas", "tencent/Hy-Nomas"),
        "tags": ("world-model", "text-to-video", "camera-control", "native-diffusion"),
    },
    "abot-world-0-5b-lf": {
        "display_name": "ABot-World-0-5B-LF",
        "category": "Video Generation",
        "summary": "In-tree causal action-conditioned world-video generation with resident KV cache.",
        "default_model_ref": lambda: str(checkpoint_root_path("ABot-World-0-5B-LF")),
        "default_prompt": "Move forward through the scene while preserving geometry and appearance.",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "num_frames": 57,
            "num_blocks": 5,
            "seed": 42,
            "fps": 16,
        },
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "reference_images",
            "num_frames",
            "num_blocks",
            "seed",
            "fps",
            "output_path",
            "return_dict",
        ),
        "supports_stream": True,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("abot-world", "abot"),
        "tags": ("world-model", "action-control", "causal-video", "in-tree-runtime"),
        "notes": "Uses the released full 57-frame rollout: one nine-frame warmup block followed by four twelve-frame causal blocks.",
    },
    "alayaworld": {
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": _alayaworld_default_load_kwargs,
        "default_call_kwargs": {"return_dict": True},
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "rounds",
            "height",
            "width",
            "sampling_steps",
            "seed",
            "camera_path",
            "camera_trajectory",
            "camera_translation_step",
            "camera_rotation_step_degrees",
            "intrinsic",
            "action_scale",
            "action_freq_scale",
            "action_history_memory",
            "spatial_enabled",
            "depth_backend",
            "spatial_num_context_frames",
            "spatial_retrieval_views",
            "spatial_downsample",
            "spatial_maximum_coverage",
            "spatial_retrieval_depth_threshold",
            "spatial_constant_depth",
            "spatial_include_sink",
            "spatial_require_full_context",
            "da3_process_res",
            "context_parallel",
            "decode_rank0_only",
            "compile_mode",
            "compile_backend",
            "compile_fullgraph",
            "compile_dynamic",
        ),
        "tags": ("world-model", "camera-control", "autoregressive-video", "in-tree-runtime"),
        "notes": "Uses the released Alaya checkpoint, local Gemma shards, and a compatible nested-giant DA3 checkpoint.",
    },
    "leworldmodel": {
        "display_name": "LeWorldModel",
        "category": "Video Generation",
        "summary": "Official PushT environment evaluation through the in-tree stable-worldmodel wrapper.",
        "default_task_type": "pusht-eval",
        "default_load_kwargs": _leworldmodel_default_load_kwargs,
        "default_call_kwargs": {"plan_only": False, "timeout_seconds": 21600},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "config_dir",
            "config_name",
            "policy",
            "cache_dir",
            "dataset_name",
            "num_eval",
            "eval_budget",
            "goal_offset_steps",
            "img_size",
            "seed",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("le-wm", "lewm"),
        "tags": ("world-model", "pusht", "official-runtime", "in-tree-runtime"),
        "notes": (
            "The default demo uses a locally staged PushT trajectory and the official random policy; "
            "learned-policy evaluation requires a compatible policy checkpoint."
        ),
    },
    "hma": {
        "display_name": "HMA",
        "category": "Video Generation",
        "summary": "STMAR continuous action-conditioned robot video generation.",
        "default_model_ref": lambda: str(checkpoint_root_path("liruiw--hma-base-cont")),
        "default_input_path": lambda: str(
            resolve.PIPELINES_ROOT.parent
            / "synthesis/visual_generation/hma/hma_runtime/assets/langtable_prompt/frame_00.png"
        ),
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _hma_default_load_kwargs,
        "default_call_kwargs": {
            "generated_frames": 6,
            "prompt_horizon": 3,
            "maskgit_steps": 2,
            "direction": "right",
            "action_scale": 0.05,
            "fps": 2,
            "seed": 42,
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "trajectory_path",
            "trajectory_json",
            "generated_frames",
            "num_frames",
            "prompt_horizon",
            "maskgit_steps",
            "direction",
            "action_scale",
            "seed",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "checkpoint_path",
            "hma_checkpoint_dir",
            "hma_magvit_checkpoint",
            "base_model_dir",
            "svd_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("hma-magvit-362m", "hma-continuous"),
        "tags": ("world-model", "robot-video", "action-conditioned", "official-runtime"),
        "notes": (
            "The default repeats one LangTable image and one raw two-dimensional action for smoke testing. "
            "For a paired rollout, pass trajectory_path with prompt_frames and raw_language_table_delta_yx actions; "
            "the official backbone normalizes actions internally. Continuous uses SVD VAE; discrete uses MAGVIT2."
        ),
    },
    "giga-world-0": {
        "display_name": "GigaWorld-0",
        "category": "Video Generation",
        "summary": "Released GR1 robot-world image-to-video generation with local T5 and Wan components.",
        "default_model_ref": lambda: str(checkpoint_root_path("open-gigaai--GigaWorld-0-Video-GR1-2b", "transformer")),
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")
        ),
        "default_prompt": "A robot arm moves toward the objects on the workbench.",
        "default_task_type": "robot-image-to-video",
        "default_load_kwargs": _giga_world_0_default_load_kwargs,
        "default_call_kwargs": {
            "attention_backend": "natten",
            "height": 480,
            "width": 640,
            "num_frames": 61,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "fps": 16,
            "seed": 6666,
            "max_text_length": 128,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "attention_backend",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "max_text_length",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "transformer_model_dir",
            "transformer_model_path",
            "checkpoint_path",
            "checkpoint",
            "text_encoder_model_dir",
            "text_encoder_model_path",
            "vae_model_dir",
            "vae_model_path",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("giga_world_0", "gigaworld-0", "gigaworld0"),
        "tags": ("world-model", "robot-video", "image-to-video", "official-runtime"),
        "notes": (
            "The default uses the released 480x640, 61-frame, 30-step NATTEN sampling settings. "
            "Use a checkpoint-matched GR1 observation for dataset parity."
        ),
    },
    "genie-envisioner": {
        "display_name": "Genie Envisioner",
        "category": "Video Generation",
        "summary": "Official GE-base multiview robot world-video generation with local LTX components.",
        "default_model_ref": lambda: str(
            checkpoint_root_path("agibot-world--Genie-Envisioner", "GE_base_fast_v0.1.safetensors")
        ),
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_prompt": "Move the robot gripper toward the objects on the workbench.",
        "default_task_type": "robot-multiview-video",
        "default_load_kwargs": _genie_envisioner_default_load_kwargs,
        "default_call_kwargs": {
            "input_mode": "explicit-three-view",
            "input_views": [
                str(resolve._data_path("test_cases", "test_vla_case1", "aloha", name))
                for name in (
                    "observation_images_cam_high.png",
                    "observation_images_cam_left_wrist.png",
                    "observation_images_cam_right_wrist.png",
                )
            ],
            "height": 192,
            "width": 256,
            "n_previous": 4,
            "latent_chunk": 2,
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_timestep": 0.03,
            "decode_noise_scale": 0.025,
            "fps": 8,
            "seed": 42,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "input_views",
            "view_images",
            "input_mode",
            "height",
            "width",
            "n_previous",
            "latent_chunk",
            "num_inference_steps",
            "guidance_scale",
            "decode_timestep",
            "decode_noise_scale",
            "seed",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "checkpoint",
            "ckpt_path",
            "genie_checkpoint_path",
            "base_model_dir",
            "ltx_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("genie_envisioner", "genieenvisioner"),
        "tags": ("world-model", "robot-video", "multiview", "official-runtime"),
        "notes": (
            "For official GE-base history, pass input_views as three directories in head, left-wrist, "
            "right-wrist order, each with 0.png through 3.png. The default three ALOHA still "
            "images repeat across four history slots and are smoke inputs only. "
            "Use a checkpoint-matched AgiBot or LeRobot trajectory for action-quality parity."
        ),
    },
    "ctrl-world": {
        "display_name": "Ctrl-World",
        "category": "Video Generation",
        "summary": "Official DROID action-conditioned three-camera robot world-video generation.",
        "default_model_ref": lambda: str(checkpoint_root_path("yjguo--Ctrl-World", "checkpoint-10000.pt")),
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_prompt": "Move the robot gripper toward the objects on the workbench.",
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _ctrl_world_default_load_kwargs,
        "default_call_kwargs": {
            "input_mode": "explicit-three-view",
            "input_views": [
                str(resolve._data_path("test_cases", "test_vla_case1", "aloha", name))
                for name in (
                    "observation_images_cam_high.png",
                    "observation_images_cam_left_wrist.png",
                    "observation_images_cam_right_wrist.png",
                )
            ],
            "action_mode": "absolute-pose",
            "action_direction": "right",
            "action_distance": 0.08,
            "height": 192,
            "width": 320,
            "num_frames": 5,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "motion_bucket_id": 127,
            "fps": 4,
            "seed": 42,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "input_views",
            "view_images",
            "input_mode",
            "action_mode",
            "initial_pose",
            "action_direction",
            "action_distance",
            "action_scale",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "motion_bucket_id",
            "seed",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "checkpoint",
            "ckpt_path",
            "ctrl_world_checkpoint_path",
            "base_model_dir",
            "svd_model_path",
            "clip_model_dir",
            "clip_model_path",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("ctrl_world", "ctrlworld"),
        "tags": ("world-model", "robot-video", "action-conditioned", "official-runtime"),
        "notes": (
            "Physical action generation requires a seven-value DROID Cartesian initial_pose; the default ALOHA images are demonstration views only. "
            "Use synchronized DROID cameras and the corresponding physical pose for action-quality evaluation. "
            "The old zero-pose conditioning is available only with action_mode='synthetic-zero-smoke'."
        ),
    },
    "tesseract": {
        "display_name": "TesserAct",
        "category": "Video Generation",
        "summary": "Official RGB-depth-normal robot world-video generation with local CogVideoX components.",
        "default_model_ref": lambda: str(checkpoint_root_path("anyeZHY--tesseract", "tesseract_v01e_rgbdn_sft")),
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")
        ),
        "default_prompt": "The robot moves its gripper toward the object on the table.",
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _tesseract_default_load_kwargs,
        "default_call_kwargs": {
            "geometry_mode": "synthetic-gradient",
            "height": 256,
            "width": 320,
            "num_frames": 49,
            "num_inference_steps": 4,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "use_dynamic_cfg": False,
            "fps": 8,
            "seed": 23,
            "mixed_precision": "bf16",
            "memory_efficient": True,
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "depth_path",
            "normal_path",
            "geometry_mode",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "image_guidance_scale",
            "use_dynamic_cfg",
            "seed",
            "mixed_precision",
            "memory_efficient",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "checkpoint_path",
            "tesseract_checkpoint_dir",
            "weights_path",
            "base_model_dir",
            "cogvideox_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("tesser-act", "tesseract-rgbdn"),
        "tags": ("world-model", "robot-video", "rgb-depth-normal", "official-runtime"),
        "notes": (
            "The default synthetic-gradient geometry is a deterministic checkpoint smoke aid, not an official depth/normal estimate. "
            "Supply depth_path and normal_path with geometry_mode=files for released-model input parity."
        ),
    },
    "diamond": {
        "default_interactions": ("forward_left",),
        "supports_stream": True,
        "aliases": ("wmfactory-diamond",),
        "tags": ("interactive-world", "navigation", "game-video", "wmfactory"),
        "notes": "WMFactory-compatible Diamond id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "mira": {
        "display_name": "MIRA",
        "category": "Video Generation",
        "default_model_ref": lambda: str(checkpoint_root_path("mira")),
        "default_load_kwargs": {
            "dataset_path": str(local_data_root_path() / "rocket-science" / "test"),
        },
        "default_prompt": (
            "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
            "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "rocket-science-rollout",
        "call_params": (
            "prompt",
            "interactions",
            "output_path",
            "fps",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "dataset_path",
            "actions",
            "actions_file",
            "seed",
            "clip_index",
            "n_context_frames",
            "num_unrolled_frames",
            "n_diffusion_steps",
            "schedule_type",
            "noise_level",
            "compile",
            "overlay_actions",
            "generated_only",
        ),
        "aliases": ("mira-wm", "multiplayer-interactive-world-model"),
        "tags": ("world-model", "rocket-league", "multiplayer", "action-conditioned"),
        "notes": "Inference-only MIRA Workspace entry backed by the vendored upstream checkpoint and Rocket Science loaders.",
    },
    "oasis-500m": {
        "display_name": "Oasis 500M",
        "default_model_ref": _oasis_500m_default_ref,
        "default_input_path": lambda: _oasis_500m_default_load_kwargs()["prompt_path"],
        "default_load_kwargs": _oasis_500m_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 32,
            "fps": 20,
            "ddim_steps": 10,
            "return_dict": True,
        },
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "ddim_steps",
            "prompt_path",
            "actions_path",
            "timeout_seconds",
            "plan_only",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "oasis_ckpt",
            "vae_ckpt",
            "vae_checkpoint_path",
            "prompt_path",
            "actions_path",
        ),
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("open-oasis", "openoasis", "open_oasis"),
        "tags": ("interactive-world", "navigation", "minecraft", "wmfactory"),
        "notes": "WMFactory Open-Oasis compatibility id; WorldFoundry canonical model id is oasis-500m.",
    },
    "vid2world": {
        "display_name": "Vid2World",
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("vid-2-world",),
        "tags": ("interactive-world", "navigation", "game-video", "wmfactory"),
        "notes": "WMFactory-compatible Vid2World id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "mineworld": {
        "display_name": "MineWorld",
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("mine-world",),
        "tags": ("interactive-world", "navigation", "minecraft", "wmfactory"),
        "notes": "WMFactory-compatible MineWorld id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "causal-rcm": {
        "display_name": "Causal-rCM",
        "default_model_ref": lambda: str(checkpoint_root_path("rcm")),
        "default_load_kwargs": _causal_rcm_default_load_kwargs,
        "default_call_kwargs": _causal_rcm_default_call_kwargs,
        "default_prompt": "A cinematic shot of a snowy mountain at sunrise",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "autoregressive-video-generation",
        "call_params": (
            "prompt",
            "negative_prompt",
            "images",
            "output_path",
            "fps",
            "dit_path",
            "checkpoint_path",
            "vae_path",
            "text_encoder_path",
            "image_path",
            "model_size",
            "distilled",
            "first_chunk_t",
            "chunk_t",
            "num_steps",
            "steps_per_chunk",
            "mid_t",
            "mid_t_schedules",
            "sigma_max",
            "guidance_scale",
            "timestep_shift",
            "num_frames",
            "num_samples",
            "resolution",
            "aspect_ratio",
            "adaptive_resolution",
            "context_from_last_step",
            "context_from_last_step_start_chunk",
            "kv_cache_policy",
            "kv_cache_window_blocks",
            "kv_cache_sink_blocks",
            "warmup_iters",
            "num_runs",
            "seed",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "dit_path",
            "checkpoint_path",
            "checkpoint_dir",
            "vae_path",
            "text_encoder_path",
        ),
        "aliases": ("rcm", "causal_rcm", "rcm-causal"),
        "tags": ("interactive-world", "streaming", "autoregressive-video", "distillation", "wan"),
        "notes": (
            "Vendored NVlabs Causal-rCM streaming runtime (Apache-2.0). Block-causal Wan2.1 distilled to "
            "1-4 steps per chunk; chunk_t selects the streaming schedule. Requires a distilled DiT "
            "checkpoint plus the Wan2.1 VAE and umT5 text encoder."
        ),
    },
    "matrix-game-1": {
        "default_model_ref": _matrix_game_1_default_ref,
        "default_input_path": "worldfoundry/data/test_cases/matrix-game-1/official_initial_image/forest_00.jpg",
        "default_load_kwargs": _matrix_game_1_default_load_kwargs,
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "fps": 16,
            "execute": True,
            "inference_steps": 50,
            "max_images": 1,
            "max_conditions": 1,
            "video_length": 17,
            "i2v_type": "refiner",
            "timeout_seconds": 3600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "image_path",
            "output_path",
            "fps",
            "return_dict",
            "execute",
            "timeout_seconds",
            "inference_steps",
            "max_images",
            "max_conditions",
            "video_length",
            "gpu_id",
            "i2v_type",
        ),
        "aliases": ("matrix-game1",),
        "tags": ("navigation", "interaction-heavy", "in-tree-runtime", "gpu-validation-recorded", "visual-qa-pending"),
        "notes": "Defaults use the official 50-step Matrix-Game-1 runner with a bounded 17-frame Studio regression shape and an in-domain GameWorldScore initial image staged in-tree. Generic Studio images generated noisy visuals and are not accepted as official visual parity.",
    },
    "matrix-game-2": {
        "default_model_ref": _matrix_game_2_default_ref,
        "default_load_kwargs": _matrix_game_2_default_load_kwargs,
        "default_interactions": (),
        "default_call_kwargs": _matrix_game_2_default_call_kwargs,
        "aliases": ("matrixgame", "matrixgame2", "matrix-game2"),
        "tags": ("navigation", "stream", "interaction-heavy", "wmfactory", "state-init"),
        "notes": "Resident interactive runtime. The initial image and all navigation/camera actions come from the user; no private benchmark fixture or action trajectory is selected by default.",
    },
    "matrix-game-3": {
        "default_model_ref": _matrix_game_3_default_ref,
        "default_prompt": MATRIX_GAME_3_OFFICIAL_PROMPT,
        "default_load_kwargs": _matrix_game_3_default_load_kwargs,
        "default_interactions": ("forward", "camera_r"),
        "default_call_kwargs": _matrix_game_3_default_call_kwargs,
        "aliases": ("matrixgame3", "matrix-game3"),
        "tags": ("navigation", "stream", "wmfactory", "state-init", "official-demo"),
        "notes": "Defaults mirror the Matrix-Game-3 README cityscape demo: 12 iterations, 3 denoising steps, seed 42, 704*1280, INT8 LightVAE. Async VAE remains off by default because it fails when CUDA_VISIBLE_DEVICES remaps GPUs.",
    },
    "matrix-game-3.5-first-person": {
        "display_name": "Matrix-Game 3.5 First-Person Base",
        "default_model_ref": lambda: _matrix_game_35_default_ref("first"),
        "default_task_type": "camera-controlled-image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves gently forward and "
            "to the right, preserving his face, clothing, and the illuminated architecture."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": (),
        "default_load_kwargs": _matrix_game_35_default_load_kwargs,
        "default_call_kwargs": _matrix_game_35_third_person_default_call_kwargs,
        "input_params": ("image_path", "camera"),
        "call_params": (
            "prompt",
            "images",
            "image",
            "image_path",
            "input_path",
            "camera_path",
            "camera",
            "trajectory_npz",
            "caption_path",
            "output_path",
            "fps",
            "num_blocks",
            "steps",
            "cfg_scale",
            "seed",
            "camera_convention",
            "keep_workspace",
            "timeout_seconds",
            "return_dict",
        ),
        "aliases": ("matrix-game-3.5-fp", "matrix-game-35-first-person"),
        "tags": ("world-model", "camera-control", "first-person", "official-default-steps"),
        "notes": (
            "Studio uses the released one-block 25-step first-person base model and a deterministic "
            "86-pose c2w dolly trajectory materialized by the runtime."
        ),
    },
    "matrix-game-3.5-third-person": {
        "display_name": "Matrix-Game 3.5 Third-Person Base",
        "default_model_ref": lambda: _matrix_game_35_default_ref("third"),
        "default_task_type": "camera-controlled-image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves gently around him, "
            "preserving his identity, clothing, and the illuminated architecture."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": (),
        "default_load_kwargs": _matrix_game_35_default_load_kwargs,
        "default_call_kwargs": _matrix_game_35_third_person_default_call_kwargs,
        "input_params": ("image_path", "camera"),
        "call_params": (
            "prompt",
            "images",
            "image",
            "image_path",
            "input_path",
            "camera_path",
            "camera",
            "trajectory_npz",
            "refs",
            "subject_refs",
            "caption_path",
            "output_path",
            "fps",
            "num_blocks",
            "steps",
            "cfg_scale",
            "seed",
            "camera_convention",
            "keep_workspace",
            "timeout_seconds",
            "return_dict",
        ),
        "aliases": ("matrix-game-3.5-tp", "matrix-game-35-third-person"),
        "tags": ("world-model", "camera-control", "subject-reference", "official-default-steps"),
        "notes": (
            "Studio uses the released one-block 25-step base-model defaults and a deterministic 86-pose "
            "c2w dolly trajectory materialized by the runtime. Subject references remain optional."
        ),
    },
    "self-forcing": {
        "display_name": "Self-Forcing",
        "category": "Video Generation",
        "default_model_ref": _self_forcing_default_ref,
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_prompt": (
            "A cinematic shot of a red sports car driving along a coastal highway at sunset, "
            "with realistic motion, rich detail, and stable lighting."
        ),
        "default_call_kwargs": _FORCING_DEFAULT_CALL_KWARGS,
        "call_params": _FORCING_T2V_CALL_PARAMS,
        "stream_params": ("prompt",),
        "load_params": (*_native_diffusion_video_load_params(), "kv_window"),
        "aliases": ("selfforcing", "gdhe17/self-forcing"),
        "tags": ("autoregressive-video", "native", "prompt-scheduled", "wan2.1", "world-model"),
        "notes": (
            "Native four-step Self-Forcing DMD rollout on the autoregressive-window runner: 480x832, 81 frames "
            "at 16 fps in three-latent-frame blocks. The generator loads from the staged gdhe17/Self-Forcing "
            "release; the Wan2.1 T2V 1.3B text encoder, tokenizer and VAE resolve from their recipe sources. "
            "Realtime commits and decodes one block per chunk and reopens the rollout on a new prompt."
        ),
    },
    "causal-forcing": {
        "display_name": "Causal-Forcing",
        "category": "Video Generation",
        "default_model_ref": _causal_forcing_default_ref,
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_prompt": (
            "A cinematic shot of a futuristic train crossing a mountain valley under dramatic clouds, "
            "with coherent motion and high visual detail."
        ),
        "default_call_kwargs": _FORCING_DEFAULT_CALL_KWARGS,
        "call_params": (*_FORCING_T2V_CALL_PARAMS, "images", "image_path"),
        "stream_params": ("prompt",),
        "load_params": (*_native_diffusion_video_load_params(), "variant"),
        "aliases": ("causalforcing", "thu-ml/causal-forcing"),
        "tags": ("autoregressive-video", "native", "prompt-scheduled", "wan2.1", "world-model"),
        "notes": (
            "Native Causal-Forcing rollout on the autoregressive-window runner: 480x832, 81 frames at 16 fps. "
            "The default is the chunk-wise student; variant selects causal-forcing-framewise, which also takes "
            "one image as its first frame, or the Causal Forcing++ causal-forcing-framewise-1step and -2step "
            "students. Each variant loads its generator from the staged zhuhz22/Causal-Forcing release."
        ),
    },
    "gen3c": {
        "display_name": "GEN3C",
        "default_model_ref": _gen3c_default_ref,
        "default_input_path": "worldfoundry/data/test_cases/gen3c/image.png",
        "default_prompt": (
            "A cinematic view of a futuristic science-fiction city with coherent structures, realistic "
            "lighting, and smooth natural camera motion."
        ),
        "default_load_kwargs": _gen3c_default_load_kwargs,
        "default_interactions": ("left",),
        "default_call_kwargs": _gen3c_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "prompt",
            "negative_prompt",
            "trajectory",
            "camera_path",
            "region_hint",
            "camera_rotation",
            "movement_distance",
            "rendered_warp_images",
            "rendered_warp_masks",
            "camera_to_world",
            "camera_intrinsics",
            "scene_name",
            "output_dir",
            "return_dict",
            "guidance",
            "num_steps",
            "num_video_frames",
            "fps",
            "height",
            "width",
            "seed",
            "num_gpus",
            "noise_aug_strength",
            "save_buffer",
            "filter_points_threshold",
            "foreground_masking",
            "disable_prompt_upsampler",
            "disable_guardrail",
            "disable_prompt_encoder",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
        ),
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": (
            "navigation",
            "stream",
            "camera-control",
        ),
        "notes": ("GEN3C supports camera trajectories and RGBD-conditioned video generation."),
    },
    "lingbot-world": {
        "display_name": "LingBot-World",
        "default_model_ref": _lingbot_world_default_ref,
        "default_load_kwargs": _lingbot_world_default_load_kwargs,
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "max_area": 480 * 832,
            "num_frames": 161,
            "sampling_steps": 70,
            "seed": 42,
        },
        "load_params": (
            "task",
            "runtime_variant",
            "fast_model_path",
            "t5_fsdp",
            "dit_fsdp",
            "t5_cpu",
            "ulysses_size",
            "nproc_per_node",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "convert_model_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "action_path",
            "resize_H",
            "resize_W",
            "max_area",
            "num_frames",
            "vis_ui",
            "allow_act2cam",
            "action_string",
            "wmfactory_action_controls",
            "sampling_steps",
            "offload_model",
            "seed",
            "output_path",
            "fps",
            "return_dict",
        ),
        "aliases": ("lingbot-fast", "lingbot-world-fast", "lingbotworldfast", "lingbotworld-fast"),
        "tags": ("navigation", "stream", "camera-control", "interaction-heavy", "wmfactory", "state-init"),
        "notes": (
            "Base-cam uses native DiT/T5 FSDP plus Ulysses sequence parallelism. Workspace selects the official "
            "8-GPU topology when available and supports 4 GPUs as the compact topology; only rank 0 decodes "
            "and materializes the output."
        ),
    },
    "sana-wm": {
        "display_name": "SANA-WM",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _sana_wm_default_ref,
        "default_prompt": "",
        "default_task_type": "image-camera-video",
        "default_interactions": ("forward", "left", "right", "camera_up", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "window_frames": 81,
            "fps": 16,
            "step": 60,
            "cfg_scale": 5.0,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": ("checkpoint_source", "checkpoint_overrides", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "window_frames",
            "fps",
            "step",
            "cfg_scale",
            "seed",
            "return_dict",
        ),
        "stream_params": ("interactions", "realtime_segments", "seed"),
        "aliases": ("sana-world-model", "sana-wm-2.6b"),
        "tags": ("world-model", "image-to-video", "camera-control", "stream", "bidirectional", "in-tree-runtime"),
        "notes": "Resident two-stage SANA-WM runtime with user-provided image/prompt, native 8k+1 temporal windows, and configurable component placement.",
    },
    "dreamx-world-5b-cam": {
        "display_name": "DreamX-World 5B Cam",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _dreamx_world_default_ref,
        "default_load_kwargs": _dreamx_world_default_load_kwargs,
        "default_prompt": (
            "A blocky coastal cliff overlooks a calm ocean at sunset as the camera glides "
            "smoothly along the shoreline, with stable terrain, vegetation, and warm lighting."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "dreamx_world", "007.jpg")),
        "default_task_type": "image-camera-video",
        "default_interactions": (),
        "default_call_kwargs": {
            "num_frames": 33,
            "fps": 16,
            "height": 704,
            "width": 1280,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": (
            "checkpoint_source",
            "wan_model_path",
            "nproc_per_node",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "num_frames",
            "fps",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "return_dict",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments", "seed"),
        "aliases": ("dreamx-5b-cam",),
        "tags": (
            "interactive-world",
            "image-to-video",
            "camera-control",
            "stream",
            "in-tree-runtime",
            "user-input-only",
        ),
        "notes": (
            "Resident 1+4k camera-conditioned segments with user-provided image and prompt. "
            "Weights, VAE, T5 embeddings, and Ulysses topology stay resident; each continuation "
            "starts from the previous final frame without file round trips."
        ),
    },
    "dreamx-world-5b": {
        "display_name": "DreamX-World 5B AR",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _dreamx_world_ar_default_ref,
        "default_load_kwargs": _dreamx_world_ar_default_load_kwargs,
        "default_prompt": (
            "A blocky coastal cliff overlooks a calm ocean at sunset as the camera glides "
            "smoothly along the shoreline, with stable terrain, vegetation, and warm lighting."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "dreamx_world", "007.jpg")),
        "default_task_type": "image-camera-video",
        "default_interactions": (),
        "default_call_kwargs": {
            "fps": 16,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": (
            "checkpoint_source",
            "wan_model_path",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "fps",
            "seed",
            "return_dict",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments", "seed"),
        "aliases": ("dreamx-world", "dreamx", "dreamx-ar", "dreamx-5b"),
        "tags": (
            "interactive-world",
            "image-to-video",
            "camera-control",
            "stream",
            "causal",
            "autoregressive",
            "in-tree-runtime",
            "user-input-only",
        ),
        "notes": (
            "Resident distilled causal runtime with three-latent blocks, persistent attention/KV "
            "and VAE caches, continuous camera pose, and no decoded-RGB feedback loop."
        ),
    },
    "lingbot-world-v2": {
        "display_name": "LingBot-World-V2",
        "default_model_ref": _lingbot_world_v2_default_ref,
        "default_load_kwargs": _lingbot_world_v2_default_load_kwargs,
        "default_prompt": (
            "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's "
            "blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle "
            "approaches steadily, its stone details becoming clearer against the backdrop of floating islands "
            "and distant waterfalls."
        ),
        "default_task_type": "image-camera-video",
        "default_call_kwargs": {
            "seed": 42,
            "sample_shift": 10.0,
            "nproc_per_node": 1,
            "offload_model": False,
            "return_dict": True,
            "timeout_seconds": 7200,
        },
        "load_params": (
            "checkpoint_source",
            "python_executable",
            "t5_fsdp",
            "dit_fsdp",
            "t5_cpu",
            "convert_model_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "action_path",
            "input_dir",
            "output_path",
            "return_dict",
            "frame_num",
            "num_frames",
            "size",
            "chunk_size",
            "seed",
            "sample_shift",
            "local_attn_size",
            "sink_size",
            "nproc_per_node",
            "t5_fsdp",
            "dit_fsdp",
            "offload_model",
            "timeout_seconds",
        ),
        "aliases": ("lingbot-world-infinity", "lingbot-v2", "lingbot_world_v2"),
        "tags": ("world-model", "image-to-video", "camera-control", "causal", "official-runtime"),
        "notes": (
            "Fully in-tree causal-fast runtime. action_path must contain poses.npy and intrinsics.npy. The "
            "official profile uses 8 GPUs; Workspace also supports 4 GPUs, reuses its existing process group "
            "without nested torchrun, and materializes output only on rank 0."
        ),
    },
    "infinite-world": {
        "display_name": "Infinite-World",
        "default_model_ref": _infinite_world_default_ref,
        "default_prompt": (
            "A young man holds a sparkling firework at night while the camera explores the illuminated "
            "architecture with smooth, coherent motion."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {"num_frames": 81, "seed": 42, "fps": 30, "return_dict": True},
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "num_chunks",
            "num_frames",
            "negative_prompt",
            "seed",
            "output_path",
            "output_dir",
            "fps",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "args",
            "weight_dtype",
            "config_path",
            "checkpoint_path",
            "bucket_config_name",
            "num_sampling_steps",
            "shift",
            "guidance_scale",
            "negative_prompt",
            "seed",
        ),
        "aliases": ("infiniteworld", "infinite_world"),
        "tags": ("navigation", "stream", "camera-control", "interaction-heavy", "wmfactory", "state-init"),
        "notes": "Supports action-conditioned stream continuation in Studio. Start from an image-conditioned first turn, then continue with short navigation tokens.",
    },
    "yume": {
        "display_name": "YUME",
        "default_model_ref": _yume_default_ref,
        "default_prompt": (
            "A first-person journey follows the stone path toward an arched bridge beside a calm lake, "
            "with coherent forward motion and stable natural scenery."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_interactions": ("forward", "camera_l"),
        "default_load_kwargs": {"fsdp": False, "t5_cpu": True},
        "default_call_kwargs": {
            "interactions": ["forward", "camera_l"],
            "task_type": "i2v",
            "size": "544*960",
            "sampling_method": "ode",
            "interaction_speeds": [100, 4],
            "interaction_distances": [4, None],
            "num_euler_timesteps": 50,
            "seed": 42,
        },
        "aliases": ("yume-i2v",),
        "tags": ("stream", "egocentric", "wmfactory", "state-init"),
    },
    "yume-1p5": {
        "display_name": "YUME-1.5",
        "default_model_ref": _yume_1p5_default_ref,
        "default_prompt": (
            "A cinematic first-person journey follows the sunlit stone path toward an arched bridge beside "
            "reflective water, with stable forward motion and detailed natural scenery."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"fsdp": False},
        "default_call_kwargs": {
            "interactions": ["forward", "left", "camera_r"],
            "task_type": "i2v",
            "size": "704*1280",
            "interaction_speeds": [100, 100, 4],
            "interaction_distances": [4, 4, None],
            "num_euler_timesteps": 4,
            "seed": 42,
        },
        "aliases": ("yume1.5", "yume-1.5"),
        "tags": ("stream", "egocentric", "wmfactory", "state-init"),
    },
    "wow": {
        "display_name": "WoW",
        "default_model_ref": _wow_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_image_case1", "init_frame.png")),
        "default_prompt": "The Franka robot grasps the red bottle on the table.",
        "default_call_kwargs": {
            "no_tiled": False,
            "enable_vram_management": True,
            "persistent_param_gb": 70,
        },
        "call_params": (
            "input_path",
            "text_prompt",
            "prompt",
            "steps",
            "seed",
            "num_frames",
            "frames",
            "no_tiled",
            "enable_vram_management",
            "no_vram_management",
            "persistent_param_gb",
            "output_path",
            "return_dict",
        ),
        "aliases": ("wow-world-model", "world-omniscient-world-model", "wow-world-model/wow-1-wan-14b-600k"),
        "tags": ("world-model", "image-to-video", "robotics", "official-runtime"),
        "notes": "Defaults match the official WoW Wan demo checkpoint X-Humanoid/WoW-1-Wan-14B-600k. Use a local Hugging Face cache or repo id with image conditioning, prompt text, seed 42, and video generation.",
    },
    "evoke": {
        "display_name": "Evoke",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _evoke_default_ref,
        "default_load_kwargs": _evoke_default_load_kwargs,
        "default_prompt": (
            "A cinematic camera glides through a sunlit forest trail while leaves move gently "
            "and the scene geometry remains stable."
        ),
        "default_call_kwargs": {
            "mode": "t2v",
            "num_frames": 33,
            "num_chunks": 1,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "video",
            "video_path",
            "output_path",
            "return_dict",
            "trajectory_npz",
            "pose_path",
            "camera_trajectory",
            "mode",
            "sample_type",
            "use_geometric_state",
            "width",
            "height",
            "num_chunks",
            "num_frames",
            "fps",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "ref_seconds",
            "start_seconds",
            "lingbot_pose_source_fps",
            "lingbot_pose_source_resolution",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "checkpoint_path",
            "base_model_path",
            "transformer_path",
            "vigeo_path",
            "runtime_root",
            "python_executable",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("alayalab/evoke", "evoke-world-model"),
        "tags": ("world-model", "text-to-video", "image-to-video", "video-to-video", "in-tree-runtime"),
        "notes": "Uses the local post-distillation Evoke checkpoint and the shared WorldFoundry Python environment; camera-trajectory runs additionally use local ViGeo1.1 weights.",
    },
    "minwm-hy-action2v": {
        "default_model_ref": _minwm_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "minwm", "first_frame.png")),
        "default_prompt": (
            "A rain-soaked brick path runs beside a long wooden aviary building as the camera "
            "moves smoothly forward, with wet reflections and stable geometry."
        ),
        "default_load_kwargs": _minwm_hy_default_load_kwargs,
        "default_call_kwargs": {
            "trajectory": "a*4,w*8,s*7",
            "num_frames": 77,
            "num_inference_steps": 4,
            "fps": 16,
            "width": 832,
            "height": 480,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "trajectory",
            "output_path",
            "num_frames",
            "num_inference_steps",
            "fps",
            "width",
            "height",
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
        "tags": ("world-model", "action-conditioned-video", "camera-control"),
    },
    "minwm-wan-action2v": {
        "display_name": "minWM Wan Action2V",
        "default_model_ref": _minwm_default_ref,
        "default_prompt": "A serene garden path winds through manicured greenery under a soft overcast sky.",
        "default_load_kwargs": _minwm_wan_default_load_kwargs,
        "default_call_kwargs": {
            "trajectory": "a*4,w*8,s*7",
            "num_output_frames": 20,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "trajectory",
            "output_path",
            "num_output_frames",
            "seed",
            "sp_size",
            "master_port",
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
        "tags": ("world-model", "action-conditioned-video", "camera-control"),
    },
    "magicworld": {
        "display_name": "MagicWorld",
        "default_model_ref": _magicworld_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "minwm", "first_frame.png")),
        "default_prompt": (
            "A rain-soaked brick path beside a long wooden aviary building, viewed with a "
            "smooth forward camera move and stable wet-weather geometry."
        ),
        "default_load_kwargs": _magicworld_default_load_kwargs,
        "default_call_kwargs": lambda: {
            "native_rows": str(
                resolve._data_path(
                    "benchmarks",
                    "assets",
                    "iworld-bench",
                    "camera_trajectories",
                    "inference_txt",
                    "camera_1_1_0.txt",
                )
            ),
            "num_frames": 81,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "native_rows",
            "output_path",
            "num_frames",
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
        "tags": ("world-model", "image-to-video", "camera-control"),
    },
    "gamma-world": {
        "display_name": "Gamma-World",
        "default_input_path": str(resolve._data_path("test_cases", "gamma_world", "buildTower_normal")),
        "default_prompt": "Two Minecraft players buildTower in normal world",
        "default_load_kwargs": _gamma_world_default_load_kwargs,
        "default_call_kwargs": {
            "n_players": 2,
            "num_frames": 189,
            "num_conditional_frames": 1,
            "height": 320,
            "width": 480,
            "guidance": 5.0,
            "seed": 1,
            "fps": 16,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "guidance",
            "guidance_scale",
            "actions",
            "action_paths",
            "eval_dir",
            "n_players",
            "num_frames",
            "frames",
            "num_conditional_frames",
            "height",
            "width",
            "num_inference_steps",
            "infer_steps",
            "negative_prompt",
            "seed",
            "fps",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "mode",
            "checkpoint_overrides",
            "text_encoder_path",
            "reason1_path",
            "torch_dtype",
            "weight_dtype",
            "dtype",
            "offload_mode",
        ),
        "tags": ("multi-agent-world-model", "action-conditioned-video", "official-runtime"),
    },
    "solaris": {
        "display_name": "Solaris",
        "default_model_ref": _solaris_default_ref,
        "default_load_kwargs": _solaris_default_load_kwargs,
        # Leave eval_num_samples unset so Solaris derives it inside the leased
        # worker from CUDA_VISIBLE_DEVICES. Hard-coding the host's eight-sample
        # upstream batch on a one-GPU Workspace lease exhausts the card during
        # the first 257-frame evaluation step.
        "default_call_kwargs": {"eval_types": "translation"},
        "call_params": (
            "eval_types",
            "experiment_name",
            "output_dir",
            "eval_num_samples",
            "num_workers",
            "num_frames_eval",
            "enable_jax_cache",
            "checkpoint_dir",
            "jax_cache_dir",
            "model_weights_path",
            "return_dict",
            "show_progress",
        ),
        "tags": ("world-model", "official-runtime"),
        "notes": "Uses the installed Solaris package; required_components.runtime_root anchors relative data paths and the working directory.",
    },
    "spatia": {
        "display_name": "Spatia",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _spatia_default_ref,
        "default_load_kwargs": _spatia_default_load_kwargs,
        "default_input_path": str(resolve._data_path("test_cases", "neoverse", "videos", "jungle.png")),
        "default_prompt": (
            "A smooth lateral camera move through a dense jungle, preserving the trees, rocks, foliage, "
            "lighting, and scene geometry across the full trajectory."
        ),
        "default_call_kwargs": _spatia_default_call_kwargs,
        "input_params": ("image_path",),
        "call_params": (
            "prompt",
            "images",
            "video",
            "image_path",
            "video_path",
            "w2c_trajectory_file",
            "intrinsics",
            "num_frames",
            "fps",
            "width",
            "height",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "base_model_path",
            "lora_path",
            "map_model_path",
            "python_executable",
            "device",
        ),
        "tags": ("camera-control", "image-to-video", "world-model", "in-tree-runtime"),
    },
    "joyai-echo-wm": {
        "display_name": "Echo-WM",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _joyai_echo_wm_default_ref,
        "default_load_kwargs": _joyai_echo_wm_default_load_kwargs,
        "default_interactions": ("w-60", "a-60"),
        "default_prompt": (
            "Environment: A warm fantasy workshop with arched windows, hanging lanterns, herbs, and glass "
            "flasks. Character: One woman in a lavender dress reads at a wooden workbench. Style: whimsical "
            "hand-painted fantasy. Perspective: wide third-person interior view. Sounds: paper rustle, soft "
            "lantern chimes, and quiet room ambience. Speech: no dialogue."
        ),
        "default_call_kwargs": {
            "action_str": "w-60,a-60",
            "execute": True,
            "seed": 4,
            "return_dict": True,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "action_str",
            "num_frames",
            "width",
            "height",
            "fps",
            "seed",
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
            "variant",
            "device",
        ),
        "tags": ("action-control", "audio-video", "camera-control", "world-model", "official-runtime"),
    },
    "hunyuan-game-craft": {
        "display_name": "Hunyuan Game Craft",
        "default_model_ref": _hunyuan_gamecraft_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "hunyuan_game_craft", "village.png")),
        "default_prompt": (
            "A charming medieval village with cobblestone streets, thatched-roof houses, "
            "and vibrant flower gardens under a bright blue sky."
        ),
        "default_interactions": ("forward", "backward", "right", "left"),
        "default_load_kwargs": {"seed": 250160},
        "load_params": ("cpu_offload", *DISPATCH_LOAD_PARAMS),
        "default_call_kwargs": {
            "interactions": ("forward", "backward", "right", "left"),
            "interaction_speed": (0.2, 0.2, 0.2, 0.2),
            "size": (704, 1216),
            "num_frames": 129,
            "infer_steps": 50,
            "cfg_scale": 2.0,
        },
        "aliases": ("hunyuan-gamecraft", "gamecraft", "tencent/Hunyuan-GameCraft-1.0"),
        "tags": ("interactive-world", "camera-control", "game-video", "local-checkpoint"),
    },
    "scope": {
        "display_name": "SCOPE",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "SCOPE",
            "zizhaotong--SCOPE",
            "custom--SCOPE",
            fallback="zizhaotong/SCOPE",
        ),
        "default_task_type": "image-action-to-video",
        "default_prompt": (
            "In a whimsical, toy-inspired garden, the first-person view reveals a tactical weapon aimed forward "
            "along a sunlit path."
        ),
        "default_input_path": str(resolve._data_path("test_cases", "scope", "example_0", "image.png")),
        "default_call_kwargs": {
            "execute": True,
            "action_path": str(resolve._data_path("test_cases", "scope", "example_0", "action.parquet")),
            "max_frames": 81,
            "height": 480,
            "width": 832,
            "num_inference_steps": 30,
            "seed": 0,
            "return_dict": True,
        },
        "default_load_kwargs": lambda: {
            "model_dir": resolve._checkpoint_model_ref("SCOPE", "zizhaotong--SCOPE", "custom--SCOPE")
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "execute",
            "action_path",
            "operator_kwargs",
            "height",
            "width",
            "max_frames",
            "num_inference_steps",
            "seed",
        ),
        "input_params": (
            "prompt",
            "input_path",
            "action_path",
            "max_frames",
            "height",
            "width",
            "num_inference_steps",
            "seed",
        ),
        "load_params": (
            "model_path",
            "model_dir",
            "checkpoint_root",
            "checkpoint_dir",
            "python_executable",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("scope-world-model", "zizhaotong/scope"),
        "tags": ("world-model", "action-conditioned-video", "official-demo", "in-tree-runtime"),
        "notes": "Defaults use the official SCOPE example_0 image/action parquet and run execute=True against Hugging Face checkpoint assets.",
    },
    "hunyuan-worldplay": {
        "display_name": "HY-WorldPlay",
        "category": "Video Generation",
        "default_model_ref": _hunyuan_worldplay_default_ref,
        "default_input_path": str(resolve._data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_prompt": (
            "A paved pathway leads towards a stone arch bridge spanning a calm body of water.  "
            "Lush green trees and foliage line the path and the far bank of the water. "
            "A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. "
            "The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, "
            "natural light, creating a tranquil and serene atmosphere. The pathway is composed of "
            "large, rectangular stones, and the bridge is constructed of light gray stone.  The overall "
            "composition emphasizes the peaceful and harmonious nature of the landscape."
        ),
        "default_interactions": ("w-31",),
        "default_load_kwargs": {
            "enable_offloading": True,
            "enable_group_offloading": True,
            "overlap_group_offloading": False,
        },
        "default_call_kwargs": _hunyuan_worldplay_default_call_kwargs,
        "load_params": (
            "enable_offloading",
            "enable_group_offloading",
            "overlap_group_offloading",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "nproc_per_node",
            *DISPATCH_LOAD_PARAMS,
        ),
        "supports_stream": True,
        "stream_params": (),
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "interactions",
            "pose",
            "num_frames",
            "aspect_ratio",
            "num_inference_steps",
            "negative_prompt",
            "seed",
            "fps",
            "output_type",
            "prompt_rewrite",
            "enable_sr",
            "sr_num_inference_steps",
            "return_pre_sr_video",
            "few_step",
            "chunk_latent_frames",
            "model_type",
            "transformer_resident_ar_rollout",
            "user_height",
            "user_width",
            "forward_speed",
            "yaw_speed_deg",
            "pitch_speed_deg",
            "output_path",
            "return_dict",
        ),
        "aliases": ("worldplay", "hy-worldplay", "hyworldplay", "tencent/HY-WorldPlay"),
        "tags": ("interactive-world", "camera-control", "image-to-video", "wmfactory"),
    },
    "multiworld-ittakestwo": {
        "display_name": "MultiWorld ItTakesTwo",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "default_input_path": str(resolve._data_path("test_cases", "multiworld_ittakestwo", "input.png")),
        "default_load_kwargs": _multiworld_ittakestwo_default_load_kwargs,
        "default_call_kwargs": _multiworld_ittakestwo_default_call_kwargs,
        "call_params": (
            "images",
            "action",
            "action_path",
            "env_obv",
            "output_dir",
            "save_name",
            "return_dict",
            "num_frames",
            "height",
            "width",
            "fps",
            "num_inference_steps",
            "inference_seed",
            "derive_env_obv_from_image",
            "show_progress",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "runtime_root",
            "config_path",
            "checkpoint_path",
            "base_model_root",
            "vggt_root",
            "python_executable",
            "derive_env_obv_from_image",
            "num_inference_steps",
            "inference_seed",
            "fps",
        ),
        "aliases": ("multiworld", "multiword"),
        "tags": ("multi-agent-world-model", "interactive-video", "game-video"),
        "notes": "Uses the official MultiWorld ItTakesTwo toy inference config and checkpoint.",
    },
}

OVERRIDES.update(
    {
        "sana-wm-streaming": {
            "display_name": "SANA-WM Streaming",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "supports_stream": True,
            "default_model_ref": _sana_wm_streaming_default_ref,
            "default_load_kwargs": _sana_wm_streaming_default_load_kwargs,
            "default_task_type": "image-camera-video",
            "default_input_path": str(resolve._data_path("test_cases", "sana_wm", "example_image.png")),
            "default_prompt": ("Move forward through a coherent scene with stable geometry and natural camera motion."),
            "default_interactions": ("forward",),
            "default_call_kwargs": {
                "window_frames": 241,
                "fps": 16,
                "step": 4,
                "cfg_scale": 1.0,
                "seed": 42,
                "return_dict": True,
            },
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoint_overrides",
                "gemma_path",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "images",
                "prompt",
                "interactions",
                "window_frames",
                "fps",
                "step",
                "cfg_scale",
                "seed",
                "return_dict",
            ),
            "tags": ("world-model", "image-to-video", "camera-control", "stream", "in-tree-runtime"),
        },
    }
)

OVERRIDES["lingbot-world-v2-1.3b"] = {
    **OVERRIDES["lingbot-world-v2"],
    "display_name": "LingBot-World-V2 1.3B",
    "default_model_ref": _lingbot_world_v2_small_default_ref,
    "default_load_kwargs": _lingbot_world_v2_small_default_load_kwargs,
    "default_call_kwargs": {
        **OVERRIDES["lingbot-world-v2"]["default_call_kwargs"],
        "frame_num": 317,
        "sample_shift": 5.0,
        "nproc_per_node": 1,
        "t5_fsdp": False,
        "dit_fsdp": False,
    },
    "load_params": (
        *OVERRIDES["lingbot-world-v2"]["load_params"],
        "shared_assets_path",
        "attention_backend",
        "vae_streaming_compile",
    ),
    "aliases": ("lingbot-v2-1.3b",),
    "notes": "Independent 1.3B causal-fast runtime with shared Wan T5/VAE assets and a four-latent resident rollout.",
}
