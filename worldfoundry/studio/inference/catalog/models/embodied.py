"""Curated overrides and defaults for embodied policies (VLA, WAM, robot action)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    resolve_worldfoundry_path,
)

from .. import resolve
from ..params import OFFICIAL_ACTION_CALL_PARAMS, OFFICIAL_ACTION_LOAD_PARAMS
from ..resolve import PIPELINES_ROOT


def _lingbot_va_default_call_kwargs() -> dict[str, Any]:
    demo_root = resolve._data_path("test_cases", "test_vla_case1", "libero")
    return {
        "operator_kwargs": {
            "lingbot_va_rgb_views": {
                "observation.images.agentview_rgb": str(demo_root / "main_view.png"),
                "observation.images.eye_in_hand_rgb": str(demo_root / "wrist_view.png"),
            },
            "obs_cam_keys": (
                "observation.images.agentview_rgb",
                "observation.images.eye_in_hand_rgb",
            ),
        }
    }


def _giga_brain_0_default_call_kwargs() -> dict[str, Any]:
    demo_root = resolve._data_path("test_cases", "test_vla_case1", "aloha")
    image_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    return {
        "compile_policy": False,
        "operator_kwargs": {
            "giga_brain_0_rgb_views": {
                image_keys[0]: str(demo_root / "observation_images_cam_high.png"),
                image_keys[1]: str(demo_root / "observation_images_cam_left_wrist.png"),
                image_keys[2]: str(demo_root / "observation_images_cam_right_wrist.png"),
            },
            "image_keys": image_keys,
            "state": [0.0] * 14,
            "embodiment_id": 0,
            "original_action_dim": 14,
            "action_chunk": 50,
        },
    }


def _openvla_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "openvla-7b",
        "openvla--openvla-7b",
        fallback="openvla/openvla-7b",
    )


def _openvla_default_call_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _openvla_default_ref(),
        "unnorm_key": "bridge_orig",
        "torch_dtype": "auto",
        "attn_implementation": "eager",
    }


def _openvla_oft_default_call_kwargs() -> dict[str, Any]:
    libero_root = resolve._data_path("test_cases", "test_vla_case1", "libero")
    return {
        "official_policy_observation": {
            "full_image": str(libero_root / "main_view.png"),
            "wrist_image": str(libero_root / "wrist_view.png"),
            "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "task_description": "put the object on the target area",
        },
        "unnorm_key": "libero_spatial_no_noops",
        "task_suite_name": "libero_spatial",
        "num_images_in_input": 2,
        "use_proprio": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "eager",
    }


def _openvla_oft_variant_call_kwargs(
    *,
    unnorm_key: str,
    task_suite_name: str,
) -> dict[str, Any]:
    """Return the executable Studio input contract shared by OFT variants."""

    values = _openvla_oft_default_call_kwargs()
    values.update(unnorm_key=unnorm_key, task_suite_name=task_suite_name)
    return values


def _libero_policy_observation(prompt: str) -> dict[str, Any]:
    libero_root = resolve._data_path("test_cases", "test_vla_case1", "libero")
    return {
        "full_image": str(libero_root / "main_view.png"),
        "wrist_image": str(libero_root / "wrist_view.png"),
        "state": [0.0] * 8,
        "task_description": prompt,
        "prompt": prompt,
    }


def _cogact_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        str(checkpoint_root_path("hfd_models", "CogACT--CogACT-Base")),
        str(checkpoint_root_path("hfd", "CogACT--CogACT-Base")),
        "CogACT/CogACT-Base",
    )


def _last_r1_default_ref() -> str:
    suffix = Path("chenhao01--LaST-R1") / "LaST-R1-RL" / "last-r1-rl-libero_10"
    configured = resolve_worldfoundry_path(Path("${WORLDFOUNDRY_CKPT_DIR}") / suffix)
    if not (configured / "config.json").is_file() and not os.environ.get("WORLDFOUNDRY_CKPT_DIR"):
        sibling = PIPELINES_ROOT.parents[2] / "ckpts" / suffix
        if (sibling / "config.json").is_file():
            return str(sibling)
    return str(configured)


def _cogact_default_call_kwargs() -> dict[str, Any]:
    prompt = "move sponge near apple"
    image_path = str(resolve._data_path("test_cases", "test_vla_image_case1", "init_frame.png"))
    kwargs: dict[str, Any] = {
        "official_policy_observation": {
            "image": image_path,
            "full_image": image_path,
            "prompt": prompt,
            "task_description": prompt,
            "unnorm_key": "bridge_orig",
        },
        "unnorm_key": "bridge_orig",
        "cfg_scale": 1.5,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "action_model_type": "DiT-B",
        "future_action_window_size": 15,
    }
    checkpoint = resolve._checkpoint_file_if_present(
        "hfd_models/CogACT--CogACT-Base/checkpoints/CogACT-Base.pt",
        "hfd/CogACT--CogACT-Base/checkpoints/CogACT-Base.pt",
    )
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    else:
        kwargs["checkpoint_ref"] = "CogACT/CogACT-Base"
    return kwargs


def _db_cogact_default_ref() -> str:
    return resolve._prefer_existing_model_ref(
        str(checkpoint_root_path("hfd_models", "Dexmal--libero-db-cogact")),
        str(checkpoint_root_path("hfd", "Dexmal--libero-db-cogact")),
        "Dexmal/libero-db-cogact",
    )


def _db_cogact_default_call_kwargs() -> dict[str, Any]:
    prompt = "put the object on the target area"
    libero_root = resolve._data_path("test_cases", "test_vla_case1", "libero")
    return {
        "official_policy_observation": {
            "prompt": prompt,
            "image/1": str(libero_root / "main_view.png"),
            "image/2": str(libero_root / "wrist_view.png"),
        },
        "camera_order": ["front", "left_wrist"],
        "num_steps": 10,
        "cfg_scale": 1.5,
        "seed": 42,
    }


def _vlanext_default_ref() -> str:
    checkpoint = resolve._checkpoint_file_if_present(
        "hfd_models/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
        "hfd/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
    )
    if checkpoint:
        return str(Path(checkpoint).parent)
    return "DravenALG/VLANeXt"


def _vlanext_default_call_kwargs() -> dict[str, Any]:
    prompt = "put the object on the target area"
    kwargs: dict[str, Any] = {
        "official_policy_observation": _libero_policy_observation(prompt),
        "task_suite_name": "libero_spatial",
        "diffusion_steps": 10,
        "num_steps_execute": 1,
    }
    checkpoint = resolve._checkpoint_file_if_present(
        "hfd_models/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
        "hfd/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
    )
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    else:
        kwargs["checkpoint_ref"] = "DravenALG/VLANeXt"
    return kwargs


def _molmobot_default_ref() -> str:
    repo_root = checkpoint_root_path("hfd_models", "allenai--MolmoBot-DROID")
    if (repo_root / "config.yaml").is_file():
        return str(repo_root)
    repo_root = checkpoint_root_path("hfd", "allenai--MolmoBot-DROID")
    if (repo_root / "config.yaml").is_file():
        return str(repo_root)
    return "allenai/MolmoBot-DROID"


def _molmobot_default_call_kwargs() -> dict[str, Any]:
    droid_root = resolve._data_path("test_cases", "test_vla_case1", "droid")
    prompt = "put the mug in the bowl"
    return {
        "official_policy_observation": {
            "exo_camera_1": str(droid_root / "exterior_image_1_left.png"),
            "wrist_camera": str(droid_root / "wrist_image_left.png"),
            "qpos": [0.0] * 8,
            "task": prompt,
            "task_description": prompt,
            "camera_keys": ["exo_camera_1", "wrist_camera"],
        },
        "camera_keys": ["exo_camera_1", "wrist_camera"],
        "norm_repo_id": "synthmanip",
        "use_bfloat16": True,
        "compile_model": False,
    }


def _mme_vla_default_ref() -> str:
    step_dir = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999")
    if step_dir.is_dir():
        return str(step_dir)
    zip_path = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999.zip")
    if zip_path.is_file():
        return str(zip_path.parent)
    return "Yinpei/perceptual-framesamp-modul"


def _mme_vla_default_call_kwargs() -> dict[str, Any]:
    libero_root = resolve._data_path("test_cases", "test_vla_case1", "libero")
    prompt = "put the object on the target area"
    kwargs: dict[str, Any] = {
        "official_policy_observation": {
            "observation/image": str(libero_root / "main_view.png"),
            "observation/wrist_image": str(libero_root / "wrist_view.png"),
            "observation/state": [0.0] * 8,
            "prompt": prompt,
        },
        "policy_config": "mme_vla_suite",
        "seed": 7,
    }
    step_dir = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999")
    if step_dir.is_dir():
        kwargs["checkpoint_path"] = str(step_dir)
    else:
        kwargs["checkpoint_ref"] = "Yinpei/perceptual-framesamp-modul"
    return kwargs


def _starvla_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "Qwen3-VL-OFT-LIBERO-4in1",
        "StarVLA--Qwen3-VL-OFT-LIBERO-4in1",
        fallback="StarVLA/Qwen3-VL-OFT-LIBERO-4in1",
    )


def _starvla_default_call_kwargs() -> dict[str, Any]:
    in_tree_source = resolve._project_root() / "worldfoundry" / "synthesis" / "action_generation" / "starvla"
    return {
        "checkpoint_dir": _starvla_default_ref(),
        "base_vlm": resolve._checkpoint_model_ref(
            "Qwen3-VL-4B-Instruct",
            "Qwen--Qwen3-VL-4B-Instruct",
            fallback="Qwen/Qwen3-VL-4B-Instruct",
        ),
        "source_repo_dir": str(in_tree_source),
        "enable_official_runtime": True,
        "attn_implementation": "sdpa",
    }


def _gr00t_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "GR00T-N1.7-LIBERO/libero_10",
        "GR00T-N1.7-LIBERO/libero_goal",
        "GR00T-N1.7-LIBERO",
        fallback="nvidia/GR00T-N1.7-LIBERO",
    )


def _openpi_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "openpi-assets/checkpoints/pi05_libero",
        fallback="gs://openpi-assets/checkpoints/pi05_libero",
    )


def _openpi_default_call_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _openpi_default_ref(),
        "config_name": "pi05_libero",
        "seed": 0,
    }


def _lapa_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "latent-action-pretraining--LAPA-7B-openx",
        "LAPA-7B-openx",
        "hfd/latent-action-pretraining--LAPA-7B-openx",
        fallback="latent-action-pretraining/LAPA-7B-openx",
    )


def _lapa_default_call_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _lapa_default_ref(),
        "dtype": "bf16",
        "image_size": 256,
        "mesh_dim": "1,1,1,1",
        "seed": 1234,
        "tokens_per_delta": 4,
    }


def _octo_default_ref() -> str:
    return resolve._checkpoint_model_ref(
        "octo-small-1.5",
        "hfd/rail-berkeley--octo-small-1.5",
        fallback="rail-berkeley/octo-small-1.5",
    )


def _octo_default_call_kwargs() -> dict[str, Any]:
    return {
        "checkpoint_dir": _octo_default_ref(),
        "variant": "small",
        "dataset_key": "bridge_dataset",
        "image_key": "image_primary",
        "image_size": [256, 256],
        "jax_platform": "cpu",
        "seed": 0,
    }


def _diffusion_policy_default_call_kwargs() -> dict[str, Any]:
    # Official low-dimensional PushT checkpoint: n_obs_steps=2, obs_dim=20.
    return {
        "device": "cuda",
        "observation": {
            "obs": [
                [
                    256.0,
                    256.0,
                    218.0,
                    224.0,
                    254.0,
                    212.0,
                    292.0,
                    224.0,
                    306.0,
                    258.0,
                    292.0,
                    292.0,
                    254.0,
                    304.0,
                    218.0,
                    292.0,
                    204.0,
                    258.0,
                    230.0,
                    258.0,
                ],
                [
                    262.0,
                    258.0,
                    224.0,
                    226.0,
                    260.0,
                    214.0,
                    298.0,
                    226.0,
                    312.0,
                    260.0,
                    298.0,
                    294.0,
                    260.0,
                    306.0,
                    224.0,
                    294.0,
                    210.0,
                    260.0,
                    236.0,
                    260.0,
                ],
            ]
        },
        "action_space": {"type": "low_dim_pusht", "action_dim": 2, "n_action_steps": 8},
    }


def _molmoact2_default_call_kwargs() -> dict[str, Any]:
    demo_dir = resolve._data_path("test_cases", "test_vla_case1", "droid")
    state = [0.0] * 8
    operator_kwargs = {
        "camera_keys": ["external_cam", "external_cam_2", "wrist_cam"],
        "embodiment": "droid",
        "norm_tag": "franka_droid",
        "state": state,
        "external_cam": str(demo_dir / "exterior_image_1_left.png"),
        "external_cam_2": str(demo_dir / "exterior_image_1_left.png"),
        "wrist_cam": str(demo_dir / "wrist_image_left.png"),
    }
    return {
        **operator_kwargs,
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_yam_call_kwargs() -> dict[str, Any]:
    demo_dir = resolve._data_path("test_cases", "test_vla_case1")
    state = [0.0] * 14
    operator_kwargs = {
        "camera_keys": ["top_cam", "left_cam", "right_cam"],
        "embodiment": "yam",
        "norm_tag": "yam_dual_molmoact2",
        "state": state,
        "top_cam": str(demo_dir / "droid" / "exterior_image_1_left.png"),
        "left_cam": str(demo_dir / "aloha" / "observation_images_cam_left_wrist.png"),
        "right_cam": str(demo_dir / "aloha" / "observation_images_cam_right_wrist.png"),
    }
    return {
        **operator_kwargs,
        "variant": "yam",
        "embodiment": "yam",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_so100_call_kwargs() -> dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-SO100_101")
    state = [
        -0.52734375,
        189.140625,
        181.40625,
        60.64453125,
        -3.603515625,
        1.0971786975860596,
    ]
    operator_kwargs = {
        "camera_keys": ["top_cam", "side_cam"],
        "embodiment": "so100",
        "norm_tag": "so100_so101_molmoact2",
        "state": state,
        "top_cam": str(checkpoint_dir / "assets" / "sample_realsense_top_rgb.png"),
        "side_cam": str(checkpoint_dir / "assets" / "sample_realsense_side_rgb.png"),
    }
    return {
        **operator_kwargs,
        "variant": "so100",
        "embodiment": "so100",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_libero_call_kwargs() -> dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-LIBERO")
    state = [
        -0.05338004603981972,
        0.007029631175100803,
        0.6783280968666077,
        3.1407692432403564,
        0.0017593271331861615,
        -0.08994418382644653,
        0.03878866136074066,
        -0.03878721222281456,
    ]
    operator_kwargs = {
        "camera_keys": ["agentview_cam", "wrist_cam"],
        "embodiment": "libero",
        "norm_tag": "libero",
        "state": state,
        "agentview_cam": str(checkpoint_dir / "assets" / "sample_agentview_rgb.png"),
        "wrist_cam": str(checkpoint_dir / "assets" / "sample_wrist_rgb.png"),
        "enable_depth_reasoning": False,
    }
    return {
        **operator_kwargs,
        "variant": "libero",
        "embodiment": "libero",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
        "enable_depth_reasoning": False,
    }


def _molmoact2_think_libero_call_kwargs() -> dict[str, Any]:
    call_kwargs = _molmoact2_libero_call_kwargs()
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-Think-LIBERO")
    operator_kwargs = dict(call_kwargs["operator_kwargs"])
    operator_kwargs.update(
        {
            "embodiment": "think_libero",
            "agentview_cam": str(checkpoint_dir / "assets" / "sample_agentview_rgb.png"),
            "wrist_cam": str(checkpoint_dir / "assets" / "sample_wrist_rgb.png"),
            "enable_depth_reasoning": True,
            "enable_adaptive_depth": True,
            "depth_cache": None,
        }
    )
    call_kwargs.update(
        {
            **operator_kwargs,
            "variant": "think-libero",
            "embodiment": "think_libero",
            "operator_kwargs": operator_kwargs,
            "enable_depth_reasoning": True,
            "enable_adaptive_depth": True,
            "depth_cache": None,
        }
    )
    return call_kwargs


def _action_checkpoint_variant(
    *,
    variant_id: str,
    label: str,
    repo_id: str,
    call_kwargs: Mapping[str, Any],
    aliases: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one local-first Workspace variant for an in-tree action policy."""

    return {
        "variant_id": variant_id,
        "label": label,
        "status": "checkpoint-configured",
        "checkpoints": (
            {
                "role": "primary",
                "uri": resolve._hf_checkpoint_model_ref(f"hfd/{repo_id.replace('/', '--')}", repo_id),
                "status": "checkpoint-configured",
            },
        ),
        "call_kwargs": dict(call_kwargs),
        "aliases": tuple(aliases),
        "notes": tuple(notes),
    }


def _xvla_workspace_variants() -> tuple[dict[str, Any], ...]:
    common = {
        "state": [0.0] * 20,
        "denoising_steps": 10,
        "seed": 0,
        "torch_dtype": "float32",
        "attention_backend": "auto",
        "plan_only": False,
    }
    specs = (
        ("google-robot", "Google Robot", "2toINF/X-VLA-Google-Robot", 1),
        ("libero", "LIBERO", "2toINF/X-VLA-Libero", 3),
        ("calvin-abc-d", "CALVIN ABC→D", "2toINF/X-VLA-Calvin-ABC_D", 2),
        ("robotwin2", "RoboTwin 2", "2toINF/X-VLA-RoboTwin2", 6),
        ("vlabench", "VLABench", "2toINF/X-VLA-VLABench", 8),
        ("agiworld", "AgiWorld Challenge", "2toINF/X-VLA-AgiWorld-Challenge", 9),
        ("softfold", "SoftFold", "2toINF/X-VLA-SoftFold", 5),
        ("foundation-domain-0", "Foundation (domain 0)", "2toINF/X-VLA-Pt", 0),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={**common, "domain_id": domain_id},
            aliases=(repo_id.rsplit("/", 1)[-1],),
            notes=(
                "The foundation checkpoint needs an embodiment domain ID; this Workspace preset uses domain 0."
                if variant_id == "foundation-domain-0"
                else f"Uses the released domain ID {domain_id} for this embodiment.",
            ),
        )
        for variant_id, label, repo_id, domain_id in specs
    )


def _xiaomi_robotics_0_workspace_variants() -> tuple[dict[str, Any], ...]:
    common = {
        "num_steps": 5,
        "seed": 42,
        "torch_dtype": "auto",
        "attn_implementation": "auto",
        "plan_only": False,
    }
    specs: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
        (
            "libero",
            "LIBERO",
            "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO",
            {
                "variant": "libero",
                "state": [0.0] * 32,
                "wrist_image": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            },
        ),
        (
            "calvin-abcd",
            "CALVIN ABCD→D",
            "XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D",
            {
                "variant": "calvin-abcd",
                "state": [0.0] * 32,
                "wrist_image": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            },
        ),
        (
            "calvin-abc",
            "CALVIN ABC→D",
            "XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABC_D",
            {
                "variant": "calvin-abc",
                "state": [0.0] * 32,
                "wrist_image": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            },
        ),
        (
            "simplerenv-google",
            "SimplerEnv Google Robot",
            "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot",
            {"variant": "simplerenv-google", "state": [0.0] * 7},
        ),
        (
            "simplerenv-widowx",
            "SimplerEnv WidowX",
            "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX",
            {"variant": "simplerenv-widowx", "state": [0.0] * 7},
        ),
        (
            "pretrain-droid",
            "Pretrain (DROID)",
            "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
            {
                "variant": "pretrain",
                "robot_type": "droid_pt",
                "camera_keys": ["base"],
                "view_labels": ["External Camera View"],
                "state": [0.0] * 32,
            },
        ),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={**common, **variant_kwargs},
            aliases=(repo_id.rsplit("/", 1)[-1],),
        )
        for variant_id, label, repo_id, variant_kwargs in specs
    )


def _spatial_forcing_workspace_variants() -> tuple[dict[str, Any], ...]:
    image = str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg"))
    specs = (
        (
            "libero-spatial",
            "LIBERO Spatial",
            "haofuly/spatial-forcing-7b-finetuned-libero-spatial",
            "libero_spatial",
            "libero_spatial_no_noops",
        ),
        (
            "libero-object",
            "LIBERO Object",
            "haofuly/spatial-forcing-7b-finetuned-libero-object",
            "libero_object",
            "libero_object_no_noops",
        ),
        (
            "libero-goal",
            "LIBERO Goal",
            "haofuly/spatial-forcing-7b-finetuned-libero-goal",
            "libero_goal",
            "libero_goal_no_noops",
        ),
        ("libero-10", "LIBERO 10", "haofuly/spatial-forcing-7b-finetuned-libero-10", "libero_10", "libero_10_no_noops"),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={
                "variant": variant_id.removeprefix("libero-"),
                "checkpoint_variant": variant_id.removeprefix("libero-"),
                "state": [0.0] * 8,
                "wrist_image": image,
                "task_suite_name": suite,
                "unnorm_key": unnorm_key,
                "torch_dtype": "auto",
                "attn_implementation": "auto",
                "plan_only": False,
            },
            aliases=(repo_id.rsplit("/", 1)[-1],),
        )
        for variant_id, label, repo_id, suite, unnorm_key in specs
    )


def _fastwam_workspace_variants() -> tuple[dict[str, Any], ...]:
    aloha = resolve._data_path("test_cases", "test_vla_case1", "aloha")
    libero = resolve._data_path("test_cases", "test_vla_case1", "libero")
    common = {
        "torch_dtype": "auto",
        "num_inference_steps": 10,
        "seed": 42,
        "plan_only": False,
    }
    return (
        _action_checkpoint_variant(
            variant_id="robotwin",
            label="RoboTwin",
            repo_id="yuanty/fastwam",
            call_kwargs={
                **common,
                "variant": "robotwin",
                "state": [0.0] * 14,
                "head_camera": str(aloha / "observation_images_cam_high.png"),
                "left_camera": str(aloha / "observation_images_cam_left_wrist.png"),
                "right_camera": str(aloha / "observation_images_cam_right_wrist.png"),
            },
        ),
        _action_checkpoint_variant(
            variant_id="libero",
            label="LIBERO",
            repo_id="yuanty/fastwam",
            call_kwargs={
                **common,
                "variant": "libero",
                "state": [0.0] * 8,
                "wrist_image": str(libero / "wrist_view.png"),
            },
        ),
    )


def _x_wam_workspace_variants() -> tuple[dict[str, Any], ...]:
    image = str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg"))
    common = {
        "rgb_views": [image] * 3,
        "torch_dtype": "auto",
        "denoise_steps": 50,
        "action_denoise_steps": 10,
        "cfg_scale": 0.0,
        "compile_model": False,
        "generate_world": False,
        "run_depth": False,
        "plan_only": False,
    }
    return (
        _action_checkpoint_variant(
            variant_id="robocasa-sft",
            label="RoboCasa SFT",
            repo_id="sharinka0715/X-WAM-checkpoints",
            call_kwargs={**common, "variant": "robocasa_sft", "state": [0.0] * 8},
        ),
        _action_checkpoint_variant(
            variant_id="robotwin-sft",
            label="RoboTwin SFT",
            repo_id="sharinka0715/X-WAM-checkpoints",
            call_kwargs={**common, "variant": "robotwin_sft", "state": [0.0] * 16},
        ),
    )


OVERRIDES: dict[str, dict[str, Any]] = {
    "xiaomi-robotics-0": {
        "display_name": "Xiaomi-Robotics-0",
        "category": "Embodied Action",
        "summary": "In-tree MiBoT VLA inference for Xiaomi-Robotics-0 robot-action checkpoints.",
        "default_model_ref": "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "simplerenv-google",
            "state": [0.0] * 7,
            "num_steps": 5,
            "seed": 42,
            "torch_dtype": "auto",
            "attn_implementation": "auto",
            "plan_only": False,
        },
        "extra_variants": _xiaomi_robotics_0_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "wrist_image",
            "variant",
            "robot_type",
            "camera_keys",
            "view_labels",
            "num_steps",
            "seed",
            "torch_dtype",
            "attn_implementation",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("xiaomi-robotics-zero", "xr0"),
        "tags": ("vla", "policy", "robot-action", "in-tree-runtime"),
        "notes": "SimplerEnv Google Robot checkpoint-backed A100 BF16/SDPA inference is validated.",
    },
    "hy-embodied-vla": {
        "display_name": "Hy-Embodied-0.5-VLA",
        "category": "Embodied Action",
        "summary": "Tencent Hy-Embodied-0.5 multi-view flow-matching VLA policy.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/tencent--Hy-Embodied-0.5-VLA-RoboTwin",
            "tencent/Hy-Embodied-0.5-VLA-RoboTwin",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "robotwin",
            "state": [0.0] * 20,
            "state_format": "normalized",
            "blend_mode": "auto",
            "history_size": 6,
            "replicate_single_image": True,
            "torch_dtype": "auto",
            "plan_only": False,
        },
        "extra_variants": (
            {
                "variant_id": "umi",
                "label": "UMI",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref(
                            "hfd/tencent--Hy-Embodied-0.5-VLA-UMI",
                            "tencent/Hy-Embodied-0.5-VLA-UMI",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": {
                    "variant": "umi",
                    "state": [0.0] * 20,
                    "state_format": "normalized",
                    "blend_mode": "auto",
                    "history_size": 1,
                    "replicate_single_image": True,
                    "torch_dtype": "auto",
                    "plan_only": False,
                },
                "aliases": ("hy-embodied-vla-umi", "hy-vla-umi"),
                "notes": ("Released UMI 50-step relative-action policy.",),
            },
        ),
        "call_params": (
            "prompt",
            "images",
            "state",
            "variant",
            "state_format",
            "blend_mode",
            "history_size",
            "replicate_single_image",
            "torch_dtype",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("hy-vla", "hy-embodied-0.5-vla"),
        "tags": ("vla", "policy", "multi-view", "robot-action", "in-tree-runtime"),
        "notes": "RoboTwin checkpoint-backed A100 BF16 inference is validated.",
    },
    "lda-1b": {
        "display_name": "LDA-1B",
        "category": "Embodied Action",
        "summary": "In-tree LDA-1B Qwen3-VL/DINOv3 flow-matching action policy.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/Wayer2--LDA-robocasa",
            "Wayer2/LDA-robocasa",
        ),
        "default_prompt": "move the object to the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "ego_view": [
                    str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
                    str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
                ],
                "state": [0.0] * 29,
            },
            "torch_dtype": "auto",
            "attention_backend": "auto",
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("lda1b", "lda-robocasa"),
        "tags": ("vla", "policy", "robot-action", "flow-matching", "in-tree-runtime"),
        "notes": "The default Workspace contract uses the released RoboCasa GR1 29-D state/action layout.",
    },
    "internvla-a1": {
        "display_name": "InternVLA-A1-3B",
        "category": "Embodied Action",
        "summary": "In-tree InternVLA-A1 understanding, foresight, and flow-action policy.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/InternRobotics--InternVLA-A1-3B",
            "InternRobotics/InternVLA-A1-3B",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "cam_high": str(
                    resolve._data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
                ),
                "cam_left_wrist": str(
                    resolve._data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_left_wrist.png")
                ),
                "cam_right_wrist": str(
                    resolve._data_path(
                        "test_cases", "test_vla_case1", "aloha", "observation_images_cam_right_wrist.png"
                    )
                ),
                "state": [0.0] * 14,
                "reset": True,
            },
            "torch_dtype": "auto",
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("internvla-a1-3b", "internvla-a-series"),
        "tags": ("vla", "policy", "robot-action", "world-model", "in-tree-runtime"),
        "notes": "The default Workspace contract uses the released ALOHA three-camera 14-D statistics block.",
    },
    "fastwam": {
        "display_name": "FastWAM",
        "category": "Embodied Action",
        "summary": "In-tree FastWAM first-frame-cached world-action policy for RoboTwin and LIBERO.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/yuanty--fastwam",
            "yuanty/fastwam",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(
            resolve._data_path(
                "test_cases",
                "test_vla_case1",
                "aloha",
                "observation_images_cam_high.png",
            )
        ),
        "default_call_kwargs": {
            "variant": "robotwin",
            "state": [0.0] * 14,
            "head_camera": str(
                resolve._data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_high.png",
                )
            ),
            "left_camera": str(
                resolve._data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_left_wrist.png",
                )
            ),
            "right_camera": str(
                resolve._data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_right_wrist.png",
                )
            ),
            "torch_dtype": "auto",
            "num_inference_steps": 10,
            "seed": 42,
            "plan_only": False,
        },
        "extra_variants": _fastwam_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "variant",
            "head_camera",
            "left_camera",
            "right_camera",
            "wrist_image",
            "combined_image",
            "model_image",
            "context",
            "context_mask",
            "torch_dtype",
            "num_inference_steps",
            "denoising_steps",
            "sigma_shift",
            "seed",
            "rand_device",
            "tiled",
            "binarize_libero_gripper",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("fast-wam",),
        "tags": ("wam", "robot-action", "libero", "robotwin", "in-tree-runtime"),
        "notes": "Both released checkpoints are staged for full Workspace GPU validation.",
    },
    "spatial-forcing": {
        "display_name": "Spatial-Forcing",
        "category": "Embodied Action",
        "summary": "Inference-only Spatial-Forcing OpenVLA-OFT policy for LIBERO action chunks.",
        "default_model_ref": "haofuly/spatial-forcing-7b-finetuned-libero-spatial",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "state": [0.0] * 8,
            "wrist_image": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "task_suite_name": "libero_spatial",
            "unnorm_key": "libero_spatial_no_noops",
            "torch_dtype": "auto",
            "attn_implementation": "auto",
            "plan_only": False,
        },
        "extra_variants": _spatial_forcing_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "wrist_image",
            "state",
            "variant",
            "checkpoint_variant",
            "task_suite_name",
            "unnorm_key",
            "torch_dtype",
            "attn_implementation",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("spatial_forcing",),
        "tags": ("vla", "openvla", "libero", "robot-action", "in-tree-runtime"),
        "notes": "LIBERO-Spatial checkpoint-backed A100 inference is validated; training-only VGGT alignment is excluded.",
    },
    "x-wam": {
        "display_name": "X-WAM",
        "category": "Embodied Action",
        "summary": "Joint multi-view world-action inference with action, proprioception, and optional future video.",
        "default_model_ref": "sharinka0715/X-WAM-checkpoints",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "robocasa_sft",
            "rgb_views": [str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg"))] * 3,
            "state": [0.0] * 8,
            "torch_dtype": "auto",
            "denoise_steps": 50,
            "action_denoise_steps": 10,
            "cfg_scale": 0.0,
            "compile_model": False,
            "generate_world": False,
            "run_depth": False,
            "plan_only": False,
        },
        "extra_variants": _x_wam_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "rgb_views",
            "state",
            "variant",
            "torch_dtype",
            "denoise_steps",
            "action_denoise_steps",
            "cfg_scale",
            "generate_world",
            "run_depth",
            "world_video_path",
            "compile_model",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("x_wam",),
        "tags": ("wam", "robot-action", "video-prediction", "multi-view", "in-tree-runtime"),
        "notes": "RoboCasa checkpoint-backed A100 action inference and the optional 50-step world-video decode are validated; world decode remains opt-in.",
    },
    "xvla": {
        "display_name": "X-VLA",
        "category": "Embodied Action",
        "summary": "Cross-embodiment Florence-2 VLA policy with a domain-aware flow action head.",
        "default_model_ref": "2toINF/X-VLA-WidowX",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(resolve._data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "state": [0.0] * 20,
            "domain_id": 0,
            "denoising_steps": 10,
            "seed": 0,
            "torch_dtype": "float32",
            "attention_backend": "auto",
            "plan_only": False,
        },
        "extra_variants": _xvla_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "domain_id",
            "denoising_steps",
            "seed",
            "torch_dtype",
            "attention_backend",
            "adapter_path",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("x-vla", "2toinf/x-vla"),
        "tags": ("vla", "cross-embodiment", "robot-action", "in-tree-runtime"),
        "notes": "WidowX checkpoint-backed A100 FP32/SDPA action inference is validated.",
    },
    "openvla-oft": {
        "display_name": "OpenVLA-OFT",
        "category": "Embodied Action",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/moojink--openvla-7b-oft-finetuned-libero-spatial",
            "moojink/openvla-7b-oft-finetuned-libero-spatial",
        ),
        "default_prompt": "put the object on the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _openvla_oft_default_call_kwargs,
        "extra_variants": (
            {
                "variant_id": "libero-object",
                "label": "LIBERO Object",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-object",
                            "moojink/openvla-7b-oft-finetuned-libero-object",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_object_no_noops",
                    task_suite_name="libero_object",
                ),
                "aliases": ("openvla-oft-libero-object",),
            },
            {
                "variant_id": "libero-goal",
                "label": "LIBERO Goal",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-goal",
                            "moojink/openvla-7b-oft-finetuned-libero-goal",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_goal_no_noops",
                    task_suite_name="libero_goal",
                ),
                "aliases": ("openvla-oft-libero-goal",),
            },
            {
                "variant_id": "libero-10",
                "label": "LIBERO 10",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-10",
                            "moojink/openvla-7b-oft-finetuned-libero-10",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_10_no_noops",
                    task_suite_name="libero_10",
                ),
                "aliases": ("openvla-oft-libero-10",),
            },
            {
                "variant_id": "libero-spatial-object-goal-10",
                "label": "LIBERO Combined",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_spatial_no_noops",
                    task_suite_name="libero_spatial",
                ),
                "aliases": ("openvla-oft-libero-spatial-object-goal-10",),
            },
        ),
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("openvla_oft", "openvla-oft-libero"),
        "tags": ("vla", "policy", "robot-action", "libero"),
        "notes": "All five declared LIBERO checkpoints completed real A100 Workspace inference with finite 8x7 action chunks.",
    },
    "smolvla": {
        "display_name": "SmolVLA LIBERO",
        "category": "Embodied Action",
        "summary": "In-tree SmolVLA flow-matching policy with the released LIBERO task checkpoint.",
        "default_model_ref": lambda: resolve._hf_checkpoint_model_ref(
            "hfd/HuggingFaceVLA--smolvla_libero",
            "HuggingFaceVLA/smolvla_libero",
        ),
        "default_prompt": "put the object on the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "image": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
                "image2": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "wrist_view.png")),
                "state": [0.0] * 8,
            },
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("smolvla-libero", "huggingfacevla/smolvla_libero"),
        "tags": ("vla", "policy", "robot-action", "flow-matching", "in-tree-runtime"),
        "notes": "The Workspace default uses the task-trained LIBERO checkpoint, two RGB views, and an 8-D state vector.",
    },
    "last-r1": {
        "display_name": "LaST-R1",
        "category": "Embodied Action",
        "summary": "Offline image-and-instruction inference with the libero_10 RL policy.",
        "default_model_ref": _last_r1_default_ref,
        "default_prompt": "Pick up the red block and place it in the bowl.",
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "tags": ("vla", "policy", "robot-action", "libero"),
        "notes": "Offline action inference is verified; LIBERO task success is unmeasured.",
    },
    "cogact": {
        "display_name": "CogACT",
        "category": "Embodied Action",
        "default_model_ref": _cogact_default_ref,
        "default_prompt": "move sponge near apple",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_image_case1", "init_frame.png")),
        "default_call_kwargs": _cogact_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("cogact-base", "cogact-small", "cogact-large"),
        "tags": ("vla", "policy", "robot-action", "diffusion-action"),
        "notes": "Independent CogACT action entry with an in-tree vendored VLA/DiT runtime. Real inference requires CogACT checkpoints plus gated meta-llama/Llama-2-7b-hf access.",
    },
    "db-cogact": {
        "category": "Embodied Action",
        "default_model_ref": _db_cogact_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _db_cogact_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("dexbotic", "db_cogact", "dexbotic-cogact"),
        "tags": ("vla", "policy", "robot-action", "dexbotic"),
        "notes": "Independent DB-CogACT action entry with an in-tree vendored Dexbotic policy runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "vlanext": {
        "display_name": "VLANeXt",
        "category": "Embodied Action",
        "default_model_ref": _vlanext_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _vlanext_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("vla-next", "vlanext-libero"),
        "tags": ("vla", "policy", "robot-action", "libero"),
        "notes": "Independent VLANeXt action entry with an in-tree vendored model/processor runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "molmobot": {
        "display_name": "MolmoBot",
        "category": "Embodied Action",
        "default_model_ref": _molmobot_default_ref,
        "default_prompt": "put the mug in the bowl",
        "default_input_path": str(
            resolve._data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")
        ),
        "default_call_kwargs": _molmobot_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("molmob0t", "molmobot-droid", "molmobot-pi0"),
        "tags": ("vla", "policy", "robot-action", "molmo"),
        "notes": "Independent MolmoBot action entry with a flattened in-tree inference-only policy runtime. Pi0 remains dependency-gated; release demo status depends on checkpoint-backed action validation.",
    },
    "mme-vla": {
        "display_name": "MME-VLA",
        "category": "Embodied Action",
        "default_model_ref": _mme_vla_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(resolve._data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _mme_vla_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("mme_vla", "robomme-policy-learning", "perceptual-framesamp-modul"),
        "tags": ("vla", "policy", "robot-action", "robomme"),
        "notes": "Independent MME-VLA action entry with an in-tree vendored MME-VLA/OpenPI policy runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "openvla": {
        "display_name": "OpenVLA",
        "category": "Embodied Action",
        "default_model_ref": _openvla_default_ref,
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _openvla_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "ckpt_path",
            "pretrained_model_path",
            "unnorm_key",
            "torch_dtype",
            "attn_implementation",
            "use_cache",
            "plan_only",
            "openvla_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("openvla-7b",),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Structured VLA policy pipeline. It emits action_trace artifacts, not generated video.",
    },
    "openpi": {
        "display_name": "OpenPI",
        "category": "Embodied Action",
        "default_model_ref": _openpi_default_ref,
        "default_prompt": "complete the manipulation instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _openpi_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "config_name",
            "pytorch_device",
            "seed",
            "openpi_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("pi0", "pi05"),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Structured pi0/pi0.5 policy pipeline. The default Studio demo mirrors the official pi05_libero checkpoint/config path.",
    },
    "giga-brain-0": {
        "display_name": "GigaBrain-0",
        "category": "Embodied Action",
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action", "action_chunk"),
        "default_call_kwargs": _giga_brain_0_default_call_kwargs,
        "aliases": ("giga_brain_0", "gigabrain0", "giga-brain-0.1"),
        "tags": ("vla", "policy", "world-model-powered", "robot-action"),
        "notes": "GigaBrain-0 is a world-model-powered VLA policy for multi-view RGB/RGBD action traces. Official execution requires staged norm stats, a LeRobot dataset, and the upstream giga-models stack.",
    },
    "gr00t": {
        "display_name": "GR00T N1.7",
        "category": "Embodied Action",
        "default_model_ref": _gr00t_default_ref,
        "default_prompt": "execute the humanoid manipulation skill",
        "default_interactions": ("robot_action",),
        "aliases": ("groot", "gr00t-n1", "gr00t-n1.7"),
        "tags": ("vla", "policy", "humanoid", "in-tree-runtime"),
        "notes": "In-tree GR00T N1.7 runtime with explicit local Qwen3-VL classes; checkpoint Python is never executed.",
    },
    "molmoact2": {
        "display_name": "MolmoAct2",
        "category": "Embodied Action",
        "default_model_ref": lambda: resolve._checkpoint_model_ref(
            "hfd/allenai--MolmoAct2-DROID",
            fallback="allenai/MolmoAct2-DROID",
        ),
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action", "action_chunk"),
        "default_call_kwargs": _molmoact2_default_call_kwargs,
        "extra_variants": (
            {
                "variant_id": "yam",
                "label": "Bimanual YAM",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-BimanualYAM",
                            fallback="allenai/MolmoAct2-BimanualYAM",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_yam_call_kwargs,
                "aliases": ("bimanual-yam", "molmoact2-bimanual-yam", "molmoact2-yam"),
                "notes": (
                    "Official YAM server schema: top_cam, left_cam, right_cam, 14-D state, yam_dual_molmoact2 normalization.",
                ),
            },
            {
                "variant_id": "so100",
                "label": "SO100/SO101",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-SO100_101",
                            fallback="allenai/MolmoAct2-SO100_101",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_so100_call_kwargs,
                "aliases": ("so101", "so100_101", "molmoact2-so100-101"),
                "notes": (
                    "Official SO100/SO101 model-card schema: top_rgb, side_rgb, 6-D state, so100_so101_molmoact2 normalization.",
                ),
            },
            {
                "variant_id": "libero",
                "label": "LIBERO",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-LIBERO",
                            fallback="allenai/MolmoAct2-LIBERO",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_libero_call_kwargs,
                "aliases": ("molmoact2-libero", "libero-10"),
                "notes": (
                    "Official LIBERO model-card schema: agentview_rgb, wrist_rgb, 8-D state, libero normalization.",
                ),
            },
            {
                "variant_id": "think-libero",
                "label": "Think LIBERO",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: resolve._checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-Think-LIBERO",
                            fallback="allenai/MolmoAct2-Think-LIBERO",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_think_libero_call_kwargs,
                "aliases": ("think_libero", "molmoact2-think-libero", "depth-libero"),
                "notes": (
                    "Official Think-LIBERO model-card schema: agentview_rgb, wrist_rgb, 8-D state, libero normalization, depth reasoning and adaptive depth enabled.",
                ),
            },
        ),
        "aliases": ("molmo-act2", "molmoact", "molmoact2-droid", "molmoact2-yam"),
        "tags": ("vla", "policy", "robot-action", "action-reasoning"),
        "notes": "MolmoAct2 DROID/YAM action-reasoning VLA. The default Studio demo mirrors the official DROID server schema: external_cam, wrist_cam, 8-D Franka state, franka_droid normalization.",
    },
    "dreamzero": {
        "category": "Embodied Action",
        "default_prompt": "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        "default_interactions": ("world_action", "robot_action"),
        "default_call_kwargs": {
            "plan_only": True,
            "variant": "droid",
            "dreamzero_server_host": "",
            "dreamzero_server_port": 8000,
            "num_chunks": 15,
            "use_zero_images": True,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "variant",
            "dreamzero_server_host",
            "dreamzero_server_port",
            "server_host",
            "server_port",
            "plan_only",
            "debug_video_dir",
            "num_chunks",
            "session_id",
            "use_zero_images",
        ),
        "aliases": ("dreamzero-droid", "dreamzero-agibot"),
        "tags": ("wam", "world-action", "robot-action", "video-prediction"),
        "notes": "NVIDIA GEAR DreamZero world-action policy. Official runtime uses a distributed RoboArena WebSocket server and emits action traces with server-side video predictions.",
    },
    "starvla": {
        "display_name": "StarVLA",
        "category": "Embodied Action",
        "default_model_ref": _starvla_default_ref,
        "default_prompt": "follow the embodied instruction with action-conditioned state updates",
        "default_interactions": ("robot_action", "world_action"),
        "default_call_kwargs": _starvla_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "base_vlm",
            "source_repo_dir",
            "enable_official_runtime",
            "attn_implementation",
            "state",
            "track",
            "variant_id",
        ),
        "aliases": ("star-vla", "wm4a"),
        "tags": ("vla", "wam", "policy", "world-action"),
        "notes": "StarVLA spans VLA and WAM variants. The default Studio demo uses the official Qwen3-VL-OFT-LIBERO-4in1 checkpoint, Qwen3-VL-4B base VLM, and local starVLA official source checkout.",
    },
    "lingbot-va": {
        "display_name": "LingBot-VA",
        "category": "Embodied Action",
        "default_prompt": "put both the alphabet soup and the tomato sauce in the basket",
        "default_interactions": ("robot_action", "video_action"),
        "default_call_kwargs": _lingbot_va_default_call_kwargs,
        "aliases": (
            "lingbot_va",
            "lingbot-va-base",
            "lingbot-va-posttrain-robotwin",
            "lingbot-va-posttrain-libero-long",
        ),
        "tags": ("va", "vam", "wam", "robot-action", "video-action"),
        "notes": "LingBot-VA is a video-action world model for LIBERO/RoboTwin-style action traces; official execution remains checkpoint and CUDA-12.6 profile gated.",
    },
    "lapa": {
        "category": "Visual Action",
        "default_model_ref": _lapa_default_ref,
        "default_prompt": "infer latent action tokens from the video context",
        "default_interactions": ("latent_action_tokens",),
        "default_call_kwargs": _lapa_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "dtype",
            "image_size",
            "mesh_dim",
            "seed",
            "tokens_per_delta",
            "lapa_observation",
        ),
        "aliases": ("lapa-7b-openx",),
        "tags": ("va", "vam", "latent-action"),
        "notes": "Visual-action model profile. It consumes video context but is not categorized as video generation.",
    },
    "octo": {
        "display_name": "Octo",
        "category": "Embodied Action",
        "default_model_ref": _octo_default_ref,
        "default_prompt": "complete the manipulation instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _octo_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "variant",
            "dataset_key",
            "image_key",
            "image_size",
            "jax_platform",
            "seed",
            "octo_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("octo-base", "octo-base-1.5", "octo-small-1.5"),
        "tags": ("vla", "policy", "generalist", "robot-action"),
        "notes": "Generalist robot policy pipeline. It emits action_trace artifacts for embodied rollouts.",
    },
    "rt-1": {
        "display_name": "RT-1",
        "category": "Embodied Action",
        "default_prompt": "follow the language-conditioned robot instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": {"plan_only": False},
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "language_embedding",
            "rt1_observation",
            "action_space",
            "policy_controls",
            "plan_only",
        ),
        "aliases": ("rt-1", "robotics-transformer"),
        "tags": ("vla", "policy", "robotics-transformer"),
        "notes": "Robotics Transformer policy pipeline. Action inference requires a checkpoint-compatible 512-D language embedding; prompt text alone is not encoded.",
    },
    "diffusion-policy": {
        "display_name": "Diffusion Policy",
        "category": "Embodied Action",
        "default_prompt": "execute the visuomotor policy rollout",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _diffusion_policy_default_call_kwargs,
        "call_params": (
            "prompt",
            "interactions",
            "output_path",
            "checkpoint_path",
            "observation",
            "action_space",
            "device",
        ),
        "aliases": ("diffusion_policy", "dp"),
        "tags": ("visuomotor", "action-diffusion", "policy"),
        "notes": "Diffusion-based action policy profile. It is not a diffusion video-generation model.",
    },
    "act": {
        "category": "Embodied Action",
        "default_prompt": "produce the next robot action chunk",
        "default_interactions": ("action_chunk",),
        "default_call_kwargs": {
            "camera_names": ["head_cam", "left_cam", "right_cam"],
            "state_dim": 14,
            "chunk_size": 100,
            "temporal_agg": False,
            "plan_only": True,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "checkpoint_path",
            "act_observation",
            "camera_names",
            "state_dim",
            "chunk_size",
            "temporal_agg",
            "plan_only",
        ),
        "aliases": ("action-chunking-transformer",),
        "tags": ("action-chunking", "policy", "imitation-learning"),
        "notes": "Action Chunking Transformer policy pipeline. Defaults to plan-only unless a trained policy checkpoint and dataset stats are provided.",
    },
    "roboflamingo": {
        "display_name": "RoboFlamingo",
        "category": "Embodied Action",
        "default_prompt": "follow the vision-language manipulation instruction",
        "default_interactions": ("robot_action",),
        "aliases": ("robo-flamingo",),
        "tags": ("vla", "vlm-policy", "robot-action"),
        "notes": "VLM/VLA policy pipeline for robot action traces.",
    },
    "being-h05": {
        "category": "Embodied Action",
        "default_prompt": "complete the robot manipulation instruction",
        "default_interactions": ("robot_action",),
        "aliases": ("beingh05", "being-h-05", "being-h05-2b"),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Being-H05 is a VLA policy runtime that emits action traces rather than generated video.",
    },
}
