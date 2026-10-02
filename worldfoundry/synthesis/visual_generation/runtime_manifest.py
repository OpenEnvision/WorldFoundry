from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import project_root
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.runtime.assets import expand_worldfoundry_path

from ..base_synthesis import BaseSynthesis


@dataclass(frozen=True)
class WorldModelRuntimeSpec:
    model_id: str
    display_name: str
    runtime_module: str | None = None
    runtime_root_attr: str | None = "RUNTIME_DIR"
    runtime_root_func: str | None = None
    entrypoint_attr: str | None = "OFFICIAL_ENTRYPOINT"
    entrypoint_relative: str | None = None
    command_builder_attr: str | None = "build_command"
    missing_requirements_attr: str | None = "missing_requirements"
    pythonpath_builder_attr: str | None = "pythonpath_entries"
    blocked_reason_attr: str | None = "BLOCKED_REASON"
    blocked_reason: str | None = None
    source_dir_names: tuple[str, ...] = ()
    source_subdir: str = ""
    official_repo_url: str = ""
    required_env: tuple[str, ...] = ()
    required_assets: tuple[str, ...] = ()
    backend: str = "worldfoundry.world_model.in_tree_runtime_manifest"
    input_schema: Mapping[str, Any] = field(default_factory=dict)


_ROBOT_INPUTS = {
    "prompt": True,
    "image": True,
    "video": True,
    "actions": ["world_action", "robot_action", "interaction"],
}
_MINECRAFT_INPUTS = {
    "prompt": True,
    "image": True,
    "video": True,
    "actions": ["minecraft_action", "world_action", "interaction"],
}


WORLD_MODEL_RUNTIME_SPECS: Mapping[str, WorldModelRuntimeSpec] = {
    "adaworld": WorldModelRuntimeSpec(
        model_id="adaworld",
        display_name="AdaWorld",
        runtime_module="worldfoundry.synthesis.visual_generation.adaworld.worldfoundry_runtime",
        blocked_reason="",
        required_assets=("Little-Podi/AdaWorld adaworld.safetensors", "source video"),
        input_schema={"prompt": False, "image": False, "video": True, "actions": []},
    ),
    "ctrl-world": WorldModelRuntimeSpec(
        model_id="ctrl-world",
        display_name="Ctrl-World",
        runtime_module="worldfoundry.synthesis.visual_generation.ctrl_world.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        blocked_reason="",
        official_repo_url="https://github.com/Robert-gyj/Ctrl-World",
        required_assets=(
            "yjguo/Ctrl-World checkpoint-10000.pt",
            "stabilityai/stable-video-diffusion-img2vid",
            "openai/clip-vit-base-patch32",
            "an official DROID three-view trajectory or explicitly labeled synthetic three-view/action smoke input",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["robot_action"]},
    ),
    "causal-rcm": WorldModelRuntimeSpec(
        model_id="causal-rcm",
        display_name="Causal-rCM",
        runtime_module="worldfoundry.synthesis.visual_generation.rcm.worldfoundry_runtime",
        runtime_root_attr="RUNTIME_DIR",
        entrypoint_attr="INFERENCE_ENTRYPOINT",
        # Source is vendored under Apache-2.0, so only checkpoints gate execution;
        # the adapter's missing_requirements owns that check.
        blocked_reason="",
        official_repo_url="https://github.com/NVlabs/rcm",
        required_assets=(
            "Causal-rCM distilled DiT checkpoint",
            "Wan2.1 VAE (Wan2.1_VAE.pth)",
            "umT5 text encoder (models_t5_umt5-xxl-enc-bf16.pth)",
        ),
        # The released causal entrypoint is prompt/image conditioned; the block
        # schedule is the streaming control surface, not a discrete action space.
        input_schema={"prompt": True, "image": True, "video": False, "actions": []},
    ),
    "diamond": WorldModelRuntimeSpec(
        model_id="diamond",
        display_name="DIAMOND",
        runtime_module="worldfoundry.synthesis.visual_generation.diamond.worldfoundry_runtime",
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["game_action", "atari_action"]},
    ),
    "dino-wm": WorldModelRuntimeSpec(
        model_id="dino-wm",
        display_name="DINO-WM",
        runtime_module="worldfoundry.synthesis.visual_generation.dino_wm.worldfoundry_runtime",
        input_schema=_ROBOT_INPUTS,
    ),
    "genie-envisioner": WorldModelRuntimeSpec(
        model_id="genie-envisioner",
        display_name="Genie Envisioner",
        runtime_module="worldfoundry.synthesis.visual_generation.genie_envisioner.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        blocked_reason="",
        official_repo_url="https://github.com/AgibotTech/Genie-Envisioner",
        required_assets=(
            "agibot-world/Genie-Envisioner GE-base checkpoint",
            "Lightricks/LTX-Video 0.9 tokenizer, T5 text encoder, and VAE",
            "an official AgiBot/LeRobot three-view trajectory or explicitly labeled synthetic three-view smoke input",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["robot_action"]},
    ),
    "giga-world-0": WorldModelRuntimeSpec(
        model_id="giga-world-0",
        display_name="GigaWorld-0",
        runtime_module="worldfoundry.synthesis.visual_generation.giga_world_0.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        blocked_reason="",
        official_repo_url="https://github.com/open-gigaai/giga-world-0",
        required_assets=(
            "open-gigaai/GigaWorld-0-Video-GR1-2b transformer checkpoint",
            "google-t5/t5-11b encoder converted locally to safetensors",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers VAE",
            "an official GR1 observation or explicitly labeled single-image checkpoint smoke input",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": []},
    ),
    "leworldmodel": WorldModelRuntimeSpec(
        model_id="leworldmodel",
        display_name="LeWorldModel",
        runtime_module="worldfoundry.synthesis.visual_generation.le_wm.worldfoundry_runtime",
        input_schema=_ROBOT_INPUTS,
    ),
    "mira": WorldModelRuntimeSpec(
        model_id="mira",
        display_name="MIRA",
        runtime_module="worldfoundry.synthesis.visual_generation.mira_wm.worldfoundry_runtime",
        blocked_reason="",
        official_repo_url="https://github.com/mira-wm/mira",
        required_assets=("MIRA world-model checkpoint", "kyutai/rocket-science local dataset split"),
        input_schema={
            "prompt": False,
            "image": False,
            "video": False,
            "actions": ["rocket_league_keyboard", "game_action"],
        },
    ),
    "mineworld": WorldModelRuntimeSpec(
        model_id="mineworld",
        display_name="MineWorld",
        runtime_module="worldfoundry.synthesis.visual_generation.mineworld.worldfoundry_runtime",
        entrypoint_attr="INFERENCE_ENTRYPOINT",
        input_schema=_MINECRAFT_INPUTS,
    ),
    "nwm": WorldModelRuntimeSpec(
        model_id="nwm",
        display_name="Navigation World Models",
        runtime_module="worldfoundry.synthesis.visual_generation.nwm.worldfoundry_runtime",
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["navigation_action", "world_action"]},
    ),
    "oasis-500m": WorldModelRuntimeSpec(
        model_id="oasis-500m",
        display_name="Oasis 500M",
        runtime_module="worldfoundry.synthesis.visual_generation.open_oasis.worldfoundry_runtime",
        input_schema=_MINECRAFT_INPUTS,
    ),
    "open-dreamer": WorldModelRuntimeSpec(
        model_id="open-dreamer",
        display_name="Open Dreamer",
        runtime_module="worldfoundry.synthesis.visual_generation.open_dreamer.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        entrypoint_attr=None,
        entrypoint_relative="inference.py",
        # Gating lives in the adapter's missing_requirements so a fully staged
        # checkout, JAX environment, and checkpoint can actually execute.
        blocked_reason="",
        official_repo_url="https://github.com/next-state/open-dreamer",
        required_assets=(
            "reactor-team/open-dreamer checkout staged under ${WORLDFOUNDRY_MODEL_SOURCE_DIR}/open-dreamer-inference",
            "Open Dreamer Orbax checkpoint containing tokenizer and dynamics weights",
            "Minecraft/VPT context clip at 368x640 (or 360x640)",
        ),
        input_schema=_MINECRAFT_INPUTS,
    ),
    "sana-wm": WorldModelRuntimeSpec(
        model_id="sana-wm",
        display_name="SANA-WM",
        runtime_module="worldfoundry.synthesis.visual_generation.sana_wm.realtime",
        runtime_root_attr=None,
        entrypoint_attr=None,
        blocked_reason="",
        backend="worldfoundry.native_diffusion",
        input_schema={
            "prompt": True,
            "image": True,
            "video": False,
            "actions": ["camera_action", "world_action", "interaction"],
        },
    ),
    "starwm": WorldModelRuntimeSpec(
        model_id="starwm",
        display_name="StarWM",
        runtime_module="worldfoundry.synthesis.visual_generation.starwm.worldfoundry_runtime",
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["starcraft_action", "world_action"]},
    ),
    "tesseract": WorldModelRuntimeSpec(
        model_id="tesseract",
        display_name="TesserAct",
        runtime_module="worldfoundry.synthesis.action_generation.tesseract.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        blocked_reason="",
        official_repo_url="https://github.com/UMass-Embodied-AGI/TesserAct",
        required_assets=(
            "anyeZHY/tesseract RGB-depth-normal SFT transformer checkpoint",
            "THUDM/CogVideoX-5b-I2V tokenizer, text encoder, VAE, and scheduler",
            "an RGB input plus official depth/normal conditioning or explicit synthetic-gradient smoke conditioning",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["robot_action"]},
    ),
    "dreamx-world-5b-cam": WorldModelRuntimeSpec(
        model_id="dreamx-world-5b-cam",
        display_name="DreamX-World 5B Cam",
        runtime_module="worldfoundry.synthesis.visual_generation.dreamx_world.realtime",
        runtime_root_attr=None,
        entrypoint_attr=None,
        command_builder_attr=None,
        blocked_reason_attr=None,
        required_assets=(
            "GD-ML/DreamX-World-5B-Cam checkpoint",
            "Wan2.2-TI2V-5B VAE, tokenizer, and T5 checkpoint",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_navigation", "world_action"]},
    ),
    "dreamx-world-5b": WorldModelRuntimeSpec(
        model_id="dreamx-world-5b",
        display_name="DreamX-World 5B AR",
        runtime_module="worldfoundry.synthesis.visual_generation.dreamx_world.ar_realtime",
        runtime_root_attr=None,
        entrypoint_attr=None,
        command_builder_attr=None,
        blocked_reason_attr=None,
        required_assets=(
            "GD-ML/DreamX-World-5B checkpoint",
            "Wan2.2-TI2V-5B VAE, tokenizer, and T5 checkpoint",
        ),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_navigation", "world_action"]},
    ),
    "egowm": WorldModelRuntimeSpec(
        model_id="egowm",
        display_name="EgoWM",
        runtime_module="worldfoundry.synthesis.visual_generation.egowm.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/miccooper9/egowm",
        blocked_reason="",
        required_assets=(
            "anuragba/egowm 25-DoF navigation checkpoint",
            "Stable Video Diffusion img2vid base model",
            "official EgoWM source checkout",
        ),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["navigation_action", "egocentric_action"]},
    ),
    "happyoyster": WorldModelRuntimeSpec(
        model_id="happyoyster",
        display_name="HappyOyster",
        official_repo_url="https://www.happyoyster.cn",
        blocked_reason="HappyOyster API route is registered; execution requires API credentials and a documented generation endpoint.",
        required_env=("HAPPYOYSTER_API_KEY",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["api_request", "world_action"]},
    ),
    "hma": WorldModelRuntimeSpec(
        model_id="hma",
        display_name="HMA",
        runtime_module="worldfoundry.synthesis.visual_generation.hma.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/liruiw/HMA",
        blocked_reason="",
        required_assets=(
            "liruiw/hma-base-cont checkpoint",
            "Stable Video Diffusion temporal VAE",
            "official HMA source checkout and prompt frame or paired trajectory",
        ),
        input_schema=_ROBOT_INPUTS,
    ),
    "hunyuanworld-1": WorldModelRuntimeSpec(
        model_id="hunyuanworld-1",
        display_name="HunyuanWorld 1.0",
        source_dir_names=("HunyuanWorld-1.0",),
        official_repo_url="https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0",
        entrypoint_relative="demo_scenegen.py",
        blocked_reason="HunyuanWorld 1.0 official source route is registered; execution requires tencent/HunyuanWorld-1 weights plus reviewed third-party runtime assets.",
        required_assets=("tencent/HunyuanWorld-1 checkpoint", "Real-ESRGAN/ZIM/MoGe/Open3D runtime assets"),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_navigation", "scene_generation"]},
    ),
    "mosaicmem": WorldModelRuntimeSpec(
        model_id="mosaicmem",
        display_name="MosaicMem",
        source_dir_names=("mosaicmem",),
        official_repo_url="https://github.com/AbdelStark/mosaicmem",
        blocked_reason="MosaicMem source route is registered; WorldFoundry exposes memory-plan artifacts and requires a downstream generator for video/world synthesis.",
        required_assets=("MosaicMem memory inputs",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["memory_update", "interaction"]},
    ),
    "motionbricks": WorldModelRuntimeSpec(
        model_id="motionbricks",
        display_name="MotionBricks",
        source_dir_names=("GR00T-WholeBodyControl",),
        source_subdir="motionbricks",
        official_repo_url="https://github.com/NVlabs/GR00T-WholeBodyControl",
        blocked_reason="MotionBricks official source route is registered; execution requires motion checkpoints, robot assets, and a validated whole-body-control runtime configuration.",
        required_assets=("MotionBricks checkpoint/assets", "robot morphology/task assets"),
        input_schema={"prompt": True, "image": False, "video": True, "actions": ["whole_body_motion", "control_sequence"]},
    ),
    "omniforcing": WorldModelRuntimeSpec(
        model_id="omniforcing",
        display_name="OmniForcing",
        source_dir_names=("OmniForcing",),
        official_repo_url="https://github.com/OmniForcing/OmniForcing",
        blocked_reason="OmniForcing official source route is registered; execution requires the official audio-visual runtime stack and checkpoints.",
        required_assets=("OmniForcing checkpoint/assets",),
        input_schema={"prompt": True, "image": True, "video": True, "audio": True, "actions": ["forcing_signal", "interaction"]},
    ),
    "pointworld": WorldModelRuntimeSpec(
        model_id="pointworld",
        display_name="PointWorld",
        source_dir_names=("PointWorld",),
        official_repo_url="https://github.com/NVlabs/PointWorld",
        blocked_reason="PointWorld official source route is registered; execution requires nvidia/PointWorld_models assets and point-cloud/robot task inputs.",
        required_assets=("nvidia/PointWorld_models checkpoint assets", "point-cloud or robot task input"),
        input_schema={"prompt": True, "image": True, "video": True, "point_cloud": True, "actions": ["robot_action", "point_cloud_condition"]},
    ),
    "shotstream": WorldModelRuntimeSpec(
        model_id="shotstream",
        display_name="ShotStream",
        runtime_module="worldfoundry.synthesis.visual_generation.shotstream.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/KlingAIResearch/ShotStream",
        blocked_reason="",
        required_assets=(
            "KlingTeam/ShotStream causal checkpoint bundle",
            "Wan-AI/Wan2.1-T2V-1.3B base model",
            "official ShotStream source checkout",
        ),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["stream_control", "edit"]},
    ),
    "solarwm": WorldModelRuntimeSpec(
        model_id="solarwm",
        display_name="SolarWM",
        runtime_module="worldfoundry.synthesis.visual_generation.solarwm.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/Junchao-cs/SolarWM",
        blocked_reason="",
        required_assets=(
            "junchaoh-cs/SolarWM-Wan2.2-5B base and Stage2 checkpoint assets",
            "junchaoh-cs/SolarWM-Data release indexes and raw shards",
            "official SolarWM source checkout",
        ),
        input_schema={
            "prompt": False,
            "image": False,
            "video": True,
            "actions": ["camera_trajectory"],
        },
    ),
    "simworld": WorldModelRuntimeSpec(
        model_id="simworld",
        display_name="SimWorld",
        source_dir_names=("SimWorld",),
        official_repo_url="https://github.com/SimWorld-AI/SimWorld",
        entrypoint_relative="setup.py",
        blocked_reason="SimWorld route is registered; execution requires a running simulator/UnrealCV backend and task assets.",
        required_assets=("SimWorld simulator assets", "UnrealCV/runtime server"),
        input_schema={"prompt": True, "image": False, "video": False, "actions": ["simulator_action", "multi_agent_action"]},
    ),
    "uwm": WorldModelRuntimeSpec(
        model_id="uwm",
        display_name="Unified World Models",
        source_dir_names=("uwm", "unified-world-model"),
        official_repo_url="https://github.com/WEIRDLabUW/unified-world-model",
        blocked_reason="UWM official evaluation route is registered; execution requires distributed UWM checkpoints, robot datasets, and the official runtime environment.",
        required_assets=("UWM checkpoint assets", "LIBERO/DROID/robomimic evaluation data"),
        input_schema=_ROBOT_INPUTS,
    ),
    "vggt-world": WorldModelRuntimeSpec(
        model_id="vggt-world",
        display_name="VGGT-World",
        source_dir_names=("VGGT-World",),
        official_repo_url="https://github.com/SimonSun0810/VGGT-World",
        blocked_reason="VGGT-World official source route is registered; execution requires released VGGT-World checkpoints/assets and validated inference arguments.",
        required_assets=("VGGT-World checkpoint/assets",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["camera_navigation", "view_sequence"]},
    ),
    "vid2world": WorldModelRuntimeSpec(
        model_id="vid2world",
        display_name="Vid2World",
        runtime_module="worldfoundry.synthesis.visual_generation.vid2world.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/thuml/Vid2World",
        blocked_reason="Vid2World official evaluation route is registered; execution requires Vid2World checkpoints, conditioning videos/actions, and official config assets.",
        required_assets=("Vid2World checkpoint/assets",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["action", "world_action"]},
    ),
    "wilddet3d": WorldModelRuntimeSpec(
        model_id="wilddet3d",
        display_name="WildDet3D",
        runtime_module="worldfoundry.synthesis.visual_generation.wilddet3d.worldfoundry_runtime",
        runtime_root_attr=None,
        runtime_root_func="runtime_root",
        official_repo_url="https://github.com/allenai/WildDet3D",
        blocked_reason="",
        required_assets=("allenai/WildDet3D checkpoint", "robbyant/lingbot-depth-postrain-dc-vitl14 model.pt", "official source checkout and submodules", "RGB image"),
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["detection_query"]},
    ),
    "wildworld": WorldModelRuntimeSpec(
        model_id="wildworld",
        display_name="WildWorld",
        source_dir_names=("WildWorld",),
        official_repo_url="https://github.com/ShandaAI/WildWorld",
        blocked_reason="WildWorld official source route is registered; execution requires benchmark/dataset assets and evaluator configuration.",
        required_assets=("WildWorld dataset/benchmark assets",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["evaluation_query"]},
    ),
    "worldgrow": WorldModelRuntimeSpec(
        model_id="worldgrow",
        display_name="WorldGrow",
        source_dir_names=("WorldGrow",),
        official_repo_url="https://github.com/world-grow/WorldGrow",
        entrypoint_relative="trellis/pipelines/world_grow.py",
        blocked_reason="WorldGrow official source route is registered; execution requires UranusITS/WorldGrow checkpoints and TRELLIS runtime assets.",
        required_assets=("UranusITS/WorldGrow checkpoint/assets", "TRELLIS assets"),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["camera_navigation", "scene_expansion"]},
    ),
    "worldmem": WorldModelRuntimeSpec(
        model_id="worldmem",
        display_name="WorldMem",
        source_dir_names=("WorldMem",),
        official_repo_url="https://github.com/xizaoqu/WorldMem",
        blocked_reason="WorldMem route is registered; upstream inference model/weights/data are still required before generation can run.",
        required_assets=("WorldMem checkpoint/data assets",),
        input_schema={"prompt": True, "image": True, "video": True, "actions": ["minecraft_action", "memory_action"]},
    ),
}


class WorldModelRuntimeSynthesis(BaseSynthesis):
    """Synthesis facade for vendored world-model runtimes that are asset gated."""

    def __init__(
        self,
        *,
        spec: WorldModelRuntimeSpec,
        runtime_root: Path,
        entrypoint: Path | None,
        blocked_reason: str,
        profile: Any = None,
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.profile = spec
        self.runtime_profile = profile
        self.model_id = spec.model_id
        self.model_name = spec.display_name
        self.runtime_root = runtime_root
        self.entrypoint = entrypoint
        self.blocked_reason = blocked_reason
        self.device = device
        self.options = dict(options or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "WorldModelRuntimeSynthesis":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or "")
        if not resolved_model_id:
            raise ValueError("WorldModelRuntimeSynthesis requires model_id/profile_id.")
        spec = runtime_spec(resolved_model_id)
        runtime_root, entrypoint, blocked_reason = resolve_runtime_manifest(spec)
        profile = load_runtime_profile(resolved_model_id, check_conda_env_exists=False)
        return cls(
            spec=spec,
            runtime_root=runtime_root,
            entrypoint=entrypoint,
            blocked_reason=blocked_reason,
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            options=options,
        )

    def _checkpoint_paths(self) -> list[Path]:
        paths: list[Path] = []
        for key in ("checkpoint_path", "checkpoint_dir", "ckpt_path", "ckpt_base_path", "model_path", "pretrained_model_path"):
            value = self.options.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                paths.append(expand_worldfoundry_path(value))
        profile = self.runtime_profile
        for item in getattr(profile, "checkpoints", ()) or ():
            if not isinstance(item, Mapping):
                continue
            for key in ("local_dir", "path", "checkpoint_path", "checkpoint_dir"):
                value = item.get(key)
                if isinstance(value, (str, Path)) and str(value).strip():
                    paths.append(expand_worldfoundry_path(value))
        return list(dict.fromkeys(paths))

    def _artifact_kind(self) -> str:
        """Return the runtime profile's declared artifact kind."""
        return str(getattr(self.runtime_profile, "artifact_kind", None) or "generated_world")

    def _missing_runtime_requirements(self) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        if not self.runtime_root.is_dir():
            missing.append(
                {
                    "kind": "source_repo",
                    "path": str(self.runtime_root),
                    "reason": "runtime source root does not exist",
                }
            )
        if self.entrypoint is None:
            missing.append({"kind": "entrypoint", "path": "", "reason": "runtime entrypoint is not configured"})
        elif not self.entrypoint.is_file():
            missing.append(
                {
                    "kind": "entrypoint",
                    "path": str(self.entrypoint),
                    "reason": "runtime entrypoint does not exist",
                }
            )
        for env_name in self.profile.required_env:
            if not os.getenv(env_name):
                missing.append({"kind": "environment", "path": env_name, "reason": "required environment variable is not set"})

        checkpoint_paths = self._checkpoint_paths()
        if checkpoint_paths:
            if not any(path.exists() for path in checkpoint_paths):
                missing.append(
                    {
                        "kind": "checkpoint",
                        "path": ", ".join(str(path) for path in checkpoint_paths),
                        "reason": "none of the configured checkpoint paths exist",
                    }
                )
        else:
            profile_status = str(getattr(self.runtime_profile, "runtime_status", ""))
            notes = " ".join(str(item) for item in getattr(self.runtime_profile, "notes", ()) or ())
            if "checkpoint" in f"{profile_status} {notes}".lower():
                missing.append(
                    {
                        "kind": "checkpoint",
                        "path": "",
                        "reason": "runtime profile declares checkpoint-gated execution but no checkpoint path is configured",
                    }
                )
        missing.extend(self._model_specific_missing_requirements())
        return missing

    def _module_hook(self, attr_name: str | None):
        if not attr_name or not self.profile.runtime_module:
            return None
        module = importlib.import_module(self.profile.runtime_module)
        hook = getattr(module, attr_name, None)
        return hook if callable(hook) else None

    def _model_specific_missing_requirements(self) -> list[dict[str, str]]:
        hook = self._module_hook(self.profile.missing_requirements_attr)
        if hook is None:
            return []
        hook_options = dict(self.options)
        hook_options.setdefault("device", self.device)
        extra_missing = hook(
            options=hook_options,
            runtime_root=self.runtime_root,
            entrypoint=self.entrypoint,
            profile=self.runtime_profile,
        )
        return list(extra_missing or [])

    def _command(self, *, output_path: Path, prompt: str, plan_path: Path) -> list[str]:
        context = {
            "python": sys.executable,
            "runtime_root": str(self.runtime_root),
            "entrypoint": str(self.entrypoint or ""),
            "output_path": str(output_path),
            "output_dir": str(output_path.parent),
            "prompt": prompt,
            "device": self.device,
            "plan_path": str(plan_path),
            "options": dict(self.options),
            "profile": self.runtime_profile,
            **{str(key): value for key, value in self.options.items()},
        }
        template = self.options.get("command_template") or getattr(self.runtime_profile, "command_template", ())
        if template:
            return [str(item).format(**context) for item in template]
        hook = self._module_hook(self.profile.command_builder_attr)
        if hook is not None:
            return [str(item) for item in hook(context)]
        if self.entrypoint is None:
            return []
        return [sys.executable, str(self.entrypoint)]

    def _subprocess_env(self, *, output_path: Path, prompt: str, plan_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath_entries: list[str] = []
        hook = self._module_hook(self.profile.pythonpath_builder_attr)
        if hook is not None:
            extra_entries = hook(runtime_root=self.runtime_root, options=self.options, profile=self.runtime_profile)
            pythonpath_entries.extend(str(item) for item in extra_entries or ())
        pythonpath_entries.extend(
            [str(self.runtime_root), str(project_root(__file__)), env.get("PYTHONPATH", "")]
        )
        env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath_entries if item)
        env["WORLDFOUNDRY_MODEL_ID"] = self.model_id
        env["WORLDFOUNDRY_OUTPUT_PATH"] = str(output_path)
        env["WORLDFOUNDRY_OUTPUT_DIR"] = str(output_path.parent)
        env["WORLDFOUNDRY_PROMPT"] = prompt
        env["WORLDFOUNDRY_RUNTIME_PLAN"] = str(plan_path)
        env["WORLDFOUNDRY_DEVICE"] = self.device
        return env

    @staticmethod
    def _resolve_produced_artifact(output_path: Path) -> Path | None:
        if output_path.is_file():
            return output_path
        suffixes = {".mp4", ".mov", ".webm", ".gif", ".json", ".jsonl", ".txt", ".ply", ".glb", ".npz"}
        candidates = [
            path
            for path in output_path.parent.glob("*")
            if path.is_file() and path.suffix.lower() in suffixes and not path.name.endswith(".runtime_plan.json")
        ]
        if not candidates:
            return None
        same_suffix = [path for path in candidates if path.suffix.lower() == output_path.suffix.lower()]
        if same_suffix:
            return max(same_suffix, key=lambda path: path.stat().st_mtime)
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def plan(
        self,
        *,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        fps: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        missing = self._missing_runtime_requirements()
        if not missing and self.blocked_reason:
            missing.append(
                {
                    "kind": "runtime_requirement",
                    "path": self.model_id,
                    "reason": self.blocked_reason,
                }
            )
        status = "blocked" if missing or self.blocked_reason else "ready"
        return {
            "status": status,
            "model_id": self.model_id,
            "display_name": self.model_name,
            "artifact_kind": self._artifact_kind(),
            "backend": self.profile.backend,
            "backend_quality": "in_tree_runtime_manifest",
            "runtime_root": str(self.runtime_root),
            "runtime_root_exists": self.runtime_root.is_dir(),
            "entrypoint": str(self.entrypoint) if self.entrypoint is not None else None,
            "entrypoint_exists": self.entrypoint.is_file() if self.entrypoint is not None else False,
            "official_repo_url": self.profile.official_repo_url,
            "required_env": list(self.profile.required_env),
            "missing_env": [name for name in self.profile.required_env if not os.getenv(name)],
            "required_assets": list(self.profile.required_assets),
            "checkpoint_paths": [str(path) for path in self._checkpoint_paths()],
            "missing_assets": missing,
            "device": self.device,
            "prompt": prompt,
            "has_images": images is not None,
            "has_video": video is not None,
            "interactions": _jsonable(interactions or []),
            "fps": fps,
            "extra": _jsonable(dict(extra or {})),
            "blocked_reason": self.blocked_reason,
        }

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.options.get("plan_only"))
        plan = self.plan(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            fps=fps,
            extra=kwargs,
        )
        plan_path = _plan_path(output_path, self.model_id)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output = _artifact_output_path(output_path, self.model_id, getattr(self.runtime_profile, "artifact_filename", "world.mp4"))
        missing = list(plan["missing_assets"])
        if plan_only:
            return {
                "status": "blocked" if missing else "prepared",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",
                "runtime": self.profile.backend,
                "backend_quality": "official_entrypoint_plan",
                "artifact_path": str(plan_path),
                "plan_path": str(plan_path),
                "blocked_reason": self.blocked_reason if missing else None,
                "missing_assets": missing,
                "metadata": {
                    "runtime_root": str(self.runtime_root),
                    "entrypoint": str(self.entrypoint) if self.entrypoint is not None else None,
                },
            }
        if missing:
            return {
                "status": "blocked",
                "model_id": self.model_id,
                "artifact_kind": self._artifact_kind(),
                "runtime": self.profile.backend,
                "backend_quality": "missing_runtime_requirements",
                "artifact_path": str(plan_path),
                "plan_path": str(plan_path),
                "blocked_reason": self.blocked_reason,
                "missing_assets": missing,
                "metadata": {
                    "runtime_root": str(self.runtime_root),
                    "entrypoint": str(self.entrypoint) if self.entrypoint is not None else None,
                },
            }

        command = self._command(output_path=output, prompt=prompt, plan_path=plan_path)
        log_path = output.with_suffix(output.suffix + ".log")
        completed = subprocess.run(
            command,
            cwd=str(self.runtime_root),
            env=self._subprocess_env(output_path=output, prompt=prompt, plan_path=plan_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(kwargs.pop("timeout_seconds", 21600)),
            check=False,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self._artifact_kind(),
                "runtime": self.profile.backend,
                "backend_quality": "official_entrypoint_failed",
                "artifact_path": str(output),
                "plan_path": str(plan_path),
                "error": f"official entrypoint exited with code {completed.returncode}; see {log_path}",
                "metadata": {"command": command, "log_path": str(log_path)},
            }
        produced = self._resolve_produced_artifact(output)
        if produced is None:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self._artifact_kind(),
                "runtime": self.profile.backend,
                "backend_quality": "official_entrypoint_no_artifact",
                "artifact_path": str(output),
                "plan_path": str(plan_path),
                "error": f"official entrypoint completed but no artifact was found at or near {output}",
                "metadata": {"command": command, "log_path": str(log_path)},
            }
        if produced != output:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, output)
        return {
            "status": "succeeded",
            "model_id": self.model_id,
            "artifact_kind": self._artifact_kind(),
            "runtime": self.profile.backend,
            "backend_quality": "official_entrypoint",
            "artifact_path": str(output),
            "plan_path": str(plan_path),
            "metadata": {
                "runtime_root": str(self.runtime_root),
                "entrypoint": str(self.entrypoint) if self.entrypoint is not None else None,
                "command": command,
                "log_path": str(log_path),
            },
        }


def plan_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    """Load the runtime plan a command builder was handed, or an empty mapping.

    :meth:`WorldModelRuntimeSynthesis.predict` writes the plan to disk before it
    builds the command, so ``context["plan_path"]`` is readable from inside a
    ``build_command`` hook. A missing or malformed plan is not fatal: the caller
    falls back to load-time options.
    """
    plan_path = context.get("plan_path")
    if not plan_path:
        return {}
    path = Path(str(plan_path))
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def command_settings(context: Mapping[str, Any]) -> dict[str, Any]:
    """Merge load-time options with the call-time kwargs recorded in the plan.

    Only ``self.options`` reaches a runtime manifest hook directly; per-call
    kwargs arrive through ``plan["extra"]``. Call-time values win so a single
    loaded pipeline can serve several differently-parameterized requests.
    """
    settings = dict(context.get("options") or {})
    extra = plan_payload(context).get("extra")
    if isinstance(extra, Mapping):
        settings.update(extra)
    return settings


def runtime_spec(model_id: str) -> WorldModelRuntimeSpec:
    key = model_id.lower().replace("_", "-")
    try:
        return WORLD_MODEL_RUNTIME_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(WORLD_MODEL_RUNTIME_SPECS))
        raise KeyError(f"Unknown world-model runtime manifest {model_id!r}. Known ids: {known}") from exc


def resolve_runtime_manifest(spec: WorldModelRuntimeSpec) -> tuple[Path, Path | None, str]:
    module = importlib.import_module(spec.runtime_module) if spec.runtime_module else None
    if module is not None and spec.runtime_root_func:
        root_value = getattr(module, spec.runtime_root_func)()
    elif module is not None and spec.runtime_root_attr:
        root_value = getattr(module, spec.runtime_root_attr)
    elif spec.source_dir_names:
        root_value = _resolve_source_root(spec)
    else:
        root_value = _resolve_source_root(spec)

    runtime_root = Path(root_value).expanduser().resolve()

    entrypoint: Path | None = None
    if module is not None and spec.entrypoint_attr and hasattr(module, spec.entrypoint_attr):
        entrypoint_value = getattr(module, spec.entrypoint_attr)
        if entrypoint_value not in (None, ""):
            entrypoint = Path(entrypoint_value).expanduser().resolve()
    elif spec.entrypoint_relative:
        entrypoint = (runtime_root / spec.entrypoint_relative).expanduser().resolve()

    blocked_reason = spec.blocked_reason
    if blocked_reason is None and module is not None and spec.blocked_reason_attr and hasattr(module, spec.blocked_reason_attr):
        blocked_reason = str(getattr(module, spec.blocked_reason_attr))
    if blocked_reason is None:
        blocked_reason = f"{spec.display_name} runtime is vendored in-tree but remains checkpoint/environment gated."
    return runtime_root, entrypoint, blocked_reason


def _resolve_source_root(spec: WorldModelRuntimeSpec) -> Path:
    """Return the expected in-tree runtime root for source-bound manifests.

    Official repositories remain provenance/acquisition metadata. The model
    inference facade resolves source-bound manifests to WorldFoundry's in-tree
    runtime layout instead of machine-local checkouts.
    """
    _ = (spec.source_dir_names, spec.source_subdir)
    runtime_dir_name = spec.model_id.replace("-", "_").replace(".", "_")
    return Path(__file__).resolve().parent / runtime_dir_name


def _plan_path(output_path: str | Path | None, model_id: str) -> Path:
    if output_path is None:
        return (Path.cwd() / f"{model_id}_runtime_plan.json").resolve()
    target = Path(output_path).expanduser()
    if target.suffix:
        return target.with_suffix(".json").resolve()
    return (target / f"{model_id}_runtime_plan.json").resolve()


def _artifact_output_path(output_path: str | Path | None, model_id: str, filename: str) -> Path:
    if output_path is None:
        return (Path.cwd() / filename).resolve()
    target = Path(output_path).expanduser()
    if target.suffix:
        return target.resolve()
    return (target / filename).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "WORLD_MODEL_RUNTIME_SPECS",
    "WorldModelRuntimeSpec",
    "WorldModelRuntimeSynthesis",
    "command_settings",
    "plan_payload",
    "resolve_runtime_manifest",
    "runtime_spec",
]
