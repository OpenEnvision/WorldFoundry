"""Runtime Manifest visual generation pipeline module."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ...core.io.paths import checkpoint_root_candidates
from ...operators.world_model_runtime_operator import WorldModelRuntimeOperator
from ...synthesis.visual_generation.memory.runtime import RuntimeMemory
from ...synthesis.visual_generation.runtime_manifest import WorldModelRuntimeSynthesis
from ..pipeline_utils import PipelineABC


def _promote_three_view_paths(kwargs: dict) -> dict:
    """Normalize generic multiview inputs to the runtime manifest contract."""

    if not kwargs.get("input_views"):
        for key in (
            "view_images",
            "reference_image_paths",
            "reference_files",
            "reference_images",
            "images",
        ):
            candidate = kwargs.get(key)
            if (
                isinstance(candidate, Sequence)
                and not isinstance(candidate, (str, bytes, bytearray))
                and len(candidate) == 3
                and all(isinstance(item, (str, os.PathLike)) for item in candidate)
            ):
                kwargs["input_views"] = [os.fspath(item) for item in candidate]
                kwargs.setdefault("input_mode", "explicit-three-view")
                break
    if not kwargs.get("image_path"):
        candidate = kwargs.get("images")
        if isinstance(candidate, (str, os.PathLike)):
            kwargs["image_path"] = os.fspath(candidate)
    return kwargs


class WorldModelRuntimePipeline(PipelineABC):
    """Pipeline surface for vendored world-model runtimes with asset-gated execution."""

    MODEL_ID = "world-model-runtime"
    OPERATOR_CLS = WorldModelRuntimeOperator
    MEMORY_CLS = RuntimeMemory
    SYNTHESIS_CLS = WorldModelRuntimeSynthesis
    MEMORY_RECORD_TYPE = "world_model_runtime_plan"
    generation_type = "world_model"

    # Call kwargs that the runtime gate has to see. The synthesis facade only
    # forwards load-time options to a runtime's ``missing_requirements`` hook, so
    # a path supplied per call would otherwise read as missing and block the run.
    # Subclasses list the option names their adapter gates on; empty is a no-op.
    RUNTIME_GATED_OPTION_KEYS: tuple[str, ...] = ()

    def __call__(self, *args, **kwargs):
        """Execute the complete pipeline generation flow."""
        kwargs.setdefault("return_dict", True)
        kwargs = self._promote_call_options(kwargs)

        options = getattr(self.synthesis_model, "options", None)
        overrides = {
            key: kwargs[key] for key in self.RUNTIME_GATED_OPTION_KEYS if kwargs.get(key) not in (None, "")
        }
        if not isinstance(options, dict) or not overrides:
            return super().__call__(*args, **kwargs)

        # Restore afterwards so a reused pipeline never inherits a previous request.
        previous = dict(options)
        options.update(overrides)
        try:
            return super().__call__(*args, **kwargs)
        finally:
            options.clear()
            options.update(previous)

    def _promote_call_options(self, kwargs: dict) -> dict:
        """Hook for subclasses that derive gated options from generic call inputs."""
        return kwargs


class AdaWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for AdaWorld visual generation."""
    MODEL_ID = "adaworld"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint",
        "checkpoint_path",
        "ckpt",
        "ckpt_path",
        "model_path",
        "config",
        "config_path",
        "source_video",
        "input_path",
        "video_path",
        "target_video",
        "data_root",
        "input_dir",
        "num_samples",
        "start_index",
        "resolution",
        "video_len",
        "context_frame",
        "num_steps",
        "cfg_scale",
        "aug_level",
        "fps",
        "seed",
    )

    @classmethod
    def from_pretrained(cls, model_path=None, *args, **kwargs):
        if model_path is None or isinstance(model_path, dict):
            options = dict(model_path or {})
            if not any(options.get(key) for key in ("checkpoint", "checkpoint_path", "ckpt_path", "model_path")):
                for root in checkpoint_root_candidates("Little-Podi--AdaWorld"):
                    candidate = root / "adaworld.safetensors"
                    if candidate.is_file():
                        options["checkpoint_path"] = str(candidate)
                        break
            model_path = options
        return super().from_pretrained(model_path, *args, **kwargs)

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("source_video"):
            candidate = kwargs.get("video")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["source_video"] = os.fspath(candidate)
        return kwargs


class CausalRCMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Causal-rCM streaming visual generation."""

    MODEL_ID = "causal-rcm"
    MODEL_PATH_OPTION = "dit_path"
    RUNTIME_GATED_OPTION_KEYS = (
        "dit_path",
        "checkpoint_path",
        "checkpoint_dir",
        "vae_path",
        "text_encoder_path",
        "image_path",
        "python_executable",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        """Normalize Causal-rCM inputs into the options the runtime gate reads."""
        if not kwargs.get("image_path"):
            candidate = kwargs.get("images")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["image_path"] = os.fspath(candidate)
        # The shared checkpoint check looks for `checkpoint_path`; upstream names
        # the distilled DiT `dit_path`. Mirror it so both gates agree.
        if kwargs.get("dit_path") and not kwargs.get("checkpoint_path"):
            kwargs["checkpoint_path"] = kwargs["dit_path"]
        return kwargs


class CtrlWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for CtrlWorld visual generation."""
    MODEL_ID = "ctrl-world"
    RUNTIME_GATED_OPTION_KEYS = (
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
        "fps",
        "seed",
        "mixed_precision",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        return _promote_three_view_paths(kwargs)


class DIAMONDPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DIAMOND visual generation."""
    MODEL_ID = "diamond"
    RUNTIME_GATED_OPTION_KEYS = (
        "variant",
        "spawn_dir",
        "spawn_id",
        "action",
        "actions_path",
        "sampler",
        "seed",
        "checkpoint",
        "checkpoint_path",
        "model_path",
        "pretrained",
        "pretrained_game",
        "pretrained_dir",
        "config_dir",
        "num_steps_initial_collect",
        "fps",
        "size",
        "record",
        "headless_steps",
        "dataset_dir",
    )

    @classmethod
    def _resolve_model_id(cls, options, model_id=None):
        selected = super()._resolve_model_id(options, model_id)
        if selected in {"diamond-csgo", "diamond-atari"}:
            variant = selected.removeprefix("diamond-")
            if options.get("variant", variant) != variant:
                raise ValueError(f"{selected} requires variant={variant!r}")
            options["variant"] = variant
            return "diamond"
        return selected

    def _promote_call_options(self, kwargs: dict) -> dict:
        if kwargs.get("frames") and not kwargs.get("headless_steps"):
            kwargs["headless_steps"] = kwargs["frames"]
        return kwargs


class DIAMONDCsgoPipeline(DIAMONDPipeline):
    """DIAMOND CS:GO with its released spawn and spatial upsampler."""

    MODEL_ID = "diamond-csgo"


class DinoWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DinoWM visual generation."""
    MODEL_ID = "dino-wm"
    RUNTIME_GATED_OPTION_KEYS = (
        "ckpt_base_path", "checkpoint_dir", "checkpoint_path", "model_name", "run_name",
        "model_epoch", "config", "config_path", "seed", "n_evals", "goal_source", "goal_H",
    )


class GenieEnvisionerPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for GenieEnvisioner visual generation."""
    MODEL_ID = "genie-envisioner"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint_path",
        "checkpoint",
        "ckpt_path",
        "genie_checkpoint_path",
        "base_model_dir",
        "ltx_base_model_dir",
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
        "fps",
        "seed",
        "mixed_precision",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        return _promote_three_view_paths(kwargs)


class GigaWorld0Pipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for GigaWorld0 visual generation."""
    MODEL_ID = "giga-world-0"
    RUNTIME_GATED_OPTION_KEYS = (
        "transformer_model_dir",
        "transformer_model_path",
        "checkpoint_path",
        "checkpoint",
        "model_path",
        "text_encoder_model_dir",
        "text_encoder_model_path",
        "vae_model_dir",
        "vae_model_path",
        "image_path",
        "input_image",
        "attention_backend",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "fps",
        "seed",
        "max_text_length",
        "mixed_precision",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("image_path"):
            candidate = kwargs.get("images")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["image_path"] = os.fspath(candidate)
        return kwargs


class LeWorldModelPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for LeWorldModel visual generation."""
    MODEL_ID = "leworldmodel"


class MIRAPipeline(WorldModelRuntimePipeline):
    """Inference-only Workspace pipeline for MIRA."""

    MODEL_ID = "mira"
    MODEL_PATH_OPTION = "checkpoint_path"


class MineWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MineWorld visual generation."""
    MODEL_ID = "mineworld"


class NWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for NWM visual generation."""
    MODEL_ID = "nwm"


class Oasis500MPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Oasis500M visual generation."""
    MODEL_ID = "oasis-500m"
    RUNTIME_GATED_OPTION_KEYS = (
        "oasis_ckpt",
        "checkpoint_path",
        "model_path",
        "vae_ckpt",
        "vae_checkpoint_path",
        "prompt_path",
        "actions_path",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("prompt_path"):
            for key in ("images", "video"):
                candidate = kwargs.get(key)
                if isinstance(candidate, (str, os.PathLike)):
                    kwargs["prompt_path"] = os.fspath(candidate)
                    break
        return kwargs


class OpenDreamerPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Open Dreamer (Dreamer 4) visual generation."""

    MODEL_ID = "open-dreamer"
    RUNTIME_GATED_OPTION_KEYS = ("input_mp4", "actions_path", "checkpoint_path", "python_executable")

    def _promote_call_options(self, kwargs: dict) -> dict:
        """Derive ``input_mp4`` from a filesystem context clip.

        The runtime plan records only whether a video was supplied, not its path,
        so a clip passed as ``video=``/``images=`` has to become an explicit
        option. Decoded frames and PIL images stay untouched.
        """
        if not kwargs.get("input_mp4"):
            for key in ("video", "images"):
                candidate = kwargs.get(key)
                if isinstance(candidate, (str, os.PathLike)):
                    kwargs["input_mp4"] = os.fspath(candidate)
                    break
        return kwargs


class SanaWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for SanaWM visual generation."""
    MODEL_ID = "sana-wm"


class StarWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for StarWM visual generation."""
    MODEL_ID = "starwm"


class TesserActPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for TesserAct visual generation."""
    MODEL_ID = "tesseract"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint_dir",
        "checkpoint_path",
        "tesseract_checkpoint_dir",
        "weights_path",
        "base_model_dir",
        "cogvideox_base_model_dir",
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
        "memory_efficient",
        "fps",
        "seed",
        "mixed_precision",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("image_path"):
            candidate = kwargs.get("images")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["image_path"] = os.fspath(candidate)
        return kwargs



class EgoWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for EgoWM visual generation."""
    MODEL_ID = "egowm"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint_path",
        "egowm_checkpoint_path",
        "base_model_dir",
        "svd_base_model_dir",
        "image_path",
        "input_image",
        "seed",
        "num_frames",
        "num_inference_steps",
        "fps",
        "height",
        "width",
        "motion_bucket_id",
        "action_scale",
        "conditions_path",
        "egowm_conditions_path",
        "actions_path",
        "egowm_actions_path",
        "smoke_synthetic",
        "variant",
        "egowm_variant",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("image_path"):
            candidate = kwargs.get("images")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["image_path"] = os.fspath(candidate)
        return kwargs


class HappyOysterPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HappyOyster visual generation."""
    MODEL_ID = "happyoyster"


class HMAPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HMA visual generation."""
    MODEL_ID = "hma"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint_dir",
        "checkpoint_path",
        "hma_checkpoint_dir",
        "base_model_dir",
        "svd_base_model_dir",
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
        "fps",
        "seed",
    )

    def _promote_call_options(self, kwargs: dict) -> dict:
        if not kwargs.get("image_path"):
            candidate = kwargs.get("images")
            if isinstance(candidate, (str, os.PathLike)):
                kwargs["image_path"] = os.fspath(candidate)
        return kwargs


class HunyuanWorld1Pipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HunyuanWorld1 visual generation."""
    MODEL_ID = "hunyuanworld-1"


class MosaicMemPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MosaicMem visual generation."""
    MODEL_ID = "mosaicmem"


class MotionBricksPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MotionBricks visual generation."""
    MODEL_ID = "motionbricks"


class OmniForcingPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for OmniForcing visual generation."""
    MODEL_ID = "omniforcing"


class PointWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for PointWorld visual generation."""
    MODEL_ID = "pointworld"


class ShotStreamPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for ShotStream visual generation."""
    MODEL_ID = "shotstream"
    RUNTIME_GATED_OPTION_KEYS = (
        "checkpoint_dir",
        "checkpoint_path",
        "shotstream_checkpoint_dir",
        "wan_model_dir",
        "base_model_dir",
        "data_path",
        "input_csv",
        "dataset_path",
        "seed",
    )


class SolarWMPipeline(WorldModelRuntimePipeline):
    """Inference-only pipeline for SolarWM Wan2.2-5B Stage2."""

    MODEL_ID = "solarwm"
    RUNTIME_GATED_OPTION_KEYS = (
        "base_model_dir",
        "model_base_path",
        "base_path",
        "checkpoint_dir",
        "checkpoint_path",
        "solarwm_checkpoint_dir",
        "data_root",
        "dataset_root",
        "index_root",
        "test_index",
        "dataset_index",
        "config_path",
        "solarwm_config",
        "sample_count",
        "seed",
    )


class SimWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for SimWorld visual generation."""
    MODEL_ID = "simworld"


class UWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for UWM visual generation."""
    MODEL_ID = "uwm"


class VGGTWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for VGGTWorld visual generation."""
    MODEL_ID = "vggt-world"


class Vid2WorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Vid2World visual generation."""
    MODEL_ID = "vid2world"


class WildDet3DPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WildDet3D visual generation."""
    MODEL_ID = "wilddet3d"
    RUNTIME_GATED_OPTION_KEYS = (
        "image_path", "input_image", "checkpoint_path", "lingbot_config_path",
        "intrinsics_path", "score_threshold", "score_3d_threshold", "python_executable",
    )


class WildWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WildWorld visual generation."""
    MODEL_ID = "wildworld"


class WorldGrowPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WorldGrow visual generation."""
    MODEL_ID = "worldgrow"


class WorldMemPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WorldMem visual generation."""
    MODEL_ID = "worldmem"


__all__ = [
    "AdaWorldPipeline",
    "CausalRCMPipeline",
    "CtrlWorldPipeline",
    "DIAMONDPipeline",
    "DIAMONDCsgoPipeline",
    "DinoWMPipeline",
    "EgoWMPipeline",
    "GenieEnvisionerPipeline",
    "GigaWorld0Pipeline",
    "HMAPipeline",
    "HappyOysterPipeline",
    "HunyuanWorld1Pipeline",
    "LeWorldModelPipeline",
    "MIRAPipeline",
    "MineWorldPipeline",
    "MosaicMemPipeline",
    "MotionBricksPipeline",
    "NWMPipeline",
    "OmniForcingPipeline",
    "Oasis500MPipeline",
    "OpenDreamerPipeline",
    "PointWorldPipeline",
    "SanaWMPipeline",
    "ShotStreamPipeline",
    "SimWorldPipeline",
    "SolarWMPipeline",
    "StarWMPipeline",
    "TesserActPipeline",
    "UWMPipeline",
    "VGGTWorldPipeline",
    "Vid2WorldPipeline",
    "WildDet3DPipeline",
    "WildWorldPipeline",
    "WorldModelRuntimePipeline",
    "WorldGrowPipeline",
    "WorldMemPipeline",
]
