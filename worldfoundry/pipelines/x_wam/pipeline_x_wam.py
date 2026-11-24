"""Independent world-action pipeline for X-WAM."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from worldfoundry.operators.embodied_action_operator import (
    EmbodiedActionOperator,
    _compact_dict,
    _extract_action_signal,
    _first_present,
)
from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class XWAMOperator(EmbodiedActionOperator):
    """Three-view RGB, language, and proprioception contract for X-WAM."""

    MODEL_ID = "x-wam"
    POLICY_FAMILY = "wan2p2_joint_4d_world_action_model"
    ACTION_REPRESENTATION = "32_step_relative_end_effector_action_chunk"
    OBSERVATION_LAYOUT = "three_view_rgb_language_absolute_end_effector_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "state": True,
        "camera_count": 3,
        "actions": ["actions", "robot_actions", "action_chunk", "world_actions"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(
                kwargs,
                "instruction",
                "task_instruction",
                "world_model_prompt",
            )
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "model_track": "wam",
                "benchmark": kwargs.get("variant", "robocasa_sft"),
            },
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        rgb_views = _first_present(kwargs, "x_wam_rgb_views", "rgb_views", "multi_view_rgb")
        if rgb_views is not None:
            images = rgb_views
        state = _first_present(kwargs, "proprios", "proprio", "robot_state", "state")
        observation = _compact_dict(
            {
                "rgb_views": images,
                "video_context": video,
                "state": state,
                "camera_keys": kwargs.get("camera_keys"),
                "env_rank": kwargs.get("env_rank"),
                "rollout_id": kwargs.get("rollout_id"),
                "step_id": kwargs.get("step_id"),
                "seed": kwargs.get("seed"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "observation": observation,
            "extra_inputs": {
                **kwargs,
                "official_policy_observation": observation,
                "x_wam_observation": observation,
                "x_wam_input_contract": {
                    "modalities": ["three_view_rgb", "language", "proprioception"],
                    "official_variants": ["robocasa_sft", "robotwin_sft"],
                    "action_horizon": 32,
                    "optional_outputs": ["future_multiview_rgb", "future_depth"],
                    "supports_world_action_modeling": True,
                },
            },
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "robot_actions", "action_chunk", "world_actions")
        self.interaction_history.append(actions)
        variant = raw.get("variant") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="relative_end_effector_delta_action_chunk",
                control_mode="world_action_model_policy",
                action_horizon=32,
                supports_world_action_modeling=True,
                predicts_video=True,
                chunked=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_end_effector_delta", "dimensions": "7_or_14", "horizon": 32},
            "policy_controls": {
                "policy_architecture": "wan2p2_ti2v_joint_video_action_proprio_flow_model",
                "variant": variant or "robocasa_sft",
                "action_denoise_steps": 10,
                "video_denoise_steps": 50,
                "output_semantics": "action_chunk_with_optional_future_rgb_depth_mosaic",
            },
        }


class XWAMPipeline(ComponentPipeline):
    """WorldFoundry WAM pipeline for X-WAM action and future-world synthesis."""

    MODEL_ID = "x-wam"
    MODEL_PATH_OPTION = "checkpoint_path"
    OPERATOR_CLS = XWAMOperator
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.x_wam:XWAMSynthesis"
    generation_type = "world_action_model"


__all__ = ["XWAMOperator", "XWAMPipeline"]
