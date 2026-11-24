"""Native embodied VLA operator contracts for checkpoint-backed action policies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = _nested_get(mapping, key) if "." in key else mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _as_image_list(images: Any) -> list[Any]:
    if images is None or (isinstance(images, str) and images == ""):
        return []
    if isinstance(images, Mapping):
        return [value for _, value in sorted(images.items(), key=lambda item: str(item[0])) if value is not None]
    if isinstance(images, (str, bytes, bytearray)):
        return [images]
    if isinstance(images, list):
        return images
    if isinstance(images, tuple):
        return list(images)
    return [images]


def _camera_map(camera_keys: list[str], images: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(images, Mapping):
        return {key: images[key] for key in camera_keys if key in images and images[key] is not None}
    values = _as_image_list(images)
    mapped = {key: values[index] for index, key in enumerate(camera_keys) if index < len(values)}
    for key in camera_keys:
        if key in kwargs and kwargs[key] is not None:
            mapped[key] = kwargs[key]
    return mapped


class OpenVLAOFTOperator(EmbodiedActionOperator):
    """Operator for OpenVLA-OFT LIBERO action-chunk checkpoints."""

    MODEL_ID = "openvla-oft"
    POLICY_FAMILY = "openvla_oft_action_chunk_policy"
    ACTION_REPRESENTATION = "continuous_libero_action_chunk"
    OBSERVATION_LAYOUT = "libero_full_and_wrist_rgb_state_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "state": True,
        "camera_keys": ["full_image", "wrist_image"],
        "actions": ["action_chunk", "actions", "robot_action"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(kwargs, "task_description", "instruction", "task_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {"task_description": instruction},
        }

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        camera_keys = list(kwargs.get("camera_keys") or self.input_schema.get("camera_keys") or [])
        cameras = _camera_map(camera_keys, images, kwargs)
        state = _first_present(kwargs, "state", "proprio", "robot_state", "eef_state")
        observation = _compact_dict(
            {
                **cameras,
                "state": state,
                "task_description": _first_present(kwargs, "task_description", "instruction", "task_instruction"),
                "unnorm_key": kwargs.get("unnorm_key"),
                "task_suite_name": kwargs.get("task_suite_name"),
            }
        )
        return {
            "images": cameras or images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "openvla_oft_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_chunk", "actions", "robot_action")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions,
                action_contract="8x7_libero_action_chunk",
                control_mode="libero_delta_eef_action_chunk",
                action_horizon=8,
            ),
            "action_space": {"kind": "continuous_chunk", "dimensions": 7},
            "policy_controls": {"normalization": "dataset_statistics", "uses_wrist_image": True},
        }


class CogACTOperator(EmbodiedActionOperator):
    """Operator for CogACT DiT action-chunk policies."""

    MODEL_ID = "cogact"
    POLICY_FAMILY = "cogact_dit_action_chunk_policy"
    ACTION_REPRESENTATION = "16x7_continuous_action_chunk"
    OBSERVATION_LAYOUT = "single_rgb_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "state": False,
        "actions": ["action_chunk", "actions"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "task_instruction", "task")
        )
        instruction = "" if instruction is None else str(instruction)
        return {"prompt": instruction, "task_instruction": instruction, "prompt_channels": {"instruction": instruction}}

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        image = _first_present(kwargs, "image", "full_image")
        if image is None:
            image = images
        observation = _compact_dict(
            {
                "image": image,
                "unnorm_key": kwargs.get("unnorm_key"),
                "cfg_scale": kwargs.get("cfg_scale"),
                "num_ddim_steps": kwargs.get("num_ddim_steps"),
                "use_ddim": kwargs.get("use_ddim"),
            }
        )
        return {
            "images": image,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "cogact_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_chunk", "actions")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions,
                action_contract="16x7_continuous_action_chunk",
                control_mode="dit_denoised_action_chunk",
                action_horizon=16,
            ),
            "action_space": {"kind": "continuous_chunk", "dimensions": 7},
            "policy_controls": {"sampler": "ddpm_or_ddim", "normalization": "dataset_statistics"},
        }


class DBCogACTOperator(EmbodiedActionOperator):
    """Operator for Dexbotic DB-CogACT direct policy inference."""

    MODEL_ID = "db-cogact"
    POLICY_FAMILY = "dexbotic_cogact_policy"
    ACTION_REPRESENTATION = "model_declared_relative_or_absolute_action_chunk"
    OBSERVATION_LAYOUT = "dexbotic_numbered_camera_slots_language_optional_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "state": "optional",
        "image_slots": ["1", "2", "3"],
        "actions": ["actions", "action_chunk"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "prompt", "instruction", "task_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {"prompt": instruction, "task_instruction": instruction, "prompt_channels": {"prompt": instruction}}

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        slots = list(kwargs.get("image_slots") or self.input_schema.get("image_slots") or [])
        input_images = images if images is not None else kwargs.get("images")
        slot_images = _camera_map(slots, input_images, kwargs)
        observation = {"prompt": _first_present(kwargs, "prompt", "instruction", "task_instruction")}
        observation.update({f"image/{slot}": value for slot, value in slot_images.items()})
        state = _first_present(kwargs, "state", "robot_state", "proprio")
        if state is not None:
            observation["state"] = state
        observation = _compact_dict(observation)
        return {
            "images": slot_images or images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "db_cogact_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "action_chunk")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions, action_contract="dexbotic_action_output", control_mode="dexbotic_policy_action"
            ),
            "action_space": {"kind": "continuous_chunk", "mode": "relative_or_absolute"},
            "policy_controls": {"api_contract": "direct_select_action", "normalization": "norm_stats_json"},
        }


class VLANeXtOperator(EmbodiedActionOperator):
    """Operator for VLANeXt LIBERO history-conditioned action policies."""

    MODEL_ID = "vlanext"
    POLICY_FAMILY = "vlanext_libero_action_chunk_policy"
    ACTION_REPRESENTATION = "normalized_libero_action_chunk"
    OBSERVATION_LAYOUT = "libero_rgb_wrist_history_state_action_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": True,
        "state": True,
        "actions": ["actions", "action_chunk"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(kwargs, "task_label", "task_description", "instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {"prompt": instruction, "task_instruction": instruction, "prompt_channels": {"task_label": instruction}}

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        image_map = images if isinstance(images, Mapping) else {}
        full_image = _first_present(kwargs, "full_image", "image")
        if full_image is None:
            full_image = image_map.get("full_image")
        if full_image is None:
            full_image = images
        wrist = _first_present(kwargs, "full_image_wrist", "wrist_image")
        if wrist is None:
            wrist = image_map.get("full_image_wrist")
        observation = _compact_dict(
            {
                "full_image": full_image,
                "full_image_wrist": wrist,
                "image_history": kwargs.get("image_history"),
                "image_history_wrist": kwargs.get("image_history_wrist"),
                "state_history": kwargs.get("state_history"),
                "action_history": kwargs.get("action_history"),
                "task_description": _first_present(kwargs, "task_description", "task_label", "instruction"),
            }
        )
        return {
            "images": {"full_image": full_image, "full_image_wrist": wrist},
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "vlanext_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "action_chunk")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions, action_contract="normalized_libero_action_chunk", control_mode="vlanext_policy_action"
            ),
            "action_space": {"kind": "continuous_chunk", "dimensions": 7},
            "policy_controls": {"uses_history": True, "sampler": "diffusion_or_regression"},
        }


class MolmoBotOperator(EmbodiedActionOperator):
    """Operator for MolmoBot action-chunk inference."""

    MODEL_ID = "molmobot"
    POLICY_FAMILY = "molmobot_action_chunk_policy"
    ACTION_REPRESENTATION = "robot_specific_action_chunk"
    OBSERVATION_LAYOUT = "camera_list_language_optional_qpos"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "state": "optional",
        "camera_keys": ["exo_camera_1", "wrist_camera"],
        "actions": ["actions", "action_chunk"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        task = prompt if prompt not in (None, "") else _first_present(kwargs, "task", "task_description", "instruction")
        task = "" if task is None else str(task)
        return {"prompt": task, "task_instruction": task, "prompt_channels": {"task": task}}

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        camera_keys = list(kwargs.get("camera_keys") or self.input_schema.get("camera_keys") or [])
        cameras = _camera_map(camera_keys, images, kwargs)
        state = _first_nested(kwargs, "state", "qpos", "qpos.arm", "robot_state", "joint_state")
        observation = _compact_dict(
            {
                "images": cameras or _as_image_list(images),
                "camera_keys": camera_keys,
                "state": state,
                "task": _first_present(kwargs, "task", "task_description", "instruction"),
                "norm_repo_id": kwargs.get("norm_repo_id"),
            }
        )
        return {
            "images": cameras or images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "molmobot_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "action_chunk")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions, action_contract="robot_specific_action_chunk", control_mode="molmobot_policy_action"
            ),
            "action_space": {"kind": "continuous_chunk", "dimensions": "robot_specific"},
            "policy_controls": {"normalization": "checkpoint_robot_postprocessor", "supports_pi0_variant": True},
        }


class MMEVLAOperator(EmbodiedActionOperator):
    """Operator for MME-VLA RoboMME memory-augmented policies."""

    MODEL_ID = "mme-vla"
    POLICY_FAMILY = "mme_vla_memory_augmented_openpi_policy"
    ACTION_REPRESENTATION = "8d_robomme_joint_action_chunk"
    OBSERVATION_LAYOUT = "robomme_rgb_wrist_state_optional_memory_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "state": True,
        "actions": ["actions", "action_chunk"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "prompt", "instruction", "task_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {"prompt": instruction, "task_instruction": instruction, "prompt_channels": {"prompt": instruction}}

    def process_perception(
        self, images: Any = None, video: Any = None, ref_image_path: str | Path | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        del video
        image_map = images if isinstance(images, Mapping) else {}
        base_image = _first_present(kwargs, "observation/image", "image", "base_image")
        if base_image is None:
            base_image = image_map.get("observation/image", images)
        wrist_image = _first_present(kwargs, "observation/wrist_image", "wrist_image")
        if wrist_image is None:
            wrist_image = image_map.get("observation/wrist_image")
        observation = _compact_dict(
            {
                "observation/image": base_image,
                "observation/wrist_image": wrist_image,
                "observation/state": _first_present(kwargs, "observation/state", "state", "robot_state", "proprio"),
                "prompt": _first_present(kwargs, "prompt", "instruction", "task_instruction"),
                "static_image_emb": kwargs.get("static_image_emb"),
                "static_pos_emb": kwargs.get("static_pos_emb"),
                "static_state_emb": kwargs.get("static_state_emb"),
                "static_mask": kwargs.get("static_mask"),
                "recur_image_emb": kwargs.get("recur_image_emb"),
                "recur_pos_emb": kwargs.get("recur_pos_emb"),
                "recur_state_emb": kwargs.get("recur_state_emb"),
                "recur_mask": kwargs.get("recur_mask"),
                "simple_subgoal": kwargs.get("simple_subgoal"),
                "grounded_subgoal": kwargs.get("grounded_subgoal"),
                "history_observations": kwargs.get("history_observations"),
            }
        )
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {**kwargs, "mme_vla_observation": observation},
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "action_chunk")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions, action_contract="8d_robomme_joint_action_chunk", control_mode="robomme_policy_action"
            ),
            "action_space": {"kind": "continuous_chunk", "dimensions": 8},
            "policy_controls": {"uses_memory": True, "backend": "openpi_jax_policy"},
        }


__all__ = [
    "CogACTOperator",
    "DBCogACTOperator",
    "MMEVLAOperator",
    "MolmoBotOperator",
    "OpenVLAOFTOperator",
    "VLANeXtOperator",
]
