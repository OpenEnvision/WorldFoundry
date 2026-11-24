"""Model-specific operator contract for X-VLA."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from worldfoundry.operators.embodied_action_operator import (
    _compact_dict,
    _extract_action_signal,
    _first_present,
)
from worldfoundry.operators.official_policy_operator import OfficialPolicyOperator


class XVLAOperator(OfficialPolicyOperator):
    """Normalize X-VLA multi-view, language, proprio, and domain inputs."""

    MODEL_ID = "xvla"
    POLICY_FAMILY = "xvla_cross_embodiment_policy"
    ACTION_REPRESENTATION = "30_step_xvla_action_chunk"
    OBSERVATION_LAYOUT = "up_to_three_rgb_views_language_proprio_domain"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "model_id": "xvla",
        "prompt": True,
        "image": True,
        "video": False,
        "state": True,
        "camera_keys": ["image0", "image1", "image2"],
        "domain_id": True,
        "denoising_steps": True,
        "policy_family": POLICY_FAMILY,
        "action_representation": ACTION_REPRESENTATION,
        "action_contract": "xvla_model_declared_action_chunk",
        "action_horizon": 30,
        "runtime_backend": "xvla_in_tree_processor_model",
    }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video
        camera_keys = list(kwargs.get("camera_keys") or self.input_schema["camera_keys"])
        if isinstance(images, Mapping):
            views = {key: images[key] for key in camera_keys if images.get(key) is not None}
        else:
            values = list(images) if isinstance(images, (list, tuple)) else [images]
            views = {
                key: values[index]
                for index, key in enumerate(camera_keys)
                if index < len(values) and values[index] is not None
            }
        for key in camera_keys:
            if kwargs.get(key) is not None:
                views[key] = kwargs[key]
        existing = kwargs.get("observation")
        if not isinstance(existing, Mapping):
            existing = {}
        observation = _compact_dict(
            {
                **dict(existing),
                **views,
                "state": _first_present(kwargs, "state", "proprio", "robot_state", "eef_state"),
                "domain_id": kwargs.get("domain_id", 0),
                "denoising_steps": kwargs.get("denoising_steps", kwargs.get("steps", 10)),
            }
        )
        selected_images: Any = views or images
        return {
            "images": selected_images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "official_policy_observation": observation,
                "official_policy_input_contract": {
                    "modalities": ["multi_view_rgb", "language", "proprio", "domain_id"],
                    "backend": "xvla_in_tree_processor_model",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "actions", "action_chunk", "robot_action", "trajectory")
        self.interaction_history.append(actions)
        metadata = raw.get("metadata", {}) if isinstance(raw, Mapping) else {}
        return {
            **self.action_payload(
                actions,
                action_contract="xvla_model_declared_action_chunk",
                control_mode="xvla_flow_denoised_action",
                action_horizon=30,
            ),
            "action_space": {
                "kind": "continuous_chunk",
                "mode": metadata.get("action_mode", "checkpoint_declared"),
            },
            "policy_controls": {
                "domain_id": metadata.get("domain_id"),
                "denoising_steps": metadata.get("denoising_steps"),
                "normalization": "checkpoint_action_space",
            },
        }


__all__ = ["XVLAOperator"]
