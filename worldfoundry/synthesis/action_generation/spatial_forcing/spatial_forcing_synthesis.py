"""WorldFoundry synthesis adapter for Spatial-Forcing action policies."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.core.model_loading.checkpoints import select_profile_checkpoint, selected_checkpoint_options
from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis
from worldfoundry.synthesis.action_generation.official_policy.runtime import (
    OfficialPolicyRuntimeConfig,
    build_runtime_config,
)


class SpatialForcingSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed adapter for the in-tree Spatial-Forcing callable runtime."""

    MODEL_ID = "spatial-forcing"
    OBSERVATION_KEYS = (
        "full_image",
        "image",
        "rgb",
        "ref_image_path",
        "wrist_image",
        "full_image_wrist",
        "image_wrist",
        "wrist",
        "state",
        "proprio",
        "robot_state",
    )
    DEFAULT_RUNTIME_OPTIONS = {
        "backend": "callable_entrypoint",
        "policy_target": "worldfoundry.synthesis.action_generation.spatial_forcing.runtime:predict_action",
        "predict_method": "predict_action",
        "require_checkpoint": True,
        "trust_remote_code": False,
        "torch_dtype": "auto",
        "attn_implementation": "auto",
        "num_images_in_input": 2,
        "center_crop": True,
        "local_files_only": True,
    }
    CHECKPOINT_ALIASES = {
        "spatial": "libero_spatial_checkpoint",
        "libero-spatial": "libero_spatial_checkpoint",
        "object": "libero_object_checkpoint",
        "libero-object": "libero_object_checkpoint",
        "goal": "libero_goal_checkpoint",
        "libero-goal": "libero_goal_checkpoint",
        "10": "libero_10_checkpoint",
        "libero-10": "libero_10_checkpoint",
    }

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Build the observation accepted by the runtime from Workspace flat inputs."""

        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key in SpatialForcingSynthesis.OBSERVATION_KEYS:
            value = kwargs.get(key)
            if value is not None and key not in observation:
                observation[key] = value
        return observation or None

    def _runtime_config(self, options: Mapping[str, Any]) -> OfficialPolicyRuntimeConfig:
        requested = {**self.runtime_options, **dict(options)}
        merged = {
            **self.DEFAULT_RUNTIME_OPTIONS,
            **requested,
        }
        selector = requested.get("checkpoint_variant") or requested.get("variant_id") or requested.get("variant")
        has_explicit_path = any(
            requested.get(key) not in (None, "")
            for key in ("checkpoint_path", "checkpoint_dir", "ckpt_path")
        )
        has_explicit_ref = any(
            requested.get(key) not in (None, "")
            for key in ("checkpoint_ref", "checkpoint_repo_id", "repo_id", "hf_repo_id")
        )
        if selector not in (None, "") and not has_explicit_path and not has_explicit_ref:
            selected = select_profile_checkpoint(
                self.profile.checkpoints,
                selector,
                aliases=self.CHECKPOINT_ALIASES,
            )
            merged.update(selected_checkpoint_options(selected))
        return build_runtime_config(
            model_id=self.model_id,
            profile_checkpoints=() if has_explicit_ref and not has_explicit_path else self.profile.checkpoints,
            defaults={},
            options=merged,
            device=self.device,
        )


__all__ = ["SpatialForcingSynthesis"]
