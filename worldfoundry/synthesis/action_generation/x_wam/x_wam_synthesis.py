"""WorldFoundry synthesis adapter for X-WAM world-action policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis
from worldfoundry.synthesis.action_generation.official_policy.runtime import (
    OfficialPolicyRuntimeConfig,
    build_runtime_config,
)
from worldfoundry.synthesis.action_generation.runtime_config import (
    load_vla_va_wam_runtime_config,
    variant_defaults,
)


class XWAMSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed adapter for the in-tree X-WAM callable runtime."""

    MODEL_ID = "x-wam"

    def _runtime_config(self, options: Mapping[str, Any]) -> OfficialPolicyRuntimeConfig:
        requested = {**self.runtime_options, **dict(options)}
        data_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            options.get("runtime_config_path") or self.runtime_options.get("runtime_config_path"),
        )
        selected_variant = options.get("variant") or self.runtime_options.get("variant") or data_defaults.get("variant")
        merged = {
            **data_defaults,
            **variant_defaults(data_defaults, selected_variant),
            **requested,
        }
        has_explicit_path = any(
            requested.get(key) not in (None, "") for key in ("checkpoint_path", "checkpoint_dir", "ckpt_path")
        )
        has_explicit_ref = any(
            requested.get(key) not in (None, "")
            for key in ("checkpoint_ref", "checkpoint_repo_id", "repo_id", "hf_repo_id")
        )
        if has_explicit_ref and not has_explicit_path:
            merged.pop("checkpoint_path", None)
            # ``variant_defaults`` exposes the policy's nested subtree as
            # checkpoint_dir metadata; it is not an alternate checkpoint root.
            merged.pop("checkpoint_dir", None)
            merged.pop("ckpt_path", None)
        has_explicit_base_path = requested.get("base_model_path") not in (None, "")
        has_explicit_base_ref = any(
            requested.get(key) not in (None, "") for key in ("base_model_ref", "base_model_repo_id")
        )
        if has_explicit_base_ref and not has_explicit_base_path:
            merged.pop("base_model_path", None)
        return build_runtime_config(
            model_id=self.model_id,
            profile_checkpoints=() if has_explicit_ref and not has_explicit_path else self.profile.checkpoints,
            defaults={},
            options=merged,
            device=self.device,
        )


__all__ = ["XWAMSynthesis"]
