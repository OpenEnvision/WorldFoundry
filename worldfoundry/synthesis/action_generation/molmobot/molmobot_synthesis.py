from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis
from worldfoundry.synthesis.action_generation.runtime_config import (
    load_vla_va_wam_runtime_config,
    variant_defaults,
)


class MolmoBotSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    MODEL_ID = "molmobot"

    def _runtime_config(self, options: Mapping[str, Any]):
        explicit = {**self.runtime_options, **dict(options)}
        selected = explicit.get("variant") or explicit.get("variant_id") or explicit.get("runtime_variant")
        defaults = load_vla_va_wam_runtime_config(self.model_id, explicit.get("runtime_config_path"))
        variant = variant_defaults(defaults, selected)
        if variant:
            merged = {**variant, **explicit}
            # The shared policy builder gives the root checkpoint_ref precedence
            # over a variant's repo_id, so carry the selected provenance forward.
            if "checkpoint_ref" not in explicit and variant.get("repo_id"):
                merged["checkpoint_ref"] = variant["repo_id"]
            return super()._runtime_config(merged)
        return super()._runtime_config(explicit)
