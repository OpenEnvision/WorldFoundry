"""WorldFoundry synthesis facade for X-VLA."""

from typing import Any, Mapping

from worldfoundry.core.model_loading.checkpoints import select_profile_checkpoint, selected_checkpoint_options
from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis
from worldfoundry.synthesis.action_generation.official_policy.runtime import (
    OfficialPolicyRuntimeConfig,
    build_runtime_config,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config


class XVLASynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct X-VLA action generation."""

    MODEL_ID = "xvla"
    CHECKPOINT_ALIASES = {
        "widowx": "widowx_checkpoint",
        "bridge": "widowx_checkpoint",
        "google": "google_robot_checkpoint",
        "google-robot": "google_robot_checkpoint",
        "fractal": "google_robot_checkpoint",
        "libero": "libero_checkpoint",
        "calvin": "calvin_checkpoint",
        "calvin-abc-d": "calvin_checkpoint",
        "robotwin": "robotwin2_checkpoint",
        "robotwin2": "robotwin2_checkpoint",
        "vlabench": "vlabench_checkpoint",
        "agiworld": "agiworld_checkpoint",
        "agiworld-challenge": "agiworld_checkpoint",
        "softfold": "softfold_checkpoint",
        "pt": "foundation_checkpoint",
        "foundation": "foundation_checkpoint",
        "pretrained": "foundation_checkpoint",
    }

    def _runtime_config(self, options: Mapping[str, Any]) -> OfficialPolicyRuntimeConfig:
        requested = {**self.runtime_options, **dict(options)}
        selector = requested.get("checkpoint_variant") or requested.get("variant_id") or requested.get("variant")
        has_explicit_path = any(
            requested.get(key) not in (None, "")
            for key in ("checkpoint_path", "checkpoint_dir", "ckpt_path")
        )
        has_explicit_ref = any(
            requested.get(key) not in (None, "")
            for key in ("checkpoint_ref", "checkpoint_repo_id", "repo_id", "hf_repo_id")
        )
        selected = None
        if selector not in (None, ""):
            selected = select_profile_checkpoint(
                self.profile.checkpoints,
                selector,
                aliases=self.CHECKPOINT_ALIASES,
            )
        if selected is not None and not has_explicit_path and not has_explicit_ref:
            requested.update(selected_checkpoint_options(selected))
        explicit_record = None
        if has_explicit_path or has_explicit_ref:
            explicit_value = next(
                (
                    requested[key]
                    for key in (
                        "checkpoint_path",
                        "checkpoint_dir",
                        "ckpt_path",
                        "checkpoint_ref",
                        "checkpoint_repo_id",
                        "repo_id",
                        "hf_repo_id",
                    )
                    if requested.get(key) not in (None, "")
                ),
                None,
            )
            for candidate in (explicit_value, str(explicit_value).rstrip("/").rsplit("/", 1)[-1]):
                try:
                    explicit_record = select_profile_checkpoint(self.profile.checkpoints, candidate)
                    break
                except ValueError:
                    pass
        if selected is not None and explicit_record is not None:
            if selected.get("repo_id") != explicit_record.get("repo_id"):
                raise ValueError(
                    f"X-VLA variant {selector!r} conflicts with explicit checkpoint "
                    f"{explicit_record.get('repo_id')!r}"
                )
        domain_record = selected or explicit_record
        if requested.get("domain_id") in (None, "") and (
            domain_record is not None or has_explicit_path or has_explicit_ref
        ):
            selected_domain = domain_record.get("domain_id") if domain_record is not None else None
            if selected_domain is None:
                raise ValueError(
                    "an X-VLA foundation or custom checkpoint requires an explicit domain_id for the target embodiment"
                )
            requested["domain_id"] = int(selected_domain)
        defaults = dict(
            load_vla_va_wam_runtime_config(
                self.model_id,
                requested.get("runtime_config_path"),
            )
        )
        if has_explicit_ref and not has_explicit_path:
            defaults.pop("checkpoint_path", None)
        return build_runtime_config(
            model_id=self.model_id,
            profile_checkpoints=() if has_explicit_ref and not has_explicit_path else self.profile.checkpoints,
            defaults=defaults,
            options=requested,
            device=self.device,
        )

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Accept Workspace's flat action fields as a policy observation."""

        selected = OfficialPolicySynthesis._select_observation(kwargs)
        observation = dict(selected or {})
        for key in (
            "state",
            "proprio",
            "robot_state",
            "eef_state",
            "image0",
            "image1",
            "image2",
            "full_image",
            "wrist_image",
            "third_person_image",
        ):
            if kwargs.get(key) is not None and key not in observation:
                observation[key] = kwargs[key]
        return observation or None


__all__ = ["XVLASynthesis"]
