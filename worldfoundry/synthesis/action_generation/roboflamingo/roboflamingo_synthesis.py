"""RoboFlamingo action synthesis module.

This module provides a wrapper for the RoboFlamingo model, allowing it to be used
within the worldfoundry action synthesis framework. It handles model loading,
configuration, and interaction with the RoboFlamingo runtime for generating
actions based on visual and textual prompts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from ..base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.roboflamingo.runtime import (
    RoboFlamingoRuntime,
    RoboFlamingoRuntimeConfig,
    select_roboflamingo_runtime_config,
)


def _first_observation_value(*values: Any) -> Any:
    """Select an input without evaluating a NumPy array as a boolean."""
    for value in values:
        if value is None or (isinstance(value, str) and not value):
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        return value
    return None


class RoboFlamingoSynthesis(ActionModelSynthesis):
    """Action synthesis wrapper for the RoboFlamingo model.

    This class extends `ActionModelSynthesis` to provide an interface for
    generating actions using the RoboFlamingo model. It manages the lifecycle
    of the RoboFlamingo runtime, including lazy loading, configuration
    resolution, and interaction with the underlying inference engine.
    """

    MODEL_ID = "roboflamingo"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a lazy RoboFlamingo synthesis wrapper.

        Args:
            profile: Runtime profile with source and checkpoint metadata.
            device: Device string used by the in-tree runtime.
            command_template: Optional base-class command template.
            env: Optional runtime environment overrides.
            runtime_options: Optional checkpoint and CALVIN runtime options.
        """

        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store additional runtime options specific to RoboFlamingo.
        self.runtime_options = dict(runtime_options or {})
        # Initialize runtime and key to None for lazy loading.
        self._runtime: RoboFlamingoRuntime | None = None
        self._runtime_key: tuple[str, str, str, str, str, str, str] | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "RoboFlamingoSynthesis":
        """Create RoboFlamingo from profile metadata without loading torch.

        This factory method simplifies instantiation by resolving model paths
        and configuration from a profile.

        Args:
            pretrained_model_path: Optional policy checkpoint path or option mapping.
            args: Unused compatibility parameter.
            device: Runtime device selector.
            model_id: Optional runtime profile id.
            profile_path: Optional runtime profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional acquisition cache root.
            hf_models_root: Optional Hugging Face cache root.
            command_template: Optional base-class command template.
            **kwargs: Additional runtime options.
        """

        # Unused parameter, explicitly deleted for clarity.
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as the policy checkpoint path.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["policy_checkpoint_path"] = str(pretrained_model_path)
        # Merge additional keyword arguments into the options.
        options.update(kwargs)
        # Resolve the model ID using a priority order: options, model_id, or default class MODEL_ID.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile using the resolved model ID and various path overrides.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        # Instantiate the class with the resolved profile and options.
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any], *, require_existing: bool) -> RoboFlamingoRuntimeConfig:
        """Resolve RoboFlamingo runtime paths and architecture options.

        Args:
            options: Per-call options that override constructor runtime options.
            require_existing: Whether selected checkpoint paths must exist now.
        """

        # Merge constructor runtime options with per-call options, with per-call having precedence.
        merged = {**self.runtime_options, **dict(options)}
        # Ensure a device is set, defaulting to the instance's device if not present in merged options.
        merged.setdefault("device", self.device)
        # Select the appropriate RoboFlamingo runtime configuration based on checkpoints and merged options.
        return select_roboflamingo_runtime_config(
            checkpoints=self.profile.checkpoints,
            options=merged,
            require_existing=require_existing,
        )

    def _runtime_for(self, config: RoboFlamingoRuntimeConfig) -> RoboFlamingoRuntime:
        """Return a cached RoboFlamingo runtime for one resolved config.

        Lazily initializes the runtime and caches it based on the configuration key.

        Args:
            config: Resolved runtime configuration.
        """

        # Create a unique key from essential configuration paths and parameters
        # to determine if a new runtime instance is needed.
        key = (
            str(config.policy_checkpoint_path or ""),
            str(config.openflamingo_checkpoint_path or ""),
            str(config.lang_encoder_path or ""),
            str(config.tokenizer_path or ""),
            config.device,
            config.precision,
            config.architecture.signature,
        )
        # If the runtime is not initialized or the configuration key has changed, create a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = RoboFlamingoRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_observation(images: Any, video: Any, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build the RoboFlamingo observation mapping from pipeline inputs.

        This method normalizes various input formats into the structured
        observation dictionary expected by the RoboFlamingo runtime.

        Args:
            images: Direct image input.
            video: Optional visual-history input.
            kwargs: Operator and runtime extras.
        """

        # Prioritize an explicit 'roboflamingo_observation' or 'observation' mapping if provided.
        observation = kwargs.get("roboflamingo_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping):
            return dict(observation)
        # Otherwise, construct the observation dictionary by mapping various input keys
        # to RoboFlamingo's expected observation keys, with fallback logic.
        return {
            "visual_history": _first_observation_value(video, kwargs.get("visual_history"), images),
            "current_image": _first_observation_value(images, kwargs.get("image"), kwargs.get("image_path"), kwargs.get("ref_image_path")),
            "gripper_image": _first_observation_value(kwargs.get("gripper_image"), kwargs.get("wrist_image"), kwargs.get("gripper")),
            "proprio": _first_observation_value(kwargs.get("proprio"), kwargs.get("robot_state")),
            "gripper_state": kwargs.get("gripper_state"),
        }

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare or execute the in-tree RoboFlamingo action runtime.

        This method generates an action plan or executes the RoboFlamingo model
        to produce an action trace, based on the provided prompt and observations.

        Args:
            prompt: Natural-language task instruction.
            images: RGB observation image or image path.
            video: Optional visual history.
            interactions: Operator-provided action context.
            output_path: Optional action_trace output path.
            fps: Unused compatibility frame rate.
            timeout_seconds: Unused compatibility timeout.
            **kwargs: Additional runtime and planning options.
        """

        # Unused parameter, explicitly deleted for clarity.
        del timeout_seconds
        # Determine if only a plan should be generated, checking call-specific then constructor options.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Resolve or create a temporary run directory for artifacts.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="roboflamingo_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Prepare the context dictionary, including run directory and other parameters.
        context = self._context(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )
        runtime_options = dict(kwargs)
        # Resolve the runtime configuration, without requiring checkpoints to exist yet if plan_only.
        runtime_config = self._runtime_config(runtime_options, require_existing=False)
        # Build the structured observation dictionary from various inputs.
        observation = self._select_observation(images, video, kwargs)
        # Define the path for the plan file within the run directory.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the plan payload, which captures all necessary information
        # to recreate or understand the planned execution.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-roboflamingo-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": runtime_config.to_plan_dict(),
            "observation_contract": {
                "instruction": prompt,
                "observation_keys": sorted(key for key, value in observation.items() if value is not None),
                "action_space": kwargs.get("action_space") or {"kind": "continuous", "dimensions": 7},
                "policy_controls": kwargs.get("policy_controls"),
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # If only a plan is requested, return the planning artifacts and status.
        if plan_only:
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # For actual execution, resolve runtime configuration, now requiring checkpoints to exist.
        runtime_config = self._runtime_config(runtime_options, require_existing=True)
        # Get or initialize the RoboFlamingo runtime and call its predict_action method.
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            observation=observation,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    "observation_keys": sorted(key for key, value in observation.items() if value is not None),
                },
            },
        )
        # Return the prediction result augmented with run directory, plan path, and profile information.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
