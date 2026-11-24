"""Implements action synthesis using Diffusion Policy models.

This module provides the `DiffusionPolicySynthesis` class, which extends `ActionModelSynthesis`
to integrate Diffusion Policy models for generating actions based on observations. It handles
loading model profiles, configuring the Diffusion Policy runtime, and executing predictions.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.diffusion_policy.runtime import (
    DiffusionPolicyRuntime,
    DiffusionPolicyRuntimeConfig,
    select_diffusion_policy_checkpoint,
)
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class DiffusionPolicySynthesis(ActionModelSynthesis):
    """Synthesizes actions using a Diffusion Policy model.

    This class extends `ActionModelSynthesis` to provide a standardized interface
    for interacting with Diffusion Policy models. It manages model loading,
    runtime configuration, and action prediction, encapsulating the specifics
    of the Diffusion Policy backend.
    """

    MODEL_ID = "diffusion-policy"  # Identifier for the Diffusion Policy model.

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the DiffusionPolicySynthesis instance.

        Args:
            profile: The runtime profile containing model configuration and metadata.
            device: The device to run the model on (e.g., "cuda", "cpu").
            command_template: An optional sequence of strings forming a command template
                for external processes (not directly used by in-process runtime).
            env: An optional mapping of environment variables for external processes.
            runtime_options: Additional options specific to the Diffusion Policy runtime.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        self._runtime: DiffusionPolicyRuntime | None = None
        self._runtime_key: tuple[str, str] | None = None

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
    ) -> DiffusionPolicySynthesis:
        """Creates a DiffusionPolicySynthesis instance from a pretrained model.

        This class method streamlines the process of loading a model by resolving
        its path and configuration from various sources.

        Args:
            pretrained_model_path: Path to the pretrained model checkpoint or a mapping
                of options including 'checkpoint_path'.
            args: Legacy argument, will be ignored.
            device: The device to run the model on.
            model_id: Explicit model identifier.
            profile_path: Path to the runtime profile JSON file.
            manifest_path: Path to the manifest file for model discovery.
            acquisition_root: Root directory for acquiring models.
            hf_models_root: Root directory for Hugging Face models.
            command_template: Command template for external processes.
            **kwargs: Additional options to pass to the synthesis constructor or
                runtime profile loader.

        Returns:
            An initialized DiffusionPolicySynthesis instance.
        """
        del args
        # Initialize options from pretrained_model_path, handling both dict and string forms.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)

        # Resolve the model ID from multiple possible sources, falling back to class default.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and other paths.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(
        self,
        options: Mapping[str, Any],
        *,
        require_checkpoint: bool = True,
    ) -> DiffusionPolicyRuntimeConfig:
        """Constructs a `DiffusionPolicyRuntimeConfig` from provided options and instance settings.

        Args:
            options: A mapping of options specific to the current prediction call.

        Returns:
            A DiffusionPolicyRuntimeConfig instance.
        """
        # Resolve the checkpoint path by checking options, instance runtime_options, and profile.
        checkpoint_path = (
            options.get("checkpoint_path")
            or options.get("ckpt_path")
            or options.get("pretrained_model_path")
            or self.runtime_options.get("checkpoint_path")
            or self.runtime_options.get("ckpt_path")
        )
        checkpoint = select_diffusion_policy_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoints=self.profile.checkpoints,
            require_exists=require_checkpoint,
        )
        return DiffusionPolicyRuntimeConfig(
            checkpoint_path=checkpoint,
            device=str(options.get("device") or self.runtime_options.get("device") or self.device),
        )

    def _runtime_for(self, config: DiffusionPolicyRuntimeConfig) -> DiffusionPolicyRuntime:
        """Retrieves or creates a `DiffusionPolicyRuntime` instance for a given configuration.

        This method caches the runtime instance, only creating a new one if the configuration
        has changed, to optimize resource usage.

        Args:
            config: The configuration for the Diffusion Policy runtime.

        Returns:
            A DiffusionPolicyRuntime instance.
        """
        # Create a unique key for the runtime based on critical configuration parameters.
        key = (
            str(config.checkpoint_path),
            config.device,
        )
        # If the cached runtime is None or the key does not match, create a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = DiffusionPolicyRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Any:
        """Selects the observation data from various possible keyword arguments.

        This method provides flexibility by checking for common observation keys
        in a specific order.

        Args:
            kwargs: A mapping of keyword arguments that may contain observation data.

        Returns:
            The selected observation data, or None if not found.
        """
        # Prioritize 'observation', then 'diffusion_policy_observation', then 'obs'.
        observation = kwargs.get("observation")
        if observation is not None:
            return observation
        policy_observation = kwargs.get("diffusion_policy_observation")
        if policy_observation is not None:
            return policy_observation
        return kwargs.get("obs")

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
        """Generates action predictions using the Diffusion Policy model.

        Args:
            prompt: A textual prompt or instruction for the policy.
            images: Image data for observation.
            video: Video data for observation.
            interactions: A sequence of interaction descriptions.
            output_path: The desired path to save any generated artifacts.
            fps: Frames per second for video processing, if applicable.
            timeout_seconds: Maximum time allowed for the prediction.
            **kwargs: Additional keyword arguments for the runtime or observation.

        Returns:
            A dictionary containing the prediction results, including status, model_id,
            artifact paths, and profile details.
        """
        del timeout_seconds

        # Determine if only a plan should be generated, and set up a temporary run directory.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="diffusion_policy_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create a context dictionary for the prediction, including all relevant inputs.
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

        # Prepare runtime options for inference.
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        runtime_config = self._runtime_config(
            runtime_options,
            require_checkpoint=not plan_only,
        )

        # Construct and save a plan file detailing the intended execution.
        plan_path = run_dir / "runtime_profile_plan.json"
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-diffusion-policy-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.diffusion_policy.in_tree_runtime.predict_action",
                "checkpoint_path": str(runtime_config.checkpoint_path),
                "device": runtime_config.device,
            },
        }
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If only planning is requested, return the plan details without executing the model.
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

        # Select the actual observation data from the provided kwargs.
        observation = self._select_observation(kwargs)

        # Execute the action prediction using the configured Diffusion Policy runtime.
        result = self._runtime_for(runtime_config).predict_action(
            observation=observation,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "instruction": prompt,
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                },
            },
        )

        # Augment the prediction result with run directory, plan path, and profile information.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
