"""
Module for OpenVLA (Open-Vocabulary Language Agent) action model synthesis.

This module provides the `OpenVLASynthesis` class, which extends `ActionModelSynthesis`
to integrate and manage the OpenVLA runtime for generating actions based on
visual and textual inputs. It handles model loading, configuration, and execution
of prediction tasks, supporting both planning and direct action generation.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.openvla.engine import (
    OpenVLARuntime,
    OpenVLARuntimeConfig,
    select_openvla_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class OpenVLASynthesis(ActionModelSynthesis):
    """
    Synthesizes actions using the OpenVLA (Open-Vocabulary Language Agent) model.

    This class orchestrates the loading, configuration, and execution of an OpenVLA
    model to generate actions based on provided prompts, images, and other context.
    It extends the `ActionModelSynthesis` base class and integrates with the
    `OpenVLARuntime` to manage the underlying model inference.
    """

    MODEL_ID = "openvla"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initializes the OpenVLASynthesis instance.

        Args:
            profile: The runtime profile containing model configuration and metadata.
            device: The device to run the model on (e.g., "cuda", "cpu").
            command_template: A template for generating commands, if applicable.
            env: Environment variables to set for the model's runtime.
            runtime_options: Additional options specific to the OpenVLA runtime.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store runtime-specific options, defaulting to an empty dict if None
        self.runtime_options = dict(runtime_options or {})
        # Internal cache for the OpenVLARuntime instance
        self._runtime: OpenVLARuntime | None = None
        # Key representing the configuration used to create the cached runtime
        self._runtime_key: tuple[str, str, str, str, str, str | bool | None] | None = None

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
    ) -> OpenVLASynthesis:
        """
        Loads a pretrained OpenVLA model.

        This class method acts as a factory to create an `OpenVLASynthesis` instance
        by resolving model paths and configurations from various input sources.

        Args:
            pretrained_model_path: Path to the pretrained model, or a mapping of options.
            args: Legacy argument, will be deleted.
            device: The device to run the model on.
            model_id: Explicit model identifier.
            profile_path: Path to a custom runtime profile.
            manifest_path: Path to the manifest file.
            acquisition_root: Root directory for model acquisition.
            hf_models_root: Root directory for Hugging Face models.
            command_template: A template for generating commands.
            **kwargs: Additional keyword arguments passed to the runtime options.

        Returns:
            An initialized `OpenVLASynthesis` instance.
        """
        # Delete unused legacy argument
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise empty.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a simple path, set it as checkpoint_dir in options.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Update options with any additional kwargs
        options.update(kwargs)

        # Determine the final model ID, prioritizing explicit options or class default
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile based on the resolved model ID and other paths
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        # Instantiate the OpenVLASynthesis class with the resolved profile and options
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(
        self, options: Mapping[str, Any], *, allow_missing_checkpoint: bool = False
    ) -> OpenVLARuntimeConfig:
        """
        Constructs an `OpenVLARuntimeConfig` object from internal and provided options.

        This method merges default runtime configurations, instance-specific options,
        and prediction-specific options to create a complete configuration for the
        OpenVLA runtime. It also handles checkpoint selection and error handling
        for missing checkpoints.

        Args:
            options: A mapping of runtime options for the current prediction.
            allow_missing_checkpoint: If True, allows a missing checkpoint and sets
                                      it to None instead of raising an error.

        Returns:
            An `OpenVLARuntimeConfig` instance.

        Raises:
            FileNotFoundError: If a checkpoint is not found and `allow_missing_checkpoint` is False.
        """
        # Combine instance's runtime_options with current call's options, prioritizing current
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configuration for the model ID
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options, explicit options override defaults
        merged = {**runtime_defaults, **explicit_options}
        unnorm_key = str(merged["unnorm_key"])

        # Determine checkpoint directory from multiple possible keys
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        try:
            # Select the appropriate OpenVLA checkpoint based on resolved directory and profile
            checkpoint = select_openvla_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoints=self.profile.checkpoints,
                unnorm_key=unnorm_key,
            )
        except FileNotFoundError:
            # If a checkpoint is not found and not allowed to be missing, re-raise the error
            if not allow_missing_checkpoint:
                raise
            # Otherwise, set checkpoint to None to proceed without it (e.g., for plan-only mode)
            checkpoint = None
        # Return a new OpenVLARuntimeConfig object with the resolved parameters
        return OpenVLARuntimeConfig(
            checkpoint_dir=checkpoint,
            unnorm_key=unnorm_key,
            device=str(merged.get("device") or self.device),
            torch_dtype=str(merged["torch_dtype"]),
            attn_implementation=str(merged["attn_implementation"]),
            use_cache=merged.get("use_cache"),
        )

    def _runtime_for(self, config: OpenVLARuntimeConfig) -> OpenVLARuntime:
        """
        Retrieves or creates an `OpenVLARuntime` instance for the given configuration.

        This method implements a caching mechanism to reuse `OpenVLARuntime` instances
        if the configuration matches the previously loaded runtime, avoiding redundant
        model loading.

        Args:
            config: The `OpenVLARuntimeConfig` specifying the desired runtime setup.

        Returns:
            An `OpenVLARuntime` instance configured as specified.
        """
        # Create a unique key from the configuration parameters for caching
        key = (
            str(config.checkpoint_dir),
            config.unnorm_key,
            config.device,
            config.torch_dtype,
            config.attn_implementation,
            config.use_cache,
        )
        # If no runtime is cached, or the cached runtime's key does not match the current config,
        # create a new OpenVLARuntime and update the cache.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = OpenVLARuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """
        Selects the primary image from various possible input sources.

        This static method provides a robust way to find an image by checking
        multiple keyword arguments and the `images` parameter in a defined
        priority order.

        Args:
            images: The direct `images` parameter.
            kwargs: Additional keyword arguments that might contain image data.

        Returns:
            The selected image data or path, or `None` if no image is found.
        """
        # Prioritize the direct 'images' parameter
        if images is not None:
            return images
        # Check for 'image' in kwargs
        if kwargs.get("image") is not None:
            return kwargs["image"]
        # Check for 'image_path' in kwargs
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Check for 'rgb' within an 'openvla_observation' mapping
        observation = kwargs.get("openvla_observation")
        if isinstance(observation, Mapping) and observation.get("rgb") is not None:
            return observation["rgb"]
        # As a last resort, check for 'ref_image_path'
        return kwargs.get("ref_image_path")

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
        """
        Generates an action prediction using the OpenVLA model.

        This method orchestrates the full prediction pipeline, including
        setting up the execution environment, configuring the OpenVLA runtime,
        generating a plan, and performing the actual action inference.
        It can operate in a "plan-only" mode to just prepare the execution plan
        without running the model.

        Args:
            prompt: The text instruction or prompt for the action.
            images: Image data to provide as visual input.
            video: Video data to provide as visual input.
            interactions: A sequence of previous interactions or context.
            output_path: The desired path for the output artifact.
            fps: Frames per second for video input, if applicable.
            timeout_seconds: Timeout for the prediction process (currently unused).
            **kwargs: Additional keyword arguments for runtime configuration or context.

        Returns:
            A dictionary containing the prediction result and metadata,
            including `run_dir`, `plan_path`, and `profile`.
        """
        # Delete unused argument to prevent linting warnings or confusion
        del timeout_seconds

        # Determine if the operation should only generate a plan without executing the model
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Create a temporary run directory for outputs and artifacts, or use a provided one
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="openvla_"))
        # Resolve and ensure the run directory exists
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the context dictionary, including prompt, images, and other inputs
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
        # Copy kwargs into runtime_options and set default device if not specified
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        # Generate the OpenVLA runtime configuration
        runtime_config = self._runtime_config(runtime_options, allow_missing_checkpoint=plan_only)

        # Define the path for the plan file within the run directory
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the plan file, including profile, context, and runtime details
        plan_payload = {
            "schema_version": "worldfoundry-openvla-runtime-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.openvla.in_tree_runtime.predict_action",
                "checkpoint_dir": str(runtime_config.checkpoint_dir) if runtime_config.checkpoint_dir else None,
                "unnorm_key": runtime_config.unnorm_key,
                "device": runtime_config.device,
                "torch_dtype": runtime_config.torch_dtype,
                "attn_implementation": runtime_config.attn_implementation,
            },
        }
        # Write the plan payload to the plan file
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If in plan-only mode, return the plan details without executing the model
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

        # Select the input image from various potential sources
        image = self._select_image(images, kwargs)
        # If no image was selected but context has an image_path, use that
        if image is None and context.get("image_path"):
            image = context["image_path"]

        # Get or create the OpenVLA runtime instance with the determined configuration
        # Perform the actual action prediction
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            image=image,
            output_path=context["output_path"],
            # Provide extra metadata for logging and debugging purposes
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    # Extract sorted keys from openvla_observation if it's a mapping
                    "observation_keys": sorted(kwargs.get("openvla_observation", {}).keys())
                    if isinstance(kwargs.get("openvla_observation"), Mapping)
                    else [],
                },
            },
        )
        # Return the prediction result augmented with run directory, plan path, and profile
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
