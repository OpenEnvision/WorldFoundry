"""Module for synthesizing actions using the Octo model."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class OctoSynthesis(ActionModelSynthesis):
    """
    Implements the ActionModelSynthesis interface for the Octo model.

    This class provides methods to load, configure, and predict actions using the
    Octo visual language agent (VLA) model, handling its specific runtime
    configurations and inference process.
    """

    MODEL_ID = "octo"

    def __init__(
        self,
        profile,
        *,
        device: str = "cpu",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initializes an OctoSynthesis instance.

        Args:
            profile: The runtime profile describing the model.
            device: The device to run the model on (e.g., "cpu", "cuda"). Defaults to "cpu".
            command_template: An optional template for the command to execute the model.
            env: Optional environment variables to set for the model process.
            runtime_options: Additional options specific to the Octo runtime.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store Octo-specific runtime options, merging with any existing ones.
        self.runtime_options = dict(runtime_options or {})
        # Private attributes for lazy initialization and caching of the Octo runtime.
        self._runtime: Any | None = None
        self._runtime_key: tuple[str, str, str, str, tuple[int, int], str, int] | None = None

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
    ) -> OctoSynthesis:
        """
        Creates an OctoSynthesis instance from a pre-trained model.

        This class method handles various ways of specifying the model and its
        configuration, including a path, a dictionary of options, or individual arguments.

        Args:
            pretrained_model_path: Path to the pre-trained model checkpoint, or a mapping
                                   of options including 'checkpoint_dir'.
            args: Legacy argument, will be ignored.
            device: The device to run the model on.
            model_id: Explicit model ID to use.
            profile_path: Path to the runtime profile.
            manifest_path: Path to the manifest file.
            acquisition_root: Root directory for acquiring models.
            hf_models_root: Root directory for Hugging Face models.
            command_template: An optional template for the command to execute the model.
            **kwargs: Additional options that will be passed to the runtime configuration.

        Returns:
            An initialized OctoSynthesis instance.
        """
        # The 'args' parameter is deprecated and not used; it's explicitly deleted.
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided but not a mapping, treat it as the checkpoint directory.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options.
        options.update(kwargs)
        # Determine the model ID, prioritizing from options, then 'model_id' arg, falling back to class default.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile based on the resolved model ID and other paths.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cpu"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any], *, require_checkpoint: bool = True):
        """
        Constructs an OctoRuntimeConfig object from various sources.

        Merges instance-level options, provided options, and default runtime
        configurations, then resolves the appropriate model checkpoint.

        Args:
            options: A mapping of runtime options specific to this prediction call.
            require_checkpoint: If True, raises an error if no checkpoint can be found.

        Returns:
            An instance of OctoRuntimeConfig.

        Raises:
            ValueError: If `image_size` is not a tuple of two integers.
        """
        from .runtime import OctoRuntimeConfig, select_octo_checkpoint

        # Combine instance-level runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configuration for Octo.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options, explicit options take precedence.
        merged = {**runtime_defaults, **explicit_options}
        variant = str(merged["variant"])
        # Determine checkpoint directory, checking multiple possible keys in order of precedence.
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        # Select the specific Octo checkpoint based on directory, profile, and variant.
        checkpoint = select_octo_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
            variant=variant,
            require_exists=require_checkpoint,
        )
        # Parse and validate image_size from the merged options.
        image_size_value = merged["image_size"]
        image_size = tuple(int(item) for item in image_size_value)
        if len(image_size) != 2:
            raise ValueError("Octo image_size must contain width and height.")
        jax_platform = explicit_options.get("jax_platform")
        if jax_platform is None:
            jax_platform = "cuda" if str(self.device).lower().startswith("cuda") else merged["jax_platform"]
        return OctoRuntimeConfig(
            checkpoint_dir=checkpoint,
            variant=variant,
            dataset_key=str(merged["dataset_key"]),
            image_key=str(merged["image_key"]),
            image_size=(image_size[0], image_size[1]),
            jax_platform=str(jax_platform),
            seed=int(merged["seed"]),
        )

    def _runtime_for(self, config):
        """
        Lazily initializes and caches the OctoRuntime instance based on the provided config.

        Ensures that only one runtime instance is created for a given unique configuration.

        Args:
            config: An instance of OctoRuntimeConfig.

        Returns:
            An initialized OctoRuntime instance.
        """
        from .runtime import OctoRuntime

        # Create a unique key from the configuration parameters for caching.
        key = (
            str(config.checkpoint_dir),
            config.variant,
            config.dataset_key,
            config.image_key,
            config.image_size,
            config.jax_platform,
            config.seed,
        )
        # If the runtime is not initialized or the configuration key has changed, re-initialize.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = OctoRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """
        Selects the primary image input from various possible arguments and observation structures.

        Args:
            images: Direct image input.
            kwargs: Additional keyword arguments, potentially containing image or observation data.

        Returns:
            The selected image, image path, or None if no image can be found.
        """
        # Prioritize 'images' parameter directly.
        if images is not None:
            return images
        # Check for 'image' in kwargs.
        if kwargs.get("image") is not None:
            return kwargs["image"]
        # Check for 'image_path' in kwargs.
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Look into 'octo_observation' or generic 'observation' for specific keys.
        observation = kwargs.get("octo_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping):
            # Prioritize 'image_primary' within observation.
            if observation.get("image_primary") is not None:
                return observation["image_primary"]
            # Fallback to 'rgb' within observation.
            if observation.get("rgb") is not None:
                return observation["rgb"]
        # Final fallback to 'ref_image_path' in kwargs.
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
        Generates actions using the Octo model based on a prompt and visual inputs.

        This method can either execute the prediction directly or generate a plan
        for later execution, depending on the `plan_only` option.

        Args:
            prompt: The instruction or prompt for the action synthesis.
            images: Primary image input for the model.
            video: Video input for the model (not typically used by Octo directly but passed to context).
            interactions: A sequence of past interactions.
            output_path: Desired path for the output artifacts.
            fps: Frames per second, if video input is provided.
            timeout_seconds: Maximum time allowed for prediction (currently unused by Octo).
            **kwargs: Additional options, including 'plan_only', 'run_dir', and Octo runtime parameters.

        Returns:
            A dictionary containing the prediction results or the generated plan details.
        """
        # The 'timeout_seconds' parameter is not currently used by the Octo runtime.
        del timeout_seconds

        # Determine if the call is for planning only, checking both call-specific and instance-level options.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Set up a dedicated temporary directory for the current prediction run.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="octo_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create a context dictionary containing all relevant input parameters.
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
        # Prepare runtime options, prioritizing those passed directly to predict.
        runtime_options = dict(kwargs)
        # Obtain the Octo runtime configuration, conditionally requiring a checkpoint.
        runtime_config = self._runtime_config(
            runtime_options,
            require_checkpoint=not plan_only,  # Checkpoint not strictly needed if only planning
        )
        # Define the path for saving the runtime plan.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-octo-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.octo.in_tree_runtime.sample_actions",
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "dataset_key": runtime_config.dataset_key,
                "image_key": runtime_config.image_key,
                "image_size": list(runtime_config.image_size),
                "jax_platform": runtime_config.jax_platform,
                "seed": runtime_config.seed,
                "variant": runtime_config.variant,
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If 'plan_only' is True, return the plan details without executing the model.
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

        # Select the actual image input from available sources.
        image = self._select_image(images, kwargs)
        # Fallback to image_path from context if no direct image was selected.
        if image is None and context.get("image_path"):
            image = context["image_path"]
        # Execute the action prediction using the Octo runtime.
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            image=image,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {  # Contextual information for the operator handling the action.
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    "observation_keys": sorted((kwargs.get("octo_observation") or {}).keys())
                    if isinstance(kwargs.get("octo_observation"), Mapping)
                    else [],
                },
            },
        )
        # Return the prediction result augmented with run details and profile.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
