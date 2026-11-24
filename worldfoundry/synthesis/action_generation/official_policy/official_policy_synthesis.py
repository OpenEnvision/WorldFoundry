"""
This module provides the `OfficialPolicySynthesis` class, which acts as an adapter
for official Vision-Language-Action (VLA) or action generation runtimes.
It leverages runtime profiles to configure and execute action generation
processes, supporting both planning and actual prediction, and handling
asset management and configuration specifics.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis
from .runtime import OfficialPolicyRuntime, OfficialPolicyRuntimeConfig, build_runtime_config


def _looks_like_hub_repo_id(value: str) -> bool:
    """Distinguish an ``owner/repository`` model reference from a local path."""

    text = value.strip()
    if not text or text.startswith(("/", "./", "../", "~", "$")) or "://" in text:
        return False
    owner, separator, repository = text.partition("/")
    if not separator or not owner or not repository or "/" in repository:
        return False
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(character in allowed for character in owner) and all(character in allowed for character in repository)


class OfficialPolicySynthesis(ActionModelSynthesis):
    """
    Profile-backed official VLA/action runtime adapter.

    This class extends `ActionModelSynthesis` to provide an interface for
    interacting with official policy runtimes for action generation.
    It manages runtime configuration, asset loading, and execution of
    planning and prediction steps, optionally caching runtime instances.
    """

    MODEL_ID = "official-policy"

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
        Initializes the OfficialPolicySynthesis instance.

        Args:
            profile: The runtime profile object containing model configuration.
            device: The device to use for model execution (e.g., 'cuda', 'cpu').
            command_template: An optional sequence of strings forming a command template.
            env: An optional mapping of environment variables to set for the runtime.
            runtime_options: A mapping of additional options specific to the runtime.
        """
        super().__init__(profile, device=device, command_template=command_template, env=env)
        self.runtime_options = dict(runtime_options or {})
        self._runtime: OfficialPolicyRuntime | None = None
        self._runtime_key: tuple[Any, ...] | None = None

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
    ) -> OfficialPolicySynthesis:
        """
        Loads an `OfficialPolicySynthesis` instance from pretrained model configurations.

        This class method consolidates various configuration sources (paths, arguments,
        keyword arguments) to construct a runtime profile and then initializes
        the synthesis object.

        Args:
            pretrained_model_path: Path to the pretrained model checkpoint or a mapping
                                   of options, potentially including 'checkpoint_path'.
            args: Legacy argument, currently ignored.
            device: The device to use, e.g., 'cuda'.
            model_id: Explicit identifier for the model.
            profile_path: Path to the runtime profile configuration.
            manifest_path: Path to the manifest file for asset acquisition.
            acquisition_root: Root directory for acquiring external assets.
            hf_models_root: Root directory for Hugging Face models.
            command_template: A sequence of strings defining a command template.
            **kwargs: Additional keyword arguments to pass as runtime options.

        Returns:
            An initialized `OfficialPolicySynthesis` instance.
        """
        del args  # This parameter is deprecated and not used.

        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # Studio's ``model_ref`` is passed through this argument as a plain
        # string.  Treat an ``owner/repository`` value as a repository
        # reference; classifying it as a relative filesystem path makes it
        # resolve under the current working directory and blocks every
        # non-default checkpoint variant.  Existing/local path spellings keep
        # their original path semantics.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            checkpoint_value = str(pretrained_model_path)
            if _looks_like_hub_repo_id(checkpoint_value):
                options["checkpoint_ref"] = checkpoint_value
            else:
                options["checkpoint_path"] = checkpoint_value
        # Merge any additional keyword arguments into the options dictionary.
        options.update(kwargs)

        # Resolve the model ID by prioritizing explicit model_id, then profile_id from options,
        # then the provided model_id argument, finally falling back to the class's default MODEL_ID.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and various path configurations.
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

    def _runtime_config(self, options: Mapping[str, Any]) -> OfficialPolicyRuntimeConfig:
        """
        Generates an `OfficialPolicyRuntimeConfig` object based on current and provided options.

        This method merges instance-level runtime options with prediction-specific options,
        then loads default configurations from a specified path (if any), and finally
        builds a comprehensive runtime configuration.

        Args:
            options: A mapping of additional runtime options for this specific prediction.

        Returns:
            An `OfficialPolicyRuntimeConfig` instance ready for runtime initialization.
        """
        # Combine instance-level runtime options with method-specific options.
        explicit = {**self.runtime_options, **dict(options)}
        # Load default VLA/VA/WAM runtime configuration based on the model ID and an optional config path.
        defaults = load_vla_va_wam_runtime_config(
            self.model_id,
            explicit.get("runtime_config_path"),
        )
        return build_runtime_config(
            model_id=self.model_id,
            profile_checkpoints=self.profile.checkpoints,
            defaults=defaults,
            options=explicit,
            device=self.device,
        )

    def _runtime_for(self, config: OfficialPolicyRuntimeConfig) -> OfficialPolicyRuntime:
        """
        Retrieves or initializes an `OfficialPolicyRuntime` instance for the given configuration.

        This method implements a caching mechanism to reuse runtime instances
        if the configuration (identified by a key derived from its attributes)
        has not changed, avoiding redundant and potentially expensive re-initialization.

        Args:
            config: The `OfficialPolicyRuntimeConfig` to use for the runtime.

        Returns:
            An `OfficialPolicyRuntime` instance corresponding to the configuration.
        """
        # Create a unique key from essential configuration attributes for caching purposes.
        key = (
            config.model_id,
            config.backend,
            str(config.checkpoint_path),
            str(config.checkpoint_ref),
            config.device,
            config.torch_dtype,
            config.policy_target,
            config.processor_target,
            config.predict_method,
            config.required_assets,
            json.dumps(config.runtime_options, ensure_ascii=False, sort_keys=True, default=str),
        )
        # If no runtime is cached or the configuration has changed, initialize a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = OfficialPolicyRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """
        Selects the most relevant image input from various possible sources.

        Prioritizes the `images` argument, then `image` or `image_path` from kwargs.
        If not found, it checks `official_policy_observation` or `observation` for an 'image'
        or 'ref_image_path' key. Finally, falls back to `ref_image_path` from kwargs.

        Args:
            images: Direct image data (e.g., PIL Image, numpy array).
            kwargs: Additional keyword arguments potentially containing image data or paths.

        Returns:
            The selected image data or path, or None if no image is found.
        """
        if images is not None:
            return images
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Check for image within observation dictionaries.
        observation = kwargs.get("official_policy_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping):
            return observation.get("image") or observation.get("full_image") or observation.get("ref_image_path")
        return kwargs.get("ref_image_path")

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Selects the most relevant observation dictionary from keyword arguments.

        Prioritizes `official_policy_observation` over a generic `observation` key.

        Args:
            kwargs: Additional keyword arguments potentially containing an observation.

        Returns:
            The selected observation mapping, or None if not found or not a mapping.
        """
        observation = kwargs.get("official_policy_observation") or kwargs.get("observation")
        return observation if isinstance(observation, Mapping) else None

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generates an action prediction based on the provided prompt and context.

        This method first prepares a plan (which can be returned directly in "plan_only" mode),
        then checks for missing assets, and finally executes the prediction using the
        configured runtime. It manages temporary directories and output paths.

        Args:
            prompt: The natural language instruction or prompt for the action model.
            images: Image data to be processed by the model.
            video: Video data (currently ignored).
            interactions: A sequence of previous interactions providing context.
            output_path: Optional path where the prediction artifact should be saved.
            fps: Frames per second (currently ignored for image-based policies).
            timeout_seconds: Timeout for the prediction process (currently ignored).
            **kwargs: Additional keyword arguments for runtime configuration or context.

        Returns:
            A dictionary containing the prediction results, status, artifact paths,
            and other relevant metadata.
        """
        # These parameters are not used by the current implementation of official policy prediction.
        del video, fps, timeout_seconds

        # Determine if only a plan should be generated, prioritizing kwargs over instance runtime options.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Resolve the run directory. Create a temporary directory if not specified.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix=f"{self.model_id.replace('-', '_')}_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the final output path for the artifact.
        # If output_path is None, default to a file named after the profile artifact in the run_dir.
        resolved_output = Path(output_path) if output_path is not None else run_dir / self.profile.artifact_filename
        # Ensure the resolved output path is absolute and its parent directory exists.
        if not resolved_output.is_absolute():
            resolved_output = (Path.cwd() / resolved_output).resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        # Build the runtime configuration for this prediction.
        config = self._runtime_config(kwargs)
        # Get (or create) the runtime instance based on the configuration.
        runtime = self._runtime_for(config)

        # Define the path for saving the runtime plan.
        plan_path = run_dir / "runtime_profile_plan.json"

        # Generate the plan payload from the runtime.
        plan_payload = runtime.plan_payload(
            instruction=prompt,
            image=self._select_image(images, kwargs),
            observation=self._select_observation(kwargs),
            action_context=interactions,
            output_path=resolved_output,
            extra_metadata={
                "profile": self.profile.to_dict(),
                "policy_controls": kwargs.get("policy_controls"),
                "action_space": kwargs.get("action_space"),
            },
        )
        # Write the plan payload to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If only planning is requested, return the plan details.
        if plan_only:
            artifact_path = plan_path
            # If an output_path was specified and it targets a JSON file, write the plan there as well.
            if output_path is not None and resolved_output.suffix.lower() == ".json":
                resolved_output.write_text(
                    json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                artifact_path = resolved_output
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",
                "artifact_path": str(artifact_path),
                "plan_path": str(plan_path),
                "run_dir": str(run_dir),
                "runtime": "worldfoundry.official_policy.plan",
                "backend_quality": "plan",
                "missing_assets": plan_payload["missing_assets"],
                "profile": self.profile.to_dict(),
            }

        # Check for any missing required assets before proceeding with prediction.
        missing = runtime.missing_assets()
        if missing:
            # If assets are missing, return a "blocked" status with details.
            return {
                "status": "blocked",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",  # Still refers to the plan as the primary artifact.
                "artifact_path": str(plan_path),
                "plan_path": str(plan_path),
                "run_dir": str(run_dir),
                "runtime": "worldfoundry.official_policy.checkpoint_gated",
                "backend_quality": "checkpoint_missing",
                "blocked_reason": "required checkpoint/assets are missing",
                "missing_assets": missing,
                "profile": self.profile.to_dict(),
            }

        # Execute the actual action prediction using the runtime.
        result = runtime.predict_action(
            instruction=prompt,
            image=self._select_image(images, kwargs),
            observation=self._select_observation(kwargs),
            action_context=interactions,
            output_path=resolved_output,
            extra_metadata={
                "plan_path": str(plan_path),
                "policy_controls": kwargs.get("policy_controls"),
                "action_space": kwargs.get("action_space"),
            },
        )
        # Merge the prediction result with additional metadata including the plan path and run directory.
        return {**result, "plan_path": str(plan_path), "run_dir": str(run_dir), "profile": self.profile.to_dict()}

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.predict(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )
