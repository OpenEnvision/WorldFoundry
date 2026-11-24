"""Provides an interface for synthesizing actions using the Being-H0.5 model.

This module defines the `BeingH05Synthesis` class, which acts as a wrapper for
the Being-H0.5 action generation model. It handles model initialization,
configuration, and execution, allowing for lazy loading of the underlying
runtime and managing different runtime configurations based on provided options.
It integrates with the worldfoundry framework for profile management and action
trace generation.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.being_h05.runtime import (
    BeingH05Runtime,
    BeingH05RuntimeConfig,
    select_being_h05_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class BeingH05Synthesis(ActionModelSynthesis):
    """A synthesis wrapper for the Being-H0.5 action generation model.

    This class extends `ActionModelSynthesis` to provide a specific interface for
    the Being-H0.5 model. It manages model configuration, lazy loading of the
    BeingH05Runtime, and orchestrates the prediction process, including input
    handling and output formatting.
    """

    MODEL_ID = "being-h05"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the lazy Being-H0.5 synthesis wrapper.

        Args:
            profile: Profile-backed metadata and checkpoint contract.
            device: Runtime device selector used when executing BeingHPolicy.
            command_template: Optional external template retained for base compatibility.
            env: Optional runtime environment overrides.
            runtime_options: Being-H0.5 runtime and planning options.
        """

        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store runtime-specific options for later use during prediction.
        self.runtime_options = dict(runtime_options or {})
        # Lazily loaded runtime instance to avoid importing heavy dependencies until needed.
        self._runtime: BeingH05Runtime | None = None
        # Key to track the configuration used to create the current `_runtime` instance,
        # enabling efficient caching and reuse of the runtime.
        self._runtime_key: tuple[str, str, str, str, str, bool, str | None, str, str] | None = None

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
    ) -> BeingH05Synthesis:
        """Create Being-H0.5 from profile metadata without importing torch.

        This factory method simplifies the instantiation of `BeingH05Synthesis`
        by resolving model paths and loading the associated runtime profile.
        It handles various ways of specifying the model and its configuration.

        Args:
            pretrained_model_path: Optional checkpoint directory path or a mapping
                containing configuration options (e.g., {"checkpoint_dir": "path/to/ckpt"}).
            args: Unused compatibility parameter.
            device: Runtime device selector.
            model_id: Optional profile id override.
            profile_path: Optional runtime profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional acquisition cache root.
            hf_models_root: Optional Hugging Face cache root.
            command_template: Optional external command template.
            **kwargs: Additional runtime or planning options passed directly to the model.
        """

        del args  # This parameter is deprecated and unused.

        # Initialize options from pretrained_model_path if it's a mapping, otherwise empty.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a string (path), set it as "checkpoint_dir".
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)  # Merge additional keyword arguments into options.

        # Determine the effective model ID, prioritizing explicit options over defaults.
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
            runtime_options=options,  # Pass all resolved options as runtime_options.
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> BeingH05RuntimeConfig:
        """Resolve Being-H0.5 runtime settings without loading model dependencies.

        This method aggregates configuration from multiple sources:
        instance-level `runtime_options`, call-specific `options`,
        default runtime configurations, and the model's profile.
        It prioritizes more specific options over general ones.

        Args:
            options: Per-call runtime options.
        Returns:
            A `BeingH05RuntimeConfig` object containing the resolved configuration.
        """

        # Extract policy controls, prioritizing specific controls if present.
        policy_controls = options.get("policy_controls") if isinstance(options.get("policy_controls"), Mapping) else {}
        # Combine instance-level runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}

        # Load default runtime configurations for VLA/VA/WAM models.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options, explicit options take precedence.
        merged = {**runtime_defaults, **explicit_options}

        # Resolve `data_config_name` from multiple sources, with a clear priority order.
        data_config_name = str(
            explicit_options.get("data_config_name")
            or policy_controls.get("data_config_name")
            or runtime_defaults["data_config_name"]
            or self.profile.input_schema.get("data_config_name")
        )
        # Resolve `dataset_name` from multiple sources.
        dataset_name = str(
            explicit_options.get("dataset_name")
            or policy_controls.get("dataset_name")
            or runtime_defaults["dataset_name"]
            or self.profile.input_schema.get("dataset_name")
        )
        # Resolve `embodiment_tag` from multiple sources.
        embodiment_tag = str(
            explicit_options.get("embodiment_tag")
            or policy_controls.get("embodiment_tag")
            or runtime_defaults["embodiment_tag"]
            or self.profile.input_schema.get("embodiment_tag")
        )
        # Identify the checkpoint directory from various possible keys.
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        # Select the specific Being-H0.5 checkpoint based on the directory and dataset.
        checkpoint = select_being_h05_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
            dataset_name=dataset_name,
        )
        # Resolve `metadata_variant` from explicit options, policy controls, or runtime defaults.
        metadata_variant = (
            explicit_options.get("metadata_variant")
            or policy_controls.get("metadata_variant")
            or runtime_defaults.get("metadata_variant")
        )
        return BeingH05RuntimeConfig(
            checkpoint_dir=checkpoint,
            data_config_name=data_config_name,
            dataset_name=dataset_name,
            embodiment_tag=embodiment_tag,
            instruction_template=str(
                explicit_options.get("instruction_template") or runtime_defaults["instruction_template"]
            ),
            device=str(merged.get("device") or self.device),
            torch_dtype=str(merged.get("torch_dtype") or merged.get("dtype") or "auto"),
            enable_rtc=bool(merged["enable_rtc"]),
            metadata_variant=str(metadata_variant) if metadata_variant is not None else None,
            stats_selection_mode=str(
                explicit_options.get("stats_selection_mode") or runtime_defaults["stats_selection_mode"]
            ),
            attention_mask_kind=str(
                explicit_options.get("attention_mask_kind")
                or policy_controls.get("attention_mask_kind")
                or runtime_defaults["attention_mask_kind"]
            ),
        )

    def _runtime_for(self, config: BeingH05RuntimeConfig) -> BeingH05Runtime:
        """Return a cached Being-H0.5 runtime for one checkpoint and dataset.

        This method ensures that the Being-H0.5 runtime is initialized only once
        for a given configuration, preventing redundant model loading.

        Args:
            config: Resolved Being-H0.5 runtime configuration.
        Returns:
            An instance of `BeingH05Runtime` configured with the provided settings.
        """

        # Create a unique key from the configuration to identify the runtime instance.
        key = (
            str(config.checkpoint_dir),
            config.data_config_name,
            config.dataset_name,
            config.embodiment_tag,
            config.device,
            config.torch_dtype,
            config.enable_rtc,
            config.metadata_variant,
            config.stats_selection_mode,
            config.attention_mask_kind,
        )
        # If no runtime is loaded or the configuration has changed, create a new one.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = BeingH05Runtime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """Select the Being-H0.5 primary image input.

        This method determines the most appropriate image source from the
        various inputs provided, following a specific priority order.

        Args:
            images: Direct image input from the pipeline.
            kwargs: Additional model-specific inputs, potentially containing image paths or observation dictionaries.
        Returns:
            The selected image input (path or data), or None if no image is found.
        """

        if images is not None:
            return images
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Check for image within a structured observation dictionary.
        observation = kwargs.get("being_h05_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping):
            # Prioritize specific image views from the observation.
            for key in ("video.top_view", "video.left_view", "video.wrist_view", "rgb"):
                if observation.get(key) is not None:
                    return observation[key]
        return kwargs.get("ref_image_path")

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Select the Being-H0.5 observation mapping from pipeline inputs.

        This method extracts the structured observation dictionary from the
        provided keyword arguments.

        Args:
            kwargs: Additional model-specific inputs.
        Returns:
            The observation mapping if present and valid, otherwise None.
        """

        observation = kwargs.get("being_h05_observation") or kwargs.get("observation")
        return observation if isinstance(observation, Mapping) else None

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
        """Prepare or execute the in-tree Being-H0.5 action runtime.

        This method orchestrates the action prediction process. It can either
        prepare a plan for execution or directly run the Being-H0.5 model
        to generate an action, depending on the `plan_only` flag.

        Args:
            prompt: Natural-language task instruction.
            images: RGB observation image path or array.
            video: Unused compatibility video input.
            interactions: Operator-provided action context (e.g., previous actions or observations).
            output_path: Optional action_trace output path for saving results.
            fps: Unused compatibility frame rate.
            timeout_seconds: Unused compatibility timeout.
            **kwargs: Additional runtime and planning options for the model.
        Returns:
            A dictionary containing the prediction results or the plan details.
        Raises:
            FileNotFoundError: If the Being-H0.5 checkpoint directory is missing.
        """

        del timeout_seconds  # This parameter is unused in BeingH05Synthesis.

        # Determine if only a plan should be generated, not actual execution.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Resolve the run directory, creating a temporary one if not provided.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="being_h05_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the context dictionary, which encapsulates all relevant input information.
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
        # Ensure the device is explicitly set in runtime options if not already present.
        runtime_options.setdefault("device", self.device)
        runtime_config = self._runtime_config(runtime_options)

        # Define the path for saving the runtime plan.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, including profile and context details.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-being-h05-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.being_h05.in_tree_runtime.BeingHPolicy.get_action",
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "data_config_name": runtime_config.data_config_name,
                "dataset_name": runtime_config.dataset_name,
                "embodiment_tag": runtime_config.embodiment_tag,
                "instruction_template": runtime_config.instruction_template,
                "device": runtime_config.device,
                "enable_rtc": runtime_config.enable_rtc,
                "metadata_variant": runtime_config.metadata_variant,
                "stats_selection_mode": runtime_config.stats_selection_mode,
                "attention_mask_kind": runtime_config.attention_mask_kind,
                "runtime_package": "worldfoundry.synthesis.action_generation.being_h05.being_h05_runtime",
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If `plan_only` is true, return the plan details without executing the model.
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

        # Verify that the checkpoint directory exists before attempting to load the model.
        if not runtime_config.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Being-H0.5 checkpoint directory is missing: {runtime_config.checkpoint_dir}")

        # Select the primary image input, prioritizing explicit `images` argument, then context.
        image = self._select_image(images, kwargs)
        if image is None and context.get("image_path"):
            image = context["image_path"]

        # Get the Being-H0.5 runtime instance (cached or newly created) and predict an action.
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            image=image,
            observation=self._select_observation(kwargs),
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    "observation_keys": sorted(self._select_observation(kwargs) or {}),
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
