"""
Module for defining the OpenPISynthesis class, which integrates OpenPI models for action synthesis.

This module provides a wrapper around the OpenPI runtime, allowing for the generation of actions
based on prompts, images, and other contextual information. It handles configuration,
runtime initialization, and execution of OpenPI policies.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.openpi.runtime import (
    OpenPIRuntime,
    OpenPIRuntimeConfig,
    select_openpi_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class OpenPISynthesis(ActionModelSynthesis):
    """
    A class that wraps the OpenPI model for action synthesis.

    This class extends ActionModelSynthesis to provide an interface for interacting
    with OpenPI policies. It manages runtime configuration, model loading, and
    action prediction based on a given profile and input context.
    """

    MODEL_ID = "openpi"

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
        Initializes the OpenPISynthesis instance.

        Args:
            profile: The runtime profile defining the OpenPI model and its configurations.
            device (str): The device to run the model on (e.g., "cuda", "cpu"). Defaults to "cuda".
            command_template (Sequence[str] | None): Optional template for executing a command.
            env (Mapping[str, str] | None): Optional environment variables for the runtime.
            runtime_options (Mapping[str, Any] | None): Additional runtime-specific options.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        # Stores the lazily loaded OpenPIRuntime instance.
        self._runtime: OpenPIRuntime | None = None
        # Stores the key (config parameters) associated with the currently loaded runtime.
        # Used for caching to avoid reloading the runtime if config doesn't change.
        self._runtime_key: tuple[str, str, str | None, str, int, str, str] | None = None

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
    ) -> OpenPISynthesis:
        """
        Creates a lazy OpenPISynthesis wrapper from profile and runtime options.

        This class method allows instantiating the OpenPISynthesis model by resolving
        the appropriate runtime profile and merging configuration options. It supports
        loading from a pretrained model path or explicit profile details.

        Args:
            pretrained_model_path (str | Path | Mapping | None): Path to a pretrained model
                checkpoint or a mapping of runtime options.
            args: Legacy argument, currently ignored.
            device (str | None): Device to use (e.g., "cuda", "cpu").
            model_id (str | None): Identifier for the model.
            profile_path (str | Path | None): Explicit path to the runtime profile.
            manifest_path (str | Path | None): Path to the manifest file.
            acquisition_root (str | Path | None): Root directory for model acquisition.
            hf_models_root (str | Path | None): Root directory for HuggingFace models.
            command_template (Sequence[str] | None): Optional command template.
            **kwargs (Any): Additional runtime options passed directly to the OpenPI runtime.

        Returns:
            OpenPISynthesis: An initialized instance of OpenPISynthesis.
        """
        del args
        # Normalize pretrained_model_path into a dictionary of options.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)

        # Resolve the model ID from various sources, prioritizing explicit options.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and paths.
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

    def _runtime_config(self, options: Mapping[str, Any]) -> OpenPIRuntimeConfig:
        """
        Resolves the OpenPI policy configuration and selects the appropriate checkpoint.

        This method merges instance-level runtime options with prediction-specific
        options to create a complete OpenPIRuntimeConfig, without loading the actual model.

        Args:
            options (Mapping[str, Any]): Specific options for the current prediction call.

        Returns:
            OpenPIRuntimeConfig: A configuration object for the OpenPI runtime.
        """
        # Merge instance-level runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configurations, then merge explicit options over them.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.model_id,
            explicit_options.get("runtime_config_path"),
        )
        merged = {**runtime_defaults, **explicit_options}

        config_name = str(merged["config_name"])
        data_family = config_name.rsplit("_", 1)[-1].lower()
        if data_family not in {"aloha", "droid", "libero"}:
            raise ValueError(f"OpenPI config {config_name!r} does not identify a supported data family")
        configured_family = str(merged.get("data_family") or data_family).lower()
        if configured_family != data_family:
            raise ValueError(
                f"OpenPI config {config_name!r} implies data family {data_family!r}, not {configured_family!r}"
            )
        # Resolve the checkpoint directory from multiple possible keys in explicit_options.
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        # Select the specific OpenPI checkpoint based on resolved directory, profile, and config.
        checkpoint = select_openpi_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
            config_name=config_name,
            require_exists=bool(merged.get("require_checkpoint_exists", True)),
        )
        pytorch_device = merged.get("pytorch_device") or merged.get("device")
        seed = int(merged["seed"])
        return OpenPIRuntimeConfig(
            checkpoint_dir=checkpoint,
            config_name=config_name,
            data_family=data_family,
            pytorch_device=str(pytorch_device) if pytorch_device is not None else None,
            torch_dtype=str(merged.get("torch_dtype") or "auto"),
            seed=seed,
            paligemma_tokenizer_path=str(merged["paligemma_tokenizer_path"]),
            fast_tokenizer_path=str(merged["fast_tokenizer_path"]),
        )

    def _runtime_for(self, config: OpenPIRuntimeConfig) -> OpenPIRuntime:
        """
        Lazily loads and caches the OpenPIRuntime instance based on the provided configuration.

        If the runtime configuration has changed or the runtime has not been loaded yet,
        a new OpenPIRuntime instance is created. Otherwise, the cached instance is returned.

        Args:
            config (OpenPIRuntimeConfig): The configuration for the OpenPI runtime.

        Returns:
            OpenPIRuntime: The instantiated OpenPI runtime.
        """
        # Create a unique key from the configuration parameters to identify the runtime instance.
        key = (
            str(config.checkpoint_dir),
            config.config_name,
            config.data_family,
            config.pytorch_device,
            config.torch_dtype,
            config.seed,
            config.paligemma_tokenizer_path,
            config.fast_tokenizer_path,
        )
        # Check if runtime is not loaded or if the configuration key has changed.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = OpenPIRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """
        Selects the primary image input from various possible arguments.

        Prioritizes an explicit 'images' argument, then checks 'image', 'image_path',
        'rgb_views' within 'openpi_observation', and finally 'ref_image_path'.

        Args:
            images (Any): Direct image input.
            kwargs (Mapping[str, Any]): Additional keyword arguments that might contain image data.

        Returns:
            Any: The selected image data or path, or None if no image is found.
        """
        if images is not None:
            return images
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        observation = kwargs.get("openpi_observation")
        if isinstance(observation, Mapping):
            rgb_views = observation.get("rgb_views")
            if rgb_views is not None:
                return rgb_views
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
        Prepares an OpenPI run plan and optionally executes in-tree policy inference.

        This method generates a plan for an OpenPI execution, including model configuration
        and context. If `plan_only` is False, it then proceeds to execute the OpenPI policy
        to predict actions.

        Args:
            prompt (str): The natural language instruction for the action. Defaults to "".
            images (Any): Input image data. Defaults to None.
            video (Any): Input video data. Defaults to None.
            interactions (Sequence[str]): A sequence of prior interactions. Defaults to ().
            output_path (str | Path | None): Path to save the output artifacts. If None, a
                temporary path will be generated.
            fps (int | None): Frames per second for video input. Defaults to None.
            timeout_seconds (int): Maximum time allowed for the prediction. Defaults to 21600.
            **kwargs (Any): Additional runtime options and context data for the prediction.

        Returns:
            dict[str, Any]: A dictionary containing the prediction results, run directory,
                plan path, and profile information.
        """
        del timeout_seconds

        # Determine if only a plan should be generated, or if inference should also run.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Create a temporary directory for this run's artifacts.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="openpi_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate the context dictionary for the OpenPI run.
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
        # Ensure device is set and checkpoint existence is checked unless plan_only is true.
        runtime_options.setdefault("device", self.device)
        runtime_options.setdefault("require_checkpoint_exists", not plan_only)
        runtime_config = self._runtime_config(runtime_options)

        # Define the path for the runtime plan JSON file.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, detailing the profile, context, and runtime config.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-openpi-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.openpi.in_tree_runtime.create_trained_policy_infer",
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "config_name": runtime_config.config_name,
                "data_family": runtime_config.data_family,
                "pytorch_device": runtime_config.pytorch_device,
                "torch_dtype": runtime_config.torch_dtype,
                "seed": runtime_config.seed,
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

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

        # Select the primary image input for the prediction.
        image = self._select_image(images, kwargs)
        # Fallback to image_path from context if no direct image was selected.
        if image is None and context.get("image_path"):
            image = context["image_path"]

        observation_payload = dict(kwargs.get("openpi_observation") or {})
        for key in (
            "state",
            "proprio",
            "robot_state",
            "joint_state",
            "joint_position",
            "gripper_position",
            "observation/state",
            "observation/joint_position",
            "observation/gripper_position",
        ):
            if key in kwargs and kwargs[key] is not None:
                observation_payload.setdefault(key, kwargs[key])

        # Execute the OpenPI policy prediction using the resolved runtime and inputs.
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            image=image,
            output_path=context["output_path"],
            openpi_observation=observation_payload,
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    # Log the keys present in the openpi_observation for debugging/analysis.
                    "observation_keys": sorted(kwargs.get("openpi_observation", {}).keys())
                    if isinstance(kwargs.get("openpi_observation"), Mapping)
                    else [],
                },
            },
        )
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
