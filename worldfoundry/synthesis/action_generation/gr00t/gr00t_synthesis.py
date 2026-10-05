from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config, variant_defaults
from worldfoundry.synthesis.action_generation.gr00t.runtime import GR00TRuntime, GR00TRuntimeConfig, select_gr00t_checkpoint
from ..base_action_synthesis import ActionModelSynthesis


class GR00TSynthesis(ActionModelSynthesis):
    """
    A synthesis wrapper for the GR00T action model, extending ActionModelSynthesis.

    This class provides methods to load, configure, and run the GR00T model for
    action prediction, handling model initialization, runtime configuration,
    and prediction execution. It manages model caching and plan generation.
    """
    MODEL_ID = "gr00t"

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
        Initializes the GR00TSynthesis wrapper.

        Args:
            profile: The runtime profile for the GR00T model.
            device: The device to run the model on (e.g., 'cuda', 'cpu').
            command_template: An optional sequence of strings to template command execution.
            env: An optional mapping of environment variables for the runtime.
            runtime_options: A dictionary of additional options specific to the GR00T runtime.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store runtime options, converting to a mutable dict if None is provided
        self.runtime_options = dict(runtime_options or {})
        # Initialize internal cache for the GR00T runtime instance
        self._runtime: GR00TRuntime | None = None
        # Initialize internal cache key for the GR00T runtime instance, used to determine if a new runtime is needed
        self._runtime_key: tuple[str, str, str, str, int] | None = None

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
    ) -> "GR00TSynthesis":
        """
        Create a lazy in-tree GR00T synthesis wrapper from profile/runtime options.

        This class method serves as a factory to instantiate `GR00TSynthesis`
        by resolving model configurations from various sources, including a
        `pretrained_model_path` (which can be a path or a dictionary of options),
        `profile_path`, `manifest_path`, and other keyword arguments.

        Args:
            pretrained_model_path: Path to the pretrained model checkpoint or a dictionary of options.
            args: Deprecated argument, will be ignored.
            device: The device to run the model on.
            model_id: The ID of the model to load.
            profile_path: Path to the runtime profile YAML file.
            manifest_path: Path to the manifest file for model discovery.
            acquisition_root: Root directory for acquiring models.
            hf_models_root: Root directory for Hugging Face models.
            command_template: Optional command template for execution.
            **kwargs: Additional keyword arguments for runtime options.

        Returns:
            An instance of `GR00TSynthesis`.
        """
        del args  # This argument is deprecated and intentionally ignored.

        # Normalize pretrained_model_path into an options dictionary
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)  # Merge additional kwargs into options

        # Resolve the model ID from options or fall back to class default
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and paths
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )

        # Instantiate GR00TSynthesis with the resolved profile and options
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> GR00TRuntimeConfig:
        """
        Resolve GR00T checkpoint and embodiment settings without loading heavy deps.

        This method merges default runtime configurations with instance-specific
        and call-specific options to produce a complete `GR00TRuntimeConfig`.
        It determines the correct checkpoint directory and other parameters
        needed for `GR00TRuntime` instantiation.

        Args:
            options: A dictionary of runtime options specific to the current call.

        Returns:
            A `GR00TRuntimeConfig` object.
        """
        # Load default runtime configurations for the model ID
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            options.get("runtime_config_path") or self.runtime_options.get("runtime_config_path"),
        )
        # Combine defaults, instance-level runtime options, and call-specific options
        initial = {**runtime_defaults, **self.runtime_options, **dict(options)}

        # Determine the model variant, prioritizing call-specific then instance-specific
        variant = str(initial.get("variant") or initial.get("variant_id") or initial.get("default_variant"))
        # Load variant-specific defaults, if any
        variant_config = variant_defaults(runtime_defaults, variant)

        # Merge all configurations: runtime_defaults -> variant_config -> instance_options -> call_options
        merged = {**runtime_defaults, **variant_config, **self.runtime_options, **dict(options)}

        # Resolve the checkpoint directory, checking multiple possible keys
        checkpoint_dir = (
            merged.get("checkpoint_dir")
            or merged.get("ckpt_path")
            or merged.get("pretrained_model_path")
        )
        # Select the final GR00T checkpoint path based on the resolved directory and variant
        checkpoint = select_gr00t_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
            variant=variant,
        )
        if (
            json.loads((checkpoint / "config.json").read_text()).get("model_type") == "gr00t_n1"
            and "embodiment_tag" not in self.runtime_options
            and "embodiment_tag" not in options
        ):
            merged["embodiment_tag"] = "GR1"

        # Construct and return the GR00TRuntimeConfig
        return GR00TRuntimeConfig(
            checkpoint_dir=checkpoint,
            embodiment_tag=str(merged["embodiment_tag"]),
            device=str(merged.get("device") or self.device),
            torch_dtype=str(merged["torch_dtype"]),
            seed=int(merged.get("seed", 0)),
        )

    def _runtime_for(self, config: GR00TRuntimeConfig) -> GR00TRuntime:
        """
        Return the cached GR00T runtime for a given configuration.

        This method ensures that a `GR00TRuntime` instance is reused if its
        configuration (checkpoint, embodiment tag, device, dtype, seed)
        matches the previously loaded runtime, avoiding redundant loading
        of potentially heavy model assets. If the configuration differs,
        a new runtime is created and cached.

        Args:
            config: The `GR00TRuntimeConfig` specifying the desired runtime.

        Returns:
            An instance of `GR00TRuntime`.
        """
        # Create a unique key from the configuration parameters for caching
        key = (
            str(config.checkpoint_dir),
            config.embodiment_tag,
            config.device,
            config.torch_dtype,
            config.seed,
        )
        # If no runtime is cached, or the cached runtime's key does not match the current config,
        # create a new GR00TRuntime instance and update the cache.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = GR00TRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """
        Select the RGB observation source from WorldFoundry or GR00T-specific kwargs.

        This static method provides a flexible way to extract image data from
        various potential input locations, prioritizing direct `images` argument,
        then `image`, `image_path`, `gr00t_observation` (specifically `camera_views`),
        and finally `ref_image_path` from the provided keyword arguments.

        Args:
            images: Direct image input (e.g., PIL Image, numpy array).
            kwargs: A dictionary of additional keyword arguments, potentially containing image sources.

        Returns:
            The selected image data or path, or None if no image source is found.
        """
        if images is not None:
            return images
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Check for image within a structured observation dictionary (common for GR00T)
        observation = kwargs.get("gr00t_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping):
            camera_views = observation.get("camera_views")
            if camera_views is not None:
                return camera_views
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
        Prepare a GR00T in-tree run plan and optionally execute policy inference.

        This method orchestrates the prediction process for the GR00T model.
        It first prepares a run directory, generates a runtime plan, and
        then, if not in 'plan_only' mode, executes the GR00T model's
        `predict_action` method using the appropriate runtime and configuration.

        Args:
            prompt: The text instruction for the model.
            images: Image input for the model.
            video: Video input for the model.
            interactions: A sequence of previous interaction strings.
            output_path: Desired path for saving model outputs.
            fps: Frames per second for video processing, if applicable.
            timeout_seconds: Maximum time allowed for the original N1 worker.
            **kwargs: Additional arguments that can influence runtime configuration or model input.

        Returns:
            A dictionary containing the prediction results and metadata,
            including `run_dir`, `plan_path`, and the model's profile.
        """
        # Determine if only a plan should be generated, not actual inference
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Create or resolve a run directory for storing artifacts
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="gr00t_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate a structured context dictionary for the run
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

        # Prepare runtime options, ensuring device is set
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)

        # Resolve the GR00T runtime configuration
        runtime_config = self._runtime_config(runtime_options)

        # Describe the checkpoint architecture for inclusion in the plan
        runtime_spec = GR00TRuntime.describe_checkpoint(runtime_config.checkpoint_dir)
        is_original_n1 = runtime_spec.get("model_type") == "gr00t_n1"
        backend = (
            "worldfoundry.gr00t.official_n1_release_subprocess"
            if is_original_n1
            else "worldfoundry.gr00t.in_tree_runtime.predict_action"
        )

        # Define the path for the runtime plan JSON file
        plan_path = run_dir / "runtime_profile_plan.json"

        # Construct the plan payload with model, context, and runtime details
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-gr00t-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": backend,
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "device": runtime_config.device,
                "embodiment_tag": runtime_config.embodiment_tag,
                "seed": runtime_config.seed,
                "torch_dtype": runtime_config.torch_dtype,
            },
            "architecture": runtime_spec,
        }
        # Write the plan to a JSON file
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if plan_only:
            # If only a plan is requested, return the plan details
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # Select the most appropriate image input source
        image = self._select_image(images, kwargs)
        # If no image was selected directly, check if context has an image path
        if image is None and context.get("image_path"):
            image = context["image_path"]

        # Get or create the GR00T runtime instance based on the configuration
        # and execute the predict_action method
        observation = kwargs.get("gr00t_observation") or kwargs.get("observation")
        extra_metadata = {
            "run_dir": str(run_dir),
            "profile": self.profile.to_dict(),
            "interactions": list(interactions),
            "operator_context": {
                "action_space": kwargs.get("action_space"),
                "policy_controls": kwargs.get("policy_controls"),
                "observation_keys": sorted(observation.keys()) if isinstance(observation, Mapping) else [],
            },
        }
        if is_original_n1:
            from worldfoundry.synthesis.action_generation.gr00t.legacy_n1 import predict_n1_action

            n1_options = {**self.runtime_options, **kwargs}
            result = predict_n1_action(
                checkpoint_dir=runtime_config.checkpoint_dir,
                source_dir=n1_options.get("n1_source_dir"),
                python_executable=n1_options.get("n1_python"),
                data_config_name=str(n1_options.get("n1_data_config_name") or "gr1_arms_only"),
                embodiment_tag=runtime_config.embodiment_tag,
                device=runtime_config.device,
                seed=runtime_config.seed,
                instruction=prompt,
                image=image,
                observation=observation,
                output_path=context["output_path"],
                run_dir=run_dir,
                extra_metadata=extra_metadata,
                timeout_seconds=timeout_seconds,
            )
        else:
            result = self._runtime_for(runtime_config).predict_action(
                instruction=prompt,
                image=image,
                output_path=context["output_path"],
                gr00t_observation=observation,
                extra_metadata=extra_metadata,
            )
        # Merge additional run metadata with the prediction result
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
