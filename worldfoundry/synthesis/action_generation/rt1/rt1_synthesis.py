"""
This module defines the RT-1 synthesis class, a wrapper for the RT-1 action generation model.

It provides functionality to initialize, configure, and run RT-1 models, including
lazy loading of TensorFlow components and handling of model checkpoints and profiles.
The class integrates with the worldfoundry framework for model acquisition and
runtime management, allowing for both planning and direct execution of RT-1 policies.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.action_generation.rt1.runtime import RT1RuntimeConfig, RT1SavedModelRuntime


class RT1Synthesis(ActionModelSynthesis):
    """
    RT-1 specific implementation for action model synthesis.

    This class extends `ActionModelSynthesis` to provide a lazy-loaded
    interface for the RT-1 TensorFlow SavedModel runtime. It manages model
    configuration, checkpoint resolution, and execution of RT-1 policies,
    supporting both plan generation and direct action prediction.
    """

    MODEL_ID = "rt-1"

    def __init__(
        self,
        profile,
        *,
        device: str = "cpu",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the lazy RT-1 synthesis wrapper.

        Args:
            profile: Profile-backed metadata and checkpoint contract.
            device: Runtime device selector used when executing TensorFlow.
            command_template: Optional external command template, kept for base compatibility.
            env: Optional runtime environment overrides.
            runtime_options: Optional RT-1 runtime and planning options.
        """

        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        self._runtime: RT1SavedModelRuntime | None = None
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
    ) -> "RT1Synthesis":
        """Create RT-1 from profile metadata without importing TensorFlow.

        This class method is the primary way to instantiate an RT-1 synthesis
        object, allowing it to load configuration from a profile, potentially
        overriding parts with provided arguments.

        Args:
            pretrained_model_path: Optional checkpoint path or option mapping. If a mapping,
                                   it's treated as additional options.
            args: Unused compatibility parameter.
            device: Runtime device selector.
            model_id: Optional profile id override.
            profile_path: Optional runtime profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional acquisition cache root.
            hf_models_root: Optional Hugging Face cache root.
            command_template: Optional external command template.
            **kwargs: Additional runtime or planning options which can override
                      profile defaults or be merged into the final configuration.

        Returns:
            An instance of `RT1Synthesis` configured with the specified options and profile.
        """

        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as the checkpoint directory.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options.
        options.update(kwargs)
        # Determine the effective model ID, prioritizing explicit options, then `model_id` arg, then class default.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile based on the resolved model ID and paths.
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

    def _select_checkpoint(self, options: Mapping[str, Any]) -> Path:
        """Resolve the RT-1 SavedModel checkpoint directory.

        This method determines the active checkpoint directory, prioritizing
        runtime options, then instance-level options, and finally profile data.

        Args:
            options: Per-call runtime options that may override profile checkpoints.

        Returns:
            The resolved and absolute path to the RT-1 SavedModel checkpoint directory.

        Raises:
            ValueError: If no valid RT-1 checkpoint directory can be found.
        """

        # Prioritize checkpoint_dir from call-specific options, then instance-level runtime_options.
        checkpoint_dir = (
            options.get("checkpoint_dir")
            or options.get("ckpt_path")
            or options.get("pretrained_model_path")
            or self.runtime_options.get("checkpoint_dir")
            or self.runtime_options.get("ckpt_path")
        )

        def expand(value: str | Path) -> Path:
            """Helper to resolve WorldFoundry paths and ensure they are absolute and resolved."""
            path = resolve_worldfoundry_path(value)
            if not path.is_absolute():
                path = project_root() / path
            return path.resolve()

        if checkpoint_dir:
            return expand(str(checkpoint_dir))
        # Fallback to the first checkpoint defined in the profile if no explicit checkpoint_dir is found.
        if self.profile.checkpoints:
            checkpoint = dict(self.profile.checkpoints[0])
            local_dir = checkpoint.get("local_dir")
            if local_dir:
                return expand(str(local_dir))
        raise ValueError("RT-1 requires a TensorFlow SavedModel checkpoint directory.")

    def _runtime_config(self, options: Mapping[str, Any]) -> "RT1RuntimeConfig":
        """Build RT-1 runtime config without loading the SavedModel.

        This method prepares the configuration required to instantiate the
        RT-1 runtime, including the checkpoint directory and device.

        Args:
            options: Per-call runtime options.

        Returns:
            An `RT1RuntimeConfig` object containing the resolved configuration.
        """

        from worldfoundry.synthesis.action_generation.rt1.runtime import RT1RuntimeConfig

        checkpoint_dir = self._select_checkpoint(options)
        return RT1RuntimeConfig(
            checkpoint_dir=checkpoint_dir,
            device=str(options.get("device") or self.runtime_options.get("device") or self.device or "cpu"),
        )

    def _runtime_for(self, config: "RT1RuntimeConfig") -> "RT1SavedModelRuntime":
        """Return a cached RT-1 runtime for one checkpoint and device.

        This method ensures that the RT-1 SavedModel runtime is loaded only once
        for a given checkpoint and device configuration, caching the instance.

        Args:
            config: Resolved RT-1 runtime configuration.

        Returns:
            An instance of `RT1SavedModelRuntime` corresponding to the provided configuration.
        """

        from worldfoundry.synthesis.action_generation.rt1.runtime import RT1SavedModelRuntime

        key = (str(config.checkpoint_dir), config.device)
        # Check if the cached runtime exists and matches the current configuration key.
        if self._runtime is None or self._runtime_key != key:
            # If not, instantiate a new runtime and update the cache.
            self._runtime = RT1SavedModelRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """Select the RT-1 observation image from pipeline inputs.

        This method attempts to find the most appropriate image input from
        various possible locations in the provided arguments.

        Args:
            images: Direct image input from the pipeline.
            kwargs: Additional model-specific inputs, which may contain image data.

        Returns:
            The selected image, which could be a path, array, or None if not found.
        """

        # Prioritize direct 'images' argument.
        if images is not None:
            return images
        # Check various common keyword arguments for image data.
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Look for image data within 'rt1_observation' or 'openvla_observation' mappings.
        observation = kwargs.get("rt1_observation") or kwargs.get("openvla_observation")
        if isinstance(observation, Mapping) and observation.get("rgb") is not None:
            return observation["rgb"]
        # Fallback to 'ref_image_path'.
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
        """Prepare or execute the in-tree RT-1 action runtime.

        This method handles the full lifecycle of an RT-1 prediction request,
        from setting up the runtime environment to executing the model and
        returning the results, or just generating a plan.

        Args:
            prompt: Natural-language task instruction for the RT-1 model.
            images: RGB observation image path or array.
            video: Unused compatibility video input.
            interactions: Operator-provided action context, typically a sequence of strings.
            output_path: Optional action_trace output path where results will be saved.
            fps: Unused compatibility frame rate.
            timeout_seconds: Unused compatibility timeout.
            **kwargs: Additional runtime and planning options, which can override
                      instance-level settings or provide specific execution parameters.

        Returns:
            A dictionary containing the prediction results, including status,
            model ID, artifact paths, and runtime metadata. If `plan_only` is true,
            it returns the plan details without executing the model.

        Raises:
            FileNotFoundError: If the resolved RT-1 SavedModel checkpoint directory is missing.
        """

        del timeout_seconds
        # Determine if only a plan should be generated, without actual model execution.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # The released SavedModel consumes a precomputed 512-D embedding. Do not
        # serialize it into the generic plan or silently substitute a zero vector.
        language_embedding = kwargs.pop("language_embedding", None)
        observation = kwargs.get("rt1_observation")
        if isinstance(observation, Mapping) and "natural_language_embedding" in observation:
            if language_embedding is None:
                language_embedding = observation["natural_language_embedding"]
            kwargs["rt1_observation"] = {
                key: value for key, value in observation.items() if key != "natural_language_embedding"
            }
        # Set up a temporary directory for run artifacts if not provided.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="rt1_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Prepare the context dictionary, merging common inputs and extra kwargs.
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
        # Initialize runtime options, ensuring 'device' is set, prioritizing kwargs, then instance default.
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        # Build the RT-1 runtime configuration based on resolved options.
        runtime_config = self._runtime_config(runtime_options)
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the plan payload, which describes the model, context, and runtime configuration.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-rt1-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.rt1.in_tree_savedmodel.action_signature",
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "device": runtime_config.device,
                "runtime_package": "worldfoundry.synthesis.action_generation.rt1",
            },
        }
        # Write the plan to a JSON file in the run directory.
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if plan_only:
            # If only a plan is requested, return the planning details and exit early.
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
            raise FileNotFoundError(f"RT-1 SavedModel checkpoint directory is missing: {runtime_config.checkpoint_dir}")
        # Select the appropriate input image for the model.
        image = self._select_image(images, kwargs)
        if image is None and context.get("image_path"):
            image = context["image_path"]
        # Execute the RT-1 model's action prediction.
        result = self._runtime_for(runtime_config).predict_action(
            instruction=prompt,
            language_embedding=language_embedding,
            image=image,
            output_path=context["output_path"],
            # Aggregate additional metadata to be included in the prediction results.
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    "observation_keys": sorted(kwargs.get("rt1_observation", {}).keys())
                    if isinstance(kwargs.get("rt1_observation"), Mapping)
                    else [],
                },
            },
        )
        # Combine model results with run-specific metadata for the final output.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
