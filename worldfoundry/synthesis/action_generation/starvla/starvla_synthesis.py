from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.synthesis.action_generation.starvla.runtime import (
    StarVLAPlanRuntime,
    StarVLARuntimeConfig,
    select_starvla_base_vlm,
    select_starvla_base_world_model,
    select_starvla_checkpoint,
)
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"Invalid StarVLA boolean option: {value!r}.")
    return bool(value)


def _first_option(
    explicit: Mapping[str, Any],
    merged: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for source in (explicit, merged):
        for name in names:
            if name in source:
                return source[name]
    return default


class StarVLASynthesis(ActionModelSynthesis):
    """
    A class for synthesizing actions using the StarVLA model.

    This class extends `ActionModelSynthesis` and provides functionality
    to load, configure, and run StarVLA for action prediction based on
    given prompts and visual observations. It handles the management of
    runtime profiles, configurations, and the underlying StarVLA execution environment.
    """

    MODEL_ID = "starvla"

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
        Initialize the lazy StarVLA synthesis wrapper.

        This constructor sets up the basic parameters for the StarVLA model,
        including the profile, device, and any initial runtime options.
        It also initializes internal caches for the StarVLA runtime.

        Args:
            profile: Profile-backed metadata and checkpoint contract.
            device: Runtime device selector used for planning or inference (e.g., "cuda", "cpu").
            command_template: Optional external command template kept for base compatibility.
            env: Optional runtime environment overrides to be passed to the model execution.
            runtime_options: Optional StarVLA runtime and planning options. These can be
                             overridden by `predict` method arguments.
        """

        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store runtime options, allowing per-call overrides later.
        self.runtime_options = dict(runtime_options or {})
        # Initialize internal cache for the StarVLA runtime instance.
        self._runtime: StarVLAPlanRuntime | None = None
        # Initialize internal cache key for the StarVLA runtime to check if a new runtime is needed.
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
    ) -> StarVLASynthesis:
        """
        Create a StarVLASynthesis instance from profile metadata without importing model dependencies.

        This class method provides a convenient way to instantiate `StarVLASynthesis`
        by loading a runtime profile and resolving various configuration options
        from arguments and an optional `pretrained_model_path` dictionary.

        Args:
            pretrained_model_path: Optional checkpoint directory path or a mapping of options.
                                   If a mapping, it can contain 'checkpoint_dir', 'model_id', etc.
            args: Unused compatibility parameter, deliberately ignored.
            device: Runtime device selector (e.g., "cuda", "cpu").
            model_id: Optional profile ID override. If not provided, `cls.MODEL_ID` or
                      values from `pretrained_model_path` will be used.
            profile_path: Optional runtime profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional acquisition cache root directory.
            hf_models_root: Optional Hugging Face cache root directory.
            command_template: Optional external command template.
            **kwargs: Additional runtime or planning options which will be merged with other options.

        Returns:
            StarVLASynthesis: An initialized instance of the StarVLASynthesis class.
        """

        del args  # `args` is an unused compatibility parameter, intentionally deleted.
        # Normalize `pretrained_model_path` into a dictionary of options.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            # If `pretrained_model_path` is a string/Path, treat it as the checkpoint directory.
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge additional keyword arguments into the options dictionary.
        options.update(kwargs)

        # Determine the effective model ID by checking various sources in order of precedence.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and paths.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        # Instantiate and return the StarVLASynthesis object.
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),  # Default to 'cuda' if not specified.
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> StarVLARuntimeConfig:
        """
        Resolve StarVLA checkpoint and architecture settings into a `StarVLARuntimeConfig`.

        This method merges options from the instance's `runtime_options`,
        default runtime configurations, and per-call `options` to produce
        a final, explicit configuration for the StarVLA model.

        Args:
            options: Per-call runtime options that may override profile defaults
                     or instance-level `runtime_options`.

        Returns:
            StarVLARuntimeConfig: A dataclass containing the fully resolved
                                  configuration for the StarVLA runtime.
        """

        # Combine instance-level runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configurations for the model ID.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge all options: defaults, instance options, then explicit call options.
        merged = {**runtime_defaults, **explicit_options}

        # Resolve the checkpoint directory path from multiple possible keys.
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        track = str(merged["track"])
        # Resolve the variant ID from multiple possible keys.
        variant_id = str(
            explicit_options.get("variant_id")
            or explicit_options.get("variant")
            or explicit_options.get("starvla_variant")
            or merged.get("variant_id")
            or ""
        )
        # Select the appropriate StarVLA checkpoint based on resolved parameters.
        checkpoint = select_starvla_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
            variant_id=variant_id,
            track=track,
        )

        # Construct and return the `StarVLARuntimeConfig` object.
        return StarVLARuntimeConfig(
            checkpoint_dir=checkpoint,
            base_vlm=str(
                select_starvla_base_vlm(
                    base_vlm=merged.get("base_vlm"),
                    checkpoints=self.profile.checkpoints,
                )
            ),
            action_model_type=str(merged["action_model_type"]),
            action_dim=int(merged["action_dim"]),
            action_horizon=int(merged["action_horizon"]),
            device=str(merged.get("device") or self.device),  # Use explicit device or instance device.
            torch_dtype=str(merged.get("torch_dtype") or "auto"),
            track=track,
            attn_implementation=str(merged["attn_implementation"]),
            enable_official_runtime=_as_bool(
                _first_option(
                    explicit_options,
                    merged,
                    "enable_checkpoint_runtime",
                    "enable_official_runtime",
                    "official_runtime",
                    default=True,
                )
            ),
            base_world_model=select_starvla_base_world_model(
                base_world_model=merged.get("base_world_model"),
                checkpoints=self.profile.checkpoints,
            ),
            unnorm_key=str(merged["unnorm_key"]) if merged.get("unnorm_key") else None,
            action_normalization=str(merged.get("action_normalization") or "min_max"),
        )

    def _runtime_for(self, config: StarVLARuntimeConfig) -> StarVLAPlanRuntime:
        """
        Return a cached StarVLA plan runtime instance, creating it if necessary.

        This method ensures that a `StarVLAPlanRuntime` is instantiated only once
        for a given configuration, improving performance by reusing existing runtimes.

        Args:
            config: Resolved StarVLA runtime configuration.

        Returns:
            StarVLAPlanRuntime: An active instance of the StarVLA plan runtime.
        """

        # Create a unique key from the configuration to identify the runtime.
        key = (
            str(config.checkpoint_dir),
            config.base_vlm,
            config.action_model_type,
            config.device,
            config.torch_dtype,
            config.track,
            config.action_dim,
            config.action_horizon,
            config.attn_implementation,
            config.enable_official_runtime,
            config.base_world_model or "",
            config.unnorm_key or "",
            config.action_normalization,
        )
        # If no runtime is cached, or the configuration has changed, create a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = StarVLAPlanRuntime(config)
            self._runtime_key = key
        return self._runtime

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
        Prepare and optionally execute a StarVLA in-tree runtime plan for action prediction.

        This method generates a plan based on the provided inputs and configuration.
        If `plan_only` is false, it then executes the plan to predict actions.

        Args:
            prompt: Natural-language task instruction for the model.
            images: RGB observation image path or array, representing the current visual state.
            video: Optional video/world context, providing temporal information.
            interactions: Operator-provided action context, e.g., previous actions or demonstrations.
            output_path: Optional path where the action trace or results will be saved.
            fps: Optional video frame rate for context, if video is provided.
            timeout_seconds: Unused compatibility timeout parameter, intentionally ignored.
            **kwargs: Additional runtime and planning options, which can override
                      instance-level and default configurations.

        Returns:
            dict[str, Any]: A dictionary containing the status, model ID, artifact paths,
                            run directory, plan path, runtime details, and the resolved
                            profile (if `plan_only` is true) or the full prediction result.
        """

        del timeout_seconds  # `timeout_seconds` is an unused compatibility parameter.

        # Determine if only a plan should be generated or if prediction should also occur.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Resolve the run directory, defaulting to a temporary directory if not provided.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="starvla_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the context dictionary for the runtime plan.
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

        # Resolve runtime options and configuration.
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        runtime_config = self._runtime_config(runtime_options)

        # Define the path for the runtime plan file.
        plan_path = run_dir / "runtime_profile_plan.json"

        # Write the runtime plan to disk and get its payload.
        plan_payload = self._runtime_for(runtime_config).write_plan(
            context=context,
            profile=self.profile.to_dict(),
            runtime_options=runtime_options,
            plan_path=plan_path,
        )

        if plan_only:
            # If only planning is requested, return the planning details.
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "runtime": plan_payload["runtime"]["backend"],
                "profile": self.profile.to_dict(),
            }

        # If full prediction is requested, execute the action prediction.
        result = self._runtime_for(runtime_config).predict_action(
            prompt=prompt,
            images=images,
            state=kwargs.get("state"),
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
            },
        )
        # Add run directory, plan path, and profile to the result.
        result["run_dir"] = str(run_dir)
        result["plan_path"] = str(plan_path)
        result["profile"] = self.profile.to_dict()
        return result
