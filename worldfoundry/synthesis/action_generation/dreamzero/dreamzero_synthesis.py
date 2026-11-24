"""
This module defines a wrapper for the DreamZero action synthesis model, designed for in-tree execution within the WorldFoundry framework.

It handles configuration, plan generation, and interaction with a DreamZero server for generating actions based on prompts and other inputs.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.dreamzero.runtime import (
    DreamZeroRuntimeConfig,
    build_server_command,
    describe_in_tree_runtime,
    run_default_client_demo,
    select_dreamzero_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class DreamZeroSynthesis(ActionModelSynthesis):
    """A wrapper for the DreamZero action synthesis model, designed for in-tree execution within the WorldFoundry framework.

    This class extends `ActionModelSynthesis` to provide specific logic for configuring,
    planning, and executing DreamZero model inference, primarily for local development
    and testing scenarios where the model is run directly from its source tree.
    """

    MODEL_ID = "dreamzero"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a lazy in-tree DreamZero synthesis wrapper.

        Initializes the DreamZero synthesis wrapper, setting up the base configuration
        and storing DreamZero-specific runtime options.

        Args:
            profile: WorldFoundry runtime profile containing model metadata.
            device: Target execution device label (e.g., "cuda", "cpu").
            command_template: Optional command override for starting the DreamZero server.
            env: Optional runtime environment variables to be passed to the server process.
            runtime_options: DreamZero checkpoint and server options specific to this instance.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        # Store DreamZero-specific runtime configuration options.
        self.runtime_options = dict(runtime_options or {})

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
    ) -> DreamZeroSynthesis:
        """Create a DreamZero wrapper without importing model-heavy deps.

        This factory method simplifies the creation of a DreamZeroSynthesis instance by
        resolving configuration from a variety of sources, including a pretrained model path
        or a mapping of options, and loading the appropriate runtime profile.

        Args:
            pretrained_model_path: Optional checkpoint directory or options mapping.
            args: Unused compatibility argument.
            device: Target execution device label.
            model_id: Runtime profile id.
            profile_path: Optional profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional source checkout cache root.
            hf_models_root: Optional Hugging Face checkpoint root.
            command_template: Optional command override.
            **kwargs: Additional options that will be merged into the runtime configuration.
        """
        # `args` is an unused compatibility argument and is explicitly deleted.
        del args
        # Initialize options from `pretrained_model_path` if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If `pretrained_model_path` is a path (not a mapping), treat it as the checkpoint directory.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options, allowing kwargs to override previous settings.
        options.update(kwargs)

        # Resolve the model ID, prioritizing explicit `model_id` or `profile_id` from options,
        # falling back to the `model_id` argument, and finally to the class's default `MODEL_ID`.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and any provided path overrides.
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

    def _runtime_config(self, options: Mapping[str, Any], *, require_checkpoint: bool) -> DreamZeroRuntimeConfig:
        """Resolve DreamZero checkpoint and server settings.

        This method merges instance-level runtime options with call-specific options and
        default configurations to produce a complete `DreamZeroRuntimeConfig`. It also
        selects the appropriate DreamZero checkpoint.

        Args:
            options: Per-call runtime options that override instance-level or default settings.
            require_checkpoint: Whether the checkpoint must exist locally for the configuration to be valid.

        Returns:
            A `DreamZeroRuntimeConfig` object containing the resolved settings.
        """
        # Combine instance-level runtime options with call-specific options, with call-specific
        # options taking precedence.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configuration specific to VLA/VA/WAM models, potentially from a custom path.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge runtime defaults with explicit options, ensuring explicit options override defaults.
        merged = {**runtime_defaults, **explicit_options}

        # Select the DreamZero checkpoint, prioritizing explicit paths from options,
        # then using the profile's checkpoints, and finally considering a variant.
        checkpoint = select_dreamzero_checkpoint(
            checkpoint_dir=(
                explicit_options.get("checkpoint_dir")
                or explicit_options.get("ckpt_path")
                or explicit_options.get("model_path")
                or explicit_options.get("pretrained_model_path")
            ),
            checkpoints=self.profile.checkpoints,
            variant=str(merged.get("variant") or ""),
            require_exists=require_checkpoint,
        )

        # Construct the DreamZeroRuntimeConfig object, resolving host, port, and other
        # parameters with fallback logic from explicit options, then runtime defaults.
        return DreamZeroRuntimeConfig(
            checkpoint_dir=checkpoint,
            host=str(
                explicit_options.get("dreamzero_server_host")
                or explicit_options.get("server_host")
                or explicit_options.get("host")
                or runtime_defaults["server_host"]
            ),
            port=int(
                explicit_options.get("dreamzero_server_port")
                or explicit_options.get("server_port")
                or explicit_options.get("port")
                or runtime_defaults["server_port"]
            ),
            nproc_per_node=int(merged["nproc_per_node"]),
            enable_dit_cache=bool(merged["enable_dit_cache"]),
            max_chunk_size=merged.get("max_chunk_size"),
            client_demo=dict(merged.get("client_demo") or {}),
        )

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
        """Prepare DreamZero in-tree runtime plan and optionally call the server client.

        This method orchestrates the DreamZero action synthesis process. It first generates
        a runtime plan detailing how the model should be executed. Depending on the `plan_only`
        option, it can either return the plan details or proceed to launch a client
        to interact with a DreamZero server for actual inference.

        Args:
            prompt: Language instruction for the action synthesis.
            images: Optional image inputs to be recorded in the plan.
            video: Optional video inputs to be recorded in the plan.
            interactions: Optional prior action/state signals as a sequence of strings.
            output_path: Destination path for action-trace artifacts.
            fps: Optional video frames per second metadata.
            timeout_seconds: Reserved compatibility option (currently unused).
            **kwargs: Additional parameters, including 'plan_only', 'run_dir', and
                      DreamZero server host/port for live inference.

        Returns:
            A dictionary containing the status of the operation, path to artifacts,
            run directory, plan path, and profile details. If live inference is run,
            it includes results from the client demo.
        """
        # `timeout_seconds` is an unused compatibility argument and is explicitly deleted.
        del timeout_seconds

        # Determine if only a plan should be generated or if full inference should run.
        # This can be specified via kwargs or instance runtime options.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Resolve the run directory. If not provided in kwargs, a temporary directory is created.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="dreamzero_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Gather all relevant context for the prediction, including inputs and output paths.
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
        # Resolve the full DreamZero runtime configuration based on instance and call options.
        # If only planning, a checkpoint isn't strictly required to exist locally.
        runtime_config = self._runtime_config(
            kwargs,
            require_checkpoint=not plan_only,
        )
        # Build the command necessary to start the DreamZero server.
        server_command = build_server_command(python=str(context["python"]), config=runtime_config)
        # Describe the in-tree runtime environment to gather evidence like the server module path.
        runtime_evidence = describe_in_tree_runtime(runtime_config.checkpoint_dir)

        # Define the path where the runtime plan JSON file will be stored.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, containing schema, profile, context, and runtime details.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-dreamzero-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.dreamzero.in_tree_official_server_client",
                "backend_quality": "in_tree_official_runtime_server_client",
                "server_module": runtime_evidence["official_server_module"],
                "client_class": "worldfoundry.synthesis.action_generation.dreamzero.runtime.DreamZeroWebsocketClient",
                "server_command": server_command,
                **runtime_config.to_dict(),
                "evidence": runtime_evidence,
            },
        }
        # Write the generated plan payload to the specified plan file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If `plan_only` is true, return the plan details without initiating inference.
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

        # For live inference, resolve the server host and port, prioritizing kwargs.
        server_host = kwargs.pop("dreamzero_server_host", None) or kwargs.pop("server_host", None)
        server_port = kwargs.pop("dreamzero_server_port", None) or kwargs.pop("server_port", None)

        # If no server host is provided for live inference, raise an error, as the client cannot connect.
        if not server_host:
            raise RuntimeError(
                f"DreamZero in-tree runtime is planned at {plan_path}. "
                "Start the in-tree server_command, then pass dreamzero_server_host/server_host for live inference."
            )

        # Run the default client demo to interact with the DreamZero server and perform inference.
        result = run_default_client_demo(
            host=str(server_host),
            port=int(server_port or runtime_config.port),
            prompt=prompt,
            output_path=context["output_path"],
            model_id=self.model_id,
            artifact_kind=self.profile.artifact_kind,
            client_demo_config=runtime_config.client_demo,
            debug_video_dir=kwargs.pop("debug_video_dir", None),
            num_chunks=kwargs.pop("num_chunks", None),
            use_zero_images=bool(kwargs.pop("use_zero_images", False)),
            session_id=kwargs.pop("session_id", None),
            # Pass remaining kwargs and plan/server command for debugging/context.
            extra={**kwargs, "plan_path": str(plan_path), "server_command": server_command},
        )
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
