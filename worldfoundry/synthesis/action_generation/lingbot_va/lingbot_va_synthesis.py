from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import project_root
from worldfoundry.synthesis.action_generation.lingbot_va.runtime import (
    RUNTIME_ROOT,
    WAN_VA_PACKAGE,
    LingBotVARuntimeConfig,
    LingBotVAWebsocketRuntime,
    build_server_command,
    config_name_for_checkpoint,
    read_transformer_attn_mode,
    select_lingbot_va_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


def _string_mapping(value: Any, *, name: str) -> dict[str, str]:
    """Converts a mapping's keys and values to strings.

    Args:
        value: The mapping to convert.
        name: The name of the mapping, used for error messages.

    Returns:
        A new dictionary with all keys and values converted to strings.

    Raises:
        TypeError: If `value` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): str(item) for key, item in value.items()}


class LingBotVASynthesis(ActionModelSynthesis):
    """
    A wrapper for LingBot-VA model synthesis, handling its runtime configuration,
    server management, and inference.
    """

    MODEL_ID = "lingbot-va"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a lazy in-tree LingBot-VA synthesis wrapper.

        Args:
            profile: WorldFoundry runtime profile.
            device: Target execution device label.
            command_template: Optional command override.
            env: Optional runtime environment overrides.
            runtime_options: LingBot-VA checkpoint/server options.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        # Private attributes for caching the LingBotVAWebsocketRuntime instance
        # and its configuration key to avoid redundant re-initialization.
        self._runtime: LingBotVAWebsocketRuntime | None = None
        self._runtime_key: tuple[str, str, str, int, int, int] | None = None

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
    ) -> LingBotVASynthesis:
        """Create a LingBot-VA wrapper without importing model-heavy deps.

        Args:
            pretrained_model_path: Optional checkpoint directory or options mapping.
            args: Unused compatibility argument.
            device: Target execution device label.
            model_id: Runtime profile id.
            profile_path: Optional profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional local source checkout cache root.
            hf_models_root: Optional Hugging Face checkpoint root.
            command_template: Optional command override.
        """
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as the checkpoint directory.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)
        # Resolve the model ID from multiple possible sources, prioritizing explicit options.
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
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any], *, require_checkpoint: bool = True) -> LingBotVARuntimeConfig:
        """Resolve LingBot-VA checkpoint and server settings.

        This method merges default runtime configurations, class-level `runtime_options`,
        and per-call `options` to produce a final `LingBotVARuntimeConfig`.

        Args:
            options: Per-call runtime options.
            require_checkpoint: Whether the selected checkpoint must exist locally.

        Returns:
            A `LingBotVARuntimeConfig` object with resolved settings.
        """
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configuration for the model ID.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options, explicit options take precedence.
        merged = {**runtime_defaults, **explicit_options}

        default_config_name = str(merged["default_config_name"])
        explicit_config_name = explicit_options.get("config_name")
        preferred_config_name = str(explicit_config_name or default_config_name)

        # Select the appropriate LingBot-VA checkpoint path.
        checkpoint = select_lingbot_va_checkpoint(
            checkpoint_dir=(
                explicit_options.get("checkpoint_dir")
                or explicit_options.get("ckpt_path")
                or explicit_options.get("pretrained_model_path")
            ),
            checkpoints=self.profile.checkpoints,
            config_name=preferred_config_name,
            require_exists=require_checkpoint,
        )
        # Resolve the final configuration name, potentially from the checkpoint's metadata.
        resolved_config_name = str(
            explicit_config_name
            or config_name_for_checkpoint(
                checkpoint,
                self.profile.checkpoints,
                config_by_role=_string_mapping(
                    merged["checkpoint_role_config_names"],
                    name="checkpoint_role_config_names",
                ),
                fallback=default_config_name,
            )
        )
        return LingBotVARuntimeConfig(
            checkpoint_dir=checkpoint,
            config_name=resolved_config_name,
            host=str(merged["server_host"]),
            port=int(merged["server_port"]),
            nproc_per_node=int(merged["nproc_per_node"]),
            master_port=int(merged["master_port"]),
        )

    def _runtime_for(self, config: LingBotVARuntimeConfig) -> LingBotVAWebsocketRuntime:
        """
        Provides a cached `LingBotVAWebsocketRuntime` instance based on the given configuration.
        A new runtime is created only if the configuration has changed.

        Args:
            config: The runtime configuration to use.

        Returns:
            An instance of `LingBotVAWebsocketRuntime`.
        """
        # Create a unique key from the configuration to identify if the runtime needs to be re-initialized.
        key = (
            str(config.checkpoint_dir),
            config.config_name,
            config.host,
            config.port,
            config.nproc_per_node,
            config.master_port,
        )
        # If no runtime is initialized or the configuration key has changed, create a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = LingBotVAWebsocketRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _server_env(wait_timeout_seconds: int) -> dict[str, str]:
        """
        Prepare environment variables for the in-tree LingBot-VA server.

        Args:
            wait_timeout_seconds: Timeout for the server to become ready.

        Returns:
            A dictionary of environment variables.
        """
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered for real-time logging.
        env["WORLDFOUNDRY_LINGBOT_VA_SERVER_WAIT_TIMEOUT"] = str(wait_timeout_seconds)
        return env

    @staticmethod
    def _stop_server_process(process: subprocess.Popen[Any] | None) -> None:
        """
        Gracefully stops a running subprocess, with a fallback to forceful termination.

        Args:
            process: The subprocess.Popen object to stop, or None if no process.
        """
        if process is None or process.poll() is not None:
            # If the process is None or has already exited, do nothing.
            return
        # Attempt to gracefully terminate the process.
        process.terminate()
        try:
            # Wait for the process to exit after termination.
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            # If it doesn't exit within the timeout, forcefully kill it.
            process.kill()
            process.wait(timeout=20)  # Wait again for the killed process.

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Extracts an observation mapping from keyword arguments, checking for multiple possible keys.

        Args:
            kwargs: The keyword arguments dictionary potentially containing an observation.

        Returns:
            The extracted observation mapping.

        Raises:
            ValueError: If no valid observation mapping is found.
        """
        # Check for 'observation' key.
        observation = kwargs.get("observation")
        if isinstance(observation, Mapping):
            return observation
        # Check for 'lingbot_va_observation' key.
        lingbot_observation = kwargs.get("lingbot_va_observation")
        if isinstance(lingbot_observation, Mapping):
            return lingbot_observation
        # Check for 'obs' key.
        obs = kwargs.get("obs")
        if isinstance(obs, Mapping):
            return obs
        raise ValueError("LingBot-VA live inference requires observation, lingbot_va_observation, or obs mapping.")

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
        """Prepare a LingBot-VA run plan and optionally call the server client.

        Args:
            prompt: Language instruction.
            images: Optional image inputs recorded in the plan.
            video: Optional video inputs recorded in the plan.
            interactions: Optional prior action/state signals.
            output_path: Action-trace artifact destination.
            fps: Optional video fps metadata.
            timeout_seconds: Upper bound used for waiting on the local server.

        Returns:
            A dictionary containing the results of the prediction or the run plan details.
        """
        # Determine if only a plan should be generated without running inference.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Prepare a temporary or specified directory for the run artifacts.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="lingbot_va_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate the context dictionary for the run.
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
        runtime_options.setdefault("device", self.device)
        # Resolve the full runtime configuration.
        runtime_config = self._runtime_config(
            runtime_options,
            require_checkpoint=not plan_only,
        )
        # Determine the root directory for saving official outputs.
        save_root = Path(runtime_options.get("save_root") or run_dir / "official_outputs").expanduser().resolve()
        # Construct the command used to launch the LingBot-VA server.
        server_command = build_server_command(
            python=str(context["python"]),
            # Default to ``python -m torch.distributed.run`` so the server
            # necessarily uses the same interpreter and CUDA/PyTorch ABI as
            # the Workspace process.  A PATH-resolved torchrun can belong to
            # an unrelated base conda environment; retain an explicit escape
            # hatch for deployments that intentionally provide one.
            torchrun=str(runtime_options.get("torchrun_executable") or ""),
            config=runtime_config,
            save_root=save_root,
        )
        # Determine the server wait timeout, capped at 900 seconds (15 minutes).
        wait_timeout_seconds = int(
            runtime_options.get("server_wait_timeout_seconds") or min(int(timeout_seconds or 900), 900)
        )
        # Check if the server should be automatically started.
        auto_start_server = bool(runtime_options.get("start_server", runtime_options.get("auto_start_server", True)))
        # Read the transformer attention mode from the checkpoint directory.
        attn_mode = read_transformer_attn_mode(runtime_config.checkpoint_dir)

        plan_path = run_dir / "runtime_profile_plan.json"
        server_log_path = run_dir / "lingbot_va_server.log"

        # Construct the detailed plan payload for inspection or external execution.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-lingbot-va-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.lingbot_va.in_tree_official_server_client",
                "runtime_root": str(RUNTIME_ROOT),
                "server_module": f"{WAN_VA_PACKAGE}.runtime",
                "client_class": f"{WAN_VA_PACKAGE}.websocket_client.WebsocketClientPolicy",
                "checkpoint_dir": str(runtime_config.checkpoint_dir),
                "config_name": runtime_config.config_name,
                "host": runtime_config.host,
                "port": runtime_config.port,
                "nproc_per_node": runtime_config.nproc_per_node,
                "master_port": runtime_config.master_port,
                "device": self.device,
                "transformer_attn_mode": attn_mode,
                "server_command": server_command,
                "auto_start_server": auto_start_server,
                "server_wait_timeout_seconds": wait_timeout_seconds,
                "server_log_path": str(server_log_path),
                "runtime_requirements": [
                    "A complete local checkpoint containing transformer, VAE, tokenizer, and text encoder assets.",
                    "One or more CUDA devices; torchrun launches the in-tree distributed inference server.",
                    "transformer/config.json must select a supported in-tree attention implementation.",
                ],
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if plan_only:
            # If plan_only is true, return the plan details without performing inference.
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # Extract the observation from kwargs, supporting multiple key names.
        observation = self._select_observation(kwargs)
        server_process: subprocess.Popen[Any] | None = None
        server_log_handle = None
        try:
            if auto_start_server:
                # Open a log file for the server process.
                server_log_handle = server_log_path.open("a", encoding="utf-8")
                # Launch the LingBot-VA server as a subprocess.
                server_process = subprocess.Popen(
                    server_command,
                    # Run from the repository root so the fully qualified
                    # WorldFoundry module resolves without PYTHONPATH surgery.
                    cwd=str(project_root()),
                    env=self._server_env(wait_timeout_seconds),
                    stdout=server_log_handle,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout for combined logging.
                    text=True,  # Ensure output is treated as text.
                )
            # Call the runtime's predict_action method to perform inference.
            result = self._runtime_for(runtime_config).predict_action(
                observation=observation,
                prompt=prompt,
                output_path=context["output_path"],
                extra_metadata={
                    "run_dir": str(run_dir),
                    "plan_path": str(plan_path),
                    "profile": self.profile.to_dict(),
                    "interactions": list(interactions),
                },
            )
        finally:
            # Ensure the server process is stopped and log handle is closed, even if an error occurs.
            self._stop_server_process(server_process)
            if server_log_handle is not None:
                server_log_handle.close()
        # Combine the inference result with additional run metadata.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
