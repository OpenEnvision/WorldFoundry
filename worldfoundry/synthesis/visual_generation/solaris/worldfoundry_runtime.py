"""
An adapter module for integrating the Solaris inference runtime into WorldFoundry.

This module provides the `SolarisRuntime` class, which acts as a wrapper around the
installed Solaris inference module. It handles path resolution, configuration,
and execution of the Solaris model for video generation based on specified
evaluation types and parameters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import package_module_command

from .runtime_env import (
    build_eval_dataset_overrides,
    build_inference_env,
    hydra_key_for_eval_type,
    normalize_eval_types,
    resolve_checkpoint_dir,
    resolve_eval_data_dir,
    resolve_jax_cache_dir,
    resolve_model_weights_path,
    resolve_output_dir,
    resolve_pretrained_model_dir,
    resolve_runtime_root,
)


def _visible_cuda_device_count() -> int:
    raw_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_devices and raw_devices not in {"-1", "none", "None"}:
        return max(1, len([item for item in raw_devices.split(",") if item.strip()]))
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return 1
    if result.returncode != 0:
        return 1
    count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))
    return max(1, count)


class SolarisRuntime:
    """WorldFoundry adapter around the installed Solaris inference module.

    This class provides a high-level interface to run Solaris inference,
    managing paths, configurations, and executing the installed Python module.
    """

    def __init__(
        self,
        runtime_root: str,
        pretrained_model_dir: str,
        eval_data_dir: str,
        output_dir: str,
        checkpoint_dir: str,
        jax_cache_dir: str,
        model_weights_path: str,
        *,
        python_executable: str | None = None,
        device: str = "cuda",
        defaults: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the SolarisRuntime instance with resolved paths and configurations.

        Args:
            runtime_root: Base directory for relative data paths and the subprocess working directory.
            pretrained_model_dir: Directory containing pre-trained model checkpoints.
            eval_data_dir: Directory containing evaluation datasets.
            output_dir: Directory where generated videos and logs will be stored.
            checkpoint_dir: Directory for storing model checkpoints (usually for training).
            jax_cache_dir: Directory for JAX compilation cache.
            model_weights_path: Explicit path to the model weights file (e.g., `.pkl`).
            python_executable: Path to the Python executable to use for running the inference module.
                               Defaults to `sys.executable`.
            device: The device to use for inference (e.g., "cuda", "cpu").
            defaults: A dictionary of default configuration parameters that can be overridden.
        """
        # Resolve all path inputs to absolute paths and store them as strings.
        self.runtime_root = str(Path(runtime_root).expanduser().resolve())
        self.pretrained_model_dir = str(Path(pretrained_model_dir).expanduser().resolve())
        self.eval_data_dir = str(Path(eval_data_dir).expanduser().resolve())
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        self.checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
        self.jax_cache_dir = str(Path(jax_cache_dir).expanduser().resolve())
        self.model_weights_path = str(Path(model_weights_path).expanduser().resolve())
        self.python_executable = python_executable or sys.executable
        self.device = device
        # Set default values for various parameters, which can be overridden by `defaults`.
        self.defaults = {
            "enable_jax_cache": False,
            "eval_num_samples": _visible_cuda_device_count(),
            "num_workers": 8,
            "num_frames_eval": 257,
        }
        if defaults:
            self.defaults.update(defaults)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | None = None,
        args=None,
        device: str | None = None,
        runtime_root: str | None = None,
        pretrained_model_dir: str | None = None,
        eval_data_dir: str | None = None,
        output_dir: str | None = None,
        checkpoint_dir: str | None = None,
        jax_cache_dir: str | None = None,
        model_weights_path: str | None = None,
        python_executable: str | None = None,
        enable_jax_cache: bool = False,
        eval_num_samples: int | None = None,
        num_workers: int = 8,
        num_frames_eval: int = 257,
        **kwargs,
    ) -> SolarisRuntime:
        """Factory method to create a `SolarisRuntime` instance with automatic path resolution.

        This method resolves asset directories relative to `runtime_root` and sets
        up default configurations. Inference code comes from the installed package.

        Args:
            pretrained_model_path: Unified loading argument; checkpoint locations are selected
                                   with pretrained_model_dir and model_weights_path.
            args: Legacy argument, currently ignored.
            device: The device to use for inference (e.g., "cuda", "cpu"). Defaults to "cuda".
            runtime_root: Data/work base; defaults to the configured environment or installed runtime directory.
            pretrained_model_dir: Directory containing pre-trained model checkpoints.
            eval_data_dir: Directory containing evaluation datasets.
            output_dir: Directory where generated videos and logs will be stored.
            checkpoint_dir: Directory for storing model checkpoints.
            jax_cache_dir: Directory for JAX compilation cache.
            model_weights_path: Explicit path to the model weights file (e.g., `.pkl`).
            python_executable: Path to the Python executable.
            enable_jax_cache: Whether to enable JAX compilation caching.
            eval_num_samples: Number of samples to evaluate.
            num_workers: Number of data loading workers.
            num_frames_eval: Number of frames to evaluate.
            **kwargs: Additional keyword arguments. Raises a ValueError if any unsupported kwargs are provided.

        Returns:
            An initialized `SolarisRuntime` instance.

        Raises:
            ValueError: If unsupported keyword arguments are provided.
        """
        del pretrained_model_path, args
        # Check for any unexpected keyword arguments to prevent silent misconfigurations.
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Solaris kwargs: {unknown}")

        # Resolve core runtime paths, deriving them if not explicitly provided.
        resolved_runtime_root = resolve_runtime_root(runtime_root)
        resolved_pretrained_model_dir = resolve_pretrained_model_dir(
            pretrained_model_dir,
            resolved_runtime_root,
        )
        resolved_eval_data_dir = resolve_eval_data_dir(
            eval_data_dir,
            resolved_runtime_root,
        )
        resolved_output_dir = resolve_output_dir(
            output_dir,
            resolved_runtime_root,
        )
        resolved_checkpoint_dir = resolve_checkpoint_dir(
            checkpoint_dir,
            resolved_runtime_root,
        )
        resolved_jax_cache_dir = resolve_jax_cache_dir(
            jax_cache_dir,
            resolved_runtime_root,
        )
        # Resolve the specific model weights path, potentially based on runtime and pretrained dirs.
        resolved_model_weights_path = resolve_model_weights_path(
            model_weights_path,
            resolved_runtime_root,
            resolved_pretrained_model_dir,
        )

        # Collect default parameter values for the `__init__` method.
        defaults = {
            "enable_jax_cache": bool(enable_jax_cache),
            "eval_num_samples": int(eval_num_samples) if eval_num_samples is not None else _visible_cuda_device_count(),
            "num_workers": int(num_workers),
            "num_frames_eval": int(num_frames_eval),
        }

        return cls(
            runtime_root=resolved_runtime_root,
            pretrained_model_dir=resolved_pretrained_model_dir,
            eval_data_dir=resolved_eval_data_dir,
            output_dir=resolved_output_dir,
            checkpoint_dir=resolved_checkpoint_dir,
            jax_cache_dir=resolved_jax_cache_dir,
            model_weights_path=resolved_model_weights_path,
            python_executable=python_executable,
            device=device or "cuda",
            defaults=defaults,
        )

    def _build_inference_command(
        self,
        *,
        experiment_name: str,
        eval_types: list[str],
        output_dir: str,
        checkpoint_dir: str,
        jax_cache_dir: str,
        model_weights_path: str,
        eval_num_samples: int,
        num_workers: int,
        num_frames_eval: int,
        enable_jax_cache: bool,
    ) -> list[str]:
        """Constructs the command-line arguments for the Solaris inference module.

        Args:
            experiment_name: Name of the experiment, used for organizing output.
            eval_types: A list of evaluation types (e.g., "recons", "interp").
            output_dir: Base directory for all inference outputs.
            checkpoint_dir: Directory for model checkpoints.
            jax_cache_dir: Directory for JAX compilation cache.
            model_weights_path: Path to the specific model weights file.
            eval_num_samples: Number of samples to generate per evaluation.
            num_workers: Number of data loading workers.
            num_frames_eval: Number of frames to evaluate for each video.
            enable_jax_cache: Whether JAX caching is enabled.

        Returns:
            A list of strings representing the full command for `subprocess.run`.
        """
        command = [
            f"experiment_name={experiment_name}",
            f"device.data_dir={self.eval_data_dir}",
            f"device.eval_data_dir={self.eval_data_dir}",
            f"device.pretrained_model_dir={self.pretrained_model_dir}",
            f"device.output_dir={output_dir}",
            f"device.checkpoint_dir={checkpoint_dir}",
            f"device.jax_cache_dir={jax_cache_dir}",
            f"device.eval_num_samples={int(eval_num_samples)}",
            f"device.num_workers={int(num_workers)}",
            f"runner.num_frames_eval={int(num_frames_eval)}",
            f"runner.params.model_weights_path={model_weights_path}",
            f"enable_jax_cache={'true' if enable_jax_cache else 'false'}",
        ]
        # Append specific Hydra overrides for each evaluation type to the command.
        command.extend(build_eval_dataset_overrides(eval_types))
        return package_module_command(
            "worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.inference",
            command,
            python_executable=self.python_executable,
        )

    def _collect_generated_videos(
        self,
        model_output_dir: Path,
        eval_types: list[str],
    ) -> dict[str, list[str]]:
        """Collects paths to generated video files from the model output directory.

        Args:
            model_output_dir: The specific output directory for the current model/experiment.
            eval_types: A list of evaluation types that were run.

        Returns:
            A dictionary where keys are evaluation types and values are lists of
            absolute paths to the generated side-by-side MP4 video files.
        """
        generated_videos: dict[str, list[str]] = {}
        for eval_type in eval_types:
            # Determine the Hydra-specific directory name for the current evaluation type.
            hydra_key = hydra_key_for_eval_type(eval_type)
            eval_dir = model_output_dir / hydra_key
            # Search for side-by-side MP4 video files within the evaluation type's directory.
            generated_videos[eval_type] = [
                str(video_path.resolve()) for video_path in sorted(eval_dir.glob("video_*_side_by_side.mp4"))
            ]
        return generated_videos

    def predict(
        self,
        eval_types: object | None = None,
        *,
        experiment_name: str = "solaris_worldfoundry",
        output_dir: str | None = None,
        eval_num_samples: int | None = None,
        num_workers: int | None = None,
        num_frames_eval: int | None = None,
        enable_jax_cache: bool | None = None,
        checkpoint_dir: str | None = None,
        jax_cache_dir: str | None = None,
        model_weights_path: str | None = None,
        return_dict: bool = False,
        show_progress: bool = True,
    ):
        """Executes Solaris inference to generate videos based on specified parameters.

        Args:
            eval_types: A single evaluation type string or a list of type strings
                        (e.g., "recons", ["recons", "interp"]). If None, all default
                        evaluation types will be run.
            experiment_name: A unique name for this inference run, used to organize output files.
            output_dir: Override the default output directory for this prediction run.
            eval_num_samples: Override the number of samples to generate per evaluation.
            num_workers: Override the number of data loading workers.
            num_frames_eval: Override the number of frames to evaluate for each video.
            enable_jax_cache: Override whether to enable JAX compilation caching.
            checkpoint_dir: Override the directory for model checkpoints.
            jax_cache_dir: Override the directory for JAX compilation cache.
            model_weights_path: Override the explicit path to the model weights file.
            return_dict: If True, returns a dictionary containing detailed results.
                         If False, returns only the path to the model's output directory.
            show_progress: If True, inference module output is shown in console.
                           If False, output is suppressed.

        Returns:
            The path to the model's output directory (str) or a dictionary with
            detailed results, depending on `return_dict`.

        Raises:
            FileNotFoundError: If the specified model weights path does not exist.
            ValueError: If `eval_num_samples` is less than 1.
            RuntimeError: If the underlying Solaris inference subprocess fails.
        """
        # Normalize the input evaluation types into a consistent list format.
        selected_eval_types = normalize_eval_types(eval_types)
        # Resolve all paths, prioritizing method-specific overrides over instance defaults.
        resolved_output_dir = str(Path(output_dir or self.output_dir).expanduser().resolve())
        resolved_checkpoint_dir = str(Path(checkpoint_dir or self.checkpoint_dir).expanduser().resolve())
        resolved_jax_cache_dir = str(Path(jax_cache_dir or self.jax_cache_dir).expanduser().resolve())
        resolved_model_weights_path = str(Path(model_weights_path or self.model_weights_path).expanduser().resolve())
        # Verify that the model weights file actually exists before proceeding.
        if not Path(resolved_model_weights_path).exists():
            raise FileNotFoundError(f"Solaris model_weights_path not found: {resolved_model_weights_path}")
        # Determine the effective JAX cache setting, prioritizing method argument then instance default.
        effective_enable_jax_cache = bool(
            enable_jax_cache if enable_jax_cache is not None else self.defaults["enable_jax_cache"]
        )

        # Create necessary output directories if they don't exist.
        output_root = Path(resolved_output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        Path(resolved_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if effective_enable_jax_cache:
            Path(resolved_jax_cache_dir).mkdir(parents=True, exist_ok=True)

        model_output_dir = output_root / experiment_name
        # Determine effective parameter values, prioritizing method arguments over instance defaults.
        effective_eval_num_samples = int(
            eval_num_samples if eval_num_samples is not None else self.defaults["eval_num_samples"]
        )
        if effective_eval_num_samples < 1:
            raise ValueError("Solaris eval_num_samples must be >= 1.")
        visible_devices = _visible_cuda_device_count()
        if effective_eval_num_samples % visible_devices != 0:
            raise ValueError(
                "Solaris eval_num_samples must be divisible by the number of visible CUDA devices "
                f"({visible_devices}); got {effective_eval_num_samples}. Set CUDA_VISIBLE_DEVICES to a "
                "single GPU for one-sample evaluation, or use eval_num_samples equal to the GPU count."
            )
        effective_num_workers = int(num_workers if num_workers is not None else self.defaults["num_workers"])
        effective_num_frames_eval = int(
            num_frames_eval if num_frames_eval is not None else self.defaults["num_frames_eval"]
        )

        # Build the command for the Solaris inference module.
        command = self._build_inference_command(
            experiment_name=experiment_name,
            eval_types=selected_eval_types,
            output_dir=resolved_output_dir,
            checkpoint_dir=resolved_checkpoint_dir,
            jax_cache_dir=resolved_jax_cache_dir,
            model_weights_path=resolved_model_weights_path,
            eval_num_samples=effective_eval_num_samples,
            num_workers=effective_num_workers,
            num_frames_eval=effective_num_frames_eval,
            enable_jax_cache=effective_enable_jax_cache,
        )
        # Configure stdout and stderr for the subprocess based on `show_progress`.
        stdout = None if show_progress else subprocess.DEVNULL
        stderr = None if show_progress else subprocess.STDOUT

        try:
            # Build the environment variables required for the inference subprocess.
            inference_env = build_inference_env(
                self.device,
                python_executable=self.python_executable,
            )
            # Execute the installed module from the configured data/work directory.
            subprocess.run(
                command,
                check=True,  # Raise CalledProcessError if the command returns a non-zero exit code.
                cwd=self.runtime_root,  # Run the command from the Solaris data/work directory.
                env=inference_env,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Solaris inference failed. Check the in-tree Solaris runtime, dataset paths, "
                "checkpoint layout, and its Python environment."
            ) from error

        # After successful inference, collect the paths to the generated videos.
        generated_videos = self._collect_generated_videos(
            model_output_dir,
            selected_eval_types,
        )
        # Find the path to the first generated video, if any, to use as a primary return value.
        primary_video = next(
            (videos[0] for videos in generated_videos.values() if videos),
            "",
        )
        # Construct the detailed result dictionary.
        result: dict[str, Any] = {
            "runtime_root": self.runtime_root,
            "experiment_name": experiment_name,
            "selected_eval_types": selected_eval_types,
            "output_root": str(output_root),
            "model_output_dir": str(model_output_dir),
            "generated_videos": generated_videos,
        }
        if primary_video:
            result["generated_video_path"] = primary_video

        # Return either the full dictionary or just the model output directory path.
        if return_dict:
            return result
        return result["model_output_dir"]


__all__ = ["SolarisRuntime"]
