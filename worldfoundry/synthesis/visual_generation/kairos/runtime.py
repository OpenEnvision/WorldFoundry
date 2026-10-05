"""
WorldFoundry in-tree official runtime wrapper for Kairos Sensenova.

This module provides an interface to interact with the Kairos Sensenova text-to-video
model, allowing for generation of videos from text prompts and optional image inputs.
It handles setting up the environment, preparing inputs, executing the Kairos
inference script as a subprocess, and processing its output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from worldfoundry.core.io.paths import checkpoint_root_path, package_data_path

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards, contorted human joints, objects "
    "floating against natural forces, abrupt shot changes"
)


def _path_values(*values: str | None) -> tuple[Path, ...]:
    """
    Converts a sequence of string paths (or None) into a tuple of expanded Path objects.

    Args:
        *values: Variable number of strings or None, representing file paths.

    Returns:
        A tuple of `Path` objects, with `None` or empty string values filtered out.
    """
    return tuple(Path(value).expanduser() for value in values if value)


def _first_existing(paths: Sequence[Path]) -> Path | None:
    """
    Finds and returns the first existing path from a sequence of `Path` objects.

    Args:
        paths: A sequence of `Path` objects to check for existence.

    Returns:
        The resolved `Path` of the first existing file/directory, or `None` if none exist.
    """
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _repo_candidates() -> tuple[Path, ...]:
    """
    Generates potential candidate paths for the Kairos Sensenova repository root.

    Returns:
        A tuple of `Path` objects representing possible repository locations.
    """
    return _path_values(
        os.environ.get("WORLDFOUNDRY_KAIROS_REPO_ROOT"),
        str(Path(__file__).resolve().parent / "kairos_runtime"),
    )


def _models_root_candidates() -> tuple[Path, ...]:
    """
    Generates potential candidate paths for the Kairos Sensenova models root directory.

    Returns:
        A tuple of `Path` objects representing possible model root locations.
    """
    values = [
        os.environ.get("WORLDFOUNDRY_KAIROS_MODELS_ROOT"),
        str(checkpoint_root_path("kairos-agi--kairos-sensenova-4B-480P-pretrained")),
        str(checkpoint_root_path("kairos-agi--kairos-sensenova-4B-720P")),
        str(checkpoint_root_path("kairos-sensenova")),
        str(checkpoint_root_path("Kairos-model")),
    ]
    return _path_values(*values)


def _default_repo_root() -> str:
    """
    Determines the default Kairos repository root path.

    It tries to find an existing repository from candidates; if none exist,
    it defaults to the first candidate.

    Returns:
        The string representation of the default Kairos repository root path.
    """
    candidates = _repo_candidates()
    existing = _first_existing(candidates)
    if existing is not None:
        return str(existing)
    raise FileNotFoundError(
        "Kairos Sensenova in-tree runtime was not found. Expected "
        "worldfoundry/synthesis/visual_generation/kairos/kairos_runtime, "
        "or provide WORLDFOUNDRY_KAIROS_REPO_ROOT/runtime_root."
    )


def _default_models_root() -> str:
    """
    Determines the default Kairos models root path.

    It tries to find an existing models directory from candidates; if none exist,
    it defaults to the first candidate.

    Returns:
        The string representation of the default Kairos models root path.
    """
    candidates = _models_root_candidates()
    return str(_first_existing(candidates) or candidates[0])


def _project_root() -> Path:
    env_root = os.environ.get("WORLDFOUNDRY_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[4]


def _default_config_path(runtime_root: str | Path, variant: str) -> str:
    """
    Determines the default Kairos configuration file path based on the runtime root and model variant.

    Args:
        runtime_root: The root directory of the Kairos runtime.
        variant: The specific model variant being used (e.g., "720p", "robot-480p-distilled").

    Returns:
        The string representation of the default configuration file path.
    """
    name = "kairos_4b_config.py" if str(variant).lower() in {"720p", "4b-720p", "pretrained"} else "kairos_4b_config_DMD.py"
    root = Path(runtime_root).expanduser()
    candidates = (
        root / "kairos" / "configs" / name,
        root / "Kairos" / "configs" / name,
        root / "configs" / name,
    )
    existing = _first_existing(candidates)
    return str(existing or package_data_path('models', 'runtime', 'configs', 'kairos') / name)


def _load_video_frames(video_path: str | Path) -> np.ndarray:
    """
    Loads all frames from a video file into a NumPy array.

    Args:
        video_path: The path to the video file.

    Returns:
        A NumPy array of video frames, typically with shape (num_frames, height, width, channels).

    Raises:
        ValueError: If no frames are found in the specified video file.
    """
    frames = [np.asarray(frame).astype(np.uint8) for frame in iio.imiter(str(video_path))]
    if not frames:
        # If the imiter yields no frames, the video might be corrupted or empty.
        raise ValueError(f"No frames found in Kairos output video: {video_path}")
    stacked = np.stack(frames, axis=0)
    if int(stacked.max()) == int(stacked.min()):
        raise ValueError(f"Kairos output video is constant-valued and likely invalid: {video_path}")
    return stacked


def _as_bool(value: Any) -> bool:
    """
    Converts a value to a boolean. Supports common string representations of true/false.

    Args:
        value: The value to convert.

    Returns:
        True if the value is boolean True or a case-insensitive match for "1", "true", "yes", "on".
        False otherwise.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _image_source(images: Any) -> Any:
    """
    Extracts a single image source from various possible input formats.

    If `images` is a list/tuple, it takes the first element. Otherwise, it returns `images` directly.

    Args:
        images: An image source, which can be None, a single image object, or a sequence of image objects.

    Returns:
        The extracted single image source, or None if the input was None or an empty sequence.
    """
    if images is None:
        return None
    if isinstance(images, (list, tuple)) and images:
        return images[0]
    return images


class KairosRuntime:
    """
    WorldFoundry in-tree official runtime wrapper for Kairos Sensenova.

    This class provides an interface to configure and run the Kairos Sensenova
    text-to-video generation model as a subprocess. It manages paths,
    configuration patching, input/output handling, and environment setup.
    """

    MODEL_ID = "kairos-sensenova"
    DISPLAY_NAME = "Kairos Sensenova"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        runtime_root: str | Path | None = None,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        variant: str = "robot-480p-distilled",
        device: str = "cuda",
        python_executable: str | None = None,
        torchrun_executable: str | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initializes the KairosRuntime instance.

        Args:
            model_id: The identifier for the model. Defaults to "kairos-sensenova".
            runtime_root: The root directory of the Kairos Sensenova repository.
                          If None, it tries to find a default.
            models_root: The directory where Kairos model checkpoints are stored.
                         If None, it tries to find a default.
            config_path: The path to the Kairos configuration file. If None, it determines a default
                         based on `runtime_root` and `variant`.
            variant: The specific model variant to use (e.g., "robot-480p-distilled", "720p").
            device: The compute device to use (e.g., "cuda", "cuda:0").
            python_executable: The path to the Python executable to use for subprocesses.
                               If None, `sys.executable` is used.
            torchrun_executable: Optional standalone ``torchrun`` command. When omitted,
                                 the current Python interpreter launches
                                 ``torch.distributed.run`` so subprocesses stay in the
                                 same environment.
            defaults: A dictionary of default parameters for video generation, which can be
                      overridden during `predict` calls.
        """
        self.model_id = str(model_id or self.MODEL_ID)
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "text_or_image_to_video"
        self.variant = str(variant)
        # Resolve and store paths for the Kairos runtime and model directories
        self.runtime_root = str(Path(runtime_root or _default_repo_root()).expanduser())
        self.models_root = str(Path(models_root or _default_models_root()).expanduser())
        self.config_path = str(config_path or _default_config_path(self.runtime_root, self.variant))
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.torchrun_executable = str(torchrun_executable) if torchrun_executable else None
        # Define default generation parameters that can be overridden
        self.defaults = {
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "seed": 0,
            "tiled": True,
            "height": 480,
            "width": 640,
            "num_frames": 81,
            "cfg_scale": 5,
            "use_prompt_rewriter": False,
            "save_fps": 16,
            "nproc_per_node": 1,
            "master_port": 29556,
            "run_manage_libs": False,
            "warmup": False,
            "enable_torch_compile": False,
        }
        if defaults:
            self.defaults.update(dict(defaults))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "KairosRuntime":
        """
        Alternative constructor to create a KairosRuntime instance from pretrained model information.

        This method allows for flexible input for model paths and other configuration options,
        mimicking common `from_pretrained` patterns.

        Args:
            pretrained_model_path: Can be a mapping of options or a string/Path directly
                                   specifying the models root.
            device: The compute device to use. Overrides any device specified in `pretrained_model_path`.
            model_id: The identifier for the model. Overrides any model ID specified.
            **kwargs: Additional keyword arguments to override or set configuration options.

        Returns:
            An initialized `KairosRuntime` instance.
        """
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["models_root"] = str(pretrained_model_path)
        options.update(kwargs)

        # Extract specific keys that should be part of the `defaults` dictionary
        defaults = {
            key: options.pop(key)
            for key in tuple(options)
            if key
            in {
                "negative_prompt",
                "seed",
                "tiled",
                "height",
                "width",
                "num_frames",
                "cfg_scale",
                "use_prompt_rewriter",
                "save_fps",
                "nproc_per_node",
                "master_port",
                "run_manage_libs",
                "warmup",
                "enable_torch_compile",
                "pretrained_dit",
                "text_encoder_path",
                "vae_path",
                "prompt_rewriter_path",
            }
        }
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            # Explicit runtime paths are part of the public loading contract and
            # must not be replaced merely because they are staged after pipeline
            # construction (or represented by a test/dry-run path).
            runtime_root=options.get("runtime_root") or options.get("repo_root"),
            models_root=options.get("models_root") or options.get("checkpoint_dir"),
            config_path=options.get("config_path"),
            variant=str(options.get("variant") or "robot-480p-distilled"),
            device=str(device or options.get("device") or "cuda"),
            python_executable=options.get("python_executable"),
            torchrun_executable=options.get("torchrun_executable"),
            defaults=defaults,
        )

    def runtime_plan(self, **overrides: Any) -> dict[str, Any]:
        """
        Provides a diagnostic plan of the Kairos runtime setup and configuration.

        This method checks for the existence of key directories and files and
        reports various configuration parameters, aiding in debugging and
        understanding the runtime environment.

        Args:
            **overrides: Optional parameters to temporarily override the instance's
                         defaults for planning purposes.

        Returns:
            A dictionary containing details about the runtime status, paths,
            model existence, and configuration.
        """
        options = {**self.defaults, **overrides}
        models_root = Path(self.models_root)
        return {
            "status": "prepared",
            "backend": "official_kairos_in_tree_runtime",
            "backend_quality": "official_in_tree_runtime",
            "model_id": self.model_id,
            "variant": self.variant,
            "runtime_root": self.runtime_root,
            "runtime_root_exists": Path(self.runtime_root).is_dir(),
            "models_root": self.models_root,
            "models_root_exists": models_root.is_dir(),
            "config_path": self.config_path,
            "config_exists": Path(self.config_path).is_file(),
            "robot_480p_distilled_exists": (
                models_root / "Kairos" / "models" / "robot" / "kairos-robot-4B-480P-16fps-distilled.safetensors"
            ).is_file(),
            "qwen_awq_exists": (models_root / "Qwen" / "Qwen2.5-VL-7B-Instruct-AWQ").is_dir(),
            "wan_vae_exists": (models_root / "Wan2.1-T2V-14B" / "Wan2.1_VAE.pth").is_file(),
            "nproc_per_node": int(options.get("nproc_per_node", 1)),
            "warmup": _as_bool(options.get("warmup", False)),
            "enable_torch_compile": _as_bool(options.get("enable_torch_compile", False)),
        }

    def _subprocess_env(self, options: Mapping[str, Any]) -> dict[str, str]:
        """
        Prepares the environment variables for the Kairos subprocess.

        This includes setting `PYTHONPATH`, `TOKENIZERS_PARALLELISM`, `PYTHONUNBUFFERED`,
        and `CUDA_VISIBLE_DEVICES` based on the instance configuration.

        Returns:
            A dictionary of environment variables to be used by the subprocess.
        """
        env = os.environ.copy()
        thirdparty_root = _project_root() / "thirdparty"
        env["WORLDFOUNDRY_THIRDPARTY_ROOT"] = str(thirdparty_root)
        # Add Kairos and WorldFoundry thirdparty roots to PYTHONPATH for in-tree runtime imports.
        pythonpath = [self.runtime_root, str(thirdparty_root), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
        # Suppress tokenizers parallelism warnings and ensure unbuffered Python output
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        # NumExpr otherwise detects every host core and emits an error when
        # its requested thread count exceeds the package safety cap.
        env.setdefault("NUMEXPR_MAX_THREADS", "64")
        env.setdefault("NUMEXPR_NUM_THREADS", "64")
        env["WORLDFOUNDRY_KAIROS_ENABLE_TORCH_COMPILE"] = (
            "1" if _as_bool(options.get("enable_torch_compile", False)) else "0"
        )
        # If a specific CUDA device is specified (e.g., "cuda:1"), set CUDA_VISIBLE_DEVICES
        if self.device.startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", self.device.split(":", 1)[1])
        return env

    def _materialize_image(self, temp_dir: Path, images: Any) -> str:
        """
        Converts various image inputs into a file path for the Kairos inference script.

        Args:
            temp_dir: The temporary directory where the image file should be saved.
            images: The image input, which can be a path, a PIL Image, or a NumPy array,
                    or a list/tuple containing one of these.

        Returns:
            The string path to the materialized image file, or an empty string if `images` is None.
        """
        source = _image_source(images)
        if source is None:
            return ""
        from PIL import Image

        image_path = temp_dir / "input_image.png"
        if isinstance(source, (str, os.PathLike)):
            # If source is a path, open it and convert to RGB
            image = Image.open(Path(source).expanduser()).convert("RGB")
        elif hasattr(source, "convert"):
            # If source is a PIL Image-like object, convert to RGB
            image = source.convert("RGB")
        else:
            # Assume source is a NumPy array or similar, convert to PIL Image and then RGB
            image = Image.fromarray(np.asarray(source).astype(np.uint8)).convert("RGB")
        image.save(image_path)
        return str(image_path)

    def _patched_config_path(self, temp_dir: Path, options: Mapping[str, Any]) -> Path:
        """
        Creates a patched version of the Kairos configuration file.

        This method reads the original configuration file, replaces the `KAIROS_MODEL_DIR`
        with the actual `models_root`, and appends additional configuration overrides
        from the `options` mapping. The patched config is saved to a temporary file.

        Args:
            temp_dir: The temporary directory where the patched configuration file will be saved.
            options: A mapping of generation options, potentially containing paths to
                     pretrained DIT, text encoder, VAE, or prompt rewriter models.

        Returns:
            The `Path` to the patched configuration file.
        """
        source = Path(self.config_path).expanduser().resolve()
        text = source.read_text(encoding="utf-8")
        models_root = str(Path(self.models_root).expanduser().resolve())

        # Replace the placeholder KAIROS_MODEL_DIR in the original config with the actual models_root
        text = text.replace("KAIROS_MODEL_DIR = 'models'", f"KAIROS_MODEL_DIR = {models_root!r}")
        text = text.replace('KAIROS_MODEL_DIR = "models"', f"KAIROS_MODEL_DIR = {models_root!r}")

        overrides: list[str] = ["", "# WorldFoundry runtime overrides"]
        # Append pipeline and model path overrides if provided in options
        if options.get("pretrained_dit"):
            overrides.append(f"pipeline['pretrained_dit'] = {str(options['pretrained_dit'])!r}")
        if options.get("text_encoder_path"):
            overrides.append(f"pipeline['pipeline_args']['text_encoder_path'] = {str(options['text_encoder_path'])!r}")
        if options.get("vae_path"):
            overrides.append(f"pipeline['pipeline_args']['vae_path'] = {str(options['vae_path'])!r}")
        if options.get("prompt_rewriter_path"):
            overrides.append(f"prompt_rewriter_path = {str(options['prompt_rewriter_path'])!r}")

        patched = temp_dir / "kairos_config.py"
        patched.write_text(text + "\n".join(overrides) + "\n", encoding="utf-8")
        return patched

    def _write_input_json(self, temp_dir: Path, prompt: str, images: Any, output_dir: Path, options: Mapping[str, Any]) -> Path:
        """
        Writes the input parameters for Kairos inference to a JSON file.

        This file is consumed by the Kairos inference script.

        Args:
            temp_dir: The temporary directory to save the JSON file.
            prompt: The text prompt for video generation.
            images: The image input (if any) to be materialized and referenced.
            output_dir: The directory where Kairos should save its output.
            options: A mapping of generation options, used to populate the JSON payload.

        Returns:
            The `Path` to the created JSON input file.
        """
        payload = {
            "prompt": prompt,
            "input_image": self._materialize_image(temp_dir, images),
            "negative_prompt": str(options.get("negative_prompt") or ""),
            "seed": int(options.get("seed", 0)),
            "tiled": _as_bool(options.get("tiled", True)),
            "height": int(options.get("height", 480)),
            "width": int(options.get("width", 640)),
            "num_frames": int(options.get("num_frames", 81)),
            "cfg_scale": float(options.get("cfg_scale", 5)),
            "use_prompt_rewriter": _as_bool(options.get("use_prompt_rewriter", False)),
            "warmup": _as_bool(options.get("warmup", False)),
            "output_dir": str(output_dir),
        }
        # Include optional keys only if they are present and not None in options
        for key in ("tea_cache_l1_thresh", "shift", "sample_solver"):
            if key in options and options[key] is not None:
                payload[key] = options[key]

        path = temp_dir / "kairos_input.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _run_manage_libs(self, env: dict[str, str]) -> None:
        """
        Executes the Kairos `manage_libs.py` script to handle third-party library dependencies.

        This is typically run once before inference to ensure all necessary
        sub-modules and libraries are correctly set up.

        Args:
            env: The environment variables to pass to the subprocess.
        """
        candidates = (
            Path(self.runtime_root) / "kairos" / "manage_libs.py",
            Path(self.runtime_root) / "kairos" / "third_party" / "manage_libs.py",
        )
        manage_libs = next((path for path in candidates if path.is_file()), None)
        if manage_libs is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"Kairos manage_libs.py not found; searched: {searched}")
        subprocess.run(
            [self.python_executable, str(manage_libs)],
            check=True,
            cwd=self.runtime_root,
            env=env,
        )

    def _command(self, input_json: Path, config_path: Path, options: Mapping[str, Any]) -> list[str]:
        """
        Constructs the `torchrun` command to launch the Kairos inference script.

        Args:
            input_json: The path to the JSON input file with generation parameters.
            config_path: The path to the patched Kairos configuration file.
            options: A mapping of generation options, including `nproc_per_node` and `master_port`.

        Returns:
            A list of strings representing the `torchrun` command and its arguments.
        """
        launcher = (
            [self.torchrun_executable]
            if self.torchrun_executable
            else [self.python_executable, "-m", "torch.distributed.run"]
        )
        return [
            *launcher,
            "--nnodes=1",
            "--master_port",
            str(int(options.get("master_port", 29556))),
            "--nproc-per-node",
            str(int(options.get("nproc_per_node", 1))),
            str((Path(self.runtime_root) / "examples" / "inference.py").resolve()),
            "--input_file",
            str(input_json),
            "--config_file",
            str(config_path),
        ]

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | np.ndarray:
        """
        Generates a video using the Kairos Sensenova model based on the provided prompt and images.

        This is the primary method for triggering video generation. It sets up temporary files,
        prepares the environment, executes the Kairos inference script as a subprocess,
        and processes its output.

        Args:
            prompt: The text prompt for video generation.
            images: Optional initial image(s) for image-to-video generation. Can be a path,
                    PIL Image, NumPy array, or a sequence thereof.
            video: This parameter is ignored but kept for API compatibility.
            interactions: A sequence of interactions; if the first element is an int, it's used as the seed.
            output_path: Optional path to save the generated video. If None, a temporary path is used.
            fps: This parameter is ignored (FPS is determined by Kairos config).
            return_dict: This parameter is ignored (always returns dict).
            show_progress: This parameter is ignored (subprocess output is always streamed).
            **kwargs: Additional keyword arguments to override generation parameters defined in `self.defaults`.

        Returns:
            A dictionary containing:
            - "artifact_path": The path to the generated video file.
            - "video": A NumPy array of the generated video frames.
            - "prompt": The prompt used for generation.
            - "model_id": The ID of the model used.
            - "runtime_plan": A dictionary detailing the runtime configuration.
            - "command": The `torchrun` command executed.

        Raises:
            RuntimeError: If the Kairos subprocess fails.
            FileNotFoundError: If Kairos does not produce an MP4 output.
            ValueError: If the generated video file contains no frames.
        """
        # Unused parameters as per WorldFoundry runtime interface, but kept for signature compatibility
        del video, fps, return_dict, show_progress
        options = {**self.defaults, **kwargs}

        # Handle `interactions` for setting the seed if not explicitly provided in kwargs
        if interactions and "seed" not in kwargs:
            first = list(interactions)[0]
            if isinstance(first, int):
                options["seed"] = first

        # Determine the final output path for the generated video
        output_target = Path(output_path).expanduser() if output_path is not None else None
        if output_target is None:
            output_target = Path(tempfile.mkdtemp(prefix="kairos_")) / "kairos.mp4"
        output_target = output_target.resolve()

        with tempfile.TemporaryDirectory(prefix="worldfoundry_kairos_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            output_dir = temp_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare input files and environment for the Kairos subprocess
            input_json = self._write_input_json(temp_dir, prompt, images, output_dir, options)
            config_path = self._patched_config_path(temp_dir, options)
            env = self._subprocess_env(options)

            # Run `manage_libs.py` if enabled in options to ensure dependencies are set up
            if _as_bool(options.get("run_manage_libs", True)):
                self._run_manage_libs(env)

            # Construct the full command for `torchrun`
            command = self._command(input_json, config_path, options)

            # Execute the Kairos inference script as a subprocess
            process = subprocess.Popen(
                command,
                cwd=str(self.runtime_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier logging
                text=True,  # Decode stdout/stderr as text
                bufsize=1,  # Line-buffered output
            )

            # Capture a tail of the output for debugging in case of failure
            output_tail: deque[str] = deque(maxlen=400)
            assert process.stdout is not None
            for line in process.stdout:
                output_tail.append(line.rstrip("\n"))
                sys.stdout.write(line)  # Stream output to main process stdout
                sys.stdout.flush()
            return_code = process.wait()

            # Check for subprocess errors
            if return_code != 0:
                tail = "\n".join(output_tail)
                raise RuntimeError(f"Kairos command failed with exit code {return_code}.\n{tail}")

            # Locate the generated video file
            generated = output_dir / "output.mp4"
            if not generated.is_file():
                # If "output.mp4" not found directly, search for any MP4 in the output directory
                candidates = sorted(output_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
                if not candidates:
                    raise FileNotFoundError(f"Kairos did not write an mp4 output under {output_dir}")
                generated = candidates[0]

            # Copy the generated video to the final output_target if it's different from the temporary path
            output_target.parent.mkdir(parents=True, exist_ok=True)
            if generated.resolve() != output_target:
                shutil.copy2(generated, output_target)

        # Load video frames and prepare the return dictionary
        frames = _load_video_frames(output_target)
        return {
            "artifact_path": str(output_target),
            "video": frames,
            "prompt": prompt,
            "model_id": self.model_id,
            "runtime_plan": self.runtime_plan(),
            "command": command,
        }


__all__ = ["KairosRuntime"]
