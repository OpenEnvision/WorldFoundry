"""
Module for providing an in-tree runtime for WorldCam video generation.

This module defines the `WorldCamRuntime` class, which acts as a facade for
loading the WorldCam model and its dependencies (Wan2.1-T2V-1.3B) and for
generating video based on prompts, conditioning videos, and camera trajectories.
It supports both direct model inference and the creation of lightweight execution plans.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.runtime.env import resolve_hfd_root

from .worldcam_runtime import runtime_root as worldcam_runtime_root

DEFAULT_WEIGHT_DTYPE = "bfloat16"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
DEFAULT_SHARED_HFD_ROOT = resolve_hfd_root()
DEFAULT_WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_WORLDCAM_REPO = "worldcam/worldcam"
DEFAULT_WAN_MODEL_DIR = DEFAULT_SHARED_HFD_ROOT / DEFAULT_WAN_REPO.replace("/", "--")
DEFAULT_WORLDCAM_CKPT_DIR = DEFAULT_SHARED_HFD_ROOT / DEFAULT_WORLDCAM_REPO.replace("/", "--")
DEFAULT_WORLDCAM_CHECKPOINT = "finetuned_dit.safetensors"
OFFICIAL_SOURCE_REPO = "https://github.com/cvlab-kaist/WorldCam"


def _hf_snapshot_dirs(root: Path) -> list[Path]:
    """Return Hugging Face cache snapshot dirs for an outer ``models--*`` root."""

    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _worldcam_checkpoint_roots(primary: str | Path | None = None) -> list[Path]:
    """Candidate local roots for the WorldCam checkpoint.

    Open-source users can pass the repo id and let Hugging Face download into the
    standard cache. Local release validation can also use pre-staged HFD or
    adjacent ``ckpt/worldcam`` folders without changing demo configs.
    """

    raw: list[Path] = []
    if primary not in {None, ""}:
        raw.append(Path(str(primary)).expanduser())
    raw.extend(
        [
            checkpoint_root_path("worldcam"),
            hfd_root_path("worldcam--worldcam"),
            hfd_root_path("custom--worldcam"),
            DEFAULT_WORLDCAM_CKPT_DIR,
            checkpoint_root_path("huggingface", "hub", "models--worldcam--worldcam"),
        ]
    )

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in raw:
        if root.is_file():
            root = root.parent
        expanded = [root, *_hf_snapshot_dirs(root)]
        for item in expanded:
            key = str(item)
            if key not in seen:
                candidates.append(item)
                seen.add(key)
    return candidates


def _require_torch():
    """
    Ensures that PyTorch is installed and importable.

    Raises:
        ModuleNotFoundError: If PyTorch cannot be imported.

    Returns:
        The torch module.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WorldCam generation requires torch. Install torch to load checkpoints or run predict()."
        ) from exc
    return torch


def _resolve_torch_dtype(weight_dtype: Any) -> Any:
    """
    Resolves a given weight dtype string to a PyTorch dtype object.

    Args:
        weight_dtype: The desired weight dtype (e.g., "bfloat16" or a torch.dtype object).

    Returns:
        The corresponding torch.dtype object.
    """
    torch = _require_torch()
    if weight_dtype is None or weight_dtype == DEFAULT_WEIGHT_DTYPE:
        return torch.bfloat16
    return weight_dtype


def _torch_no_grad(func):
    """
    Decorator that executes the wrapped function within a `torch.no_grad()` context.

    This prevents PyTorch from building a computational graph, reducing memory
    consumption and speeding up inference.

    Args:
        func: The function to be wrapped.

    Returns:
        A wrapper function that calls `func` inside `torch.no_grad()`.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        torch = _require_torch()
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


class WorldCamRuntime:
    """
    Facade for WorldCam video generation, providing capabilities to load the model
    or create execution plans.

    This class manages the configuration and loading of the WorldCam pipeline,
    which builds upon the Wan2.1-T2V-1.3B base model for video generation,
    and the fine-tuned WorldCam DiT checkpoint. It supports both direct
    video generation (`predict`) and the creation of lightweight JSON plans
    for later execution (`plan`).
    """

    def __init__(
        self,
        pipeline: Any | None,
        device: str = "cuda",
        weight_dtype: Any = DEFAULT_WEIGHT_DTYPE,
        prompt_prefix: str = "<A first-person shooter CS game> ",
        height: int = 480,
        width: int = 832,
        condition_latent_frames: int = 8,
        frames_per_latent: int = 4,
        pretrained_model_path: str | Path = DEFAULT_WAN_REPO,
        worldcam_ckpt_path: str | Path = DEFAULT_WORLDCAM_REPO,
        checkpoint_name: str = DEFAULT_WORLDCAM_CHECKPOINT,
    ):
        """
        Store a loaded WorldCam pipeline or a lightweight runtime plan wrapper.

        Args:
            pipeline: Loaded Wan video pipeline, or ``None`` for plan-only usage.
            device: Target torch device label.
            weight_dtype: Model weight dtype.
            prompt_prefix: Domain prefix prepended to generation prompts.
            height: Output frame height.
            width: Output frame width.
            condition_latent_frames: Number of latent conditioning frames.
            frames_per_latent: Decoded frames generated per autoregressive step.
            pretrained_model_path: Local Wan2.1 base model path or repo id.
            worldcam_ckpt_path: Local WorldCam checkpoint path/root or repo id.
            checkpoint_name: Expected fine-tuned DiT checkpoint filename.
        """
        self.pipeline = pipeline
        self.device = device
        self.weight_dtype = weight_dtype or DEFAULT_WEIGHT_DTYPE
        self.prompt_prefix = prompt_prefix
        self.height = int(height)
        self.width = int(width)
        self.condition_latent_frames = int(condition_latent_frames)
        self.frames_per_latent = int(frames_per_latent)
        # Calculate the total number of frames needed for conditioning and initial generation.
        self.warmup_output_frames = 1 + self.condition_latent_frames * self.frames_per_latent
        self.pretrained_model_path = str(pretrained_model_path)
        self.worldcam_ckpt_path = str(worldcam_ckpt_path)
        self.checkpoint_name = checkpoint_name

    @staticmethod
    def _default_cache_dir(path_or_repo: str | Path, repo_id: str, cache_dir: Path) -> Path | None:
        """
        Returns the default cache directory if the provided path matches the repo ID
        and the cache directory exists.

        Args:
            path_or_repo: The path or Hugging Face repo ID.
            repo_id: The expected Hugging Face repo ID.
            cache_dir: The default cache directory for the repo.

        Returns:
            The cache directory path if conditions are met, otherwise None.
        """
        value = str(path_or_repo)
        if value == repo_id and cache_dir.is_dir():
            return cache_dir
        return None

    @classmethod
    def _candidate_dir(
        cls,
        path_or_repo: str | Path,
        repo_id: str,
        cache_dir: Path,
    ) -> Path | None:
        """
        Resolves a candidate local directory for a model or checkpoint.

        Prioritizes an explicit local path, then checks the parent directory if
        a file is specified, and finally checks the default shared cache.

        Args:
            path_or_repo: Local path or known Hugging Face repo id.
            repo_id: Repo id that maps to the shared HFD cache.
            cache_dir: Shared cache directory for the repo id.

        Returns:
            A `Path` object to the resolved directory, or `None` if not found locally.
        """
        value = Path(str(path_or_repo)).expanduser()
        if value.is_dir():
            return value
        if value.is_file():
            return value.parent
        for snapshot in _hf_snapshot_dirs(value):
            return snapshot
        if str(path_or_repo) == DEFAULT_WORLDCAM_REPO:
            for candidate in _worldcam_checkpoint_roots():
                if candidate.is_dir():
                    return candidate
        # If the path_or_repo isn't an explicit path, check if it matches the default repo_id
        # and if the cache_dir is valid.
        return cls._default_cache_dir(path_or_repo, repo_id, cache_dir)

    @classmethod
    def _resolve_dir(
        cls,
        path_or_repo: str | Path,
        repo_id: str,
        cache_dir: Path,
        allow_download: bool = False,
    ) -> str:
        """
        Resolve a local model directory without relying on an external checkout.

        Args:
            path_or_repo: Local path or known Hugging Face repo id.
            repo_id: Repo id that maps to the shared HFD cache.
            cache_dir: Shared cache directory for the repo id.
            allow_download: Deprecated; runtime downloads are intentionally disabled.
        """
        candidate = cls._candidate_dir(path_or_repo, repo_id, cache_dir)
        if candidate is not None:
            return str(candidate)
        if allow_download:
            raise FileNotFoundError(
                "WorldCam runtime downloads are disabled. Stage the required assets under "
                f"{cache_dir} or pass an explicit local path."
            )
        raise FileNotFoundError(
            f"Unable to locate local WorldCam asset directory for {path_or_repo!r}. "
            f"Expected a local path or shared cache at {cache_dir}. Set allow_download=True to fetch it."
        )

    @staticmethod
    def _resolve_existing_path(candidates: Iterable[Path], description: str) -> str:
        """
        Finds the first existing path from a list of candidates.

        Args:
            candidates: An iterable of `Path` objects to check.
            description: A descriptive string for the asset being searched (for error messages).

        Returns:
            The string representation of the first existing `Path`.

        Raises:
            FileNotFoundError: If none of the candidate paths exist.
        """
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Unable to locate {description}. Checked: {[str(path) for path in candidates]}")

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        checkpoint_path_or_repo: str | Path,
        checkpoint_name: str = DEFAULT_WORLDCAM_CHECKPOINT,
        allow_download: bool = False,
    ) -> str:
        """
        Resolve the fine-tuned WorldCam DiT checkpoint.

        Args:
            checkpoint_path_or_repo: Local checkpoint file/root or ``worldcam/worldcam``.
            checkpoint_name: Expected checkpoint filename.
            allow_download: Permit Hugging Face download when no local checkpoint root exists.
        """
        checkpoint_path = Path(str(checkpoint_path_or_repo)).expanduser()
        if checkpoint_path.is_file():
            return str(checkpoint_path)

        # Resolve the root directory where the checkpoint might be located.
        root = Path(
            cls._resolve_dir(
                checkpoint_path_or_repo,
                DEFAULT_WORLDCAM_REPO,
                DEFAULT_WORLDCAM_CKPT_DIR,
                allow_download=allow_download,
            )
        )
        search_roots = _worldcam_checkpoint_roots(root)
        for search_root in search_roots:
            direct_candidates = [
                search_root / checkpoint_name,
                search_root / "weights" / checkpoint_name,
            ]
            for candidate in direct_candidates:
                if candidate.exists():
                    return str(candidate)

        for search_root in search_roots:
            recursive_matches = sorted(search_root.glob(f"**/{checkpoint_name}")) if search_root.is_dir() else []
            if recursive_matches:
                return str(recursive_matches[0])

        for search_root in search_roots:
            safetensors_matches = sorted(search_root.glob("**/*.safetensors")) if search_root.is_dir() else []
            if safetensors_matches:
                return str(safetensors_matches[0])

        raise FileNotFoundError(
            f"Unable to find WorldCam checkpoint under {root}. Expected {checkpoint_name}. "
            f"Checked: {[str(path) for path in search_roots]}. "
            "The official checkpoint is public at worldcam/worldcam, but checkpoint files are not vendored."
        )

    @classmethod
    def _find_checkpoint_path(
        cls,
        checkpoint_path_or_repo: str | Path,
        checkpoint_name: str = DEFAULT_WORLDCAM_CHECKPOINT,
    ) -> Path | None:
        """
        Attempts to find the WorldCam DiT checkpoint path without raising an error if not found.

        Args:
            checkpoint_path_or_repo: Local checkpoint file/root or ``worldcam/worldcam``.
            checkpoint_name: Expected checkpoint filename.

        Returns:
            A `Path` object to the resolved checkpoint file, or `None` if not found.
        """
        checkpoint_path = Path(str(checkpoint_path_or_repo)).expanduser()
        if checkpoint_path.is_file():
            return checkpoint_path.resolve()
        # Resolve the potential root directory for the checkpoint.
        root = cls._candidate_dir(checkpoint_path_or_repo, DEFAULT_WORLDCAM_REPO, DEFAULT_WORLDCAM_CKPT_DIR)
        if root is None:
            return None
        for search_root in _worldcam_checkpoint_roots(root):
            direct_candidates = [
                search_root / checkpoint_name,
                search_root / "weights" / checkpoint_name,
            ]
            for candidate in direct_candidates:
                if candidate.exists():
                    return candidate.resolve()
        for search_root in _worldcam_checkpoint_roots(root):
            recursive_matches = sorted(search_root.glob(f"**/{checkpoint_name}")) if search_root.is_dir() else []
            if recursive_matches:
                return recursive_matches[0].resolve()
        for search_root in _worldcam_checkpoint_roots(root):
            safetensors_matches = sorted(search_root.glob("**/*.safetensors")) if search_root.is_dir() else []
            if safetensors_matches:
                return safetensors_matches[0].resolve()
        return None

    @classmethod
    def _missing_wan_components(cls, pretrained_model_path: str | Path) -> list[str]:
        """
        Checks for the presence of essential Wan model components (text encoder, VAE, DiT).

        Args:
            pretrained_model_path: Local Wan2.1 base model path or repo id.

        Returns:
            A list of string paths for any missing required components.
        """
        # Resolve the base directory for the Wan model.
        base_root = cls._candidate_dir(pretrained_model_path, DEFAULT_WAN_REPO, DEFAULT_WAN_MODEL_DIR)
        if base_root is None:
            # If the base root itself cannot be resolved, the entire model is considered missing.
            return [str(DEFAULT_WAN_MODEL_DIR)]
        # Define expected paths for key model components.
        expected = [
            base_root / "models_t5_umt5-xxl-enc-bf16.pth",
            base_root / "Wan2.1_VAE.pth",
            base_root / "diffusion_pytorch_model.safetensors",
        ]
        # Define candidate paths for the tokenizer.
        tokenizer_candidates = [
            base_root / "google" / "umt5-xxl",
            base_root / "google",
        ]
        # Collect missing file paths.
        missing = [str(path) for path in expected if not path.exists()]
        # Check if any tokenizer candidate path exists.
        if not any(path.exists() for path in tokenizer_candidates):
            missing.append(str(tokenizer_candidates[0]))
        return missing

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = DEFAULT_WAN_REPO,
        worldcam_ckpt_path: str | Path = DEFAULT_WORLDCAM_REPO,
        device: str = "cuda",
        weight_dtype: Any = DEFAULT_WEIGHT_DTYPE,
        checkpoint_name: str = DEFAULT_WORLDCAM_CHECKPOINT,
        prompt_prefix: str = "<A first-person shooter CS game> ",
        height: int = 480,
        width: int = 832,
        load_model: bool = True,
        allow_download: bool = False,
        **kwargs,
    ) -> WorldCamRuntime:
        """
        Build WorldCam from local in-tree runtime code and local model assets.

        Args:
            pretrained_model_path: Wan2.1 base model path, repo id, or options mapping.
            worldcam_ckpt_path: WorldCam checkpoint file/root or repo id.
            device: Target torch device label.
            weight_dtype: Model weight dtype.
            checkpoint_name: Fine-tuned DiT checkpoint filename.
            prompt_prefix: Domain prefix prepended to prompts.
            height: Output frame height.
            width: Output frame width.
            load_model: Load heavy checkpoints when true; build plan wrapper when false.
            allow_download: Permit Hugging Face downloads for missing local assets.
            **kwargs: Optional path aliases such as ``wan_model_path`` and ``checkpoint_path``.
        """
        # Normalize input 'pretrained_model_path' to an options dictionary if it's a mapping.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If 'pretrained_model_path' was not a mapping, set it in the options.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["pretrained_model_path"] = pretrained_model_path
        # Resolve the base model path using various aliases and fallback to default.
        base_model_path = (
            options.get("pretrained_model_path")
            or options.get("wan_model_path")
            or options.get("base_model_path")
            or DEFAULT_WAN_REPO
        )
        # Apply any additional keyword arguments to the options.
        options.update(kwargs)
        # Resolve the WorldCam checkpoint root using various aliases and fallback to input.
        checkpoint_root = (
            options.get("worldcam_ckpt_path")
            or options.get("checkpoint_path")
            or options.get("ckpt_path")
            or options.get("model_path")
            or worldcam_ckpt_path
        )
        # Override load_model and allow_download flags if present in options.
        if "load_model" in options:
            load_model = bool(options["load_model"])
        if "allow_download" in options:
            allow_download = bool(options["allow_download"])

        # If 'load_model' is false, return a lightweight runtime wrapper without loading actual models.
        if not load_model:
            return cls(
                pipeline=None,
                device=device,
                weight_dtype=weight_dtype,
                prompt_prefix=prompt_prefix,
                height=height,
                width=width,
                pretrained_model_path=str(base_model_path),
                worldcam_ckpt_path=str(checkpoint_root),
                checkpoint_name=checkpoint_name,
            )

        # Resolve the PyTorch dtype for model weights.
        weight_dtype = _resolve_torch_dtype(weight_dtype)
        # Import necessary components for pipeline loading.
        from worldfoundry.core.model_loading import ModelConfig, load_state_dict
        from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.pipelines.wan_video_new import (
            WanVideoPipeline,
        )

        # Resolve the actual local paths for the base model and WorldCam checkpoint.
        base_root = Path(
            cls._resolve_dir(
                base_model_path,
                DEFAULT_WAN_REPO,
                DEFAULT_WAN_MODEL_DIR,
                allow_download=allow_download,
            )
        )
        checkpoint_path = cls._resolve_checkpoint_path(
            checkpoint_root,
            checkpoint_name=checkpoint_name,
            allow_download=allow_download,
        )

        # Resolve the tokenizer directory path.
        tokenizer_path = cls._resolve_existing_path(
            [
                base_root / "google" / "umt5-xxl",
                base_root / "google",
            ],
            "Wan tokenizer directory",
        )

        # Define model configurations for the base Wan pipeline components.
        base_model_configs = [
            ModelConfig(
                path=cls._resolve_existing_path(
                    [base_root / "models_t5_umt5-xxl-enc-bf16.pth"],
                    "Wan text encoder checkpoint",
                ),
                offload_device="cpu",
                offload_dtype=weight_dtype,
            ),
            ModelConfig(
                path=cls._resolve_existing_path(
                    [base_root / "Wan2.1_VAE.pth"],
                    "Wan VAE checkpoint",
                ),
                offload_device="cpu",
                offload_dtype=weight_dtype,
            ),
            ModelConfig(
                path=cls._resolve_existing_path(
                    [base_root / "diffusion_pytorch_model.safetensors"],
                    "Wan DiT checkpoint",
                ),
                offload_device="cpu",
                offload_dtype=weight_dtype,
            ),
        ]

        # Initialize the WanVideoPipeline with the base model configurations.
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=weight_dtype,
            device=device,
            model_configs=base_model_configs,
            tokenizer_config=ModelConfig(path=tokenizer_path),
        )

        # Load the fine-tuned WorldCam DiT checkpoint into the pipeline's DiT model.
        pipe.dit.load_state_dict(load_state_dict(checkpoint_path, device="cpu"), strict=True)
        # Move all pipeline components to the specified device and dtype.
        pipe.text_encoder.to(device, dtype=weight_dtype)
        pipe.vae.to(device, dtype=weight_dtype)
        pipe.dit.to(device, dtype=weight_dtype)
        pipe.to(device)
        pipe.to(dtype=weight_dtype)

        return cls(
            pipeline=pipe,
            device=device,
            weight_dtype=weight_dtype,
            prompt_prefix=prompt_prefix,
            height=height,
            width=width,
            pretrained_model_path=str(base_model_path),
            worldcam_ckpt_path=str(checkpoint_root),
            checkpoint_name=checkpoint_name,
        )

    @staticmethod
    def runtime_root() -> Path:
        """
        Returns the root path of the WorldCam runtime.
        """
        return worldcam_runtime_root()

    def plan(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write a lightweight WorldCam runtime execution plan.

        Args:
            prompt: Text prompt for world generation.
            images: Optional initial image context metadata.
            video: Optional conditioning video metadata.
            interactions: Camera/action commands preserved in the plan.
            output_path: Target JSON plan path.
            fps: Optional FPS metadata.
            **kwargs: Additional runtime options.
        """
        # Attempt to find the checkpoint path for inclusion in the plan.
        checkpoint_path = self._find_checkpoint_path(self.worldcam_ckpt_path, self.checkpoint_name)
        # Check for missing Wan components to determine the plan's status.
        missing_assets = self._missing_wan_components(self.pretrained_model_path)
        # If the WorldCam checkpoint wasn't found, add it to missing assets.
        if checkpoint_path is None:
            missing_assets.append(str(DEFAULT_WORLDCAM_CKPT_DIR / self.checkpoint_name))

        # Determine the target output path for the JSON plan.
        target = Path(output_path) if output_path is not None else Path.cwd() / "worldcam_plan.json"
        target = target.expanduser().resolve()
        # Ensure the parent directory for the output path exists.
        target.parent.mkdir(parents=True, exist_ok=True)

        # Construct the payload for the JSON plan.
        payload = {
            "status": "blocked" if missing_assets else "prepared",  # Status indicates if assets are missing.
            "model_id": "worldcam",
            "artifact_kind": "generated_world",
            "backend": "worldfoundry.worldcam.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self.runtime_root()),
            "official_source_repo": OFFICIAL_SOURCE_REPO,
            "pretrained_model_path": self.pretrained_model_path,
            "worldcam_ckpt_path": self.worldcam_ckpt_path,
            "resolved_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_name": self.checkpoint_name,
            "missing_assets": missing_assets,
            "prompt": prompt,
            "has_images": images is not None,
            "has_video": video is not None,
            "interactions": list(interactions),
            "fps": fps,
            "height": self.height,
            "width": self.width,
            "extra": {key: str(value) for key, value in kwargs.items()},
            "note": (
                "WorldCam runtime code is vendored in-tree. Full generation requires the public "
                "worldcam/worldcam fine-tuned DiT checkpoint and Wan2.1-T2V-1.3B base files in local cache."
            ),
        }
        # Write the plan payload to a JSON file.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**payload, "artifact_path": str(target)}

    @_torch_no_grad
    def predict(
        self,
        prompt: str,
        condition_video,
        intrinsics: Any,
        extrinsics: Any,
        num_ar_steps: int,
        negative_prompt: str | None = None,
        cfg_scale: float = 4.0,
        seed: int = 0,
        long_term_memory_start_step: int = 30,
        long_term_memory_num_clips: int = 4,
        long_term_memory_ref_indices: list[int] | None = None,
        attention_sink_inference: bool = False,
        trim_conditioning: bool = True,
        num_inference_steps: int = 50,
        tiled: bool = True,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Generate WorldCam video from conditioning frames and camera trajectory tensors.

        Args:
            prompt: Text prompt for the generated world rollout.
            condition_video: Conditioning video frames.
            intrinsics: Camera intrinsics tensor.
            extrinsics: Camera extrinsics tensor.
            num_ar_steps: Number of autoregressive rollout steps.
            negative_prompt: Optional negative prompt.
            cfg_scale: Classifier-free guidance scale.
            seed: Sampling seed.
            long_term_memory_start_step: AR step for memory retrieval.
            long_term_memory_num_clips: Number of retrieved memory clips.
            long_term_memory_ref_indices: Frame indices used as retrieval query.
            attention_sink_inference: Use the first conditioning frame as attention sink.
            trim_conditioning: Return only generated frames when true.
            num_inference_steps: Diffusion denoising steps.
            tiled: Enable tiled VAE processing.
            return_dict: Return metadata payload when true.
            **kwargs: Extra pipeline options.
        """
        # Ensure the model pipeline has been loaded before attempting generation.
        if self.pipeline is None:
            raise RuntimeError("WorldCam model is not loaded. Use from_pretrained(load_model=True) for generation.")
        torch = _require_torch()

        # Compose the final prompt by prepending the instance's prompt prefix.
        composed_prompt = (self.prompt_prefix or "") + (prompt or "")
        composed_prompt = composed_prompt.strip()

        # Invoke the WorldCam pipeline for video generation.
        full_video = self.pipeline(
            prompt=composed_prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            input_video=condition_video,
            intrinsics=intrinsics.to(device=self.device, dtype=torch.float32),
            extrinsics=extrinsics.to(device=self.device, dtype=torch.float32),
            cfg_scale=cfg_scale,
            seed=seed,
            tiled=tiled,
            height=self.height,
            width=self.width,
            num_frames=self.warmup_output_frames + num_ar_steps * self.frames_per_latent,  # Total frames to generate.
            num_inference_steps=num_inference_steps,
            num_ar_steps=num_ar_steps,
            long_term_memory_start_step=long_term_memory_start_step,
            long_term_memory_num_clips=long_term_memory_num_clips,
            long_term_memory_ref_indices=long_term_memory_ref_indices,
            attention_sink_inference=attention_sink_inference,
            **kwargs,
        )
        # Trim conditioning frames from the output if requested.
        video = full_video[self.warmup_output_frames :] if trim_conditioning else full_video

        # Prepare the result dictionary.
        result = {
            "video": video,
            "full_video": full_video,
            "warmup_output_frames": self.warmup_output_frames,
            "num_ar_steps": num_ar_steps,
        }
        # Return either the full dictionary or just the generated video based on `return_dict`.
        if return_dict:
            return result
        return result["video"]


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_SHARED_HFD_ROOT",
    "DEFAULT_WAN_MODEL_DIR",
    "DEFAULT_WAN_REPO",
    "DEFAULT_WEIGHT_DTYPE",
    "DEFAULT_WORLDCAM_CHECKPOINT",
    "DEFAULT_WORLDCAM_CKPT_DIR",
    "DEFAULT_WORLDCAM_REPO",
    "OFFICIAL_SOURCE_REPO",
    "WorldCamRuntime",
]
