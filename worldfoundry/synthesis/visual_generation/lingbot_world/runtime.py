from __future__ import annotations

import os
import random
import sys
import tempfile
from inspect import signature
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime import (
    runtime_root as lingbot_runtime_root,
)

DEFAULT_LINGBOT_BASE_REPO = "robbyant/lingbot-world-base-cam"
DEFAULT_LINGBOT_FAST_REPO = "robbyant/lingbot-world-fast"
DEFAULT_LINGBOT_ACT_REPO = "robbyant/lingbot-world-base-act-preview"
DEFAULT_LINGBOT_HFD_ROOT = Path(__file__).resolve().parents[6] / "cache" / "hfd"
SUPPORTED_LINGBOT_TASKS = frozenset({"i2v-A14B"})


def _candidate_official_runtime_roots() -> tuple[Path, ...]:
    return tuple()


def _official_runtime_root() -> Optional[Path]:
    for root in _candidate_official_runtime_roots():
        if (root / "wan" / "image2video.py").is_file() and (root / "wan" / "configs").is_dir():
            return root
    return None


def _load_runtime_components():
    """Load the vendored in-tree LingBot runtime modules."""

    from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime.configs import WAN_CONFIGS
    from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime.image2video import WanI2V
    from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime.image2video_fast import WanI2VFast

    return WAN_CONFIGS, WanI2V, WanI2VFast


def _patch_fast_attention_fallback(use_fast: bool) -> None:
    """Route LingBot fast sequence parallel attention through SDPA when flash-attn is absent."""

    if not use_fast:
        return
    try:
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot import (
            attention as attention_module,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot import (
            fast as model_fast_module,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot import model as model_module
    except Exception:
        return

    has_flash = bool(
        getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)
        or getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False)
    )
    if has_flash:
        return

    model_module.flash_attention = attention_module.attention
    model_fast_module.flash_attention = attention_module.attention


class LingBotWorldRuntime:
    """In-tree LingBot World runtime facade around the vendored official runtime."""

    def __init__(
        self,
        core_model: Any,
        default_offload_model: bool = True,
        runtime_variant: str = "base",
    ):
        self.core_model = core_model
        self.config = core_model.config
        self.default_offload_model = default_offload_model
        self.runtime_variant = runtime_variant

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        task="i2v-A14B",
        device="cuda",
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        ulysses_size=1,
        t5_cpu=False,
        offload_model=True,
        convert_model_dtype=False,
        runtime_variant: Optional[str] = None,
        fast_model_path: Optional[str] = None,
        control_type: Optional[str] = None,
        **kwargs,
    ):
        if task not in SUPPORTED_LINGBOT_TASKS:
            raise ValueError(f"Unsupported task: {task}")

        wan_configs, wan_i2v, wan_i2v_fast = _load_runtime_components()
        config = wan_configs[task]
        cls._validate_parallelism(
            config=config,
            rank=int(rank),
            t5_fsdp=bool(t5_fsdp),
            dit_fsdp=bool(dit_fsdp),
            ulysses_size=int(ulysses_size),
            t5_cpu=bool(t5_cpu),
            offload_model=bool(offload_model),
        )
        use_fast = cls._should_use_fast_runtime(
            pretrained_model_path=pretrained_model_path,
            runtime_variant=runtime_variant,
            fast_model_path=fast_model_path,
        )
        _patch_fast_attention_fallback(use_fast)

        base_model_path, resolved_fast_model_path = cls._resolve_model_paths(
            pretrained_model_path=pretrained_model_path,
            fast_model_path=fast_model_path,
            use_fast=use_fast,
            rank=rank,
        )

        device_id = cls._parse_device_id(device)
        use_sp = ulysses_size > 1
        core_model_cls = wan_i2v_fast if use_fast else wan_i2v
        core_model_kwargs = dict(
            config=config,
            checkpoint_dir=base_model_path,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_sp=use_sp,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
            control_type=control_type,
        )
        if use_fast and "fast_checkpoint_dir" in signature(core_model_cls.__init__).parameters:
            core_model_kwargs["fast_checkpoint_dir"] = resolved_fast_model_path
        core_model = core_model_cls(**core_model_kwargs)
        return cls(
            core_model=core_model,
            default_offload_model=offload_model,
            runtime_variant="fast" if use_fast else "base",
        )

    @staticmethod
    def _validate_parallelism(
        *,
        config: Any,
        rank: int,
        t5_fsdp: bool,
        dit_fsdp: bool,
        ulysses_size: int,
        t5_cpu: bool,
        offload_model: bool,
    ) -> None:
        """Fail closed when FSDP/Ulysses does not match the active process group."""

        dist = torch.distributed
        initialized = dist.is_available() and dist.is_initialized()
        world_size = int(dist.get_world_size()) if initialized else 1
        if not initialized and (t5_fsdp or dit_fsdp or ulysses_size > 1):
            raise RuntimeError(
                "LingBot-World FSDP/Ulysses requires an initialized torchrun process group. "
                "Launch through Workspace or initialize torch.distributed before from_pretrained()."
            )
        if initialized and world_size == 1 and (t5_fsdp or dit_fsdp):
            raise ValueError("LingBot-World FSDP requires a process group with more than one rank.")
        if t5_cpu and t5_fsdp:
            raise ValueError("LingBot-World t5_cpu and t5_fsdp are mutually exclusive.")
        if offload_model and dit_fsdp:
            raise ValueError("LingBot-World offload_model is not supported together with DiT FSDP.")
        if initialized and rank != int(dist.get_rank()):
            raise ValueError(
                f"LingBot-World rank={rank} does not match process-group rank={dist.get_rank()}."
            )
        if ulysses_size > 1 and ulysses_size != world_size:
            raise ValueError(
                "LingBot-World ulysses_size must equal torchrun WORLD_SIZE; "
                f"got ulysses_size={ulysses_size}, WORLD_SIZE={world_size}."
            )
        num_heads = int(config.num_heads)
        if ulysses_size > 1 and num_heads % ulysses_size:
            raise ValueError(
                f"LingBot-World num_heads={num_heads} is not divisible by ulysses_size={ulysses_size}."
            )

    @staticmethod
    def _looks_like_fast_repo(path_or_repo: Optional[str]) -> bool:
        """Identify LingBot fast checkpoint refs by repo id or local directory name."""

        if not isinstance(path_or_repo, str):
            return False
        lowered = path_or_repo.lower()
        return (
            "lingbot-world-fast" in lowered
            or lowered.endswith("/lingbot_world_fast")
            or Path(lowered).name == "lingbot_world_fast"
        )

    @classmethod
    def _should_use_fast_runtime(
        cls,
        *,
        pretrained_model_path,
        runtime_variant: Optional[str],
        fast_model_path: Optional[str],
    ) -> bool:
        """Resolve whether the fast runtime is requested by variant or checkpoint ref."""

        if isinstance(runtime_variant, str) and runtime_variant.strip().lower() == "fast":
            return True
        if fast_model_path:
            return True
        if isinstance(pretrained_model_path, str) and cls._looks_like_fast_repo(pretrained_model_path):
            return True
        return False

    @staticmethod
    def _candidate_hfd_roots() -> tuple[Path, ...]:
        """Return local HFD cache roots without initiating network downloads."""

        roots = [DEFAULT_LINGBOT_HFD_ROOT]
        env_root = os.environ.get("WORLDFOUNDRY_HFD_ROOT")
        if env_root:
            roots.insert(0, Path(env_root).expanduser())
        deduped = []
        for root in roots:
            resolved = root.expanduser()
            if resolved not in deduped:
                deduped.append(resolved)
        return tuple(deduped)

    @staticmethod
    def _repo_dir_names(repo_id: str) -> list[str]:
        """Map a repo id to supported local cache directory names."""

        names = [repo_id.replace("/", "--")]
        leaf_name = repo_id.split("/")[-1]
        if leaf_name not in names:
            names.append(leaf_name)
        return names

    @classmethod
    def _resolve_cached_sibling(cls, path_or_repo: Optional[str], repo_id: str) -> Optional[str]:
        """Find a locally staged checkpoint next to a requested HFD path or cache root."""

        if not isinstance(path_or_repo, str) or not path_or_repo.strip():
            return None

        repo_dir_names = cls._repo_dir_names(repo_id)
        candidate = Path(path_or_repo).expanduser()
        if candidate.is_dir() and candidate.name in repo_dir_names:
            return str(candidate)

        base = Path(path_or_repo).expanduser()
        probe_points = [base]
        if base.exists():
            probe_points.extend(base.parents)

        for point in probe_points:
            if point.name == "hfd":
                for repo_dir_name in repo_dir_names:
                    sibling = point / repo_dir_name
                    if sibling.exists():
                        return str(sibling)
                break
            if point.parent.name == "hfd":
                for repo_dir_name in repo_dir_names:
                    sibling = point.parent / repo_dir_name
                    if sibling.exists():
                        return str(sibling)
                break

        for hfd_root in cls._candidate_hfd_roots():
            for repo_dir_name in repo_dir_names:
                cached = hfd_root / repo_dir_name
                if cached.is_dir():
                    return str(cached)
        return None

    @classmethod
    def _resolve_local_repo(cls, repo_or_path: str, rank: int) -> str:
        """Resolve a local checkpoint directory and fail closed when it is absent."""

        if os.path.isdir(repo_or_path):
            return str(Path(repo_or_path).expanduser())

        resolved_cached = cls._resolve_cached_sibling(repo_or_path, repo_or_path)
        if resolved_cached is not None and os.path.isdir(resolved_cached):
            return resolved_cached

        roots = ", ".join(str(root) for root in cls._candidate_hfd_roots())
        raise FileNotFoundError(
            f"LingBot checkpoint '{repo_or_path}' is not available locally for rank {rank}. "
            "Code is vendored in-tree; only checkpoints may be external. "
            f"Stage the repo under one of: {roots}, or pass a local checkpoint directory."
        )

    @classmethod
    def _resolve_model_paths(
        cls,
        *,
        pretrained_model_path,
        fast_model_path: Optional[str],
        use_fast: bool,
        rank: int,
    ) -> tuple[str, Optional[str]]:
        """Resolve base and optional fast checkpoint directories from local storage."""

        if not isinstance(pretrained_model_path, str) or not pretrained_model_path.strip():
            raise ValueError("LingBotWorldRuntime requires a non-empty pretrained_model_path.")

        requested_path = pretrained_model_path.strip()
        resolved_fast = None

        if use_fast:
            if cls._looks_like_fast_repo(requested_path):
                resolved_fast = requested_path
                base_request = cls._resolve_cached_sibling(requested_path, DEFAULT_LINGBOT_BASE_REPO) or DEFAULT_LINGBOT_BASE_REPO
            elif fast_model_path:
                resolved_fast = fast_model_path
                base_request = requested_path
            else:
                base_request = requested_path
                resolved_fast = cls._resolve_cached_sibling(requested_path, DEFAULT_LINGBOT_FAST_REPO) or DEFAULT_LINGBOT_FAST_REPO
        else:
            base_request = requested_path

        base_model_path = cls._resolve_local_repo(base_request, rank=rank)
        if resolved_fast is not None:
            resolved_fast = cls._resolve_local_repo(resolved_fast, rank=rank)

        return base_model_path, resolved_fast

    @classmethod
    def runtime_plan(
        cls,
        pretrained_model_path,
        task: str = "i2v-A14B",
        runtime_variant: Optional[str] = None,
        fast_model_path: Optional[str] = None,
        rank: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a lightweight LingBot runtime plan without loading model weights."""

        if task not in SUPPORTED_LINGBOT_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        use_fast = cls._should_use_fast_runtime(
            pretrained_model_path=pretrained_model_path,
            runtime_variant=runtime_variant,
            fast_model_path=fast_model_path,
        )
        base_model_path, resolved_fast_model_path = cls._resolve_model_paths(
            pretrained_model_path=pretrained_model_path,
            fast_model_path=fast_model_path,
            use_fast=use_fast,
            rank=rank,
        )
        return {
            "status": "prepared",
            "backend": "worldfoundry.lingbot_world.in_tree_runtime",
            "backend_quality": "official_in_tree_runtime",
            "runtime_root": str(lingbot_runtime_root()),
            "task": task,
            "base_checkpoint_dir": base_model_path,
            "fast_checkpoint_dir": resolved_fast_model_path,
            "runtime_variant": "fast" if use_fast else "base",
            "extra": {key: str(value) for key, value in kwargs.items()},
        }

    @staticmethod
    def _parse_device_id(device: str) -> int:
        if device.startswith("cuda:"):
            return int(device.split(":")[-1])
        if device == "cuda":
            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
            return 0
        raise ValueError(f"Unsupported device for LingBotWorldRuntime: {device}")

    @staticmethod
    def _to_pil_image(
        pil_image: Optional[Image.Image] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> Image.Image:
        if pil_image is not None:
            return pil_image.convert("RGB")

        if image_tensor is None:
            raise ValueError("Either 'pil_image' or 'image_tensor' must be provided.")

        tensor = image_tensor.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3):
            raise ValueError(f"Unsupported channel count for image tensor: {tensor.shape[0]}")

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        array = tensor.permute(1, 2, 0).float().numpy()
        if array.min() < 0.0:
            array = (array + 1.0) / 2.0
        array = np.clip(array, 0.0, 1.0)
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    @staticmethod
    def _video_to_numpy(video: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if video is None:
            return None

        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"Unexpected video tensor shape: {tuple(tensor.shape)}")

        return ((tensor.permute(1, 2, 3, 0) + 1.0) / 2.0).clamp(0.0, 1.0).float().numpy()

    @staticmethod
    def _save_camera_controls(
        action_dir: str,
        c2ws: np.ndarray,
        Ks: np.ndarray,
        action_matrix: Optional[np.ndarray] = None,
    ) -> None:
        np.save(os.path.join(action_dir, "poses.npy"), np.asarray(c2ws, dtype=np.float32))
        np.save(os.path.join(action_dir, "intrinsics.npy"), np.asarray(Ks, dtype=np.float32))
        if action_matrix is not None:
            actions = np.asarray(action_matrix, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 4:
                raise ValueError(f"LingBot Base-Act action_matrix must have shape [F,4], got {actions.shape}.")
            if actions.shape[0] < len(c2ws):
                actions = np.pad(actions, ((0, len(c2ws) - actions.shape[0]), (0, 0)), mode="constant")
            np.save(os.path.join(action_dir, "action.npy"), actions[: len(c2ws)])

    @staticmethod
    def _synchronized_seed(seed: int) -> int:
        """Resolve negative seeds once so every SP rank owns one global latent."""

        resolved = int(seed)
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            payload = [
                resolved if resolved >= 0 else random.randint(0, sys.maxsize)
                if dist.get_rank() == 0
                else 0
            ]
            dist.broadcast_object_list(payload, src=0)
            return int(payload[0])
        return resolved if resolved >= 0 else random.randint(0, sys.maxsize)

    def _generate_video(self, *, prompt: str, pil_image: Image.Image, offload_model: bool, **kwargs):
        generate = self.core_model.generate
        accepted = set(signature(generate).parameters)
        payload = {
            "input_prompt": prompt,
            "img": pil_image,
            "offload_model": offload_model,
            **kwargs,
        }
        filtered_payload = {
            key: value
            for key, value in payload.items()
            if value is not None and key in accepted
        }
        return generate(**filtered_payload)

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        pil_image: Optional[Image.Image] = None,
        image_tensor: Optional[torch.Tensor] = None,
        action_path: Optional[str] = None,
        c2ws: Optional[np.ndarray] = None,
        Ks: Optional[np.ndarray] = None,
        num_output_frames: int = 81,
        max_area: int = 720 * 1280,
        shift=None,
        sample_solver: str = "unipc",
        sampling_steps=None,
        guide_scale=None,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: Optional[bool] = None,
        vis_ui: bool = False,
        allow_act2cam: bool = False,
        action_string: Optional[str] = None,
        action_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if self.runtime_variant == "fast":
            requested_frames = int(num_output_frames)
            latent_frames = (requested_frames - 1) // 4 + 1
            if requested_frames < 9 or (requested_frames - 1) % 4 or latent_frames % 3:
                raise ValueError(
                    "LingBot Fast requires frame_num=12n-3 (9, 21, 33, ...); "
                    f"got {requested_frames}. The upstream runtime otherwise silently truncates the clip."
                )
        pil_image = self._to_pil_image(pil_image=pil_image, image_tensor=image_tensor)
        offload_model = self.default_offload_model if offload_model is None else offload_model
        seed = self._synchronized_seed(seed)

        if action_path is None and (c2ws is None) != (Ks is None):
            raise ValueError("'c2ws' and 'Ks' must be provided together when 'action_path' is omitted.")

        if action_path is None and c2ws is not None and Ks is not None:
            control_type = str(getattr(self.core_model, "control_type", ""))
            if control_type == "act" and action_matrix is None:
                raise ValueError("LingBot Base-Act requires action_matrix alongside c2ws/Ks.")
            if control_type == "act" and allow_act2cam:
                raise ValueError("allow_act2cam is a Base-Cam conversion mode and cannot be used with Base-Act weights.")
            with tempfile.TemporaryDirectory(prefix="lingbot_action_") as action_dir:
                self._save_camera_controls(action_dir, c2ws, Ks, action_matrix=action_matrix)
                video = self._generate_video(
                    prompt=prompt,
                    pil_image=pil_image,
                    offload_model=offload_model,
                    action_path=action_dir,
                    allow_act2cam=allow_act2cam,
                    action_string=action_string,
                    vis_ui=vis_ui,
                    max_area=max_area,
                    frame_num=num_output_frames,
                    shift=shift if shift is not None else self.config.sample_shift,
                    sample_solver=sample_solver,
                    sampling_steps=sampling_steps if sampling_steps is not None else self.config.sample_steps,
                    guide_scale=guide_scale if guide_scale is not None else self.config.sample_guide_scale,
                    n_prompt=n_prompt,
                    seed=seed,
                )
                return self._video_to_numpy(video)

        video = self._generate_video(
            prompt=prompt,
            pil_image=pil_image,
            offload_model=offload_model,
            action_path=action_path,
            allow_act2cam=allow_act2cam,
            action_string=action_string,
            vis_ui=vis_ui,
            max_area=max_area,
            frame_num=num_output_frames,
            shift=shift if shift is not None else self.config.sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps if sampling_steps is not None else self.config.sample_steps,
            guide_scale=guide_scale if guide_scale is not None else self.config.sample_guide_scale,
            n_prompt=n_prompt,
            seed=seed,
        )
        return self._video_to_numpy(video)


LingBotRuntime = LingBotWorldRuntime


def load_runtime(*args: Any, **kwargs: Any) -> LingBotWorldRuntime:
    return LingBotWorldRuntime.from_pretrained(*args, **kwargs)


__all__ = [
    "DEFAULT_LINGBOT_ACT_REPO",
    "DEFAULT_LINGBOT_BASE_REPO",
    "DEFAULT_LINGBOT_FAST_REPO",
    "DEFAULT_LINGBOT_HFD_ROOT",
    "LingBotRuntime",
    "LingBotWorldRuntime",
    "SUPPORTED_LINGBOT_TASKS",
    "lingbot_runtime_root",
    "load_runtime",
]
