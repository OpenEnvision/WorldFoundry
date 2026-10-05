from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.core.model_loading import load_torch_state_dict

from .bucket_registry import get_bucket_config
from .plan import (
    InfiniteWorldRuntimePlan,
    find_default_model_root,
    missing_python_modules,
    missing_runtime_files,
    resolve_in_tree_model_root,
)
from .utils.import_utils import get_obj_from_str


DEFAULT_NEGATIVE_PROMPT = (
    "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, "
    "image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
    "motionless image, cluttered background, three legs, crowded background, walking backwards."
)


def default_config_path() -> str:
    """
    Return the public bundled Infinite-World runtime config path.

    Args:
        None.
    """
    return str(worldfoundry_data_path("models", "runtime", "configs", "infinite_world.yaml"))


def build_runtime_plan(
    pretrained_model_path: Any = None,
    *,
    device: str = "cuda",
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Build a lightweight in-tree Infinite-World runtime readiness plan.

    Args:
        pretrained_model_path: Local checkpoint root, model id alias, or None for default cache lookup.
        device: Target torch device label.
        config_path: Optional bundled or caller-provided config path.
    """
    model_root = _plan_model_root(pretrained_model_path)
    missing_files = missing_runtime_files(model_root) if model_root is not None else ["model_root"]
    missing_modules = missing_python_modules()
    ready = model_root is not None and not missing_files and not missing_modules
    return InfiniteWorldRuntimePlan(
        model_root=None if model_root is None else str(model_root),
        config_path=config_path or default_config_path(),
        device=device,
        backend_quality="in_tree_runtime_ready" if ready else "blocked_missing_runtime_requirements",
        status="ready" if ready else "blocked",
    ).to_dict()


def _plan_model_root(pretrained_model_path: Any = None) -> Path | None:
    if isinstance(pretrained_model_path, dict):
        candidate_value = (
            pretrained_model_path.get("pretrained_model_path")
            or pretrained_model_path.get("model_root")
            or pretrained_model_path.get("checkpoint_dir")
        )
    else:
        candidate_value = pretrained_model_path

    if candidate_value is None or str(candidate_value) in {"MeiGen-AI/Infinite-World", "Infinite-World"}:
        return find_default_model_root()

    candidate = Path(str(candidate_value)).expanduser()
    return candidate.resolve() if candidate.is_dir() else None


def _resolve_model_root(pretrained_model_path: Optional[str]) -> str:
    return str(resolve_in_tree_model_root(pretrained_model_path))


def _device_index(device: str) -> int:
    if device is None or not str(device).startswith("cuda"):
        return 0
    if ":" in str(device):
        return int(str(device).split(":", maxsplit=1)[1])
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def _resolve_path(path_value: Optional[str], model_root: str) -> Optional[str]:
    if path_value is None:
        return None
    path_value = str(path_value).strip()
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(model_root, path_value)


def _resolve_config_path(config_path: Optional[str]) -> str:
    if config_path is None:
        return default_config_path()

    candidate = Path(config_path)
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    data_candidate = Path(default_config_path()).resolve().parent / candidate
    return str(data_candidate.resolve())


def _load_dit_state_dict(checkpoint_path: str, model_root: str):
    checkpoint_path = _resolve_path(checkpoint_path, model_root)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = load_torch_state_dict(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _set_single_process_context_parallel():
    from worldfoundry.core.distributed.model_parallel import context_state as cp_util

    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    cp_util.dp_group = None
    cp_util.cp_group = None


def _setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _video_tensor_to_numpy(video_tensor: torch.Tensor) -> np.ndarray:
    video_tensor = video_tensor.detach().cpu().float()
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]
    video = video_tensor.permute(1, 2, 3, 0)
    video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    return video.numpy()


def _video_tensor_to_uint8(video_tensor: torch.Tensor) -> np.ndarray:
    video = (_video_tensor_to_numpy(video_tensor) * 255.0).round().clip(0.0, 255.0)
    return video.astype(np.uint8)


def _capture_rng_state(device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    state = {
        "cpu": torch.get_rng_state().clone(),
        "cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _restore_rng_state(device: torch.device, state: Dict[str, Optional[torch.Tensor]]):
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state, device)


def _torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


class InfiniteWorldRuntime:
    chunk_frames = 21
    chunk_stride = 20

    def __init__(
        self,
        model_root: str,
        config_path: str,
        vae,
        text_encoder,
        scheduler,
        dit,
        bucket_config,
        validation_num_frames: int,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        guidance_scale: float = 5.0,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        load_seed: Optional[int] = None,
        post_load_rng_state: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ):
        self.model_root = model_root
        self.config_path = config_path
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.dit = dit
        self.bucket_config = bucket_config
        self.validation_num_frames = validation_num_frames
        self.device = device
        self.device_torch = torch.device(device)
        self.weight_dtype = weight_dtype
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.load_seed = load_seed
        self.post_load_rng_state = post_load_rng_state

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        args=None,
        device=None,
        weight_dtype=torch.bfloat16,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        bucket_config_name: str = "ASPECT_RATIO_627_F64",
        num_sampling_steps: Optional[int] = None,
        shift: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        device = device or "cuda"
        if kwargs.pop("plan_only", False):
            return cls.plan(
                pretrained_model_path=pretrained_model_path,
                device=device,
                config_path=config_path,
            )
        if not str(device).startswith("cuda"):
            raise ValueError("Infinite-World currently requires a CUDA device.")

        model_root = _resolve_model_root(pretrained_model_path)
        _set_single_process_context_parallel()

        local_rank = _device_index(device)
        torch.cuda.set_device(local_rank)
        if seed is not None:
            _setup_seed(int(seed))

        from omegaconf import OmegaConf

        config_path = _resolve_config_path(config_path)
        cfg = OmegaConf.load(config_path) if args is None else args
        if checkpoint_path is not None:
            cfg.checkpoint_path = checkpoint_path

        cfg.checkpoint_path = _resolve_path(cfg.get("checkpoint_path"), model_root)
        if "vae_cfg" in cfg and "vae_pth" in cfg.vae_cfg:
            cfg.vae_cfg.vae_pth = _resolve_path(cfg.vae_cfg.vae_pth, model_root)
        if "text_encoder_cfg" in cfg:
            if "checkpoint_path" in cfg.text_encoder_cfg:
                cfg.text_encoder_cfg.checkpoint_path = _resolve_path(cfg.text_encoder_cfg.checkpoint_path, model_root)
            if "tokenizer_path" in cfg.text_encoder_cfg:
                cfg.text_encoder_cfg.tokenizer_path = _resolve_path(cfg.text_encoder_cfg.tokenizer_path, model_root)

        vae = get_obj_from_str(cfg.vae_target)(**cfg.vae_cfg).to(local_rank)
        text_encoder = get_obj_from_str(cfg.text_encoder_target)(device=local_rank, **cfg.text_encoder_cfg)
        text_encoder.t5.model.to(local_rank)

        scheduler = get_obj_from_str(cfg.scheduler_target)(**cfg.val_scheduler_cfg)
        scheduler.num_sampling_steps = int(num_sampling_steps or cfg.val_scheduler_cfg.get("num_sampling_steps", 30))
        scheduler.shift = float(shift if shift is not None else cfg.val_scheduler_cfg.get("shift", 7.0))

        amp_dtype_name = cfg.get("amp_dtype", None)
        amp_dtype = getattr(torch, amp_dtype_name) if amp_dtype_name else weight_dtype
        model_dtype = weight_dtype or amp_dtype

        dit = get_obj_from_str(cfg.model_target)(
            out_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_context_parallel=False,
            **cfg.model_cfg,
        )
        dit = dit.to(dtype=model_dtype)
        dit.eval()

        state_dict = _load_dit_state_dict(cfg.checkpoint_path, model_root)
        state_dict.pop("pos_embed_temporal", None)
        state_dict.pop("pos_embed", None)
        dit.load_state_dict(state_dict, strict=False)
        dit = dit.to(torch.device(device))

        bucket_config = get_bucket_config(bucket_config_name)
        validation_num_frames = int(cfg.validation_data.get("num_frames", 81))
        post_load_rng_state = _capture_rng_state(torch.device(device)) if seed is not None else None

        return cls(
            model_root=model_root,
            config_path=config_path,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            dit=dit,
            bucket_config=bucket_config,
            validation_num_frames=validation_num_frames,
            device=device,
            weight_dtype=model_dtype,
            guidance_scale=float(guidance_scale or cfg.val_scheduler_cfg.get("text_cfg_scale", 5.0)),
            negative_prompt=negative_prompt,
            load_seed=int(seed) if seed is not None else None,
            post_load_rng_state=post_load_rng_state,
        )

    @classmethod
    def plan(
        cls,
        pretrained_model_path=None,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build a local-runtime readiness plan without loading model weights.

        Args:
            pretrained_model_path: Local Infinite-World checkpoint root or mapping with runtime paths.
            device: Target torch device label.
            config_path: Optional config file path.
            **kwargs: Reserved compatibility options.
        """
        del kwargs
        return build_runtime_plan(
            pretrained_model_path=pretrained_model_path,
            device=str(device or "cuda"),
            config_path=_resolve_config_path(config_path),
        )

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        condition_video: torch.Tensor,
        move_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_chunks: int = 1,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if seed is not None:
            if self.load_seed is not None and int(seed) == int(self.load_seed) and self.post_load_rng_state is not None:
                _restore_rng_state(self.device_torch, self.post_load_rng_state)
            else:
                _setup_seed(int(seed))

        prompt = prompt or ""
        negative_prompt = negative_prompt or self.negative_prompt
        video_buffer = condition_video.detach().cpu().float()
        move_ids = move_ids.detach().cpu().long()
        view_ids = view_ids.detach().cpu().long()

        for _ in range(max(int(num_chunks), 1)):
            current_cond = video_buffer.to(self.device_torch, dtype=torch.float32)
            current_latent = self.vae.encode(current_cond)

            latent_size = list(current_latent.shape)
            latent_size[2] = self.chunk_frames

            curr_start = int(video_buffer.shape[2] - 1)
            curr_end = curr_start + self.validation_num_frames

            move = move_ids[curr_start:curr_end]
            view = view_ids[curr_start:curr_end]

            if move.shape[0] < self.validation_num_frames:
                pad_len = self.validation_num_frames - move.shape[0]
                move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long)], dim=0)

            additional_args = {
                "image_cond": current_latent,
                "move": move.unsqueeze(0).to(self.device_torch),
                "view": view.unsqueeze(0).to(self.device_torch),
            }

            _torch_gc()
            samples = self.scheduler.sample(
                model=self.dit,
                text_encoder=self.text_encoder,
                null_embedder=self.dit.y_embedder,
                z_size=torch.Size(latent_size),
                prompts=[prompt],
                guidance_scale=self.guidance_scale,
                negative_prompts=[negative_prompt],
                device=self.device_torch,
                additional_args=additional_args,
            )
            decoded_chunk = self.vae.decode(samples).detach().cpu()
            video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
            _torch_gc()

        return {
            "video": _video_tensor_to_numpy(video_buffer),
            "video_uint8": _video_tensor_to_uint8(video_buffer),
            "video_tensor": video_buffer,
            "num_frames": int(video_buffer.shape[2]),
            "num_chunks": int(num_chunks),
        }


def load_runtime(*args: Any, **kwargs: Any) -> Any:
    return InfiniteWorldRuntime.from_pretrained(*args, **kwargs)
