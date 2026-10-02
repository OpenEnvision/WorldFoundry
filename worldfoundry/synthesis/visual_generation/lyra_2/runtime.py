from __future__ import annotations

import importlib
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from worldfoundry.core.io.paths import checkpoint_root_candidates
from worldfoundry.synthesis.visual_generation.lyra_2 import (
    SOURCE_PACKAGE_ROOT as LYRA2_SOURCE_PACKAGE_ROOT,
)

DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_WEIGHT_DTYPE = "bfloat16"


class Lyra2Runtime:
    """In-tree Lyra-2 custom-trajectory wrapper with vendored runtime sources."""

    MODEL_ID = "lyra-2"
    DISPLAY_NAME = "Lyra-2"
    BLOCKED_REASONS = (
        "Full Lyra-2 execution requires the official CUDA stack, transformer-engine/megatron dependencies, and local checkpoints.",
        "The source runtime is vendored in-tree; model, text-encoder, and DA3 checkpoints remain external caller-provided assets.",
    )

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        negative_prompt_path: str | Path | None = None,
        da3_model_path_custom: str | Path | None = None,
        da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
        device: str = "cuda",
        weight_dtype: Any = DEFAULT_WEIGHT_DTYPE,
        experiment: str = "lyra2",
        context_parallel_size: int = 1,
        guidance: float = 5.0,
        shift: float = 5.0,
        num_sampling_step: int = 35,
        prompt_suffix: str = "",
        offload: bool = False,
        offload_when_prompt: bool = False,
        model: Any = None,
        da3_model: Any = None,
        negative_prompt_data: dict[str, Any] | None = None,
        model_id: str = MODEL_ID,
    ) -> None:
        """
        Initialize a Lyra-2 in-tree wrapper.

        Args:
            checkpoint_dir: Optional external Lyra-2 model checkpoint directory.
            negative_prompt_path: Optional external negative prompt embedding path.
            da3_model_path_custom: Optional external Depth Anything 3 checkpoint path.
            da3_model_name: Depth Anything 3 model identifier.
            device: Target device label.
            weight_dtype: Desired torch dtype for loaded model execution.
            experiment: Lyra-2 experiment config name.
            context_parallel_size: Context parallel size supported by runtime.
            guidance: Classifier-free guidance scale.
            shift: Scheduler shift value.
            num_sampling_step: Number of sampling steps.
            prompt_suffix: Text appended to prompts before embedding.
            offload: Whether runtime model offload is requested.
            offload_when_prompt: Whether prompt embedding offload is requested.
            model: Optional preloaded Lyra-2 model for explicit execution.
            da3_model: Optional preloaded DA3 model for explicit execution.
            negative_prompt_data: Optional preloaded negative prompt embeddings.
            model_id: Runtime profile identifier.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "image_to_world_video"
        self.checkpoint_dir = self._optional_path(checkpoint_dir)
        self.negative_prompt_path = self._optional_path(negative_prompt_path)
        self.da3_model_path_custom = self._optional_path(da3_model_path_custom)
        self.da3_model_name = da3_model_name
        self.device = device
        self.weight_dtype = weight_dtype
        self.experiment = experiment
        self.context_parallel_size = int(context_parallel_size)
        self.guidance = float(guidance)
        self.shift = float(shift)
        self.num_sampling_step = int(num_sampling_step)
        self.prompt_suffix = prompt_suffix
        self.offload = bool(offload)
        self.offload_when_prompt = bool(offload_when_prompt)
        self.model = model
        self.da3_model = da3_model
        self.negative_prompt_data = negative_prompt_data

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        weight_dtype: Any = DEFAULT_WEIGHT_DTYPE,
        checkpoint_dir: Optional[str] = None,
        negative_prompt_path: Optional[str] = None,
        da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
        da3_model_path_custom: Optional[str] = None,
        experiment: str = "lyra2",
        context_parallel_size: int = 1,
        guidance: float = 5.0,
        shift: float = 5.0,
        num_sampling_step: int = 35,
        prompt_suffix: str = "",
        offload: bool = False,
        offload_when_prompt: bool = False,
        load_runtime: bool = False,
        strict: bool = True,
        **kwargs: Any,
    ) -> "Lyra2Runtime":
        """
        Build a lazy Lyra-2 wrapper or explicitly load the vendored runtime.

        Args:
            pretrained_model_path: Optional checkpoint root or mapping with runtime options.
            args: Unused compatibility argument for factory callers.
            device: Target device label.
            weight_dtype: Desired torch dtype for explicit runtime loading.
            checkpoint_dir: Explicit external Lyra-2 checkpoint directory.
            negative_prompt_path: Explicit external negative prompt embedding path.
            da3_model_name: Depth Anything 3 model identifier.
            da3_model_path_custom: Explicit external DA3 checkpoint path.
            experiment: Lyra-2 experiment config name.
            context_parallel_size: Context parallel size supported by runtime.
            guidance: Classifier-free guidance scale.
            shift: Scheduler shift value.
            num_sampling_step: Number of sampling steps.
            prompt_suffix: Text appended to prompts before embedding.
            offload: Whether runtime model offload is requested.
            offload_when_prompt: Whether prompt embedding offload is requested.
            load_runtime: Whether to load checkpoints immediately.
            strict: Strict checkpoint loading flag.
            **kwargs: Extra runtime options preserved by the lazy wrapper.
        """
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_root"] = str(pretrained_model_path)
        options.update(kwargs)

        checkpoint_root = cls._optional_path(options.get("checkpoint_root") or options.get("weights_root"))
        resolved_checkpoint_dir = cls._resolve_asset_path(
            checkpoint_dir or options.get("checkpoint_dir") or "checkpoints/model",
            checkpoint_root,
        )
        resolved_negative_prompt_path = cls._resolve_asset_path(
            negative_prompt_path
            or options.get("negative_prompt_path")
            or "checkpoints/text_encoder/negative_prompt.pt",
            checkpoint_root,
        )
        resolved_da3_model_path = cls._resolve_asset_path(
            da3_model_path_custom or options.get("da3_model_path_custom") or "checkpoints/recon/model.pt",
            checkpoint_root,
        )
        instance = cls(
            checkpoint_dir=resolved_checkpoint_dir,
            negative_prompt_path=resolved_negative_prompt_path,
            da3_model_path_custom=resolved_da3_model_path,
            da3_model_name=da3_model_name,
            device=str(device or options.get("device") or "cuda"),
            weight_dtype=weight_dtype,
            experiment=str(options.get("experiment") or experiment),
            context_parallel_size=int(options.get("context_parallel_size", context_parallel_size)),
            guidance=float(options.get("guidance", guidance)),
            shift=float(options.get("shift", shift)),
            num_sampling_step=int(options.get("num_sampling_step", num_sampling_step)),
            prompt_suffix=str(options.get("prompt_suffix", prompt_suffix)),
            offload=bool(options.get("offload", offload)),
            offload_when_prompt=bool(options.get("offload_when_prompt", offload_when_prompt)),
            model_id=str(options.get("model_id") or options.get("profile_id") or cls.MODEL_ID),
        )
        if load_runtime:
            instance.load_runtime(strict=strict)
        return instance

    @staticmethod
    def _runtime_root() -> Path:
        """
        Return the vendored Lyra-2 runtime root.

        Args:
            None.
        """
        return LYRA2_SOURCE_PACKAGE_ROOT.parent

    @staticmethod
    def _optional_path(path_value: str | Path | None) -> str | None:
        """
        Normalize an optional filesystem path string.

        Args:
            path_value: Optional path value.
        """
        if path_value is None:
            return None
        return str(Path(path_value).expanduser())

    @staticmethod
    def _resolve_asset_path(path_value: str | Path | None, root: str | Path | None) -> str | None:
        """
        Resolve an external asset path relative to a checkpoint root.

        Args:
            path_value: Asset path or relative path under root.
            root: Optional external checkpoint root.
        """
        if path_value is None:
            return None
        candidate = Path(path_value).expanduser()
        if candidate.is_absolute() or root is None:
            return str(candidate)
        return str((Path(root).expanduser() / candidate).resolve())

    def _require_asset(self, path_value: str | None, label: str) -> str:
        """
        Validate a required external asset path.

        Args:
            path_value: External asset path to validate.
            label: Human-readable asset label.
        """
        if path_value is None:
            raise FileNotFoundError(f"{label} is not configured.")
        if not Path(path_value).exists():
            raise FileNotFoundError(f"{label} not found: {path_value}")
        return path_value

    def _install_checkpoint_aliases(self, checkpoint_root: Path) -> None:
        runtime_checkpoint_dir = self._runtime_root() / "checkpoints"
        source_checkpoint_dir = checkpoint_root / "checkpoints"
        if not source_checkpoint_dir.is_dir():
            return
        if not runtime_checkpoint_dir.exists() and not runtime_checkpoint_dir.is_symlink():
            try:
                runtime_checkpoint_dir.symlink_to(source_checkpoint_dir, target_is_directory=True)
            except FileExistsError:
                pass

        shared_root = checkpoint_root.parent
        tokenizer_roots = tuple(dict.fromkeys((shared_root, *checkpoint_root_candidates())))
        umt5_candidates = []
        xlm_candidates = []
        for root in tokenizer_roots:
            umt5_candidates.extend(
                (
                    root / "umt5-xxl",
                    root / "google" / "umt5-xxl",
                    root / "google--umt5-xxl",
                    root / "Wan-AI--Wan2.1-T2V-1.3B" / "google" / "umt5-xxl",
                    root / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl",
                )
            )
            xlm_candidates.extend(
                (
                    root / "xlm-roberta-large",
                    root / "FacebookAI--xlm-roberta-large",
                )
            )
        umt5_tokenizer = next(
            (
                candidate
                for candidate in umt5_candidates
                if (candidate / "tokenizer_config.json").is_file()
                and ((candidate / "spiece.model").is_file() or (candidate / "tokenizer.json").is_file())
            ),
            None,
        )
        xlm_tokenizer = next(
            (
                candidate
                for candidate in xlm_candidates
                if (candidate / "tokenizer_config.json").is_file()
            ),
            None,
        )
        for env_name, candidate in {
            "LYRA2_IMAGE_ENCODER_CKPT": source_checkpoint_dir / "image_encoder" / "model.pth",
            "LYRA2_TEXT_ENCODER_CKPT": source_checkpoint_dir / "text_encoder" / "encoder.pth",
            "LYRA2_CLIP_TOKENIZER": xlm_tokenizer,
            "LYRA2_UMT5_TOKENIZER": umt5_tokenizer,
        }.items():
            if candidate is not None and candidate.exists():
                os.environ.setdefault(env_name, candidate.as_posix())

    @staticmethod
    def _install_megatron_aliases() -> None:
        import sys
        import types

        from worldfoundry.core.distributed.model_parallel import megatron_compat

        parallel_state_module = sys.modules.get("megatron.core.parallel_state")
        if parallel_state_module is None:
            parallel_state_module = types.ModuleType("megatron.core.parallel_state")
            for attr in dir(megatron_compat.parallel_state):
                if attr.startswith("__"):
                    continue
                try:
                    setattr(parallel_state_module, attr, getattr(megatron_compat.parallel_state, attr))
                except Exception:
                    pass
            sys.modules["megatron.core.parallel_state"] = parallel_state_module

        megatron_module = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
        if not hasattr(megatron_module, "__path__"):
            megatron_module.__path__ = []
        core_module = sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
        if not hasattr(core_module, "__path__"):
            core_module.__path__ = []
        core_module.ModelParallelConfig = megatron_compat.ModelParallelConfig
        core_module.parallel_state = parallel_state_module
        core_module.mpu = parallel_state_module
        megatron_module.core = core_module

    def _install_runtime_aliases(self) -> None:
        """
        Register vendored Lyra-2 package aliases for official absolute imports.

        Args:
            None.
        """
        import sys

        self._install_megatron_aliases()

        from worldfoundry.synthesis.visual_generation.lyra_2 import lyra_2

        sys.modules.setdefault("lyra_2", lyra_2)
        vipe_module = importlib.import_module("worldfoundry.base_models.three_dimensions.general_3d.vipe")
        sys.modules["vipe"] = vipe_module
        from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.depth_utils import (
            install_da3_legacy_aliases,
        )

        install_da3_legacy_aliases()

    def load_runtime(self, strict: bool = True) -> "Lyra2Runtime":
        """
        Load Lyra-2 model, DA3 model, and negative prompt embeddings.

        Args:
            strict: Strict checkpoint loading flag passed to official loader.
        """
        import torch

        from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
            working_directory,
        )

        if not str(self.device).startswith("cuda"):
            raise ValueError("Lyra-2 runtime loading requires a CUDA device.")
        if self.context_parallel_size != 1:
            raise ValueError("Lyra-2 integration currently supports context_parallel_size=1 only.")

        self._install_runtime_aliases()
        checkpoint_dir = Path(self._require_asset(self.checkpoint_dir, "Lyra-2 checkpoint_dir"))
        negative_prompt_path = Path(self._require_asset(self.negative_prompt_path, "Lyra-2 negative_prompt_path"))
        da3_model_path = Path(self._require_asset(self.da3_model_path_custom, "Lyra-2 DA3 checkpoint"))
        checkpoint_root = next(
            (
                candidate
                for candidate in (checkpoint_dir, *checkpoint_dir.parents)
                if (candidate / "checkpoints" / "vae" / "vae.pth").is_file()
            ),
            checkpoint_dir,
        )
        vae_path = Path(
            self._require_asset(checkpoint_root / "checkpoints" / "vae" / "vae.pth", "Lyra-2 VAE checkpoint")
        )
        image_mean_std_path = Path(
            self._require_asset(
                checkpoint_root / "checkpoints" / "vae" / "images_mean_std.pt",
                "Lyra-2 image mean/std",
            )
        )
        video_mean_std_path = Path(
            self._require_asset(
                checkpoint_root / "checkpoints" / "vae" / "video_mean_std.pt",
                "Lyra-2 video mean/std",
            )
        )
        self._install_checkpoint_aliases(checkpoint_root)

        from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.depth_utils import (
            load_da3_model,
        )
        from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.utils.model_loader import (
            load_model_from_checkpoint,
        )

        experiment_opts = [
            "model.config.use_mp_policy_fsdp=False",
            "model.config.keep_original_net_dtype=False",
            f"+model.config.tokenizer.vae_pth={vae_path.as_posix()}",
            f"+model.config.tokenizer.image_mean_std_path={image_mean_std_path.as_posix()}",
            f"+model.config.tokenizer.video_mean_std_path={video_mean_std_path.as_posix()}",
        ]
        with working_directory(str(self._runtime_root())):
            model, _ = load_model_from_checkpoint(
                config_file="lyra_2/_src/configs/config.py",
                experiment_name=self.experiment,
                checkpoint_path=checkpoint_dir.as_posix(),
                enable_fsdp=False,
                experiment_opts=experiment_opts,
                strict=strict,
            )

        desired_dtype = model.tensor_kwargs.get("dtype", self._torch_dtype(torch))
        desired_device = model.tensor_kwargs.get("device", self.device)
        if desired_dtype is not None and hasattr(model, "net"):
            model.net = model.net.to(device=desired_device, dtype=desired_dtype)
        model.eval()

        da3_model = load_da3_model(
            da3_model_name=self.da3_model_name,
            da3_model_path_custom=da3_model_path,
            device=str(desired_device),
        )
        da3_model.eval()
        self.model = model
        self.da3_model = da3_model
        self.negative_prompt_data = torch.load(negative_prompt_path, map_location="cpu", weights_only=False)
        self.weight_dtype = desired_dtype
        return self

    def _runtime_args(
        self,
        *,
        num_frames: int,
        seed: int,
        guidance: Optional[float],
        shift: Optional[float],
        num_sampling_step: Optional[int],
        offload: Optional[bool],
    ) -> Namespace:
        """
        Build Lyra-2 runtime argument namespace.

        Args:
            num_frames: Number of output frames.
            seed: Random seed.
            guidance: Optional guidance scale override.
            shift: Optional scheduler shift override.
            num_sampling_step: Optional sampling step override.
            offload: Optional model offload override.
        """
        return Namespace(
            ablate_same_t5=False,
            context_parallel_size=self.context_parallel_size,
            da3_frame_interval=8,
            da3_max_history_frames=10,
            da3_model_name=self.da3_model_name,
            da3_model_path_custom=self.da3_model_path_custom,
            guidance=float(self.guidance if guidance is None else guidance),
            num_frames=int(num_frames),
            num_sampling_step=int(self.num_sampling_step if num_sampling_step is None else num_sampling_step),
            offload=self.offload if offload is None else bool(offload),
            seed=int(seed),
            shift=float(self.shift if shift is None else shift),
            use_dmd_scheduler=False,
        )

    def _torch_dtype(self, torch_module: Any) -> Any:
        """
        Resolve the configured torch dtype.

        Args:
            torch_module: Imported torch module.
        """
        if isinstance(self.weight_dtype, str):
            return getattr(torch_module, self.weight_dtype)
        return self.weight_dtype

    def _embed_caption(self, caption: str, *, offload_when_prompt: bool) -> Any:
        """
        Embed a prompt with the vendored Lyra-2 text encoder helpers.

        Args:
            caption: Prompt text.
            offload_when_prompt: Whether to use the offloaded embedding helper.
        """
        self._install_runtime_aliases()
        from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
            working_directory,
        )

        desired_device = self.model.tensor_kwargs.get("device", self.device)
        desired_dtype = self.model.tensor_kwargs.get("dtype", self.weight_dtype)
        if self.prompt_suffix:
            caption = caption.rstrip() + " " + self.prompt_suffix
        with working_directory(str(self._runtime_root())):
            from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.get_t5_emb import (
                get_umt5_embedding,
                get_umt5_embedding_offloaded,
            )

            embedding_fn = get_umt5_embedding_offloaded if offload_when_prompt else get_umt5_embedding
            embedding = embedding_fn(caption, device=desired_device).to(dtype=desired_dtype)
        return embedding

    def _build_prompt_batch(
        self,
        prompt: str,
        chunk_captions: Optional[dict[str, str]],
        num_frames: int,
        offload_when_prompt: bool,
    ) -> dict[str, Any]:
        """
        Build Lyra-2 prompt conditioning tensors.

        Args:
            prompt: Base prompt text.
            chunk_captions: Optional per-frame chunk captions.
            num_frames: Number of output frames.
            offload_when_prompt: Whether to use the offloaded embedding helper.
        """
        import torch

        desired_device = self.model.tensor_kwargs.get("device", self.device)
        desired_dtype = self.model.tensor_kwargs.get("dtype", self.weight_dtype)
        neg_t5 = self.negative_prompt_data["t5_text_embeddings"].to(device=desired_device, dtype=desired_dtype)
        chunk_keys_int = (
            sorted(int(key) for key in chunk_captions.keys() if int(key) < num_frames) if chunk_captions else []
        )

        if len(chunk_keys_int) > 1:
            chunk_keys = torch.tensor(chunk_keys_int, dtype=torch.long, device=desired_device)
            chunk_embs = []
            chunk_masks = []
            for key in chunk_keys_int:
                embedding = self._embed_caption(
                    chunk_captions[str(key)] or prompt or "", offload_when_prompt=offload_when_prompt
                )
                if embedding.dim() == 3:
                    embedding = embedding[0]
                seq_len, dim = embedding.shape
                seq_len = min(seq_len, 512)
                dim = min(dim, 4096)
                padded_emb = torch.zeros(512, 4096, dtype=desired_dtype, device=desired_device)
                padded_mask = torch.zeros(512, dtype=desired_dtype, device=desired_device)
                padded_emb[:seq_len, :dim] = embedding[:seq_len, :dim]
                padded_mask[:seq_len] = 1.0
                chunk_embs.append(padded_emb)
                chunk_masks.append(padded_mask)

            t5_chunk_embeddings = torch.stack(chunk_embs).unsqueeze(0)
            t5_chunk_mask = torch.stack(chunk_masks).unsqueeze(0)
            t5_chunk_keys = chunk_keys.unsqueeze(0)
            sample_frame_indices = torch.arange(num_frames, dtype=torch.long, device=desired_device).unsqueeze(0)
            return {
                "t5_text_embeddings": t5_chunk_embeddings[:, 0, :, :],
                "neg_t5_text_embeddings": neg_t5,
                "t5_chunk_keys": t5_chunk_keys,
                "t5_chunk_embeddings": t5_chunk_embeddings,
                "t5_chunk_mask": t5_chunk_mask,
                "sample_frame_indices": sample_frame_indices,
            }

        caption = prompt or ""
        if chunk_keys_int:
            caption = chunk_captions.get(str(chunk_keys_int[0]), caption) or caption
        t5 = self._embed_caption(caption, offload_when_prompt=offload_when_prompt)
        if t5.dim() == 2:
            t5 = t5.unsqueeze(0)
        elif t5.dim() == 3 and t5.shape[0] != 1:
            t5 = t5[:1]
        return {"t5_text_embeddings": t5, "neg_t5_text_embeddings": neg_t5}

    @staticmethod
    def _prepare_camera_tensor(camera_w2c: Any) -> Any:
        """
        Convert camera matrices to a CPU float tensor.

        Args:
            camera_w2c: Camera world-to-camera matrices with shape ``[T,4,4]``.
        """
        import numpy as np
        import torch

        tensor = (
            camera_w2c.detach().cpu().float()
            if torch.is_tensor(camera_w2c)
            else torch.from_numpy(np.asarray(camera_w2c, dtype=np.float32))
        )
        if tensor.ndim != 3 or tensor.shape[-2:] != (4, 4):
            raise ValueError(f"camera_w2c must have shape [T,4,4], got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _prepare_zoom_factors(zoom_factors: Any, num_frames: int) -> Any:
        """
        Convert zoom factors to a CPU float tensor.

        Args:
            zoom_factors: Optional per-frame zoom factors.
            num_frames: Expected number of frames.
        """
        import numpy as np
        import torch

        if zoom_factors is None:
            return torch.ones(num_frames, dtype=torch.float32)
        tensor = (
            zoom_factors.detach().cpu().float()
            if torch.is_tensor(zoom_factors)
            else torch.from_numpy(np.asarray(zoom_factors, dtype=np.float32))
        )
        if tensor.ndim != 1 or tensor.shape[0] != num_frames:
            raise ValueError(f"zoom_factors must have shape [T], got {tuple(tensor.shape)} and num_frames={num_frames}")
        return tensor

    def predict(
        self,
        input_image: Any = None,
        camera_w2c: Any = None,
        prompt: str = "",
        chunk_captions: Optional[dict[str, str]] = None,
        zoom_factors: Any = None,
        fps: int = 16,
        seed: int = 1,
        resolution: Sequence[int] = (480, 832),
        guidance: Optional[float] = None,
        shift: Optional[float] = None,
        num_sampling_step: Optional[int] = None,
        offload: Optional[bool] = None,
        offload_when_prompt: Optional[bool] = None,
        show_progress: bool = True,
        execute: bool = False,
        output_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare or execute a Lyra-2 custom trajectory generation plan.

        Args:
            input_image: Input image path, PIL image, tensor, array, or metadata placeholder.
            camera_w2c: Camera world-to-camera matrices with shape ``[T,4,4]``.
            prompt: Optional text prompt.
            chunk_captions: Optional per-frame chunk captions.
            zoom_factors: Optional per-frame zoom factors.
            fps: Output frames per second.
            seed: Random seed.
            resolution: Target ``(height, width)``.
            guidance: Optional guidance scale override.
            shift: Optional scheduler shift override.
            num_sampling_step: Optional sampling step override.
            offload: Optional model offload override.
            offload_when_prompt: Optional prompt embedding offload override.
            show_progress: Whether official runtime should show progress bars.
            execute: Whether to run already loaded runtime modules.
            output_path: Optional JSON plan output path when not executing.
            **kwargs: Extra metadata preserved in the plan.
        """
        import contextlib

        if not execute:
            return self._write_plan(
                output_path=output_path,
                input_image=input_image,
                camera_w2c=camera_w2c,
                prompt=prompt,
                chunk_captions=chunk_captions,
                zoom_factors=zoom_factors,
                fps=fps,
                seed=seed,
                resolution=resolution,
                guidance=guidance,
                shift=shift,
                num_sampling_step=num_sampling_step,
                offload=offload,
                offload_when_prompt=offload_when_prompt,
                extra=kwargs,
            )
        if self.model is None or self.da3_model is None or self.negative_prompt_data is None:
            raise RuntimeError("Lyra-2 execute=True requires load_runtime=True or preloaded runtime objects.")
        if camera_w2c is None:
            raise ValueError("camera_w2c is required for Lyra-2 execution.")

        import numpy as np
        import torch

        from worldfoundry.core.io import video_tensor_to_uint8_frames
        from worldfoundry.core.media.codecs.image import load_pil_image

        self._install_runtime_aliases()
        with torch.no_grad() if hasattr(torch, "no_grad") else contextlib.nullcontext():
            image = load_pil_image(input_image)
            rgb_t = torch.from_numpy(np.asarray(image, dtype=np.uint8))
        target_h = int(resolution[0])
        target_w = int(resolution[1])
        camera_w2c_t = self._prepare_camera_tensor(camera_w2c)
        num_frames = int(camera_w2c_t.shape[0])
        if (num_frames - 1) % 80 != 0:
            raise ValueError("Lyra-2 requires num_frames to follow 1 + 80k for autoregressive generation.")

        zoom_factors_t = self._prepare_zoom_factors(zoom_factors, num_frames)
        desired_device = self.model.tensor_kwargs.get("device", self.device)
        desired_dtype = self.model.tensor_kwargs.get("dtype", self.weight_dtype)

        from worldfoundry.core.execution import inference_runtime as misc
        from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.lyra2_ar_inference import (
            run_lyra2_sample,
            safe_to,
        )
        from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.lyra2_zoomgs_inference import (
            _da3_infer_depth_intrinsics_single,
        )

        image_chw01, depth_hw, base_k_33, _ = _da3_infer_depth_intrinsics_single(
            da3_model=self.da3_model,
            img_rgb_uint8=rgb_t,
            target_hw=(target_h, target_w),
        )
        intrinsics = base_k_33.unsqueeze(0).repeat(num_frames, 1, 1)
        intrinsics[:, 0, 0] *= zoom_factors_t
        intrinsics[:, 1, 1] *= zoom_factors_t

        prompt_batch = self._build_prompt_batch(
            prompt=prompt,
            chunk_captions=chunk_captions,
            num_frames=num_frames,
            offload_when_prompt=self.offload_when_prompt if offload_when_prompt is None else bool(offload_when_prompt),
        )
        img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0
        data_batch = {
            "video": img_bchw.unsqueeze(2),
            "fps": torch.tensor([fps], dtype=torch.int32, device=desired_device),
            "padding_mask": torch.zeros(
                (1, 1, image_chw01.shape[-2], image_chw01.shape[-1]), dtype=desired_dtype, device=desired_device
            ),
            "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=desired_device),
            "camera_w2c": camera_w2c_t.unsqueeze(0).to(dtype=torch.float32, device=desired_device),
            "intrinsics": intrinsics.unsqueeze(0).to(dtype=torch.float32, device=desired_device),
            "depth": depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1).to(device=desired_device),
            **prompt_batch,
        }
        data_batch = safe_to(
            data_batch,
            device=desired_device,
            dtype=desired_dtype,
            skip_keys={"camera_w2c", "intrinsics", "depth", "t5_chunk_keys", "sample_frame_indices"},
        )
        runtime_args = self._runtime_args(
            num_frames=num_frames,
            seed=seed,
            guidance=guidance,
            shift=shift,
            num_sampling_step=num_sampling_step,
            offload=offload,
        )
        misc.set_random_seed(seed=runtime_args.seed, by_rank=True)
        result = run_lyra2_sample(
            self.model,
            data_batch,
            runtime_args,
            process_group=None,
            da3_model=self.da3_model,
            show_progress=show_progress,
            log_prefix="lyra2_custom_traj",
        )
        video_tensor = result["video"]
        return {
            "video": video_tensor_to_uint8_frames(video_tensor),
            "video_tensor": video_tensor,
            "warp_video": result.get("warp_video"),
            "fps": fps,
            "prompt": prompt,
            "camera_w2c": camera_w2c_t,
            "intrinsics": intrinsics,
            "zoom_factors": zoom_factors_t,
        }

    def _write_plan(
        self,
        *,
        output_path: str | Path | None,
        input_image: Any,
        camera_w2c: Any,
        prompt: str,
        chunk_captions: Optional[dict[str, str]],
        zoom_factors: Any,
        fps: int,
        seed: int,
        resolution: Sequence[int],
        guidance: Optional[float],
        shift: Optional[float],
        num_sampling_step: Optional[int],
        offload: Optional[bool],
        offload_when_prompt: Optional[bool],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Write a Lyra-2 execution plan JSON artifact.

        Args:
            output_path: Optional target JSON path.
            input_image: Input image metadata or path.
            camera_w2c: Camera trajectory metadata.
            prompt: Optional text prompt.
            chunk_captions: Optional per-frame chunk captions.
            zoom_factors: Optional per-frame zoom factors.
            fps: Output frames per second.
            seed: Random seed.
            resolution: Target ``(height, width)``.
            guidance: Optional guidance scale override.
            shift: Optional scheduler shift override.
            num_sampling_step: Optional sampling step override.
            offload: Optional model offload override.
            offload_when_prompt: Optional prompt embedding offload override.
            extra: Extra caller metadata.
        """
        target = Path(output_path) if output_path is not None else Path.cwd() / "lyra2_plan.json"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        camera_shape = tuple(camera_w2c.shape) if hasattr(camera_w2c, "shape") else None
        zoom_shape = tuple(zoom_factors.shape) if hasattr(zoom_factors, "shape") else None
        payload = {
            "status": "blocked",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "backend": "worldfoundry.lyra2.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self._runtime_root()),
            "checkpoint_dir": self.checkpoint_dir,
            "negative_prompt_path": self.negative_prompt_path,
            "da3_model_name": self.da3_model_name,
            "da3_model_path_custom": self.da3_model_path_custom,
            "experiment": self.experiment,
            "device": self.device,
            "prompt": prompt,
            "has_input_image": input_image is not None,
            "camera_w2c_shape": camera_shape,
            "chunk_caption_keys": sorted(chunk_captions.keys()) if chunk_captions else [],
            "zoom_factors_shape": zoom_shape,
            "fps": fps,
            "seed": seed,
            "resolution": [int(resolution[0]), int(resolution[1])],
            "guidance": self.guidance if guidance is None else float(guidance),
            "shift": self.shift if shift is None else float(shift),
            "num_sampling_step": self.num_sampling_step if num_sampling_step is None else int(num_sampling_step),
            "offload": self.offload if offload is None else bool(offload),
            "offload_when_prompt": self.offload_when_prompt
            if offload_when_prompt is None
            else bool(offload_when_prompt),
            "blocked_reasons": list(self.BLOCKED_REASONS),
            "extra": {key: str(value) for key, value in extra.items()},
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "blocked",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "blocked_reasons": payload["blocked_reasons"],
        }


def load_runtime(*args: Any, **kwargs: Any) -> Lyra2Runtime:
    return Lyra2Runtime.from_pretrained(*args, **kwargs)


__all__ = [
    "DEFAULT_DA3_MODEL_NAME",
    "DEFAULT_WEIGHT_DTYPE",
    "Lyra2Runtime",
    "load_runtime",
]
