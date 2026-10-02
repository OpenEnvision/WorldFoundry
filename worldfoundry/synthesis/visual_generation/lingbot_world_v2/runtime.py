"""Lightweight process facade for the in-tree LingBot-World-V2 runtime."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.core.media.codecs.video import load_video_frames, save_image_or_video_tensor

MODEL_ID = "lingbot-world-v2"
DISPLAY_NAME = "LingBot-World-V2"
DEFAULT_CHECKPOINT = "robbyant/lingbot-world-v2-14b-causal-fast"
SOURCE_COMMIT = "94f43115de8d4a4f9f282126528c300a0b232c5f"
RUNNER_MODULE = "worldfoundry.synthesis.visual_generation.lingbot_world_v2.runner"
SOURCE_ROOT = Path(__file__).resolve().parents[4]
REQUIRED_CHECKPOINT_PATHS = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl",
    "Wan2.1_VAE.pth",
    "config.json",
    "transformers",
)
SUPPORTED_SIZES = frozenset({"720*1280", "1280*720", "480*832", "832*480"})


def _first_image(images: Any) -> Any:
    if isinstance(images, (list, tuple)):
        return images[0] if images else None
    return images


def _checkpoint_missing(root: Path) -> tuple[str, ...]:
    return tuple(relative for relative in REQUIRED_CHECKPOINT_PATHS if not (root / relative).exists())


class LingBotWorldV2Runtime:
    """Launch LingBot-World-V2 in a dedicated distributed subprocess."""

    MODEL_ID = MODEL_ID
    DISPLAY_NAME = DISPLAY_NAME

    def __init__(
        self,
        checkpoint_source: str | Path = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda",
        python_executable: str | Path | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_id = MODEL_ID
        self.model_name = DISPLAY_NAME
        self.generation_type = "image_and_camera_to_world_video"
        self.checkpoint_source = str(checkpoint_source)
        self.device = str(device)
        self.python_executable = str(Path(python_executable or sys.executable).expanduser().resolve())
        self._distributed_core: Any = None
        self._distributed_core_signature: tuple[Any, ...] | None = None
        self.defaults = {
            "size": "480*832",
            "frame_num": 361,
            "chunk_size": 4,
            "seed": 42,
            "sample_shift": 10.0,
            "local_attn_size": 18,
            "sink_size": 6,
            "nproc_per_node": 8,
            "t5_fsdp": True,
            "dit_fsdp": True,
            "offload_model": False,
            "fps": 16,
            "timeout_seconds": 7200,
        }
        self.defaults.update(dict(defaults or {}))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LingBotWorldV2Runtime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_source"] = pretrained_model_path
        options.update(kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or MODEL_ID))
        if resolved_model_id not in {MODEL_ID, "lingbot-world-infinity"}:
            raise ValueError(f"Unsupported LingBot-World-V2 model id: {resolved_model_id}")
        checkpoint_source = options.pop(
            "checkpoint_source",
            options.pop("checkpoint_dir", options.pop("pretrained_model_path", DEFAULT_CHECKPOINT)),
        )
        python_executable = options.pop("python_executable", None)
        options.pop("profile_id", None)
        return cls(
            checkpoint_source,
            device=device,
            python_executable=python_executable,
            defaults=options,
        )

    def _resolve_checkpoint(self, *, required: bool) -> Path | None:
        direct = Path(os.path.expandvars(self.checkpoint_source)).expanduser()
        candidates = [
            direct,
            checkpoint_root_path("lingbot-world-v2-14b-causal-fast"),
            hfd_root_path("robbyant--lingbot-world-v2-14b-causal-fast"),
        ]
        if "/" in self.checkpoint_source and not direct.exists():
            owner, repo = self.checkpoint_source.split("/", 1)
            candidates.extend(
                (
                    hfd_root_path(f"{owner}--{repo}"),
                    hfd_root_path(repo),
                )
            )
            try:
                from worldfoundry.core.io.assets.hf import resolve_hf_snapshot_path

                candidates.insert(
                    0,
                    resolve_hf_snapshot_path(
                        self.checkpoint_source,
                        required_files=REQUIRED_CHECKPOINT_PATHS,
                        local_files_only=True,
                    ),
                )
            except Exception:
                pass
        for candidate in candidates:
            if candidate.is_dir() and not _checkpoint_missing(candidate):
                return candidate.resolve()
        if required:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"LingBot-World-V2 checkpoint is missing or incomplete. Searched: {searched}. "
                f"Expected: {', '.join(REQUIRED_CHECKPOINT_PATHS)}."
            )
        return None

    def runtime_plan(self, **overrides: Any) -> dict[str, Any]:
        options = self._validated_options({**self.defaults, **overrides})
        checkpoint = self._resolve_checkpoint(required=False)
        nproc = int(options["nproc_per_node"])
        return {
            "status": "ready" if checkpoint is not None else "checkpoint_missing",
            "backend": "worldfoundry.lingbot_world_v2.in_tree_causal_fast",
            "backend_quality": "official_in_tree_runtime",
            "model_id": self.model_id,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_dir": str(checkpoint) if checkpoint else None,
            "checkpoint_exists": checkpoint is not None,
            "variant": "14b-causal-fast",
            "nproc_per_node": nproc,
            "sequence_parallel": nproc > 1,
            "source_commit": SOURCE_COMMIT,
            "license": "CC-BY-NC-SA-4.0",
            "options": {key: value for key, value in options.items() if key not in {"timeout_seconds"}},
        }

    def _validated_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(options)
        nproc = int(resolved["nproc_per_node"])
        frame_num = int(resolved["frame_num"])
        chunk_size = int(resolved["chunk_size"])
        local_attn_size = int(resolved["local_attn_size"])
        sink_size = int(resolved["sink_size"])
        if nproc < 1 or 40 % nproc:
            raise ValueError("nproc_per_node must be a positive divisor of the model's 40 attention heads.")
        if str(resolved["size"]) not in SUPPORTED_SIZES:
            raise ValueError(f"Unsupported size {resolved['size']!r}; choose one of {sorted(SUPPORTED_SIZES)}.")
        if frame_num < 5 or (frame_num - 1) % 4:
            raise ValueError("frame_num must be at least 5 and follow the 4n+1 layout.")
        if chunk_size < 1 or ((frame_num - 1) // 4 + 1) < chunk_size:
            raise ValueError("chunk_size must be positive and fit within the latent frame count.")
        if local_attn_size != -1 and local_attn_size < chunk_size:
            raise ValueError("local_attn_size must be -1 or at least chunk_size.")
        if sink_size < 0 or (local_attn_size != -1 and sink_size + chunk_size > local_attn_size):
            raise ValueError("sink_size must leave room for one chunk in the local attention window.")
        if nproc == 1 and (resolved.get("t5_fsdp") or resolved.get("dit_fsdp")):
            raise ValueError("FSDP requires nproc_per_node greater than one.")
        if resolved.get("t5_cpu") and resolved.get("t5_fsdp"):
            raise ValueError("t5_cpu and t5_fsdp are mutually exclusive.")
        if resolved.get("offload_model") and resolved.get("dit_fsdp"):
            raise ValueError("offload_model is not supported together with DiT FSDP.")
        if resolved.get("max_attention_size") is not None and int(resolved["max_attention_size"]) < 1:
            raise ValueError("max_attention_size must be positive when provided.")
        if not self.device.startswith("cuda"):
            raise ValueError("LingBot-World-V2 inference requires CUDA devices.")
        # Under Workspace torchrun every rank is intentionally assigned one
        # local device (``cuda:LOCAL_RANK``).  Counting that rank-local device
        # as the complete launch topology incorrectly rejects a healthy
        # multi-process job before it can reuse the already initialized group.
        active_world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                active_world_size = int(dist.get_world_size())
        except Exception:
            active_world_size = 1
        if active_world_size > 1:
            if active_world_size != nproc:
                raise ValueError(
                    "LingBot-World-V2 nproc_per_node must match the active torchrun "
                    f"WORLD_SIZE; got nproc_per_node={nproc}, WORLD_SIZE={active_world_size}."
                )
        elif self.device.startswith("cuda:"):
            visible = [item for item in self.device.split(":", 1)[1].split(",") if item.strip()]
            if visible and len(visible) < nproc:
                raise ValueError(
                    f"device exposes {len(visible)} GPU(s), but nproc_per_node={nproc}. "
                    "Use device='cuda' with CUDA_VISIBLE_DEVICES or list every GPU after 'cuda:'."
                )
        elif os.getenv("CUDA_VISIBLE_DEVICES"):
            visible = [item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
            if len(visible) < nproc:
                raise ValueError(f"CUDA_VISIBLE_DEVICES exposes {len(visible)} GPU(s), but nproc_per_node={nproc}.")
        return resolved

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(value for value in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if value)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        if self.device.startswith("cuda:"):
            visible = self.device.split(":", 1)[1].strip()
            if visible:
                env["CUDA_VISIBLE_DEVICES"] = visible
        return env

    @staticmethod
    def _materialize_image(temp_dir: Path, images: Any) -> Path:
        import numpy as np
        from PIL import Image

        source = _first_image(images)
        if source is None:
            raise ValueError("LingBot-World-V2 requires an input image.")
        if isinstance(source, (str, os.PathLike)):
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Input image not found: {source_path}")
            return source_path
        if hasattr(source, "detach"):
            tensor = source.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Expected image tensor [C,H,W], got {tuple(tensor.shape)}.")
            array = tensor.permute(1, 2, 0).float().numpy()
            if array.min() < 0:
                array = (array + 1.0) / 2.0
            source = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        image = (
            source.convert("RGB") if hasattr(source, "convert") else Image.fromarray(np.asarray(source)).convert("RGB")
        )
        target = temp_dir / "input.png"
        image.save(target)
        return target

    @staticmethod
    def _resolve_action_source(
        temp_dir: Path,
        *,
        action_path: str | Path | None,
        interactions: Sequence[Any] | Mapping[str, Any] | None,
        c2ws: Any,
        intrinsics: Any,
    ) -> Path:
        import numpy as np

        if action_path is None and isinstance(interactions, Mapping):
            action_path = interactions.get("action_path") or interactions.get("actions_path")
        if action_path is not None:
            action_dir = Path(action_path).expanduser().resolve()
        elif c2ws is not None and intrinsics is not None:
            action_dir = temp_dir / "actions"
            action_dir.mkdir()
            np.save(action_dir / "poses.npy", np.asarray(c2ws, dtype=np.float32))
            np.save(action_dir / "intrinsics.npy", np.asarray(intrinsics, dtype=np.float32))
        else:
            raise ValueError("LingBot-World-V2 requires action_path, or both c2ws and Ks/intrinsics.")
        missing = [name for name in ("poses.npy", "intrinsics.npy") if not (action_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Action directory {action_dir} is missing: {', '.join(missing)}")
        return action_dir

    def build_command(
        self,
        *,
        checkpoint_dir: Path,
        image_path: Path,
        action_path: Path,
        prompt: str,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> list[str]:
        options = self._validated_options(options)
        nproc = int(options["nproc_per_node"])
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "-m",
            RUNNER_MODULE,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--image",
            str(image_path),
            "--action-path",
            str(action_path),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
            "--size",
            str(options["size"]),
            "--frame-num",
            str(int(options["frame_num"])),
            "--chunk-size",
            str(int(options["chunk_size"])),
            "--seed",
            str(int(options["seed"])),
            "--sample-shift",
            str(float(options["sample_shift"])),
            "--local-attn-size",
            str(int(options["local_attn_size"])),
            "--sink-size",
            str(int(options["sink_size"])),
        ]
        if options.get("max_attention_size") is not None:
            command.extend(("--max-attention-size", str(int(options["max_attention_size"]))))
        if options.get("t5_fsdp"):
            command.append("--t5-fsdp")
        if options.get("dit_fsdp"):
            command.append("--dit-fsdp")
        if options.get("t5_cpu"):
            command.append("--t5-cpu")
        if options.get("convert_model_dtype"):
            command.append("--convert-model-dtype")
        command.append("--offload-model" if options.get("offload_model") else "--no-offload-model")
        return command

    @staticmethod
    def _initialized_distributed_context() -> tuple[Any, Any, int, int] | None:
        """Return the active torch process group without creating a nested one."""

        try:
            import torch
            import torch.distributed as dist
        except Exception:
            return None
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return None
        return torch, dist, int(dist.get_rank()), int(dist.get_world_size())

    def _get_distributed_core(
        self,
        *,
        checkpoint: Path,
        options: Mapping[str, Any],
        torch_module: Any,
        rank: int,
        world_size: int,
    ) -> Any:
        """Load one resident causal-fast shard per existing Workspace rank."""

        from .inference import LingBotWorldV2Inference

        local_rank = int(torch_module.cuda.current_device())
        signature = (
            str(checkpoint),
            local_rank,
            world_size,
            bool(options.get("t5_fsdp")),
            bool(options.get("dit_fsdp")),
            bool(options.get("t5_cpu")),
            bool(options.get("convert_model_dtype")),
            int(options["local_attn_size"]),
            int(options["sink_size"]),
        )
        if self._distributed_core is not None and self._distributed_core_signature == signature:
            return self._distributed_core
        if self._distributed_core is not None:
            # Configuration changes are rare, but keeping two 14B cores alive
            # concurrently can exhaust even high-memory accelerators.
            self._distributed_core = None
            self._distributed_core_signature = None
            torch_module.cuda.empty_cache()
        self._distributed_core = LingBotWorldV2Inference(
            checkpoint,
            device_id=local_rank,
            rank=rank,
            t5_fsdp=bool(options.get("t5_fsdp")),
            dit_fsdp=bool(options.get("dit_fsdp")),
            use_sp=world_size > 1,
            t5_cpu=bool(options.get("t5_cpu", False)),
            convert_model_dtype=bool(options.get("convert_model_dtype", False)),
            local_attn_size=int(options["local_attn_size"]),
            sink_size=int(options["sink_size"]),
        )
        self._distributed_core_signature = signature
        return self._distributed_core

    def _predict_in_initialized_process_group(
        self,
        *,
        distributed_context: tuple[Any, Any, int, int],
        prompt: str,
        images: Any,
        interactions: Sequence[Any] | Mapping[str, Any] | None,
        output_path: str | Path | None,
        action_path: str | Path | None,
        c2ws: Any,
        intrinsics: Any,
        pil_image: Any,
        image_tensor: Any,
        options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Execute on an existing Workspace torchrun group; rank zero writes output."""

        torch, dist, rank, world_size = distributed_context
        requested_world_size = int(options["nproc_per_node"])
        if requested_world_size != world_size:
            raise ValueError(
                "LingBot-World-V2 nproc_per_node must match the active torchrun "
                f"WORLD_SIZE; got nproc_per_node={requested_world_size}, WORLD_SIZE={world_size}."
            )
        checkpoint = self._resolve_checkpoint(required=True)
        core = self._get_distributed_core(
            checkpoint=checkpoint,
            options=options,
            torch_module=torch,
            rank=rank,
            world_size=world_size,
        )
        requested_seed = int(options["seed"])
        rank_zero_seed = requested_seed if requested_seed >= 0 else random.randint(0, sys.maxsize)
        seed_payload = [rank_zero_seed if rank == 0 else 0]
        dist.broadcast_object_list(seed_payload, src=0)
        if output_path is not None:
            target = Path(output_path).expanduser()
        elif rank == 0:
            target = Path(tempfile.mkdtemp(prefix="lingbot_world_v2_output_")) / "lingbot-world-v2.mp4"
        else:
            target = Path(tempfile.gettempdir()) / f"lingbot-world-v2-rank-{rank}-not-materialized.mp4"
        if not target.suffix:
            target = target.with_suffix(".mp4")
        target = target.resolve()
        if rank == 0:
            target.parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"worldfoundry_lingbot_v2_rank{rank}_") as temp_raw:
            temp_dir = Path(temp_raw)
            image_path = self._materialize_image(
                temp_dir,
                images if images is not None else pil_image if pil_image is not None else image_tensor,
            )
            resolved_action_path = self._resolve_action_source(
                temp_dir,
                action_path=action_path,
                interactions=interactions,
                c2ws=c2ws,
                intrinsics=intrinsics,
            )
            from PIL import Image

            video = core.generate(
                str(prompt),
                Image.open(image_path).convert("RGB"),
                resolved_action_path,
                chunk_size=int(options["chunk_size"]),
                max_area=self._size_area(str(options["size"])),
                frame_num=int(options["frame_num"]),
                shift=float(options["sample_shift"]),
                seed=int(seed_payload[0]),
                offload_model=bool(options.get("offload_model", False)),
                max_attention_size=(
                    int(options["max_attention_size"])
                    if options.get("max_attention_size") is not None
                    else None
                ),
            )
        if rank != 0:
            return None
        save_image_or_video_tensor(
            video,
            target,
            fps=16,
            value_range=(-1.0, 1.0),
        )
        duration = time.monotonic() - started
        return {
            "artifact_path": str(target),
            "video": load_video_frames(target),
            "prompt": str(prompt),
            "model_id": self.model_id,
            "duration_seconds": round(duration, 3),
            "runtime_plan": self.runtime_plan(**options),
            "execution_mode": "existing_torchrun_process_group",
            "rank": rank,
            "world_size": world_size,
        }

    @staticmethod
    def _size_area(size: str) -> int:
        parts = str(size).split("*")
        if len(parts) != 2:
            raise ValueError(f"Invalid LingBot-World-V2 size: {size!r}")
        return int(parts[0]) * int(parts[1])

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        *,
        action_path: str | Path | None = None,
        c2ws: Any = None,
        Ks: Any = None,
        intrinsics: Any = None,
        pil_image: Any = None,
        image_tensor: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        del video
        options = {**self.defaults, **kwargs}
        if fps is not None:
            if int(fps) != 16:
                raise ValueError("LingBot-World-V2 causal-fast checkpoints use a fixed output rate of 16 FPS.")
            options["fps"] = 16
        options = self._validated_options(options)
        distributed_context = self._initialized_distributed_context()
        if distributed_context is not None:
            return self._predict_in_initialized_process_group(
                distributed_context=distributed_context,
                prompt=str(prompt),
                images=images,
                interactions=interactions,
                output_path=output_path,
                action_path=action_path,
                c2ws=c2ws,
                intrinsics=intrinsics if intrinsics is not None else Ks,
                pil_image=pil_image,
                image_tensor=image_tensor,
                options=options,
            )
        checkpoint = self._resolve_checkpoint(required=True)
        target = (
            Path(output_path).expanduser()
            if output_path is not None
            else Path(tempfile.mkdtemp(prefix="lingbot_world_v2_output_")) / "lingbot-world-v2.mp4"
        )
        if not target.suffix:
            target = target.with_suffix(".mp4")
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="worldfoundry_lingbot_v2_") as temp_raw:
            temp_dir = Path(temp_raw)
            image_path = self._materialize_image(
                temp_dir,
                images if images is not None else pil_image if pil_image is not None else image_tensor,
            )
            resolved_action_path = self._resolve_action_source(
                temp_dir,
                action_path=action_path,
                interactions=interactions,
                c2ws=c2ws,
                intrinsics=intrinsics if intrinsics is not None else Ks,
            )
            generated = temp_dir / "generated.mp4"
            command = self.build_command(
                checkpoint_dir=checkpoint,
                image_path=image_path,
                action_path=resolved_action_path,
                prompt=str(prompt),
                output_path=generated,
                options=options,
            )
            stdout_path = target.with_suffix(".stdout.log")
            stderr_path = target.with_suffix(".stderr.log")
            started = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=SOURCE_ROOT,
                    env=self._subprocess_env(),
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=int(options["timeout_seconds"]),
                    check=False,
                )
            if completed.returncode != 0:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise RuntimeError(
                    f"LingBot-World-V2 runner exited with code {completed.returncode}. "
                    f"See {stdout_path} and {stderr_path}.\n{tail}"
                )
            if not generated.is_file() or generated.stat().st_size == 0:
                raise FileNotFoundError("LingBot-World-V2 runner completed without an MP4 artifact.")
            shutil.copy2(generated, target)
            duration = time.monotonic() - started

        return {
            "artifact_path": str(target),
            "video": load_video_frames(target),
            "prompt": str(prompt),
            "model_id": self.model_id,
            "duration_seconds": round(duration, 3),
            "runtime_plan": self.runtime_plan(**options),
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DISPLAY_NAME",
    "LingBotWorldV2Runtime",
    "MODEL_ID",
    "SOURCE_COMMIT",
]
