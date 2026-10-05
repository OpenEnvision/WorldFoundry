"""Independent in-tree inference runtime for ByteDance ATI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import hfd_root_path

DEFAULT_ATI_REPO = "bytedance-research/ATI"
DEFAULT_WAN_I2V_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"


def _repo_names(repo_id: str) -> tuple[str, ...]:
    leaf = repo_id.rsplit("/", 1)[-1]
    return tuple(dict.fromkeys((repo_id.replace("/", "--"), leaf)))


def _snapshot_candidates(repo_id: str) -> list[Path]:
    candidates: list[Path] = []
    hfd_root = hfd_root_path()
    for name in _repo_names(repo_id):
        candidates.append(hfd_root / name)
    hf_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ).expanduser()
    snapshots = hf_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in sorted(snapshots.iterdir(), reverse=True) if path.is_dir())
    return candidates


def _resolve_local_repo(value: str | Path | None, repo_id: str, label: str) -> Path:
    requested = str(value or repo_id)
    explicit = Path(requested).expanduser()
    if explicit.is_dir():
        return explicit.resolve()
    if "/" not in requested:
        candidate = hfd_root_path(requested)
        if candidate.is_dir():
            return candidate.resolve()
    for candidate in _snapshot_candidates(requested if "/" in requested else repo_id):
        if candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _snapshot_candidates(repo_id))
    raise FileNotFoundError(
        f"{label} checkpoint is not staged locally: {requested!r}. "
        f"WorldFoundry never downloads weights during inference; checked {searched}."
    )


def _symlink_union(primary: Path, fallback: Path | None, destination: Path) -> None:
    """Build a read-only checkpoint view with ATI files taking precedence."""

    destination.mkdir(parents=True, exist_ok=True)
    for source in tuple(path for path in (fallback, primary) if path is not None):
        for child in source.iterdir():
            target = destination / child.name
            if target.exists() or target.is_symlink():
                if source == primary:
                    target.unlink()
                else:
                    continue
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())


def _image(value: Any):
    from PIL import Image

    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0 if array.max(initial=0.0) <= 1.0 else array, 0, 255)
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise TypeError(f"ATI requires an image path, PIL image, or NumPy image; got {type(value).__name__}.")


class ATIRuntime:
    """Load ATI's official trajectory-conditioned Wan2.1 inference path."""

    MODEL_ID = "ati-wan21-14b"
    SOURCE_REVISION = "1a002caf7bb55cfb016dcc670c357bd803af3a0d"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        device: str = "cuda",
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        t5_cpu: bool = False,
        offload_model: bool = True,
    ) -> None:
        self.checkpoint_path = _resolve_local_repo(checkpoint_path, DEFAULT_ATI_REPO, "ATI")
        self.base_model_path = _resolve_local_repo(base_model_path, DEFAULT_WAN_I2V_REPO, "Wan2.1 I2V base")
        self.device = str(device)
        self.rank = int(rank)
        self.offload_model = bool(offload_model)
        self._checkpoint_overlay = tempfile.TemporaryDirectory(prefix="worldfoundry_ati_checkpoint_")
        checkpoint_view = Path(self._checkpoint_overlay.name)
        _symlink_union(self.checkpoint_path, self.base_model_path, checkpoint_view)

        import torch

        from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21 import i2v_14B

        from .ati_runtime.image2video import WanATI

        if not self.device.startswith("cuda"):
            raise ValueError("ATI's official Wan2.1 runtime currently requires a CUDA device.")
        device_id = int(self.device.split(":", 1)[1]) if ":" in self.device else int(torch.cuda.current_device())
        self.core_model = WanATI(
            config=i2v_14B,
            checkpoint_dir=str(checkpoint_view),
            device_id=device_id,
            rank=self.rank,
            t5_fsdp=bool(t5_fsdp),
            dit_fsdp=bool(dit_fsdp),
            use_usp=bool(use_usp),
            t5_cpu=bool(t5_cpu),
        )
        self.config = i2v_14B

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "ATIRuntime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = pretrained_model_path
        options.update(kwargs)
        for key in (
            "adapter",
            "env",
            "model_id",
            "pipeline_binding",
            "profile_id",
            "profile_path",
            "python_executable",
            "repo_root",
            "runtime_profile",
            "variant_id",
        ):
            options.pop(key, None)
        return cls(**options)

    def plan(self) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "source_revision": self.SOURCE_REVISION,
            "runtime": "independent_in_tree_ati",
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "wan_base_code": "worldfoundry/base_models/diffusion_model/models",
        }

    @staticmethod
    def _tracks(
        track_path: str | Path,
        *,
        image_width: int,
        image_height: int,
        track_width: int | None,
        track_height: int | None,
    ):
        from .ati_runtime.motion import load_packed_tracks, process_tracks, unzip_to_array

        packed = load_packed_tracks(str(track_path))
        raw = np.asarray(unzip_to_array(packed), dtype=np.float32).copy()
        if raw.ndim != 4 or raw.shape[1:] != (121, 1, 3):
            raise ValueError(f"ATI track payload must have shape [N,121,1,3], got {raw.shape}.")
        if raw.shape[0] == 0:
            raise ValueError("ATI track payload must contain at least one trajectory.")
        if not np.isfinite(raw).all():
            raise ValueError("ATI track coordinates and visibility must be finite.")
        visibility = raw[..., 2]
        if np.any((visibility < 0) | (visibility > 8)):
            raise ValueError("ATI track visibility must be in [0,1] or upstream quantized [0,8].")
        # Upstream serializes both pixel coordinates and visibility multiplied by 8.
        # Accept ordinary 0..1 visibility as well, so a visible value of 1 does not
        # become only 1/8 visible after process_tracks divides the payload by 8.
        if visibility.size:
            normalized_tracks = visibility.reshape(raw.shape[0], -1).max(axis=1) <= 1
            visibility[normalized_tracks] *= 8
        if track_width and track_height:
            raw[..., 0] *= float(image_width) / float(track_width)
            raw[..., 1] *= float(image_height) / float(track_height)
        return process_tracks(raw, (image_width, image_height), quant_multi=8)

    def predict(
        self,
        *,
        prompt: str,
        image: Any,
        track_path: str | Path,
        output_path: str | Path,
        track_width: int | None = None,
        track_height: int | None = None,
        num_frames: int = 81,
        fps: int = 16,
        width: int = 832,
        height: int = 480,
        seed: int = 42,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        flow_shift: float = 5.0,
        sample_solver: str = "unipc",
        negative_prompt: str = "",
        offload_model: bool | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        if int(num_frames) != 81:
            raise ValueError("ATI's released inference checkpoint and trajectory resampler require exactly 81 output frames.")
        image_value = _image(image)
        image_width, image_height = image_value.size
        tracks = self._tracks(
            track_path,
            image_width=image_width,
            image_height=image_height,
            track_width=track_width,
            track_height=track_height,
        )
        video = self.core_model.generate(
            prompt,
            image_value,
            tracks,
            max_area=int(width) * int(height),
            frame_num=81,
            shift=float(flow_shift),
            sample_solver=str(sample_solver),
            sampling_steps=int(num_inference_steps),
            guide_scale=float(guidance_scale),
            n_prompt=str(negative_prompt),
            seed=int(seed),
            offload_model=self.offload_model if offload_model is None else bool(offload_model),
        )
        if video is None:
            raise RuntimeError("ATI did not return video frames on the current rank.")
        from worldfoundry.core.io import save_image_or_video_tensor

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        saved = save_image_or_video_tensor(video, output, fps=int(fps), value_range=(-1.0, 1.0))
        artifact = Path(saved or output)
        result = {
            "status": "success",
            "model_id": self.MODEL_ID,
            "artifact_kind": "generated_video",
            "artifact_path": str(artifact),
            "generated_video_path": str(artifact),
            "video": video,
            "video_sha256": file_sha256(artifact),
            "runtime_plan": self.plan(),
        }
        return result if return_dict else video

    def close(self) -> None:
        overlay = getattr(self, "_checkpoint_overlay", None)
        if overlay is not None:
            overlay.cleanup()
            self._checkpoint_overlay = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ATIRuntime", "DEFAULT_ATI_REPO", "DEFAULT_WAN_I2V_REPO"]
