"""MegaSAM pipeline with immutable networks resident across video requests."""

from __future__ import annotations

import gc
import glob
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.base_models.three_dimensions.slam.megasam import (
    checkpoint_path,
    compute_stride,
    depth_anything_checkpoint_path,
    extract_frames,
    runtime_root,
    weights_dir,
)

SCHEMA_VERSION = "worldfoundry.resident_megasam_pipeline"
DEFAULT_DROID_BUFFER_SIZE = 1024


def droid_buffer_size(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return max(DEFAULT_DROID_BUFFER_SIZE, frame_count)


def prepend_sys_path(paths: Iterable[Path]) -> None:
    for path in reversed([str(path) for path in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)


def batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def atomic_save_pose(source: Path, destination: Path, metadata: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz")
    try:
        with np.load(source) as data:
            cam_c2w = data["cam_c2w"]
            np.savez(
                temporary,
                cam_c2w=cam_c2w,
                camera_centers=cam_c2w[:, :3, 3],
                intrinsic=data["intrinsic"],
                stride=metadata["stride"],
                original_fps=metadata["original_fps"],
                effective_fps=metadata["effective_fps"],
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def valid_pose_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            poses = data["cam_c2w"]
            return poses.ndim == 3 and poses.shape[1:] == (4, 4) and poses.shape[0] >= 2
    except Exception:  # noqa: BLE001
        return False


LONG_DIM = 640
UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"


def find_unidepth_weights(weights_root: Path) -> Path:
    explicit = os.environ.get("HARNESSEVAL_UNIDEPTH_WEIGHTS")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"HARNESSEVAL_UNIDEPTH_WEIGHTS does not exist: {path}")
        return path
    asset = BASE_MODEL_CAPABILITIES["unidepth_v2_vitl14"].assets[0]
    status = asset.check()
    if os.environ.get("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR") and not status["env_ready"]:
        raise FileNotFoundError(
            "Registered UniDepth weights missing or incomplete: "
            f"{os.environ['WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR']} (config.json and model.safetensors required)"
        )
    registered = Path(status["matched_path"] or status["local_path"]) / "model.safetensors"
    if registered.is_file():
        return registered
    snapshot = (
        weights_root
        / "huggingface"
        / "hub"
        / "models--lpiccinelli--unidepth-v2-vitl14"
        / "snapshots"
        / UNIDEPTH_REVISION
        / "model.safetensors"
    )
    if snapshot.exists():
        return snapshot
    candidates = sorted(
        (weights_root / "huggingface" / "hub" / "models--lpiccinelli--unidepth-v2-vitl14").glob(
            "snapshots/*/model.safetensors"
        )
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        "UniDepth model.safetensors not found. Set WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR "
        "(config.json and model.safetensors), HARNESSEVAL_UNIDEPTH_WEIGHTS, or populate "
        f"{weights_root}/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/."
    )


def load_unidepth_model(weights_root: Path, device: Any) -> Any:
    from worldfoundry.base_models.three_dimensions.depth.unidepth import UniDepth2Model

    weights_path = find_unidepth_weights(weights_root)
    if weights_path.name != "model.safetensors":
        raise ValueError("The canonical UniDepth loader requires model.safetensors beside config.json")
    prior = UniDepth2Model(type="l", model_path=str(weights_path.parent), device=str(device))
    return prior.model


def resize_unidepth_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[1] > rgb.shape[0]:
        final_w = LONG_DIM
        final_h = int(round(LONG_DIM * rgb.shape[0] / rgb.shape[1]))
    else:
        final_w = int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0]))
        final_h = LONG_DIM
    return cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)


class ResidentMegaSamPipeline:
    """Load all three MegaSAM networks once while recreating scene state per request."""

    def __init__(
        self,
        weights_root: Path | None = None,
        device: str = "cuda",
        target_fps: float = 15.0,
        frame_batch_size: int = 1,
    ) -> None:
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("MegaSAM requires cuda:0 in its worker; select a GPU with CUDA_VISIBLE_DEVICES")
        self.megasam_root = runtime_root()
        self.weights_root = weights_root.resolve() if weights_root else weights_dir()
        self.device_name = device
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.target_fps = target_fps
        if frame_batch_size < 1:
            raise ValueError("frame_batch_size must be positive")
        self.frame_batch_size = frame_batch_size
        configured_tmp_root = os.environ.get("HARNESSEVAL_MEGASAM_TMPDIR")
        self.tmp_root = (
            Path(configured_tmp_root).resolve()
            if configured_tmp_root
            else Path(tempfile.gettempdir()) / "harnesseval-megasam"
        )
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.load_started = time.time()
        self._configure_imports()

        import torch
        import torch.nn.functional as torch_functional
        from torchvision.transforms import Compose

        from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dpt import DPT_DINOv2
        from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.util.transform import (
            NormalizeImage,
            PrepareForNet,
            Resize,
        )

        self.torch = torch
        self.torch_functional = torch_functional
        self.device = torch.device(device)
        self.depth_anything = DPT_DINOv2(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
            localhub=True,
        ).to(self.device)
        depth_weights = self.weights_root / "depth_anything_vitl14.pth"
        if not depth_weights.is_file():
            depth_weights = depth_anything_checkpoint_path()
        self.depth_anything.load_state_dict(
            torch.load(depth_weights, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.depth_anything.eval()
        self.depth_transform = Compose(
            [
                Resize(
                    width=768,
                    height=768,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        self.unidepth = load_unidepth_model(
            self.weights_root,
            self.device,
        )
        self.droid_checkpoint = self.weights_root / "megasam_final.pth"
        if not self.droid_checkpoint.is_file():
            self.droid_checkpoint = checkpoint_path()
        self.droid_class, self.droid_net = self._load_droid_net()
        self.loaded_at = time.time()
        print(
            f"[MEGASAM] resident Depth Anything, UniDepth, and DROIDNet loaded in "
            f"{self.loaded_at - self.load_started:.1f}s "
            f"(frame_batch_size={self.frame_batch_size}, tmp_root={self.tmp_root})",
            flush=True,
        )

    def _configure_imports(self) -> None:
        prepend_sys_path(
            [
                self.megasam_root / "base",
                self.megasam_root / "base" / "droid_slam",
                self.megasam_root / "base" / "thirdparty" / "lietorch",
            ]
        )

    def _load_droid_net(self) -> tuple[Any, Any]:
        from droid import Droid
        from droid_net import DroidNet

        checkpoint = self.droid_checkpoint
        net = DroidNet()
        state_dict = OrderedDict(
            (key.replace("module.", ""), value) for key, value in self.torch.load(checkpoint, map_location="cpu", weights_only=True).items()
        )
        for key in (
            "update.weight.2.weight",
            "update.weight.2.bias",
            "update.delta.2.weight",
            "update.delta.2.bias",
        ):
            state_dict[key] = state_dict[key][:2]
        net.load_state_dict(state_dict, strict=True)
        return Droid, net.to(self.device).eval()

    @staticmethod
    def frame_paths(image_root: Path) -> list[Path]:
        return [Path(path) for path in sorted(glob.glob(str(image_root / "*.jpg")))] + [
            Path(path) for path in sorted(glob.glob(str(image_root / "*.png")))
        ]

    def run_depth_anything(self, image_root: Path, output_root: Path) -> int:
        images = self.frame_paths(image_root)
        if not images:
            raise FileNotFoundError(f"no input frames found under {image_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        with self.torch.no_grad():
            for image_paths in batches(images, self.frame_batch_size):
                transformed = []
                dimensions = []
                for image_path in image_paths:
                    raw_image = cv2.imread(str(image_path))[..., :3]
                    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                    dimensions.append(image.shape[:2])
                    transformed.append(self.depth_transform({"image": image})["image"])
                if len(set(dimensions)) != 1:
                    raise RuntimeError("MegaSAM extracted frames changed dimensions within one video")
                image_tensor = self.torch.from_numpy(np.stack(transformed)).to(self.device)
                depth = self.depth_anything(image_tensor).unsqueeze(1)
                depth = self.torch_functional.interpolate(
                    depth,
                    dimensions[0],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                depth_values = np.float32(depth.cpu().numpy())
                for image_path, depth_value in zip(image_paths, depth_values, strict=True):
                    np.save(output_root / f"{image_path.stem}.npy", depth_value)
        return len(images)

    def run_unidepth(self, image_root: Path, output_root: Path, scene_name: str) -> int:
        from PIL import Image

        images = self.frame_paths(image_root)
        if not images:
            raise FileNotFoundError(f"no input frames found under {image_root}")
        output_scene = output_root / scene_name
        output_scene.mkdir(parents=True, exist_ok=True)
        with self.torch.no_grad():
            for image_paths in batches(images, self.frame_batch_size):
                rgb_values = []
                for image_path in image_paths:
                    with Image.open(image_path) as image:
                        rgb = np.array(image)[..., :3]
                    rgb_values.append(resize_unidepth_rgb(rgb))
                rgb_tensor = self.torch.from_numpy(np.stack(rgb_values)).permute(0, 3, 1, 2).to(self.device)
                predictions = self.unidepth.infer(rgb_tensor)
                depth_values = predictions["depth"][:, 0].detach().cpu().numpy()
                intrinsics_values = predictions["intrinsics"].detach().cpu().numpy()
                prediction_width = predictions["depth"].shape[-1]
                for image_path, depth, intrinsics in zip(
                    image_paths,
                    depth_values,
                    intrinsics_values,
                    strict=True,
                ):
                    fov = np.rad2deg(2 * np.arctan(prediction_width / (2 * intrinsics[0, 0])))
                    np.savez(
                        output_scene / f"{image_path.stem}.npz",
                        depth=np.float32(depth),
                        fov=np.float32(fov),
                    )
        return len(images)

    def run_camera_tracking(
        self,
        frames_dir: Path,
        mono_root: Path,
        metric_root: Path,
        scene_name: str,
        runtime_root: Path,
        buffer_size: int,
    ) -> Path:
        runtime_root.mkdir(parents=True, exist_ok=True)
        output = runtime_root / "outputs" / f"{scene_name}_droid.npz"
        output.unlink(missing_ok=True)
        from .mega_sam_runtime.camera_tracking import run_tracking

        def resident_droid(args: Any) -> Any:
            return self.droid_class(args, net=self.droid_net)

        try:
            run_tracking(
                frames_dir, mono_root, metric_root, scene_name, output,
                buffer_size=buffer_size, droid_factory=resident_droid,
            )
        finally:
            gc.collect()
            self.torch.cuda.empty_cache()
        if not output.is_file():
            raise FileNotFoundError(f"MegaSAM output not found: {output}")
        return output

    def evaluate(self, video: Path, output: Path, target_fps: float | None = None) -> dict[str, Any]:
        target_fps = float(target_fps or self.target_fps)
        scene_name = video.stem
        stride, original_fps, effective_fps = compute_stride(video, target_fps)
        stage_times: dict[str, float] = {}
        started = time.time()
        with tempfile.TemporaryDirectory(
            prefix=f"harnesseval_resident_megasam_{scene_name}_",
            dir=str(self.tmp_root),
        ) as temporary:
            temporary_root = Path(temporary)
            frames_dir = temporary_root / "frames" / scene_name
            mono_root = temporary_root / "mono"
            mono_dir = mono_root / scene_name
            metric_root = temporary_root / "metric"

            stage = time.time()
            frame_count = extract_frames(video, frames_dir, stride)
            tracking_buffer_size = droid_buffer_size(frame_count)
            stage_times["extract_frames"] = round(time.time() - stage, 2)
            stage = time.time()
            self.run_depth_anything(frames_dir, mono_dir)
            stage_times["depth_anything"] = round(time.time() - stage, 2)
            stage = time.time()
            self.run_unidepth(frames_dir, metric_root, scene_name)
            stage_times["unidepth"] = round(time.time() - stage, 2)
            stage = time.time()
            source_pose = self.run_camera_tracking(
                frames_dir,
                mono_root,
                metric_root,
                scene_name,
                temporary_root / "tracking_runtime",
                tracking_buffer_size,
            )
            stage_times["camera_tracking"] = round(time.time() - stage, 2)
            atomic_save_pose(
                source_pose,
                output,
                {
                    "stride": stride,
                    "original_fps": original_fps,
                    "effective_fps": effective_fps,
                },
            )
            source_pose.unlink(missing_ok=True)
        if not valid_pose_cache(output):
            raise RuntimeError(f"resident MegaSAM produced an invalid pose cache: {output}")
        return {
            "schema_version": SCHEMA_VERSION,
            "video_path": str(video),
            "output_path": str(output),
            "frame_count": frame_count,
            "droid_buffer_size": tracking_buffer_size,
            "stride": stride,
            "original_fps": original_fps,
            "effective_fps": effective_fps,
            "frame_batch_size": self.frame_batch_size,
            "tmp_root": str(self.tmp_root),
            "stage_times": stage_times,
            "elapsed_seconds": round(time.time() - started, 2),
            "model_load_scope": "resident_worker_process",
            "model_load_count": {
                "depth_anything": 1,
                "unidepth": 1,
                "droid_net": 1,
            },
        }
