from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.base_models.three_dimensions.point_clouds.pi3_source import (
    SOURCE_ROOT as PI3_SOURCE_ROOT,
    ensure_import_paths as ensure_pi3_import_paths,
)
from worldfoundry.core.io import file_sha256

from .variants import (
    WARP_AS_HISTORY_VARIANTS,
    get_warp_as_history_variant,
    runtime_root,
    test_cases_root,
)


class WarpAsHistoryRuntime:
    """WorldFoundry adapter around the vendored official Warp-as-History inference code."""

    MODEL_ID = "warp-as-history"
    DISPLAY_NAME = "Warp-as-History"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        model_path: str | Path | None = None,
        lora_path: str | Path | None = None,
        pi3x_ckpt_path: str | Path | None = None,
        demo_csv_path: str | Path | None = None,
        device: str = "cuda",
        dtype: str | None = None,
        python_executable: str | Path | None = None,
        default_work_dir: str | Path | None = None,
    ) -> None:
        self.variant = get_warp_as_history_variant(model_id)
        self.model_id = self.variant.model_id
        self.model_name = self.variant.display_name
        self.generation_type = self.variant.task
        self.model_path = str(model_path or os.getenv("WAH_HELIOS_DISTILLED_PATH") or self.variant.model_path)
        self.lora_path = str(lora_path or os.getenv("WAH_LORA_PATH") or self.variant.lora_path)
        self.pi3x_ckpt_path = None if pi3x_ckpt_path is None else str(pi3x_ckpt_path)
        self.demo_csv_path = str(demo_csv_path or self.variant.demo_csv_path)
        self.device = str(device)
        self.dtype = str(dtype or self.variant.default_dtype)
        self.python_executable = str(python_executable or sys.executable)
        self.default_work_dir = (
            Path(default_work_dir).expanduser()
            if default_work_dir is not None
            else self._project_root() / "tmp" / "warp_as_history"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "WarpAsHistoryRuntime":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["model_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(
            options.get("model_id")
            or options.get("variant")
            or options.get("profile_id")
            or model_id
            or cls.MODEL_ID
        )
        return cls(
            model_id=resolved_model_id,
            model_path=options.get("model_path") or options.get("helios_path") or options.get("checkpoint_dir"),
            lora_path=options.get("lora_path") or options.get("wah_lora_path"),
            pi3x_ckpt_path=options.get("pi3x_ckpt_path") or options.get("pi3x_path"),
            demo_csv_path=options.get("demo_csv_path") or options.get("csv_path"),
            device=str(device or options.get("device") or "cuda"),
            dtype=options.get("dtype"),
            python_executable=options.get("python_executable") or options.get("python"),
            default_work_dir=options.get("work_dir") or options.get("default_work_dir"),
        )

    @staticmethod
    def available_variants() -> tuple[str, ...]:
        return tuple(sorted(WARP_AS_HISTORY_VARIANTS))

    @staticmethod
    def _project_root() -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").is_file():
                return parent
        return current.parents[5]

    @staticmethod
    def _runtime_root() -> Path:
        return runtime_root().resolve()

    @staticmethod
    def _repo_src_root() -> Path:
        root = WarpAsHistoryRuntime._project_root()
        src_root = root / "src"
        if (src_root / "worldfoundry").is_dir():
            return src_root
        if (root / "worldfoundry").is_dir():
            return root
        return Path(__file__).resolve().parents[5]

    def _resolve_runtime_path(self, value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        for candidate in (self._runtime_root() / path, test_cases_root() / path):
            if candidate.exists():
                return candidate.resolve()
        return self._runtime_root() / path

    def _prepare_work_dir(self, output_path: str | Path | None) -> Path:
        if output_path is not None:
            base = Path(output_path).expanduser()
            if base.suffix:
                return (base.parent / f".{base.stem}_wah_work").resolve()
            return (base / ".wah_work").resolve()
        return (self.default_work_dir / self.model_id).resolve()

    def _target_path(self, output_path: str | Path | None, suffix: str) -> Path:
        if output_path is None:
            target = self.default_work_dir / f"{self.model_id}{suffix}"
        else:
            target = Path(output_path).expanduser()
            if not target.suffix:
                target = target / f"{self.model_id}{suffix}"
        return target.resolve()

    def _plan_path(self, output_path: str | Path | None) -> Path:
        if output_path is None:
            target = self.default_work_dir / f"{self.model_id}.json"
        else:
            target = Path(output_path).expanduser()
            if target.suffix:
                target = target.with_suffix(".json")
            else:
                target = target / f"{self.model_id}.json"
        return target.resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        return file_sha256(path)

    @staticmethod
    def _copy_artifact(source: Path, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def _materialize_image(self, images: Any, work_dir: Path) -> Path | None:
        if images is None:
            return None
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            if not images:
                raise ValueError(f"{self.model_id} received an empty image sequence.")
            images = images[0]
        if isinstance(images, (str, os.PathLike)):
            path = Path(images).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Warp-as-History input image not found: {path}")
            return path.resolve()

        import numpy as np
        from PIL import Image

        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / "first_frame.png"
        if isinstance(images, Image.Image):
            image = images.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(images)).convert("RGB")
        image.save(target)
        return target

    @staticmethod
    def _write_video(path: Path, frames: Any, fps: int) -> Path:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image

        def to_uint8(frame: Any):
            if isinstance(frame, Image.Image):
                return np.asarray(frame.convert("RGB"))
            array = np.asarray(frame)
            if array.ndim != 3:
                raise ValueError(f"Expected video frame with shape [H, W, C], got {array.shape}")
            if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
                array = np.transpose(array, (1, 2, 0))
            if array.shape[-1] == 4:
                array = array[..., :3]
            if array.dtype != np.uint8:
                array = np.clip(array, 0.0, 1.0) * 255.0 if array.max() <= 1.0 else np.clip(array, 0.0, 255.0)
                array = array.round().astype(np.uint8)
            return array

        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(to_uint8(frame))
        return path

    def _materialize_conditioning(
        self,
        *,
        work_dir: Path,
        camera_poses: Any = None,
        camera_poses_path: str | Path | None = None,
        warp_video: Any = None,
        warp_video_path: str | Path | None = None,
        warp_visibility_mask: Any = None,
        warp_visibility_mask_path: str | Path | None = None,
        fps: int,
    ) -> tuple[str, str, str]:
        if camera_poses_path is not None:
            camera_path = str(Path(camera_poses_path).expanduser().resolve())
        elif camera_poses is not None:
            import numpy as np

            camera_path_obj = work_dir / "camera_poses.npz"
            camera_path_obj.parent.mkdir(parents=True, exist_ok=True)
            np.savez(camera_path_obj, camera_poses=np.asarray(camera_poses, dtype=np.float32), fps=int(fps))
            camera_path = str(camera_path_obj)
        else:
            camera_path = ""

        if warp_video_path is not None:
            warp_path = str(Path(warp_video_path).expanduser().resolve())
        elif warp_video is not None:
            warp_path = str(self._write_video(work_dir / "warp_video.mp4", warp_video, fps=fps))
        else:
            warp_path = ""

        if warp_visibility_mask_path is not None:
            mask_path = str(Path(warp_visibility_mask_path).expanduser().resolve())
        elif warp_visibility_mask is not None:
            mask_path = str(self._write_video(work_dir / "warp_visibility_mask.mp4", warp_visibility_mask, fps=fps))
        else:
            mask_path = ""

        return camera_path, warp_path, mask_path

    def _load_demo_row(self, csv_path: str | Path) -> dict[str, str]:
        path = self._resolve_runtime_path(csv_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"{path} must contain exactly one demo row, got {len(rows)}")
        row = {key: value for key, value in rows[0].items()}
        for key in ("first_frame_path", "camera_poses_path", "warp_video_path", "warp_visibility_mask_path"):
            value = (row.get(key) or "").strip()
            if value:
                candidate = Path(value)
                if candidate.is_absolute():
                    row[key] = str(candidate)
                    continue
                candidates = (
                    path.parent / candidate,
                    self._runtime_root() / candidate,
                    test_cases_root() / candidate,
                    Path.cwd() / candidate,
                )
                for resolved in candidates:
                    if resolved.exists():
                        row[key] = str(resolved.resolve())
                        break
                else:
                    row[key] = str((self._runtime_root() / candidate).resolve())
        return row

    def _write_csv(
        self,
        *,
        work_dir: Path,
        prompt: str,
        images: Any,
        camera_poses: Any,
        camera_poses_path: str | Path | None,
        warp_video: Any,
        warp_video_path: str | Path | None,
        warp_visibility_mask: Any,
        warp_visibility_mask_path: str | Path | None,
        fps: int,
        demo_csv_path: str | Path | None,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._materialize_image(images, work_dir)
        camera_path, warp_path, mask_path = self._materialize_conditioning(
            work_dir=work_dir,
            camera_poses=camera_poses,
            camera_poses_path=camera_poses_path,
            warp_video=warp_video,
            warp_video_path=warp_video_path,
            warp_visibility_mask=warp_visibility_mask,
            warp_visibility_mask_path=warp_visibility_mask_path,
            fps=fps,
        )
        if image_path is None and not (camera_path or warp_path):
            row = self._load_demo_row(demo_csv_path or self.demo_csv_path)
            row["prompt"] = prompt or row.get("prompt", "")
        else:
            if image_path is None:
                raise ValueError("Warp-as-History requires an input image when not using a vendored demo CSV.")
            if not (camera_path or warp_path):
                raise ValueError("Warp-as-History requires camera_poses/camera_poses_path or warp_video/warp_video_path.")
            row = {
                "first_frame_path": str(image_path),
                "prompt": prompt or "",
                "camera_poses_path": camera_path,
                "warp_video_path": warp_path,
                "warp_visibility_mask_path": mask_path,
            }

        csv_path = work_dir / "worldfoundry_warp_as_history_input.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "first_frame_path",
                "prompt",
                "camera_poses_path",
                "warp_video_path",
                "warp_visibility_mask_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        return csv_path

    def _ensure_runtime_importable(self) -> None:
        root = str(self._runtime_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        repo_src = str(self._repo_src_root())
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        pi3_root = str(PI3_SOURCE_ROOT)
        if pi3_root not in sys.path:
            sys.path.insert(0, pi3_root)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.pi3x_ckpt_path:
            os.environ["WAH_PI3X_CKPT_PATH"] = self.pi3x_ckpt_path

    def _infer_kwargs(
        self,
        *,
        csv_path: Path,
        target: Path,
        height: int,
        width: int,
        num_frames: int,
        fps: int,
        seed: int,
        dtype: str,
        lora_path: str | None,
        warp_debug_dir: str | Path | None,
        enable_optional_attention: bool,
    ) -> dict[str, Any]:
        self._ensure_runtime_importable()
        from warp_as_history.infer import infer_plan_kwargs, load_demo_row

        sample = load_demo_row(csv_path)
        if sample["warp_video_path"] is None and sample["camera_poses_path"] is not None:
            ensure_pi3_import_paths()

        return infer_plan_kwargs(
            csv_path=csv_path,
            output=target,
            model_path=self.model_path,
            lora_path=lora_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            device=self.device,
            dtype=dtype,
            warp_debug_dir=warp_debug_dir,
            enable_optional_attention=enable_optional_attention,
        )

    def _run_infer(self, infer_kwargs: dict[str, Any], target: Path) -> dict[str, Any]:
        self._ensure_runtime_importable()
        from warp_as_history.infer import run_infer_from_csv

        payload = dict(infer_kwargs)
        payload["csv_path"] = Path(payload["csv_path"])
        payload["output"] = Path(payload["output"])
        warp_debug_dir = payload.get("warp_debug_dir")
        payload["warp_debug_dir"] = None if not warp_debug_dir else Path(warp_debug_dir)

        log_path = target.with_suffix(target.suffix + ".wah.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = run_infer_from_csv(**payload)
        except Exception as exc:
            log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise RuntimeError(
                f"Warp-as-History runner for {self.model_id} failed; see {log_path}"
            ) from exc
        log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def _plan(
        self,
        *,
        prompt: str,
        output_path: str | Path | None,
        infer_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "status": "planned",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "artifact_kind": self.variant.artifact_kind,
            "backend_quality": "execution_plan",
            "runtime": "worldfoundry.synthesis.visual_generation.warp_as_history.in_tree_runtime",
            "runtime_root": str(self._runtime_root()),
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "prompt": prompt,
            "infer_kwargs": infer_kwargs,
            "notes": self.variant.notes,
        }
        if output_path is not None:
            target = self._plan_path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["artifact_path"] = str(target)
        return payload

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video, interactions
        seed_value = kwargs.pop("seed", 42)
        seed = int(42 if seed_value is None else seed_value)
        plan_only = bool(kwargs.pop("plan_only", False))
        height_value = kwargs.pop("height", self.variant.default_height)
        width_value = kwargs.pop("width", self.variant.default_width)
        num_frames_value = kwargs.pop("num_frames", self.variant.default_num_frames)
        height = int(self.variant.default_height if height_value is None else height_value)
        width = int(self.variant.default_width if width_value is None else width_value)
        num_frames = int(self.variant.default_num_frames if num_frames_value is None else num_frames_value)
        fps_value = int(fps or kwargs.pop("fps", self.variant.default_fps))
        dtype = str(kwargs.pop("dtype", self.dtype))
        lora_path = kwargs.pop("lora_path", self.lora_path)
        if kwargs.pop("no_lora", False):
            lora_path = None
        enable_optional_attention = bool(kwargs.pop("enable_optional_attention", False))
        warp_debug_dir = kwargs.pop("warp_debug_dir", None)
        work_dir = self._prepare_work_dir(output_path)
        target = self._target_path(output_path, self.variant.default_extension)
        csv_path = self._write_csv(
            work_dir=work_dir,
            prompt=prompt,
            images=images,
            camera_poses=kwargs.pop("camera_poses", None),
            camera_poses_path=kwargs.pop("camera_poses_path", None),
            warp_video=kwargs.pop("warp_video", None),
            warp_video_path=kwargs.pop("warp_video_path", None),
            warp_visibility_mask=kwargs.pop("warp_visibility_mask", None),
            warp_visibility_mask_path=kwargs.pop("warp_visibility_mask_path", None),
            fps=fps_value,
            demo_csv_path=kwargs.pop("demo_csv_path", None),
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Warp-as-History options: {unknown}")

        infer_kwargs = self._infer_kwargs(
            csv_path=csv_path,
            target=target,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps_value,
            seed=seed,
            dtype=dtype,
            lora_path=None if lora_path is None else str(lora_path),
            warp_debug_dir=warp_debug_dir,
            enable_optional_attention=enable_optional_attention,
        )
        if plan_only:
            return self._plan(prompt=prompt, output_path=output_path, infer_kwargs=infer_kwargs)

        self._run_infer(infer_kwargs, target)
        if not target.is_file():
            raise FileNotFoundError(f"Warp-as-History runner did not produce expected artifact: {target}")
        return {
            "status": "success",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "artifact_kind": self.variant.artifact_kind,
            "artifact_path": str(target),
            "artifact_sha256": self._sha256(target),
            "generated_video_path": str(target),
            "backend_quality": "official_in_tree_runtime",
            "runtime": "worldfoundry.synthesis.visual_generation.warp_as_history.in_tree_runtime",
            "runtime_root": str(self._runtime_root()),
            "model_path": self.model_path,
            "lora_path": lora_path,
            "prompt": prompt,
        }


__all__ = ["WarpAsHistoryRuntime"]
