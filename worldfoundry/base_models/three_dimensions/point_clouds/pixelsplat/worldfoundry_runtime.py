"""WorldFoundry runtime adapter for the in-tree pixelSplat source."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import package_module_root as package_root
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.evaluation.utils import worldfoundry_data_path


def _resolve_hfd_root() -> Path:
    """Helper function to resolve hfd root.

    Returns:
        The return value.
    """
    return hfd_root_path()


DEFAULT_PIXELSPLAT_CKPT = _resolve_hfd_root() / "dylanebert--pixelSplat" / "re10k.ckpt"
_PIXELSPLAT_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "pixelsplat")
DEFAULT_PIXELSPLAT_ASSET_ROOT = _PIXELSPLAT_CONFIG_ROOT / "assets"
DEFAULT_PIXELSPLAT_INDEX = DEFAULT_PIXELSPLAT_ASSET_ROOT / "evaluation_index_re10k.json"
DEFAULT_PIXELSPLAT_DEMO_ROOT = DEFAULT_PIXELSPLAT_ASSET_ROOT / "demo_re10k"
DEFAULT_PIXELSPLAT_DEMO_INDEX = DEFAULT_PIXELSPLAT_ASSET_ROOT / "evaluation_index_re10k_demo.json"


def _default_checkpoint_candidates() -> list[Path]:
    """Helper function to default checkpoint candidates.

    Returns:
        The return value.
    """
    hfd_root = _resolve_hfd_root()
    ckpt_root = hfd_root.parent
    return [
        DEFAULT_PIXELSPLAT_CKPT,
        hfd_root / "3DGeneration--pixelSplat" / "re10k.ckpt",
        ckpt_root / "pixelSplat" / "re10k.ckpt",
    ]


class PixelSplatRuntime:
    """WorldFoundry adapter for the packaged official pixelSplat runtime."""

    MODEL_ID = "pixelsplat"
    DISPLAY_NAME = "pixelSplat"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
        dataset_roots: Sequence[str | Path] = (DEFAULT_PIXELSPLAT_DEMO_ROOT,),
        experiment: str = "re10k",
        index_path: str | Path = DEFAULT_PIXELSPLAT_DEMO_INDEX,
    ) -> None:
        """Init.

        Returns:
            The return value.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "3d_reconstruction"
        self.device = device
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser())
        self.dataset_roots = [str(Path(root).expanduser()) for root in dataset_roots]
        self.experiment = experiment
        self.index_path = str(Path(index_path).expanduser())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "PixelSplatRuntime":
        """From pretrained.

        Args:
            pretrained_model_path: The pretrained model path.
            args: The args.
            device: The device.
            model_id: The model id.

        Returns:
            The return value.
        """
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path") or options.get("model_path"),
            dataset_roots=options.get("dataset_roots") or options.get("data_roots") or (DEFAULT_PIXELSPLAT_DEMO_ROOT,),
            experiment=str(options.get("experiment") or "re10k"),
            index_path=options.get("index_path") or DEFAULT_PIXELSPLAT_DEMO_INDEX,
        )

    @staticmethod
    def _runtime_root() -> Path:
        """Helper function to runtime root.

        Returns:
            The return value.
        """
        return package_root("worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat")

    @staticmethod
    def _base_source_root() -> Path:
        """Helper function to base source root.

        Returns:
            The return value.
        """
        return Path(__file__).resolve().parent

    def _resolve_checkpoint(self) -> Path:
        """Helper function to resolve checkpoint.

        Returns:
            The return value.
        """
        if self.checkpoint_path:
            return Path(self.checkpoint_path).expanduser().resolve()
        for candidate in _default_checkpoint_candidates():
            if candidate.is_file():
                return candidate.resolve()
        return DEFAULT_PIXELSPLAT_CKPT

    @staticmethod
    def _input_image_paths(images: Any, kwargs: Mapping[str, Any]) -> list[Path]:
        value = images or kwargs.get("input_path") or kwargs.get("image_path")
        if not value:
            return []
        values = [value] if isinstance(value, (str, Path)) else list(value)
        suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths: list[Path] = []
        for item in values:
            path = Path(str(item)).expanduser()
            if path.is_dir():
                paths.extend(
                    candidate
                    for candidate in sorted(path.iterdir())
                    if candidate.is_file() and candidate.suffix.lower() in suffixes
                )
            elif path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
        return paths

    @staticmethod
    def _stage_demo_re10k(image_paths: Sequence[Path]) -> tuple[Path, Path, list[str]]:
        """Package sequential images as a minimal RE10K-style inference sample."""
        if not image_paths:
            raise ValueError("pixelSplat requires at least one input image")

        import torch
        from PIL import Image, ImageOps

        selected = list(image_paths[:3])
        while len(selected) < 3:
            selected.append(selected[-1])

        stage_root = Path(tempfile.mkdtemp(prefix="worldfoundry_pixelsplat_re10k_"))
        test_root = stage_root / "test"
        test_root.mkdir(parents=True)
        scene = "worldfoundry-demo"

        encoded_images = []
        camera_rows = []
        for index, path in enumerate(selected):
            with Image.open(path) as image:
                image = ImageOps.fit(
                    image.convert("RGB"),
                    (640, 360),
                    method=Image.Resampling.LANCZOS,
                )
                buffer = BytesIO()
                image.save(buffer, format="PNG")
            encoded_images.append(
                torch.frombuffer(buffer.getbuffer(), dtype=torch.uint8).clone()
            )
            # RE10K stores normalized intrinsics followed by a 3x4 world-to-camera
            # matrix.  Sequential demo frames do not ship poses, so use a small,
            # explicit linear baseline and record that fact in result metadata.
            camera_rows.append(
                [
                    0.8,
                    0.8 * 640.0 / 360.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -0.05 * index,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ]
            )

        chunk_name = "000000.torch"
        torch.save(
            [
                {
                    "key": scene,
                    "cameras": torch.tensor(camera_rows, dtype=torch.float32),
                    "images": encoded_images,
                }
            ],
            test_root / chunk_name,
        )
        (test_root / "index.json").write_text(
            json.dumps({scene: chunk_name}, indent=2) + "\n", encoding="utf-8"
        )
        evaluation_index = stage_root / "evaluation_index.json"
        evaluation_index.write_text(
            json.dumps(
                {scene: {"context": [0, 1], "target": [2]}}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        return stage_root, evaluation_index, [str(path) for path in selected]

    def _runtime_env(self) -> dict[str, str]:
        """Helper function to runtime env.

        Returns:
            The return value.
        """
        runtime_root = self._runtime_root()
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH")
        paths = [str(package_root("worldfoundry").parent), str(self._base_source_root()), str(runtime_root)]
        if python_path:
            paths.append(python_path)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["WORLDFOUNDRY_PIXELSPLAT_RUNTIME_CONFIG_ROOT"] = str(_PIXELSPLAT_CONFIG_ROOT)
        env["WORLDFOUNDRY_PIXELSPLAT_ASSET_ROOT"] = str(DEFAULT_PIXELSPLAT_ASSET_ROOT)
        env["WORLDFOUNDRY_PIXELSPLAT_INFERENCE"] = "1"
        return env

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
        """Predict.

        Args:
            prompt: The prompt.
            images: The images.
            video: The video.
            interactions: The interactions.
            output_path: The output path.
            fps: The fps.

        Returns:
            The return value.
        """
        del prompt, fps
        if video is not None:
            raise ValueError("pixelSplat consumes posed image sequences, not input video.")

        checkpoint_path = Path(kwargs.get("checkpoint_path") or self._resolve_checkpoint()).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"pixelSplat checkpoint not found: {checkpoint_path}")

        runtime_root = self._runtime_root()
        output_dir = Path(kwargs.get("output_dir") or Path.cwd() / "pixelsplat")
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = (
            Path(output_path)
            if output_path is not None and Path(output_path).suffix.lower() == ".json"
            else output_dir / "pixelsplat_run.json"
        )
        metadata_path = metadata_path.expanduser().resolve()

        input_images = self._input_image_paths(images, kwargs)
        input_pose_mode = None
        if input_images:
            dataset_root, index_path, staged_images = self._stage_demo_re10k(input_images)
            dataset_roots = [dataset_root]
            input_pose_mode = "synthetic_linear_baseline"
        else:
            index_path = Path(kwargs.get("index_path") or self.index_path).expanduser()
            dataset_roots = kwargs.get("dataset_roots") or self.dataset_roots
            staged_images = []
            if not index_path.is_file():
                raise FileNotFoundError(
                    "pixelSplat demo index is missing and no input images were provided: "
                    f"{index_path}"
                )
        overrides = [
            f"+experiment={kwargs.get('experiment') or self.experiment}",
            "dataset/view_sampler=evaluation",
            f"dataset.view_sampler.index_path={index_path}",
            f"+checkpointing.load={checkpoint_path}",
            f"test.output_path={output_dir}",
            f"dataset.roots=[{','.join(str(Path(root).expanduser()) for root in dataset_roots)}]",
            "data_loader.test.num_workers=0",
        ]
        overrides.extend(str(item) for item in interactions)
        overrides.extend(str(item) for item in kwargs.get("overrides", ()))

        python = str(kwargs.get("python_executable") or sys.executable)
        command = [python, "-m", "src.inference", *overrides]
        subprocess.run(command, cwd=str(runtime_root), env=self._runtime_env(), check=True)

        model_candidates = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ply", ".glb", ".gltf", ".obj", ".splat", ".spz"}
        ]
        artifact_path = model_candidates[0] if model_candidates else metadata_path

        payload = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_3d_asset",
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "run_dir": str(output_dir),
            "runtime": "worldfoundry.pixelsplat.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "checkpoint_path": str(checkpoint_path),
            "index_path": str(index_path),
            "dataset_roots": [str(Path(root).expanduser()) for root in dataset_roots],
            "input_images": staged_images,
            "input_pose_mode": input_pose_mode,
            "command": command,
        }
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload


__all__ = [
    "DEFAULT_PIXELSPLAT_ASSET_ROOT",
    "DEFAULT_PIXELSPLAT_CKPT",
    "DEFAULT_PIXELSPLAT_DEMO_INDEX",
    "DEFAULT_PIXELSPLAT_DEMO_ROOT",
    "DEFAULT_PIXELSPLAT_INDEX",
    "PixelSplatRuntime",
]
