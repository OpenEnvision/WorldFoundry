"""Module for base_models -> three_dimensions -> general_3d -> splatt3r -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256, write_json
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.core.io.paths import package_module_root as package_root

DEFAULT_SPLATT3R_REPO = "brandonsmart/splatt3r_v1.0"
DEFAULT_SPLATT3R_FILENAME = "epoch=19-step=1200.ckpt"


def _resolve_hfd_root() -> Path:
    """Helper function to resolve hfd root.

    Returns:
        The return value.
    """
    return hfd_root_path()


DEFAULT_SPLATT3R_LOCAL_CKPT = _resolve_hfd_root() / "brandonsmart--splatt3r_v1.0" / DEFAULT_SPLATT3R_FILENAME


def _default_checkpoint_candidates() -> list[Path]:
    """Helper function to default checkpoint candidates.

    Returns:
        The return value.
    """
    hfd_root = _resolve_hfd_root()
    ckpt_root = hfd_root.parent
    return [
        DEFAULT_SPLATT3R_LOCAL_CKPT,
        hfd_root / "3DGeneration--splatt3r_v1.0" / DEFAULT_SPLATT3R_FILENAME,
        ckpt_root / "splatt3r_v1.0" / DEFAULT_SPLATT3R_FILENAME,
    ]


class Splatt3RRuntime:
    """WorldFoundry adapter for Splatt3R using shared MASt3R and PixelSplat integrations."""

    MODEL_ID = "splatt3r"
    DISPLAY_NAME = "Splatt3R"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
        image_size: int = 512,
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
        self.image_size = int(image_size)
        self._model = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "Splatt3RRuntime":
        """
        Build a lazy Splatt3R runtime.

        Args:
            pretrained_model_path: Optional checkpoint path or mapping with runtime options.
            args: Unused compatibility argument for pipeline factories.
            device: Target torch device.
            model_id: Runtime profile id, defaults to ``splatt3r``.
            **kwargs: Optional ``checkpoint_path`` and ``image_size`` values.
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
            image_size=int(options.get("image_size", 512)),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """Helper function to runtime root.

        Returns:
            The return value.
        """
        return package_root("worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime")

    def _resolve_checkpoint(self) -> str:
        """Helper function to resolve checkpoint.

        Returns:
            The return value.
        """
        if self.checkpoint_path:
            return self.checkpoint_path
        for candidate in _default_checkpoint_candidates():
            if candidate.is_file():
                return str(candidate.resolve())
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=DEFAULT_SPLATT3R_REPO, filename=DEFAULT_SPLATT3R_FILENAME)

    def _ensure_model(self):
        """
        Load the Splatt3R model once, with shared MASt3R and PixelSplat imports.

        Args:
            None. Runtime paths are resolved from the synthesis instance state.
        """
        if self._model is not None:
            return self._model
        import torch

        from worldfoundry.base_models.three_dimensions.general_3d.mast3r import ensure_import_paths
        from worldfoundry.core.model_loading import load_torch_checkpoint

        ensure_import_paths()
        from omegaconf import OmegaConf

        from worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime.model import (
            MAST3RGaussians,
        )

        device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        # Splatt3R ships a PyTorch-Lightning checkpoint whose ``hyper_parameters``
        # payload contains pickled config objects, so the weights-only attempt
        # needs the explicit unsafe-pickle fallback for this pinned artifact
        # (plan/code_review/11_vendored_integration.md [VI-23]).
        checkpoint = load_torch_checkpoint(
            self._resolve_checkpoint(), map_location="cpu", allow_unsafe_pickle_fallback=True
        )
        hyper_parameters = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
        config_payload = hyper_parameters.get("config", {}) if isinstance(hyper_parameters, Mapping) else {}
        config = OmegaConf.create(config_payload)
        if "use_offsets" not in config:
            config.use_offsets = False
        self._model = MAST3RGaussians(config)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(device)
        self._model.eval()
        self.device = device
        return self._model

    @staticmethod
    def _image_paths(images: Any, interactions: Sequence[str]) -> list[str]:
        """Helper function to image paths.

        Args:
            images: The images.
            interactions: The interactions.

        Returns:
            The return value.
        """
        if images is None:
            paths = list(interactions)
        elif isinstance(images, (str, Path)):
            image_path = Path(images).expanduser()
            if image_path.is_dir():
                suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                paths = [str(path) for path in sorted(image_path.iterdir()) if path.suffix.lower() in suffixes]
            else:
                paths = [str(images)]
        elif isinstance(images, Sequence):
            paths = [str(item) for item in images]
        else:
            raise TypeError(f"Splatt3R expects one or two image paths, got {type(images).__name__}.")
        paths = [path for path in paths if path]
        if len(paths) > 2:
            paths = paths[:2]
        if len(paths) == 1:
            paths = [paths[0], paths[0]]
        if len(paths) != 2:
            raise ValueError("Splatt3R requires one or two image paths.")
        return paths

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
        """
        Reconstruct a Gaussian splat PLY from one or two images.

        Args:
            prompt: Unused compatibility field.
            images: One image path or a sequence of two image paths.
            video: Must be ``None`` because Splatt3R consumes images.
            interactions: Optional fallback image path sequence.
            output_path: Target PLY file path.
            fps: Unused compatibility field.
            **kwargs: Optional runtime parameters.
        """
        plan_only = bool(kwargs.pop("plan_only", False))
        del prompt, fps
        if video is not None:
            raise ValueError("Splatt3R reconstructs from images and does not accept input video.")
        if images is None:
            images = kwargs.get("input_path") or kwargs.get("image_path") or kwargs.get("data_path")
        target = Path(output_path) if output_path is not None else Path.cwd() / "splatt3r.ply"
        target = target.expanduser().resolve()
        if target.suffix.lower() != ".ply":
            target = target.with_suffix(".ply")
        if plan_only:
            plan_path = target.with_suffix(target.suffix + ".plan.json")
            plan_payload = {
                "schema_version": "worldfoundry-splatt3r-runtime-plan",
                "model_id": self.model_id,
                "artifact_kind": "generated_3d_asset",
                "artifact_path": str(target),
                "checkpoint_path": self.checkpoint_path or str(DEFAULT_SPLATT3R_LOCAL_CKPT),
                "image_size": self.image_size,
                "runtime": "worldfoundry.splatt3r.in_tree_runtime",
            }
            write_json(plan_path, plan_payload)
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": "generated_3d_asset",
                "artifact_path": str(target),
                "plan_path": str(plan_path),
                "runtime": "worldfoundry.splatt3r.in_tree_runtime",
                "backend_quality": "in_tree_runtime",
            }

        import torch

        model = self._ensure_model()
        from dust3r.utils.image import load_images

        from worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime.utils import export

        image_paths = self._image_paths(images, interactions)
        loaded = load_images(image_paths, size=self.image_size, verbose=False)
        for item in loaded:
            item["img"] = item["img"].to(self.device)
            if "original_img" not in item:
                # DUSt3R normalizes encoder inputs to [-1, 1], while the
                # Gaussian color residual is anchored to RGB in [0, 1].
                item["original_img"] = item["img"].add(1).mul(0.5)
            elif hasattr(item["original_img"], "to"):
                item["original_img"] = item["original_img"].to(self.device)
            item["true_shape"] = torch.from_numpy(item["true_shape"])
        pred1, pred2 = model(loaded[0], loaded[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        export.save_as_ply(pred1, pred2, str(target))
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_3d_asset",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "runtime": "worldfoundry.splatt3r.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "image_paths": image_paths,
        }


__all__ = [
    "DEFAULT_SPLATT3R_FILENAME",
    "DEFAULT_SPLATT3R_LOCAL_CKPT",
    "DEFAULT_SPLATT3R_REPO",
    "Splatt3RRuntime",
]
