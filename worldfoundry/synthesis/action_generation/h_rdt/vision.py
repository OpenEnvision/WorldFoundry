# SPDX-License-Identifier: MPL-2.0
"""Local DINOv2 plus SigLIP feature encoder used by H-RDT."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any


def _first_sequence(value: Any) -> Any:
    return value[0] if isinstance(value, Sequence) else value


class DinoSigLIPEncoder:
    """Load both official TIMM backbones from local checkpoint files only."""

    DINO_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
    SIGLIP_ID = "vit_so400m_patch14_siglip_384"

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        device: str,
        dtype: Any,
        image_size: int = 384,
        resize_strategy: str = "letterbox",
    ) -> None:
        import timm
        import torch
        from torchvision.transforms import Compose, Resize

        root = Path(checkpoint_root).expanduser().resolve()
        dino_file = root / self.DINO_ID / "pytorch_model.bin"
        siglip_file = root / self.SIGLIP_ID / "open_clip_pytorch_model.bin"
        missing = [str(path) for path in (dino_file, siglip_file) if not path.is_file()]
        if missing:
            raise FileNotFoundError("H-RDT vision checkpoint files are missing: " + ", ".join(missing))

        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = int(image_size)
        self.resize_strategy = resize_strategy
        self.dino = timm.create_model(
            self.DINO_ID,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(dino_file)},
            num_classes=0,
            img_size=self.image_size,
        )
        self.siglip = timm.create_model(
            self.SIGLIP_ID,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(siglip_file)},
            num_classes=0,
            img_size=self.image_size,
        )
        self.dino.eval()
        self.siglip.eval()
        self.dino.forward = self._unpack(partial(self.dino.get_intermediate_layers, n={len(self.dino.blocks) - 2}))
        self.siglip.forward = self._unpack(
            partial(self.siglip.get_intermediate_layers, n={len(self.siglip.blocks) - 2})
        )

        dino_data = timm.data.resolve_model_data_config(self.dino)
        siglip_data = timm.data.resolve_model_data_config(self.siglip)
        dino_data["input_size"] = (3, self.image_size, self.image_size)
        siglip_data["input_size"] = (3, self.image_size, self.image_size)
        dino_transform = timm.data.create_transform(**dino_data, is_training=False)
        siglip_transform = timm.data.create_transform(**siglip_data, is_training=False)
        if not isinstance(dino_transform, Compose) or not isinstance(siglip_transform, Compose):
            raise TypeError("TIMM returned an unsupported H-RDT image transform")
        if not isinstance(siglip_transform.transforms[0], Resize):
            raise TypeError("TIMM SigLIP transform no longer begins with Resize")
        siglip_transform = Compose(
            [
                Resize(
                    self.image_size,
                    interpolation=siglip_transform.transforms[0].interpolation,
                ),
                *siglip_transform.transforms[1:],
            ]
        )
        self._dino_mean = tuple(int(float(value) * 255) for value in dino_data["mean"])
        self._siglip_mean = tuple(int(float(value) * 255) for value in siglip_data["mean"])
        self._dino_transform = self._prepare_transform(dino_transform, self._dino_mean)
        self._siglip_transform = self._prepare_transform(siglip_transform, self._siglip_mean)
        self.dino.to(device=self.device, dtype=self.dtype)
        self.siglip.to(device=self.device, dtype=self.dtype)
        self.embed_dim = int(self.dino.embed_dim + self.siglip.embed_dim)
        dino_patches = int(self.dino.patch_embed.num_patches)
        siglip_patches = int(self.siglip.patch_embed.num_patches)
        if dino_patches != siglip_patches:
            raise RuntimeError("H-RDT DINO and SigLIP backbones produced different patch grids")
        self.num_patches = dino_patches

    @staticmethod
    def _unpack(function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _first_sequence(function(*args, **kwargs))

        return wrapper

    def _prepare_transform(self, transform: Any, fill: tuple[int, int, int]) -> Any:
        from torchvision.transforms import Compose, Resize

        if self.resize_strategy == "resize-crop":
            return transform
        if self.resize_strategy == "resize-naive":
            first = transform.transforms[0]
            if not isinstance(first, Resize):
                raise TypeError("TIMM image transform no longer begins with Resize")
            return Compose(
                [
                    Resize((self.image_size, self.image_size), interpolation=first.interpolation),
                    *transform.transforms[1:],
                ]
            )
        if self.resize_strategy != "letterbox":
            raise ValueError(f"Unsupported H-RDT resize strategy: {self.resize_strategy!r}")

        class Letterbox:
            def __call__(_self, image: Any) -> Any:
                from PIL import Image, ImageOps

                if not isinstance(image, Image.Image):
                    raise TypeError("H-RDT vision encoder expects PIL RGB images")
                width, height = image.size
                side = max(width, height)
                horizontal = side - width
                vertical = side - height
                return ImageOps.expand(
                    image,
                    border=(
                        horizontal // 2,
                        vertical // 2,
                        horizontal - horizontal // 2,
                        vertical - vertical // 2,
                    ),
                    fill=fill,
                )

        return Compose([Letterbox(), *transform.transforms])

    def encode(self, images: Sequence[Any]) -> Any:
        import torch

        if not images:
            raise ValueError("H-RDT requires at least one RGB image")
        dino_pixels = torch.stack([self._dino_transform(image) for image in images]).to(
            device=self.device,
            dtype=self.dtype,
        )
        siglip_pixels = torch.stack([self._siglip_transform(image) for image in images]).to(
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            dino_features = self.dino(dino_pixels)
            siglip_features = self.siglip(siglip_pixels)
        return torch.cat([dino_features, siglip_features], dim=-1).reshape(
            1,
            -1,
            self.embed_dim,
        )


__all__ = ["DinoSigLIPEncoder"]
