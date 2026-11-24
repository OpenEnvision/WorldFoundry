"""DAP adapter for the shared in-tree DINOv3 backbone."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn
from torch.nn import functional as F

from worldfoundry.base_models.perception_core.general_perception.dinov3.hub.backbones import (
    DINOv3,
)


class DINOv3Adapter(nn.Module):
    """Expose the feature-map interface expected by the DAP checkpoint."""

    _MODEL_NAMES: ClassVar[set] = {"vits", "vitb", "vitl"}

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if model_name not in self._MODEL_NAMES:
            raise ValueError(
                f"Unsupported DAP DINOv3 backbone {model_name!r}; expected one of {sorted(self._MODEL_NAMES)}."
            )

        self.model = DINOv3(model_name)
        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_size)
        self.blocks = self.model.blocks
        self.n_blocks = int(getattr(self.model, "n_blocks", len(self.blocks)))
        self.depth = self.n_blocks
        self.norm = nn.LayerNorm(self.embed_dim)

    def get_intermediate_layers(
        self,
        value: Tensor,
        n: int | list[int] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        """Return DAP-compatible spatial maps while running shared DINO once."""
        outputs = self.model.get_intermediate_layers(
            value,
            n=n,
            reshape=False,
            return_class_token=True,
            norm=norm,
        )
        grid_height = value.shape[-2] // self.patch_size
        grid_width = value.shape[-1] // self.patch_size
        patch_maps = []
        class_tokens = []

        for patches, class_token in outputs:
            if norm:
                patches = self.norm(patches)

            # DAP's released model was trained with this first-patch removal.
            patches = patches[:, 1:]
            batch, token_count, channels = patches.shape
            square_size = int(token_count**0.5)
            if square_size * square_size == token_count:
                grid = patches.transpose(1, 2).reshape(batch, channels, square_size, square_size)
            else:
                grid = patches.transpose(1, 2).reshape(batch, channels, token_count, 1)
                grid = F.interpolate(
                    grid,
                    size=(grid_height * grid_width, 1),
                    mode="bilinear",
                ).squeeze(-1)
                grid = grid.reshape(batch, channels, grid_height, grid_width)

            if grid.shape[-2:] != (grid_height, grid_width):
                grid = F.interpolate(
                    grid,
                    size=(grid_height, grid_width),
                    mode="bilinear",
                    align_corners=False,
                )
            patch_maps.append(grid.contiguous())
            class_tokens.append(class_token)

        if return_class_token:
            return tuple(zip(patch_maps, class_tokens, strict=False))
        return tuple(patch_maps)
