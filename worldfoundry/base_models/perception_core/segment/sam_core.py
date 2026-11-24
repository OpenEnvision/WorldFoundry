from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Any,
        prompt_encoder: Any,
        mask_decoder: Any,
        pixel_mean: list[float] | None = None,
        pixel_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        resolved_pixel_mean = [123.675, 116.28, 103.53] if pixel_mean is None else pixel_mean
        resolved_pixel_std = [58.395, 57.12, 57.375] if pixel_std is None else pixel_std
        self.register_buffer("pixel_mean", torch.Tensor(resolved_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(resolved_pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: list[dict[str, Any]],
        multimask_output: bool,
    ) -> list[dict[str, torch.Tensor]]:
        input_images = torch.stack([self.preprocess(item["image"]) for item in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings, strict=False):
            points = (
                (image_record["point_coords"], image_record["point_labels"]) if "point_coords" in image_record else None
            )
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            outputs.append(
                {
                    "masks": masks > self.mask_threshold,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return F.pad(x, (0, padw, 0, padh))


__all__ = ["Sam"]
