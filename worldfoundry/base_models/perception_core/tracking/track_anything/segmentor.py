# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> segmentor.py functionality."""

import numpy as np
import torch

from worldfoundry.base_models.perception_core.segment.sam_v1 import SamAutomaticMaskGenerator, sam_model_registry


class Segmentor:
    """Segmentor implementation."""
    def __init__(self, sam_args):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            gpu_id: device
        """
        gpu_id = sam_args["gpu_id"]
        if isinstance(gpu_id, int):
            self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(gpu_id)
        self.sam = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])
        self.sam.to(device=self.device)
        self.everything_generator = SamAutomaticMaskGenerator(model=self.sam, **sam_args["generator_args"])
        self.interactive_predictor = self.everything_generator.predictor
        self.have_embedded = False

    @torch.no_grad()
    def set_image(self, image):
        """Set image.

        Args:
            image: The image.
        """
        # calculate the embedding only once per frame.
        if not self.have_embedded:
            self.interactive_predictor.set_image(image)
            self.have_embedded = True

    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):
        """Interactive predict.

        Args:
            prompts: The prompts.
            mode: The mode.
            multimask: The multimask.
        """
        assert self.have_embedded, "image embedding for sam need be set before predict."

        if mode == "point":
            masks, scores, logits = self.interactive_predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_modes"],
                multimask_output=multimask,
            )
        elif mode == "mask":
            masks, scores, logits = self.interactive_predictor.predict(
                mask_input=prompts["mask_prompt"], multimask_output=multimask
            )
        elif mode == "point_mask":
            masks, scores, logits = self.interactive_predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_modes"],
                mask_input=prompts["mask_prompt"],
                multimask_output=multimask,
            )

        return masks, scores, logits

    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        """

        return:
            mask: one-hot
        """
        self.set_image(origin_frame)

        prompts = {
            "point_coords": coords,
            "point_modes": modes,
        }
        masks, scores, logits = self.interactive_predict(prompts, "point", multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            "point_coords": coords,
            "point_modes": modes,
            "mask_prompt": logit[None, :, :],
        }
        masks, scores, logits = self.interactive_predict(prompts, "point_mask", multimask)
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8)

    def segment_with_box(self, origin_frame, bbox, reset_image=False):
        """Segment with box.

        Args:
            origin_frame: The origin frame.
            bbox: The bbox.
            reset_image: The reset image.
        """
        if reset_image:
            self.interactive_predictor.set_image(origin_frame)
        else:
            self.set_image(origin_frame)
        # coord = np.array([[int((bbox[1][0] - bbox[0][0]) / 2.),  int((bbox[1][1] - bbox[0][1]) / 2)]])
        # point_label = np.array([1])

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]),
            multimask_output=True,
        )
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]]),
            mask_input=logit[None, :, :],
            multimask_output=True,
        )
        mask = masks[np.argmax(scores)]

        return [mask]
