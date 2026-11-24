from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def segment_boxes(sam_predictor: Any, image: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Segment one mask per box using a SAM-style predictor."""
    sam_predictor.set_image(image)
    masks = []
    for box in boxes_xyxy:
        candidate_masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        masks.append(candidate_masks[int(np.argmax(scores))])
    return np.asarray(masks)


@dataclass
class GroundedSegmentAnything:
    """Compose a GroundingDINO detector with a SAM-compatible predictor."""

    grounding_model: Any
    sam_predictor: Any

    @classmethod
    def from_paths(
        cls,
        *,
        grounding_config_path: str,
        grounding_checkpoint_path: str,
        sam_predictor: Any,
        device: str = "cuda",
    ) -> GroundedSegmentAnything:
        from worldfoundry.base_models.perception_core.detection.grounding_dino.util.inference import Model

        grounding_model = Model(
            model_config_path=grounding_config_path,
            model_checkpoint_path=grounding_checkpoint_path,
            device=device,
        )
        return cls(grounding_model=grounding_model, sam_predictor=sam_predictor)

    def predict(
        self,
        *,
        image: np.ndarray,
        classes: Sequence[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> Any:
        detections = self.grounding_model.predict_with_classes(
            image=image,
            classes=list(classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        detections.mask = segment_boxes(self.sam_predictor, image, detections.xyxy)
        return detections
