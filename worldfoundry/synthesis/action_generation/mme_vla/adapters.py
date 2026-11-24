import dataclasses

import einops
import numpy as np

from ..openpi import transforms
from ..openpi.modeling import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains non-finite values")
        image = (image + 1.0) * 127.5 if image.min() < 0 else image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    """Convert a RoboMME observation to the MME-VLA inference format."""

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # "right_wrist_0_rgb": np.zeros_like(base_image), # remove the third image for memory saving
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # "right_wrist_0_rgb": np.False_,
            },
            # perceptual memory
            "static_image_emb": data.get("static_image_emb"),  # (budget, d1)
            "static_pos_emb": data.get("static_pos_emb"),  # (budget, d2)
            "static_state_emb": data.get("static_state_emb"),  # (budget, d3)
            "static_mask": data.get("static_mask"),  # (budget)
            # recurrent memory
            "recur_image_emb": data.get("recur_image_emb"),  # (max_recur_steps, views, p, d1)
            "recur_pos_emb": data.get("recur_pos_emb"),  # (max_recur_steps, views, p, d2)
            "recur_state_emb": data.get("recur_state_emb"),  # (max_recur_steps, d3)
            "recur_mask": data.get("recur_mask"),  # (max_recur_steps)
            # symbolic memory
            "simple_subgoal": data.get("simple_subgoal"),
            "grounded_subgoal": data.get("grounded_subgoal"),
        }

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    """Return RoboMME's seven joints plus gripper action."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}  # joint angles + gripper (1 open -1 close)
