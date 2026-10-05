"""Module for the Yume1p5 operator implementation."""

from pathlib import Path
from typing import Any, Dict, Optional, List

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .base_operator import BaseOperator
from ..base_models.diffusion_model.recipes.wan_configs.wan22 import SIZE_CONFIGS, SUPPORTED_SIZES


def _to_rgb_array(image: Any) -> np.ndarray:
    """To rgb array implementation."""
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError(
                "YUME image input expects a single image. Pass multi-frame inputs through the videos argument."
            )
        image = image[0]
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.array(image, copy=True)
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
    array = np.asarray(image)
    if array.ndim == 0:
        raise TypeError(f"Unsupported YUME-1.5 image input: {type(image)!r}")
    if array.ndim == 2:
        array = np.stack((array,) * 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype in (np.float16, np.float32, np.float64):
        if array.min() >= -1.0 and array.max() <= 1.0:
            array = (array + 1.0) * 127.5 if array.min() < 0.0 else array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class Yume1p5Operator(BaseOperator):
    """Lightweight operator for YUME prompt/image preprocessing."""

    def __init__(self, operation_types=None) -> None:
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = []
        super(Yume1p5Operator, self).__init__(operation_types=operation_types)
        self.interaction_template = ["forward", "left", "right", "backward", 
                                     "camera_l", "camera_r", "camera_up", "camera_down"]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True
    
    def get_interaction(self, interactions):
        """Process and append the interaction to the current sequence."""
        if not isinstance(interactions, list):
            interactions = [interactions]
        for interaction in interactions:
            self.check_interaction(interaction)
        self.current_interaction.append(interactions)

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        INTERACTION_2_CAPTION_DICT = {
                                        # movement
                                        "forward": "The camera pushes forward (W).", 
                                        "backward": "The camera pulls back (S).", 
                                        "left": "Camera turns left (←).",
                                        "right": "Camera turns right (→).",
                                        # rotation
                                        "camera_up": "Camera tilts up (↑).", 
                                        "camera_down": "Camera tilts down (↓).",
                                        "camera_l": "The camera pans to the left (←).",
                                        "camera_r": "The camera pans to the right (→).",
                                    }
        

        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return [INTERACTION_2_CAPTION_DICT[act] for act in now_interaction]

    def process_perception(
        self,
        size: Optional[str] = None,
        images: Optional[Image.Image] = None, # None or one PIL image
        videos: Optional[List[Image.Image]] = None # None or list of PIL images from one video
    ) -> Dict[str, Any]:
        
        """Process perception inputs like images, videos, and reference frames."""
        assert size in SUPPORTED_SIZES['ti2v-5B'], f"Unsupported size: {size}. Supported sizes for ti2v-5B are: {SUPPORTED_SIZES['ti2v-5B']}"
        size = SIZE_CONFIGS[size]

        if images is not None:
            images = _to_rgb_array(images)
            
            images_tensor = torch.from_numpy(images).permute(2, 0, 1).float() / 255.0
            resized_images = F.interpolate(
                images_tensor.unsqueeze(0),
                size=size,
                mode='bilinear',
                align_corners=False
            )[0]
        
        if videos:
            video_transform = transforms.ToTensor()
            normalized_frames = []
            for frame in videos:
                if isinstance(frame, Image.Image) and frame.mode != "RGB":
                    frame = frame.convert("RGB")
                normalized_frames.append(video_transform(frame))
            video_pixel_values = torch.stack(normalized_frames, dim=0)
            video_pixel_values = (torch.nn.functional.interpolate(video_pixel_values.sub_(0.5).div_(0.5), size=size, mode='bicubic')).clamp_(-1, 1)
        
        return {
            "ref_images": resized_images if images is not None else None, 
            "ref_videos": video_pixel_values if videos is not None else None
        }
