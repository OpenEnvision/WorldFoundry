# Inference-only DB-CogACT source retained in-tree.
from transformers import PretrainedConfig


# FIXME: is it necessary to build a separate function for vision tower
# or just use transformers AutoModel.from_pretrained
def build_vision_tower(mm_vision_tower, **kwargs):
    """
    Build the vision tower.
    """
    vision_tower = mm_vision_tower

    if isinstance(vision_tower, str):
        if "sig" in vision_tower.lower():
            from .siglip import SiglipVisionTower

            return SiglipVisionTower(vision_tower, **kwargs)

        elif "clip" in vision_tower.lower():
            from .clip import CLIPVisionTower

            return CLIPVisionTower(vision_tower, **kwargs)

        elif "pe" in vision_tower.lower():
            from .pe import PEVisionTower

            return PEVisionTower(vision_tower, **kwargs)
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower}")
    elif isinstance(vision_tower, PretrainedConfig):
        assert getattr(vision_tower, "model_type", None) and "processor_config" in kwargs, (
            'When vision_tower is a dict, it should contain "model_type" key, '
            "and processor_config should be provided in kwargs"
        )
        if "sig" in vision_tower.model_type.lower():
            from .siglip import SiglipVisionTower

            return SiglipVisionTower(vision_tower, **kwargs)
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower.model_type}")
