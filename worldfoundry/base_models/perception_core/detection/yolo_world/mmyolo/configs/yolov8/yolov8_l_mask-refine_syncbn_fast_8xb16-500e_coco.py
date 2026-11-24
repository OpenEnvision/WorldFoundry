_base_ = "./yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py"

# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15
copypaste_prob = 0.3

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform
affine_scale = _base_.affine_scale

model = {
    "backbone": {
        "last_stage_out_channels": last_stage_out_channels,
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "in_channels": [256, 512, last_stage_out_channels],
        "out_channels": [256, 512, last_stage_out_channels],
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor, "in_channels": [256, 512, last_stage_out_channels]}},
}

mosaic_affine_transform = [
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {"type": "YOLOv5CopyPaste", "prob": copypaste_prob},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "max_aspect_ratio": 100.0,
        "scaling_ratio_range": (1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
        "border_val": (114, 114, 114),
        "min_area_ratio": _base_.min_area_ratio,
        "use_mask_refine": _base_.use_mask2refine,
    },
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    {"type": "YOLOv5MixUp", "prob": mixup_prob, "pre_transform": [*pre_transform, *mosaic_affine_transform]},
    *last_transform,
]

train_dataloader = {"dataset": {"pipeline": train_pipeline}}
