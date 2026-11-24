_base_ = "./yolov5u_s_mask-refine_syncbn_fast_8xb16-300e_coco.py"

# This config will refine bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

affine_scale = 0.9
mixup_prob = 0.1
copypaste_prob = 0.1

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform

model = {
    "backbone": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
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

train_pipeline_stage2 = [
    *pre_transform,
    {"type": "YOLOv5KeepRatioResize", "scale": img_scale},
    {"type": "LetterResize", "scale": img_scale, "allow_scale_up": True, "pad_val": {"img": 114.0}},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (1 - affine_scale, 1 + affine_scale),
        "max_aspect_ratio": _base_.max_aspect_ratio,
        "border_val": (114, 114, 114),
        "min_area_ratio": _base_.min_area_ratio,
        "use_mask_refine": _base_.use_mask2refine,
    },
    *last_transform,
]

train_dataloader = {"dataset": {"pipeline": train_pipeline}}
_base_.custom_hooks[1].switch_pipeline = train_pipeline_stage2
