_base_ = "./yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance.py"

# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`
# ========================modified parameters======================
deepen_factor = 1.0
widen_factor = 1.0

mixup_prob = 0.1
copypaste_prob = 0.1

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale

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

pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms
mosaic_affine_pipeline = [
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {"type": "YOLOv5CopyPaste", "prob": copypaste_prob},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (1 - _base_.affine_scale, 1 + _base_.affine_scale),
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
        "border_val": (114, 114, 114),
        "min_area_ratio": _base_.min_area_ratio,
        "max_aspect_ratio": _base_.max_aspect_ratio,
        "use_mask_refine": _base_.use_mask2refine,
    },
]

# enable mixup
train_pipeline = [
    *pre_transform,
    *mosaic_affine_pipeline,
    {"type": "YOLOv5MixUp", "prob": mixup_prob, "pre_transform": [*pre_transform, *mosaic_affine_pipeline]},
    # TODO: support mask transform in albu
    # Geometric transformations are not supported in albu now.
    {
        "type": "mmdet.Albu",
        "transforms": albu_train_transforms,
        "bbox_params": {
            "type": "BboxParams",
            "format": "pascal_voc",
            "label_fields": ["gt_bboxes_labels", "gt_ignore_flags"],
        },
        "keymap": {"img": "image", "gt_bboxes": "bboxes"},
    },
    {"type": "YOLOv5HSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "Polygon2Mask", "downsample_ratio": _base_.downsample_ratio, "mask_overlap": _base_.mask_overlap},
    {"type": "PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction")},
]

train_dataloader = {"dataset": {"pipeline": train_pipeline}}
