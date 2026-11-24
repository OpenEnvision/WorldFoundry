_base_ = "yolov5_s-v61_fast_1xb12-40e_cat.py"

# This configuration is used to provide non-square training examples
# Must be a multiple of 32
img_scale = (608, 352)  # w h

anchors = [
    [(65, 35), (159, 45), (119, 80)],  # P3/8
    [(215, 77), (224, 116), (170, 166)],  # P4/16
    [(376, 108), (339, 176), (483, 190)],  # P5/32
]

# ===============================Unmodified in most cases====================
_base_.model.bbox_head.loss_obj.loss_weight = 1.0 * ((img_scale[1] / 640) ** 2)
_base_.model.bbox_head.prior_generator.base_sizes = anchors

train_pipeline = [
    *_base_.pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": _base_.pre_transform},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
        "border_val": (114, 114, 114),
    },
    {
        "type": "mmdet.Albu",
        "transforms": _base_.albu_train_transforms,
        "bbox_params": {
            "type": "BboxParams",
            "format": "pascal_voc",
            "label_fields": ["gt_bboxes_labels", "gt_ignore_flags"],
        },
        "keymap": {"img": "image", "gt_bboxes": "bboxes"},
    },
    {"type": "YOLOv5HSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

_base_.train_dataloader.dataset.pipeline = train_pipeline

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "YOLOv5KeepRatioResize", "scale": img_scale},
    {"type": "LetterResize", "scale": img_scale, "allow_scale_up": False, "pad_val": {"img": 114}},
    {"type": "mmdet.LoadAnnotations", "with_bbox": True},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "pad_param"),
    },
]

val_dataloader = {"dataset": {"pipeline": test_pipeline, "batch_shapes_cfg": None}}
test_dataloader = val_dataloader
