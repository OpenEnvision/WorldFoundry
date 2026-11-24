_base_ = "yolov5_s-v61_fast_8xb16-300e_crowdhuman.py"

model = {
    "data_preprocessor": {
        "_delete_": True,
        "type": "mmdet.DetDataPreprocessor",
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "bgr_to_rgb": True,
    },
    "bbox_head": {"ignore_iof_thr": 0.5},
}

img_scale = _base_.img_scale

albu_train_transforms = [
    {"type": "Blur", "p": 0.01},
    {"type": "MedianBlur", "p": 0.01},
    {"type": "ToGray", "p": 0.01},
    {"type": "CLAHE", "p": 0.01},
]

pre_transform = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    # only change this
    {"type": "mmdet.LoadAnnotations", "with_bbox": True},
]

train_pipeline = [
    *pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (0.5, 1.5),
        # img_scale is (width, height)
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
        "border_val": (114, 114, 114),
    },
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
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

train_dataloader = {"collate_fn": {"type": "pseudo_collate"}, "dataset": {"pipeline": train_pipeline}}
