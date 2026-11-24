_base_ = "./yolox-pose_s_8xb32-300e-rtmdet-hyp_coco.py"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco_20230210_143637-4c338102.pth"

deepen_factor = 0.33
widen_factor = 0.375
scaling_ratio_range = (0.75, 1.0)

# model settings
model = {
    "data_preprocessor": {
        "batch_augments": [
            {"type": "YOLOXBatchSyncRandomResize", "random_size_range": (320, 640), "size_divisor": 32, "interval": 1}
        ]
    },
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

# data settings
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform

train_pipeline_stage1 = [
    *pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "RandomAffine",
        "scaling_ratio_range": scaling_ratio_range,
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
    },
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "RandomFlip", "prob": 0.5},
    {"type": "FilterAnnotations", "by_keypoints": True, "min_gt_bbox_wh": (1, 1), "keep_empty": False},
    {"type": "PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape")},
]

test_pipeline = [
    *pre_transform,
    {"type": "Resize", "scale": (416, 416), "keep_ratio": True},
    {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
    {
        "type": "PackDetInputs",
        "meta_keys": ("id", "img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "flip_indices"),
    },
]

train_dataloader = {"dataset": {"pipeline": train_pipeline_stage1}}
val_dataloader = {"dataset": {"pipeline": test_pipeline}}
test_dataloader = val_dataloader
