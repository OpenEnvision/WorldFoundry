_base_ = "./yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py"

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.375

# Multi-scale training intervals
# 10 -> 1
batch_augments_interval = 1

scaling_ratio_range = (0.5, 1.5)

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform

# model settings
model = {
    "data_preprocessor": {
        "batch_augments": [
            {
                "type": "YOLOXBatchSyncRandomResize",
                "random_size_range": (320, 640),
                "size_divisor": 32,
                "interval": batch_augments_interval,
            }
        ]
    },
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}

train_pipeline_stage1 = [
    *pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "mmdet.RandomAffine",
        "scaling_ratio_range": scaling_ratio_range,  # note
        # img_scale is (width, height)
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
    },
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "mmdet.FilterAnnotations", "min_gt_bbox_wh": (1, 1), "keep_empty": False},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "mmdet.Resize", "scale": (416, 416), "keep_ratio": True},  # note
    {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
    {"type": "LoadAnnotations", "with_bbox": True, "_scope_": "mmdet"},
    {"type": "mmdet.PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")},
]

train_dataloader = {"dataset": {"pipeline": train_pipeline_stage1}}
val_dataloader = {"dataset": {"pipeline": test_pipeline}}
test_dataloader = val_dataloader
