_base_ = "../yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth"

num_keypoints = 17
scaling_ratio_range = (0.75, 1.0)
mixup_ratio_range = (0.8, 1.6)
num_last_epochs = 20

# model settings
model = {
    "bbox_head": {
        "type": "YOLOXPoseHead",
        "head_module": {
            "type": "YOLOXPoseHeadModule",
            "num_classes": 1,
            "num_keypoints": num_keypoints,
        },
        "loss_pose": {"type": "OksLoss", "metainfo": "configs/_base_/pose/coco.py", "loss_weight": 30.0},
    },
    "train_cfg": {
        "assigner": {
            "type": "PoseSimOTAAssigner",
            "center_radius": 2.5,
            "oks_weight": 3.0,
            "iou_calculator": {"type": "mmdet.BboxOverlaps2D"},
            "oks_calculator": {"type": "OksLoss", "metainfo": "configs/_base_/pose/coco.py"},
        }
    },
    "test_cfg": {"score_thr": 0.01},
}

# pipelines
pre_transform = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "LoadAnnotations", "with_keypoints": True},
]

img_scale = _base_.img_scale

train_pipeline_stage1 = [
    *pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "RandomAffine",
        "scaling_ratio_range": scaling_ratio_range,
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
    },
    {
        "type": "YOLOXMixUp",
        "img_scale": img_scale,
        "ratio_range": mixup_ratio_range,
        "pad_val": 114.0,
        "pre_transform": pre_transform,
    },
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "RandomFlip", "prob": 0.5},
    {"type": "FilterAnnotations", "by_keypoints": True, "keep_empty": False},
    {"type": "PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape")},
]

train_pipeline_stage2 = [
    *pre_transform,
    {"type": "Resize", "scale": img_scale, "keep_ratio": True},
    {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "RandomFlip", "prob": 0.5},
    {"type": "FilterAnnotations", "by_keypoints": True, "keep_empty": False},
    {"type": "PackDetInputs"},
]

test_pipeline = [
    *pre_transform,
    {"type": "Resize", "scale": img_scale, "keep_ratio": True},
    {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
    {
        "type": "PackDetInputs",
        "meta_keys": ("id", "img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "flip_indices"),
    },
]

# dataset settings
dataset_type = "PoseCocoDataset"

train_dataloader = {
    "dataset": {
        "type": dataset_type,
        "data_mode": "bottomup",
        "ann_file": "annotations/person_keypoints_train2017.json",
        "pipeline": train_pipeline_stage1,
    }
}

val_dataloader = {
    "dataset": {
        "type": dataset_type,
        "data_mode": "bottomup",
        "ann_file": "annotations/person_keypoints_val2017.json",
        "pipeline": test_pipeline,
    }
}
test_dataloader = val_dataloader

# evaluators
val_evaluator = {
    "_delete_": True,
    "type": "mmpose.CocoMetric",
    "ann_file": _base_.data_root + "annotations/person_keypoints_val2017.json",
    "score_mode": "bbox",
}
test_evaluator = val_evaluator

default_hooks = {"checkpoint": {"save_best": "coco/AP", "rule": "greater"}}

visualizer = {"type": "mmpose.PoseLocalVisualizer"}

custom_hooks = [
    {
        "type": "YOLOXModeSwitchHook",
        "num_last_epochs": num_last_epochs,
        "new_train_pipeline": train_pipeline_stage2,
        "priority": 48,
    },
    {"type": "mmdet.SyncNormHook", "priority": 48},
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0002,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    },
]
