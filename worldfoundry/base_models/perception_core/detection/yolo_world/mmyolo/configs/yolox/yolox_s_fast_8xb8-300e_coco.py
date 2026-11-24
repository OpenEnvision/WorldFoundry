_base_ = ["../_base_/default_runtime.py", "yolox_p5_tta.py"]

# ========================Frequently modified parameters======================
# -----data related-----
data_root = "data/coco/"  # Root path of data
# path of train annotation file
train_ann_file = "annotations/instances_train2017.json"
train_data_prefix = "train2017/"  # Prefix of train image path
# path of val annotation file
val_ann_file = "annotations/instances_val2017.json"
val_data_prefix = "val2017/"  # Prefix of train image path

num_classes = 80  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during tarining
train_num_workers = 8
# Presistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
max_epochs = 300  # Maximum training epochs

model_test_cfg = {
    "yolox_style": True,  # better
    # The config of multi-label for multi-class prediction
    "multi_label": True,  # 40.5 -> 40.7
    "score_thr": 0.001,  # Threshold to filter out boxes
    "max_per_img": 300,  # Max number of detections of each image
    "nms": {"type": "nms", "iou_threshold": 0.65},
}  # NMS type and threshold

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = "YOLOv5CocoDataset"
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
norm_cfg = {"type": "BN", "momentum": 0.03, "eps": 0.001}
# generate new random resize shape interval
batch_augments_interval = 10

# -----train val related-----
weight_decay = 0.0005
loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_obj_weight = 1.0
loss_bbox_aux_weight = 1.0
center_radius = 2.5  # SimOTAAssigner
num_last_epochs = 15
random_affine_scaling_ratio_range = (0.1, 2)
mixup_ratio_range = (0.8, 1.6)
# Save model checkpoint and validation intervals
save_epoch_intervals = 10
# The maximum checkpoints to keep.
max_keep_ckpts = 3

ema_momentum = 0.0001

# ===============================Unmodified in most cases====================
# model settings
model = {
    "type": "YOLODetector",
    "init_cfg": {
        "type": "Kaiming",
        "layer": "Conv2d",
        "a": 2.23606797749979,  # math.sqrt(5)
        "distribution": "uniform",
        "mode": "fan_in",
        "nonlinearity": "leaky_relu",
    },
    # TODO: Waiting for mmengine support
    "use_syncbn": False,
    "data_preprocessor": {
        "type": "YOLOv5DetDataPreprocessor",
        "pad_size_divisor": 32,
        "batch_augments": [
            {
                "type": "YOLOXBatchSyncRandomResize",
                "random_size_range": (480, 800),
                "size_divisor": 32,
                "interval": batch_augments_interval,
            }
        ],
    },
    "backbone": {
        "type": "YOLOXCSPDarknet",
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "out_indices": (2, 3, 4),
        "spp_kernal_sizes": (5, 9, 13),
        "norm_cfg": norm_cfg,
        "act_cfg": {"type": "SiLU", "inplace": True},
    },
    "neck": {
        "type": "YOLOXPAFPN",
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "in_channels": [256, 512, 1024],
        "out_channels": 256,
        "norm_cfg": norm_cfg,
        "act_cfg": {"type": "SiLU", "inplace": True},
    },
    "bbox_head": {
        "type": "YOLOXHead",
        "head_module": {
            "type": "YOLOXHeadModule",
            "num_classes": num_classes,
            "in_channels": 256,
            "feat_channels": 256,
            "widen_factor": widen_factor,
            "stacked_convs": 2,
            "featmap_strides": (8, 16, 32),
            "use_depthwise": False,
            "norm_cfg": norm_cfg,
            "act_cfg": {"type": "SiLU", "inplace": True},
        },
        "loss_cls": {
            "type": "mmdet.CrossEntropyLoss",
            "use_sigmoid": True,
            "reduction": "sum",
            "loss_weight": loss_cls_weight,
        },
        "loss_bbox": {
            "type": "mmdet.IoULoss",
            "mode": "square",
            "eps": 1e-16,
            "reduction": "sum",
            "loss_weight": loss_bbox_weight,
        },
        "loss_obj": {
            "type": "mmdet.CrossEntropyLoss",
            "use_sigmoid": True,
            "reduction": "sum",
            "loss_weight": loss_obj_weight,
        },
        "loss_bbox_aux": {"type": "mmdet.L1Loss", "reduction": "sum", "loss_weight": loss_bbox_aux_weight},
    },
    "train_cfg": {
        "assigner": {
            "type": "mmdet.SimOTAAssigner",
            "center_radius": center_radius,
            "iou_calculator": {"type": "mmdet.BboxOverlaps2D"},
        }
    },
    "test_cfg": model_test_cfg,
}

pre_transform = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "LoadAnnotations", "with_bbox": True},
]

train_pipeline_stage1 = [
    *pre_transform,
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "mmdet.RandomAffine",
        "scaling_ratio_range": random_affine_scaling_ratio_range,
        # img_scale is (width, height)
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
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "mmdet.FilterAnnotations", "min_gt_bbox_wh": (1, 1), "keep_empty": False},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

train_pipeline_stage2 = [
    *pre_transform,
    {"type": "mmdet.Resize", "scale": img_scale, "keep_ratio": True},
    {
        "type": "mmdet.Pad",
        "pad_to_square": True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        "pad_val": {"img": (114.0, 114.0, 114.0)},
    },
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "mmdet.FilterAnnotations", "min_gt_bbox_wh": (1, 1), "keep_empty": False},
    {"type": "mmdet.PackDetInputs"},
]

train_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "persistent_workers": persistent_workers,
    "pin_memory": True,
    "collate_fn": {"type": "yolov5_collate"},
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": train_ann_file,
        "data_prefix": {"img": train_data_prefix},
        "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
        "pipeline": train_pipeline_stage1,
    },
}

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "mmdet.Resize", "scale": img_scale, "keep_ratio": True},
    {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
    {"type": "LoadAnnotations", "with_bbox": True, "_scope_": "mmdet"},
    {"type": "mmdet.PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")},
]

val_dataloader = {
    "batch_size": val_batch_size_per_gpu,
    "num_workers": val_num_workers,
    "persistent_workers": persistent_workers,
    "pin_memory": True,
    "drop_last": False,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": val_ann_file,
        "data_prefix": {"img": val_data_prefix},
        "test_mode": True,
        "pipeline": test_pipeline,
    },
}
test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = {
    "type": "mmdet.CocoMetric",
    "proposal_nums": (100, 1, 10),
    "ann_file": data_root + val_ann_file,
    "metric": "bbox",
}

test_evaluator = val_evaluator

# optimizer
# default 8 gpu
optim_wrapper = {
    "type": "OptimWrapper",
    "optimizer": {"type": "SGD", "lr": base_lr, "momentum": 0.9, "weight_decay": weight_decay, "nesterov": True},
    "paramwise_cfg": {"norm_decay_mult": 0.0, "bias_decay_mult": 0.0},
}

# learning rate
param_scheduler = [
    {
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        "type": "mmdet.QuadraticWarmupLR",
        "by_epoch": True,
        "begin": 0,
        "end": 5,
        "convert_to_iter_based": True,
    },
    {
        # use cosine lr from 5 to 285 epoch
        "type": "CosineAnnealingLR",
        "eta_min": base_lr * 0.05,
        "begin": 5,
        "T_max": max_epochs - num_last_epochs,
        "end": max_epochs - num_last_epochs,
        "by_epoch": True,
        "convert_to_iter_based": True,
    },
    {
        # use fixed lr during last 15 epochs
        "type": "ConstantLR",
        "by_epoch": True,
        "factor": 1,
        "begin": max_epochs - num_last_epochs,
        "end": max_epochs,
    },
]

default_hooks = {
    "checkpoint": {
        "type": "CheckpointHook",
        "interval": save_epoch_intervals,
        "max_keep_ckpts": max_keep_ckpts,
        "save_best": "auto",
    }
}

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
        "momentum": ema_momentum,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    },
]

train_cfg = {
    "type": "EpochBasedTrainLoop",
    "max_epochs": max_epochs,
    "val_interval": save_epoch_intervals,
    "dynamic_intervals": [(max_epochs - num_last_epochs, 1)],
}

auto_scale_lr = {"base_batch_size": 8 * train_batch_size_per_gpu}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}
