_base_ = ["../_base_/default_runtime.py", "../_base_/det_p5_tta.py"]

# dataset settings
data_root = "data/coco/"
dataset_type = "YOLOv5CocoDataset"

# parameters that often need to be modified
img_scale = (640, 640)  # width, height
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 80
num_classes = 80
save_epoch_intervals = 5
train_batch_size_per_gpu = 8
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
load_from = (
    "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_s_obj365_pretrained-bcfe8478.pth"
)

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Base learning rate for optim_wrapper
base_lr = 0.001

strides = [8, 16, 32]

model = {
    "type": "YOLODetector",
    "data_preprocessor": {
        # use this to support multi_scale training
        "type": "PPYOLOEDetDataPreprocessor",
        "pad_size_divisor": 32,
        "batch_augments": [
            {
                "type": "PPYOLOEBatchRandomResize",
                "random_size_range": (320, 800),
                "interval": 1,
                "size_divisor": 32,
                "random_interp": True,
                "keep_ratio": False,
            }
        ],
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "bgr_to_rgb": True,
    },
    "backbone": {
        "type": "PPYOLOECSPResNet",
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "block_cfg": {"type": "PPYOLOEBasicBlock", "shortcut": True, "use_alpha": True},
        "norm_cfg": {"type": "BN", "momentum": 0.1, "eps": 1e-5},
        "act_cfg": {"type": "SiLU", "inplace": True},
        "attention_cfg": {"type": "EffectiveSELayer", "act_cfg": {"type": "HSigmoid"}},
        "use_large_stem": True,
    },
    "neck": {
        "type": "PPYOLOECSPPAFPN",
        "in_channels": [256, 512, 1024],
        "out_channels": [192, 384, 768],
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "num_csplayer": 1,
        "num_blocks_per_layer": 3,
        "block_cfg": {"type": "PPYOLOEBasicBlock", "shortcut": False, "use_alpha": False},
        "norm_cfg": {"type": "BN", "momentum": 0.1, "eps": 1e-5},
        "act_cfg": {"type": "SiLU", "inplace": True},
        "drop_block_cfg": None,
        "use_spp": True,
    },
    "bbox_head": {
        "type": "PPYOLOEHead",
        "head_module": {
            "type": "PPYOLOEHeadModule",
            "num_classes": num_classes,
            "in_channels": [192, 384, 768],
            "widen_factor": widen_factor,
            "featmap_strides": strides,
            "reg_max": 16,
            "norm_cfg": {"type": "BN", "momentum": 0.1, "eps": 1e-5},
            "act_cfg": {"type": "SiLU", "inplace": True},
            "num_base_priors": 1,
        },
        "prior_generator": {"type": "mmdet.MlvlPointGenerator", "offset": 0.5, "strides": strides},
        "bbox_coder": {"type": "DistancePointBBoxCoder"},
        "loss_cls": {
            "type": "mmdet.VarifocalLoss",
            "use_sigmoid": True,
            "alpha": 0.75,
            "gamma": 2.0,
            "iou_weighted": True,
            "reduction": "sum",
            "loss_weight": 1.0,
        },
        "loss_bbox": {
            "type": "IoULoss",
            "iou_mode": "giou",
            "bbox_format": "xyxy",
            "reduction": "mean",
            "loss_weight": 2.5,
            "return_iou": False,
        },
        # Since the dflloss is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        "loss_dfl": {"type": "mmdet.DistributionFocalLoss", "reduction": "mean", "loss_weight": 0.5 / 4},
    },
    "train_cfg": {
        "initial_epoch": 30,
        "initial_assigner": {
            "type": "BatchATSSAssigner",
            "num_classes": num_classes,
            "topk": 9,
            "iou_calculator": {"type": "mmdet.BboxOverlaps2D"},
        },
        "assigner": {
            "type": "BatchTaskAlignedAssigner",
            "num_classes": num_classes,
            "topk": 13,
            "alpha": 1,
            "beta": 6,
            "eps": 1e-9,
        },
    },
    "test_cfg": {
        "multi_label": True,
        "nms_pre": 1000,
        "score_thr": 0.01,
        "nms": {"type": "nms", "iou_threshold": 0.7},
        "max_per_img": 300,
    },
}

train_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "PPYOLOERandomDistort"},
    {"type": "mmdet.Expand", "mean": (103.53, 116.28, 123.675)},
    {"type": "PPYOLOERandomCrop"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

train_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "persistent_workers": persistent_workers,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "collate_fn": {"type": "yolov5_collate", "use_ms_training": True},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": "annotations/instances_train2017.json",
        "data_prefix": {"img": "train2017/"},
        "filter_cfg": {"filter_empty_gt": True, "min_size": 0},
        "pipeline": train_pipeline,
    },
}

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {
        "type": "mmdet.FixShapeResize",
        "width": img_scale[0],
        "height": img_scale[1],
        "keep_ratio": False,
        "interpolation": "bicubic",
    },
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
        "test_mode": True,
        "data_prefix": {"img": "val2017/"},
        "filter_cfg": {"filter_empty_gt": True, "min_size": 0},
        "ann_file": "annotations/instances_val2017.json",
        "pipeline": test_pipeline,
    },
}

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = {
    "type": "OptimWrapper",
    "optimizer": {"type": "SGD", "lr": base_lr, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False},
    "paramwise_cfg": {"norm_decay_mult": 0.0},
}

default_hooks = {
    "param_scheduler": {
        "type": "PPYOLOEParamSchedulerHook",
        "warmup_min_iter": 1000,
        "start_factor": 0.0,
        "warmup_epochs": 5,
        "min_lr_ratio": 0.0,
        "total_epochs": int(max_epochs * 1.2),
    },
    "checkpoint": {
        "type": "CheckpointHook",
        "interval": save_epoch_intervals,
        "save_best": "auto",
        "max_keep_ckpts": 3,
    },
}

custom_hooks = [
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0002,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    }
]

val_evaluator = {
    "type": "mmdet.CocoMetric",
    "proposal_nums": (100, 1, 10),
    "ann_file": data_root + "annotations/instances_val2017.json",
    "metric": "bbox",
}
test_evaluator = val_evaluator

train_cfg = {"type": "EpochBasedTrainLoop", "max_epochs": max_epochs, "val_interval": save_epoch_intervals}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}
