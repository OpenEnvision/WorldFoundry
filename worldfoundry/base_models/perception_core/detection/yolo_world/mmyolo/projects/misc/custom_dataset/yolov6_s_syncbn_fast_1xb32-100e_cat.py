_base_ = "../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py"

max_epochs = 100
data_root = "./data/cat/"

work_dir = "./work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth"

train_batch_size_per_gpu = 32
train_num_workers = 4  # train_num_workers = nGPU x 4

save_epoch_intervals = 2

# base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 8

class_name = ("cat",)
num_classes = len(class_name)
metainfo = {"classes": class_name, "palette": [(220, 20, 60)]}

train_cfg = {
    "max_epochs": max_epochs,
    "val_begin": 20,
    "val_interval": save_epoch_intervals,
    "dynamic_intervals": [(max_epochs - _base_.num_last_epochs, 1)],
}

model = {
    "bbox_head": {"head_module": {"num_classes": num_classes}},
    "train_cfg": {"initial_assigner": {"num_classes": num_classes}, "assigner": {"num_classes": num_classes}},
}

train_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "_delete_": True,
        "type": "RepeatDataset",
        "times": 5,
        "dataset": {
            "type": _base_.dataset_type,
            "data_root": data_root,
            "metainfo": metainfo,
            "ann_file": "annotations/trainval.json",
            "data_prefix": {"img": "images/"},
            "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
            "pipeline": _base_.train_pipeline,
        },
    },
}

val_dataloader = {
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "ann_file": "annotations/trainval.json",
        "data_prefix": {"img": "images/"},
    }
}

test_dataloader = val_dataloader

val_evaluator = {"ann_file": data_root + "annotations/trainval.json"}
test_evaluator = val_evaluator

optim_wrapper = {"optimizer": {"lr": base_lr}}

default_hooks = {
    "checkpoint": {
        "type": "CheckpointHook",
        "interval": save_epoch_intervals,
        "max_keep_ckpts": 5,
        "save_best": "auto",
    },
    "param_scheduler": {"max_epochs": max_epochs},
    "logger": {"type": "LoggerHook", "interval": 10},
}

custom_hooks = [
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0001,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    },
    {
        "type": "mmdet.PipelineSwitchHook",
        "switch_epoch": max_epochs - _base_.num_last_epochs,
        "switch_pipeline": _base_.train_pipeline_stage2,
    },
]
