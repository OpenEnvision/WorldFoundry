_base_ = "../yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py"

max_epochs = 100
data_root = "./data/cat/"

work_dir = "./work_dirs/yolov7_tiny_syncbn_fast_1xb32-100e_cat"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth"

train_batch_size_per_gpu = 32
train_num_workers = 4  # train_num_workers = nGPU x 4

save_epoch_intervals = 2

# base_lr_default * (your_bs / default_bs)
base_lr = 0.01 / 4

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)],  # P5/32
]

class_name = ("cat",)
num_classes = len(class_name)
metainfo = {"classes": class_name, "palette": [(220, 20, 60)]}

train_cfg = {
    "max_epochs": max_epochs,
    "val_begin": 20,
    "val_interval": save_epoch_intervals,
    "dynamic_intervals": [(max_epochs - 10, 1)],
}

model = {
    "bbox_head": {
        "head_module": {"num_classes": num_classes},
        "prior_generator": {"base_sizes": anchors},
        "loss_cls": {"loss_weight": 0.5 * (num_classes / 80 * 3 / _base_.num_det_layers)},
    }
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
        "max_keep_ckpts": 2,
        "save_best": "auto",
    },
    "param_scheduler": {"max_epochs": max_epochs},
    "logger": {"type": "LoggerHook", "interval": 10},
}
