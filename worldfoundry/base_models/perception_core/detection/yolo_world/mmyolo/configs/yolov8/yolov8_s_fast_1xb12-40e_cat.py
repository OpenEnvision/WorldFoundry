_base_ = "yolov8_s_syncbn_fast_8xb16-500e_coco.py"

data_root = "./data/cat/"
class_name = ("cat",)
num_classes = len(class_name)
metainfo = {"classes": class_name, "palette": [(20, 220, 60)]}

close_mosaic_epochs = 5

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth"

model = {
    "backbone": {"frozen_stages": 4},
    "bbox_head": {"head_module": {"num_classes": num_classes}},
    "train_cfg": {"assigner": {"num_classes": num_classes}},
}

train_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "data_root": data_root,
        "metainfo": metainfo,
        "ann_file": "annotations/trainval.json",
        "data_prefix": {"img": "images/"},
    },
}

val_dataloader = {
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "ann_file": "annotations/test.json",
        "data_prefix": {"img": "images/"},
    }
}

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = {"ann_file": data_root + "annotations/test.json"}
test_evaluator = val_evaluator

default_hooks = {
    "checkpoint": {"interval": 10, "max_keep_ckpts": 2, "save_best": "auto"},
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    "param_scheduler": {"max_epochs": max_epochs, "warmup_mim_iter": 10},
    "logger": {"type": "LoggerHook", "interval": 5},
}
train_cfg = {"max_epochs": max_epochs, "val_interval": 10}
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
