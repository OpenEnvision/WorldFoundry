_base_ = "./yolov6_s_syncbn_fast_8xb32-400e_coco.py"

data_root = "./data/cat/"
class_name = ("cat",)
num_classes = len(class_name)
metainfo = {"classes": class_name, "palette": [(20, 220, 60)]}

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4
num_last_epochs = 5

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth"

model = {
    "backbone": {"frozen_stages": 4},
    "bbox_head": {"head_module": {"num_classes": num_classes}},
    "train_cfg": {"initial_assigner": {"num_classes": num_classes}, "assigner": {"num_classes": num_classes}},
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

val_evaluator = {"ann_file": data_root + "annotations/test.json"}
test_evaluator = val_evaluator

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

default_hooks = {
    "checkpoint": {"interval": 10, "max_keep_ckpts": 2, "save_best": "auto"},
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    "param_scheduler": {"max_epochs": max_epochs, "warmup_mim_iter": 10},
    "logger": {"type": "LoggerHook", "interval": 5},
}
train_cfg = {"max_epochs": max_epochs, "val_interval": 10, "dynamic_intervals": [(max_epochs - num_last_epochs, 1)]}
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
