# Compared to other same scale models, this configuration consumes too much
# GPU memory and is not validated for now
_base_ = "ppyoloe_plus_s_fast_8xb8-80e_coco.py"

data_root = "./data/cat/"
class_name = ("cat",)
num_classes = len(class_name)
metainfo = {"classes": class_name, "palette": [(20, 220, 60)]}

num_last_epochs = 5

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 2

load_from = "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth"

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

default_hooks = {"param_scheduler": {"warmup_min_iter": 10, "warmup_epochs": 3, "total_epochs": int(max_epochs * 1.2)}}

val_evaluator = {"ann_file": data_root + "annotations/test.json"}
test_evaluator = val_evaluator

default_hooks = {
    "checkpoint": {"interval": 10, "max_keep_ckpts": 2, "save_best": "auto"},
    "logger": {"type": "LoggerHook", "interval": 5},
}
train_cfg = {"max_epochs": max_epochs, "val_interval": 10}
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
