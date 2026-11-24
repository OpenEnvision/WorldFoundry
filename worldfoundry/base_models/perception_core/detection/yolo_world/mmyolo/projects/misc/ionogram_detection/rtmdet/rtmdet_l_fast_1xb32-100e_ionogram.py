_base_ = "mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py"

# ======================== Modified parameters ======================
# -----data related-----
data_root = "./Iono4311/"
train_ann_file = "annotations/train.json"
train_data_prefix = "train_images/"
val_ann_file = "annotations/val.json"
val_data_prefix = "val_images/"
test_ann_file = "annotations/test.json"
test_data_prefix = "test_images/"

class_name = ("E", "Es-l", "Es-c", "F1", "F2", "Spread-F")
num_classes = len(class_name)
metainfo = {
    "classes": class_name,
    "palette": [(250, 165, 30), (120, 69, 125), (53, 125, 34), (0, 11, 123), (130, 20, 12), (120, 121, 80)],
}

train_batch_size_per_gpu = 32
train_num_workers = 8
val_batch_size_per_gpu = train_batch_size_per_gpu

# Config of batch shapes. Only on val.
batch_shapes_cfg = {"batch_size": val_batch_size_per_gpu}

# -----train val related-----
load_from = "https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth"

# default hooks
save_epoch_intervals = 10
max_epochs = 100
max_keep_ckpts = 1

# learning rate
param_scheduler = [
    {"type": "LinearLR", "start_factor": 1.0e-5, "by_epoch": False, "begin": 0, "end": 300},
    {
        # use cosine lr from 20 to 100 epoch
        "type": "CosineAnnealingLR",
        "eta_min": _base_.base_lr * 0.05,
        "begin": max_epochs // 5,
        "end": max_epochs,
        "T_max": max_epochs * 4 // 5,
        "by_epoch": True,
        "convert_to_iter_based": True,
    },
]

# train_cfg
val_interval = 2
val_begin = 20

tta_model = None
tta_pipeline = None

visualizer = {"vis_backends": [{"type": "LocalVisBackend"}, {"type": "WandbVisBackend"}]}

# ===================== Unmodified in most cases ==================
model = {
    "bbox_head": {"head_module": {"num_classes": num_classes}},
    "train_cfg": {"assigner": {"num_classes": num_classes}},
}

train_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "ann_file": train_ann_file,
        "data_prefix": {"img": train_data_prefix},
    },
}

val_dataloader = {
    "batch_size": val_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "data_prefix": {"img": val_data_prefix},
        "ann_file": val_ann_file,
    },
}

test_dataloader = {
    "batch_size": val_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "data_prefix": {"img": test_data_prefix},
        "ann_file": test_ann_file,
    },
}

default_hooks = {
    "checkpoint": {"interval": save_epoch_intervals, "max_keep_ckpts": max_keep_ckpts, "save_best": "auto"}
}

val_evaluator = {"ann_file": data_root + val_ann_file}
test_evaluator = {"ann_file": data_root + test_ann_file}

train_cfg = {
    "type": "EpochBasedTrainLoop",
    "max_epochs": max_epochs,
    "val_begin": val_begin,
    "val_interval": val_interval,
}
