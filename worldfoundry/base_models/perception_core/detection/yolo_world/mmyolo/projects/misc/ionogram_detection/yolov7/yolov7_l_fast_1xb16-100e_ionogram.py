_base_ = "mmyolo::yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py"

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

train_batch_size_per_gpu = 16
train_num_workers = 8

# -----model related-----
anchors = [[[14, 14], [35, 6], [32, 18]], [[32, 45], [28, 97], [52, 80]], [[71, 122], [185, 94], [164, 134]]]

# -----train val related-----
#  base_lr_default * (your_bs 32 / default_bs (8 x 16))
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 16)
load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth"

# default hooks
save_epoch_intervals = 10
max_epochs = 100
max_keep_ckpts = 1

# train_cfg
val_interval = 2
val_begin = 20

tta_model = None
tta_pipeline = None

visualizer = {"vis_backends": [{"type": "LocalVisBackend"}, {"type": "WandbVisBackend"}]}

# ===================== Unmodified in most cases ==================
model = {
    "bbox_head": {
        "head_module": {"num_classes": num_classes},
        "prior_generator": {"base_sizes": anchors},
        "loss_cls": {"loss_weight": _base_.loss_cls_weight * (num_classes / 80 * 3 / _base_.num_det_layers)},
    }
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
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "data_prefix": {"img": val_data_prefix},
        "ann_file": val_ann_file,
    },
}

test_dataloader = {
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "dataset": {
        "metainfo": metainfo,
        "data_root": data_root,
        "data_prefix": {"img": test_data_prefix},
        "ann_file": test_ann_file,
    },
}

optim_wrapper = {"optimizer": {"lr": base_lr, "batch_size_per_gpu": train_batch_size_per_gpu}}

default_hooks = {
    "param_scheduler": {"max_epochs": max_epochs},
    "checkpoint": {"interval": save_epoch_intervals, "max_keep_ckpts": max_keep_ckpts},
}

val_evaluator = {"ann_file": data_root + val_ann_file}
test_evaluator = {"ann_file": data_root + test_ann_file}

train_cfg = {
    "type": "EpochBasedTrainLoop",
    "max_epochs": max_epochs,
    "val_begin": val_begin,
    "val_interval": val_interval,
}
