_base_ = "./ppyoloe_plus_s_fast_8xb8-80e_coco.py"

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
checkpoint = (
    "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/cspresnet_s_imagenet1k_pretrained-2be81763.pth"
)

train_batch_size_per_gpu = 32
max_epochs = 300

# Base learning rate for optim_wrapper
base_lr = 0.01

model = {
    "data_preprocessor": {
        "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
        "std": [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
    },
    "backbone": {
        "block_cfg": {"use_alpha": False},
        "init_cfg": {"type": "Pretrained", "prefix": "backbone.", "checkpoint": checkpoint, "map_location": "cpu"},
    },
    "train_cfg": {"initial_epoch": 100},
}

train_dataloader = {"batch_size": train_batch_size_per_gpu}

optim_wrapper = {"optimizer": {"lr": base_lr}}

default_hooks = {"param_scheduler": {"total_epochs": int(max_epochs * 1.2)}}

train_cfg = {"max_epochs": max_epochs}

# PPYOLOE plus use obj365 pretrained model, but PPYOLOE not,
# `load_from` need to set to None.
load_from = None
