_base_ = "./yolov6_s_syncbn_fast_8xb32-400e_coco.py"

# ======================= Frequently modified parameters =====================
# -----train val related-----
# Base learning rate for optim_wrapper
max_epochs = 300  # Maximum training epochs
num_last_epochs = 15  # Last epoch number to switch training pipeline

# ============================== Unmodified in most cases ===================
default_hooks = {
    "param_scheduler": {
        "type": "YOLOv5ParamSchedulerHook",
        "scheduler_type": "cosine",
        "lr_factor": 0.01,
        "max_epochs": max_epochs,
    }
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
        "switch_epoch": max_epochs - num_last_epochs,
        "switch_pipeline": _base_.train_pipeline_stage2,
    },
]

train_cfg = {"max_epochs": max_epochs, "dynamic_intervals": [(max_epochs - num_last_epochs, 1)]}
