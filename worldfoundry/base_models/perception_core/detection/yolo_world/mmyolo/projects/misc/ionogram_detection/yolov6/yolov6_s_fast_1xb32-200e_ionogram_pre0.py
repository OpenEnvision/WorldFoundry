_base_ = "./yolov6_s_fast_1xb32-100e_ionogram.py"

# ======================= Modified parameters =====================
base_lr = _base_.base_lr * 4
optim_wrapper = {"optimizer": {"lr": base_lr}}
max_epochs = 200
load_from = None

# ==================== Unmodified in most cases ===================
train_cfg = {
    "max_epochs": max_epochs,
    "val_begin": 20,
}

default_hooks = {"param_scheduler": {"max_epochs": max_epochs}, "logger": {"type": "LoggerHook", "interval": 50}}
